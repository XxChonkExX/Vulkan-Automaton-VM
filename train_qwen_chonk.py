#!/usr/bin/env python3
"""
Qwen 27B Training with Chonk Buffer (VulkanVM UnifiedMemoryPool)
================================================================

This script loads EVERYTHING into the Chonk Buffer:
- Model weights
- Optimizer states (AdamW fp32)
- KV Cache (full attention layers)
- Activation buffers (for chunked forward)
- Staging buffer (host-visible)

Hardware: Strix Halo 395 (128GB unified RAM, 2GB VRAM carve)
"""

import os
import sys
import time
import contextlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_utils import PreTrainedModel

# Add VulkanVM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "vulkanvm_torch"))

from chonk import (
    ChonkPool,
    build_chonk_cache,
    reset_chonk_cache,
    build_lora_chonk_setup,
    load_model_into_chonk,
    create_optimizer_states_in_chonk,
    create_activation_buffers,
    estimate_model_memory,
    patch_linear_cache_for_chunked_training,
    install_chonk_allocator,
)

# ROCm/Strix Halo memory config
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.4"

# Stability knobs (iGPU also drives the display; sustained compute can
# starve it and reset the driver -> login screen)
CHONK_COMPILE = os.environ.get("CHONK_COMPILE", "0") == "1"   # torch.compile (risky w/ fla kernels)
CHONK_AUTOCAST = os.environ.get("CHONK_AUTOCAST", "0") == "1"  # bf16 autocast in train_step
CHONK_PAUSE = float(os.environ.get("CHONK_PAUSE", "0.02"))     # sec pause per chunk (display breathing room)
CHONK_SMOKE = os.environ.get("CHONK_SMOKE", "0") == "1"        # smoke test: tiny seq, 2 steps
CHONK_ATTN = os.environ.get("CHONK_ATTN", "eager")             # eager (stable) | sdpa | flash_attention_2
CHONK_ALLOCATOR = os.environ.get("CHONK_ALLOCATOR", "1") == "1"  # pool-backed pluggable allocator for torch/HIP

# Training config
MODEL_PATH = "/home/chonke/local_training/models/qwen36_27ablit"
DATA_PATH = "/home/chonke/local_training/qwen_tokenized_128k"
SEQ_LEN = 131072
BATCH_SIZE = 1
CHUNK_SIZE = 1024  # 1024 validated stable; 2048 froze the machine, 4096 panicked
MAX_CACHE_LEN = 131072
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 10000
GRAD_ACCUM_STEPS = 4  # EXP 1: Effective batch = 4
GRAD_CLIP_NORM = 1.0  # EXP 1: Gradient clipping
LOG_INTERVAL = 10
SAVE_INTERVAL = 500

if CHONK_SMOKE:
    SEQ_LEN = int(os.environ.get("CHONK_SMOKE_SEQ", "2048"))
    MAX_STEPS = int(os.environ.get("CHONK_SMOKE_STEPS", "2"))
    CHUNK_SIZE = int(os.environ.get("CHONK_SMOKE_CHUNK", "1024"))

# Chonk Buffer config
# Total pool size: ~80-90GB (model ~22GB + optimizer ~44GB + KV ~10GB + activations ~8GB + staging ~2GB)


def get_tokenized_dataset(data_path, seq_len, batch_size):
    """Load tokenized dataset from disk (memmap tokens.bin + index.bin).
    Tokens are packed into fixed seq_len blocks (variable-length documents
    are concatenated, cache resets at block boundaries)."""
    import numpy as np

    tokens_path = os.path.join(data_path, "tokens.bin")
    index_path = os.path.join(data_path, "index.bin")
    if not os.path.exists(tokens_path):
        # Fallback: legacy .npy files
        import glob

        files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No data found in {data_path}")

        def gen_npy():
            for f in files:
                data = np.load(f, mmap_mode="r")
                for i in range(0, len(data) - seq_len, seq_len):
                    chunk = data[i:i + seq_len]
                    if len(chunk) == seq_len:
                        yield torch.from_numpy(chunk.astype(np.int64))

        return gen_npy()

    tokens = np.memmap(tokens_path, dtype=np.uint32, mode="r")
    index = np.memmap(index_path, dtype=np.int64, mode="r")
    print(f"  Dataset: {len(tokens):,} tokens, {len(index) - 1:,} sequences "
          f"(packed into {len(tokens) // seq_len:,} blocks of {seq_len:,})")

    def generator():
        n_blocks = len(tokens) // seq_len
        for b in range(n_blocks):
            start = b * seq_len
            chunk = tokens[start:start + seq_len]
            yield torch.from_numpy(chunk.astype(np.int64))

    return generator()


class ChonkAdamW(torch.optim.Optimizer):
    """AdamW optimizer that uses pre-allocated states in Chonk Buffer."""

    def __init__(self, params, optimizer_states, lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.optimizer_states = optimizer_states
        self.step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        self.step_count += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.optimizer_states.get(p)
                if state is None:
                    # Fallback (should not happen with Chonk Buffer)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** self.step_count
                bias_correction2 = 1 - beta2 ** self.step_count

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class EMAModel:
    """Exponential Moving Average of model weights for better final quality."""
    
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.model = model
        self.register()
    
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def patch_sdpa_for_chonk():
    """Patch scaled dot product attention to work with Chonk Cache."""
    from torch.nn.functional import scaled_dot_product_attention as orig_sdpa

    def patched_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        # Use original SDPA but ensure tensors are contiguous
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        return orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched_sdpa


def train_step(model, chunk_ids, kv_cache, optimizer, chunk_start, chunk_end, seq_len):
    """Single training step on a chunk."""
    cp = torch.arange(chunk_start, chunk_end, device="cuda")

    if CHONK_AUTOCAST:
        # EXP 4: BF16 autocast for forward pass (off by default: untested with Chonk path)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        context = contextlib.nullcontext()

    with context:
        outputs = model(
            input_ids=chunk_ids,
            past_key_values=kv_cache,
            use_cache=True,
            cache_position=cp,
        )

        logits = outputs.logits
        loss = None

        # Next-token prediction loss
        if logits is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = chunk_ids[..., 1:].contiguous()
            # EXP 7: Label smoothing
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

    return loss, outputs


def main():
    print("=" * 60)
    print("Qwen 27B Chonk Buffer Training")
    print("=" * 60)
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Max cache len: {MAX_CACHE_LEN}")
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_PATH}")
    print("=" * 60)

    # 1. Load config first (to estimate memory)
    print("\n[1/6] Loading model config...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    text_cfg = config.get_text_config(decoder=True)
    print(f"  Hidden size: {text_cfg.hidden_size}")
    print(f"  Num layers: {text_cfg.num_hidden_layers}")
    print(f"  Num heads: {text_cfg.num_attention_heads}")
    print(f"  Vocab size: {text_cfg.vocab_size}")

    # 2/3. Everything happens inside build_lora_chonk_setup (single pool):
    #      KV cache, base weights, LoRA, optimizer states, staging
    print("\n[2/6] Building LoRA training setup in Chonk Buffer...")
    if CHONK_ALLOCATOR:
        print("  [+] Installing pool-backed pluggable allocator "
              "(torch segments draw from the Chonk Buffer)")
        install_chonk_allocator()
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, BATCH_SIZE, MAX_CACHE_LEN,
        lora_r=64, lora_alpha=128, lora_dropout=0.05,
        attn_implementation=CHONK_ATTN,
    )
    pool = setup["pool"]
    kv_cache = setup["kv_cache"]
    model = setup["model"]
    optimizer_states = setup["optimizer_states"]
    print(f"  Device: {pool.device_name}")
    # Release torch's setup-phase cached segments back to the pool so the
    # first forward doesn't spike on top of them (display shares the heap).
    torch.cuda.empty_cache()
    s = pool.stats()
    print(f"  Pool after setup+empty_cache: {s['totalUsed'] / 1e9:.2f} GB used, "
          f"{s['allocationCount']} allocations")
    print(f"  KV cache built for {len([l for l in kv_cache.layers if hasattr(l, 'keys')])} full-attention layers")
    model.print_trainable_parameters()

    # Patch linear-attention cache for chunked forward/backward (truncated BPTT)
    patch_linear_cache_for_chunked_training()

    if CHONK_COMPILE:
        # EXP 2: torch.compile for speedup (off by default: fla kernels are risky under dynamo)
        print("\n[Exp 2] Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("  Model compiled successfully")

    # 6. Create optimizer
    optimizer = ChonkAdamW(
        [p for p in model.parameters() if p.requires_grad],
        optimizer_states,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # EXP 7: EMA for better final quality
    ema = EMAModel(model, decay=0.9999)
    print("  EMA model initialized (decay=0.9999)")

    # 9. Patch SDPA
    patch_sdpa_for_chonk()
    print("  SDPA patched for Chonk Cache")

    # 10. Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # 11. Load dataset
    print("\n[9/6] Loading dataset...")
    dataset_gen = get_tokenized_dataset(DATA_PATH, SEQ_LEN, BATCH_SIZE)

    # 12. Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    step = 0
    model.train()
    t_seq_start = time.time()

    for input_ids in dataset_gen:
        if step >= MAX_STEPS:
            break

        input_ids = input_ids.unsqueeze(0).cuda()  # [1, seq_len]

        # Reset cache for new sequence
        reset_chonk_cache(kv_cache)

        # Gradient accumulation: track chunks per sequence
        chunks_this_seq = 0

        # Process in chunks
        for chunk_start in range(0, SEQ_LEN, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, SEQ_LEN)
            chunk_ids = input_ids[:, chunk_start:chunk_end]

            # Forward
            loss, outputs = train_step(model, chunk_ids, kv_cache, optimizer, chunk_start, chunk_end, SEQ_LEN)

            if CHONK_PAUSE > 0:
                time.sleep(CHONK_PAUSE)  # give the display pipeline breathing room

            last_loss = float("nan")
            if loss is not None:
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS
                last_loss = loss.item()
                loss.backward()
                del loss, outputs  # free the 2GB logits + graph ASAP

            chunks_this_seq += 1

            # Heartbeat + HIP heap defrag (keeps the driver heap from
            # fragmenting as the cache grows; system memory is shared)
            if chunks_this_seq % 8 == 0:
                torch.cuda.empty_cache()
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"  [seq] chunk {chunks_this_seq}/{SEQ_LEN // CHUNK_SIZE} "
                      f"({time.time() - t_seq_start:.0f}s, loss={last_loss:.4f}, "
                      f"hip_peak={peak_gb:.1f}GB)", flush=True)
                torch.cuda.reset_peak_memory_stats()

            # Step optimizer after GRAD_ACCUM_STEPS chunks (or end of sequence)
            if chunks_this_seq % GRAD_ACCUM_STEPS == 0 or chunk_end == SEQ_LEN:
                # Gradient clipping
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # EXP 7: Update EMA
                ema.update()

            # Log (count optimizer steps, not chunks)
            if step % LOG_INTERVAL == 0:
                stats = pool.stats()
                print(f"Step {step}: loss={last_loss * GRAD_ACCUM_STEPS:.4f}, "
                      f"lr={scheduler.get_last_lr()[0]:.2e}, pool_used={stats['totalUsed'] / 1e9:.2f} GB")

            step += 1

            if step >= MAX_STEPS:
                break

        # Save checkpoint
        if step % SAVE_INTERVAL == 0 and step > 0:
            save_path = f"/home/chonke/local_training/qwen_fine_tuned/chonk_step_{step}"
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            print(f"Checkpoint saved to {save_path}")

    # Final save
    # Apply EMA weights for final model
    ema.apply_shadow()
    save_path = f"/home/chonke/local_training/qwen_fine_tuned/chonk_final"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nFinal model saved to {save_path} (EMA weights applied)")

    # Cleanup
    pool.shutdown()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()