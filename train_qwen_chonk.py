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
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
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

# ROCm/Strix Halo memory config.
# NOTE: expandable_segments MUST stay False — (a) the pluggable allocator path
# bypasses expandable segments anyway (no alloc/free hooks to swap), and
# (b) expandable_segments:True crashes on ROCm/kernel 7.0
# (pytorch/pytorch#187343, drm_suballoc_helper splits) — the recurring
# display-driver resets.
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False,garbage_collection_threshold:0.4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False,garbage_collection_threshold:0.4"

# Stability knobs (iGPU also drives the display; sustained compute can
# starve it and reset the driver -> login screen)
CHONK_COMPILE = os.environ.get("CHONK_COMPILE", "0") == "1"   # torch.compile (risky w/ fla kernels)
CHONK_AUTOCAST = os.environ.get("CHONK_AUTOCAST", "0") == "1"  # bf16 autocast in train_step
CHONK_PAUSE = float(os.environ.get("CHONK_PAUSE", "0.05"))     # sec pause per chunk (display breathing room)
CHONK_SMOKE = os.environ.get("CHONK_SMOKE", "0") == "1"        # smoke test: tiny seq, 2 steps
CHONK_ATTN = os.environ.get("CHONK_ATTN", "eager")             # eager (stable) | sdpa | flash_attention_2
CHONK_ATTN_RECOMPUTE = os.environ.get("CHONK_ATTN_RECOMPUTE", "1") == "1"  # attention checkpointing (THE fix for the ~1.5MB/token eager-graph growth)
CHONK_ALLOCATOR = os.environ.get("CHONK_ALLOCATOR", "1") == "1"  # pool-backed pluggable allocator for torch/HIP
CHONK_QUANTIZE = os.environ.get("CHONK_QUANTIZE", "0") == "1"    # INT8/INT4-quantize LoRA target weights in the pool
CHONK_QUANT_GROUP = int(os.environ.get("CHONK_QUANT_GROUP", "128"))
CHONK_QUANT_BITS = int(os.environ.get("CHONK_QUANT_BITS", "8"))  # 8 or 4 bits per weight (4 = packed 2/byte, ~half the pool footprint)
CHONK_ACT_GB = float(os.environ.get("CHONK_ACT_GB", "2.0"))      # activation scratch budget (mostly unused)
CHONK_STAGING_GB = float(os.environ.get("CHONK_STAGING_GB", "2.0"))  # host-visible staging (unused w/o offload)

# Training config
MODEL_PATH = os.environ.get("CHONK_MODEL_PATH", "/home/chonke/local_training/models/Qwen3.8-AEON-Ultimate")
DATA_PATH = os.environ.get("CHONK_DATA_PATH", "/home/chonke/local_training/qwen_tokenized_128k")
OUT_DIR = os.environ.get("CHONK_OUT_DIR", "/home/chonke/local_training/qwen_fine_tuned")
STATUS_FILE = os.environ.get("CHONK_STATUS_FILE", "/home/chonke/local_training/qwen_logs/train_status.txt")
SEQ_LEN = int(os.environ.get("CHONK_SEQ_LEN", "131072"))     # 262144 = the long-context target
BATCH_SIZE = 1
CHUNK_SIZE = int(os.environ.get("CHONK_CHUNK", "1024"))  # 1024 validated stable; 2048 froze the machine, 4096 panicked
MAX_CACHE_LEN = int(os.environ.get("CHONK_MAX_CACHE_LEN", str(SEQ_LEN)))
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 10000
GRAD_ACCUM_STEPS = int(os.environ.get("CHONK_GRAD_ACCUM", "4"))  # Effective batch = 4
GRAD_CLIP_NORM = 1.0  # EXP 1: Gradient clipping
LOG_INTERVAL = 10
SAVE_INTERVAL = int(os.environ.get("CHONK_SAVE_INTERVAL", "100"))
KEEP_CHECKPOINTS = int(os.environ.get("CHONK_KEEP_CHECKPOINTS", "3"))

# Live training: epoch-based with block subsampling (for 262K feasibility)
CHONK_SUBSAMPLE = float(os.environ.get("CHONK_SUBSAMPLE", "1.0"))  # 1.0 = all blocks, 0.05 = 5%
CHONK_EPOCHS = int(os.environ.get("CHONK_EPOCHS", "1"))

# Display-starvation mitigations (Strix Halo iGPU drives display)
CHONK_PAUSE = float(os.environ.get("CHONK_PAUSE", "0.02"))     # sec pause per chunk (display breathing room)
CHONK_OPTIMIZER_PAUSE = float(os.environ.get("CHONK_OPTIMIZER_PAUSE", "0.5"))  # extra pause on optimizer step
CHONK_EMA_UPDATE_EVERY = int(os.environ.get("CHONK_EMA_UPDATE_EVERY", "1"))  # update EMA every N optimizer steps

if CHONK_SMOKE:
    SEQ_LEN = int(os.environ.get("CHONK_SMOKE_SEQ", "2048"))
    MAX_STEPS = int(os.environ.get("CHONK_SMOKE_STEPS", "2"))
    CHUNK_SIZE = int(os.environ.get("CHONK_SMOKE_CHUNK", "1024"))

# Chonk Buffer config
# Total pool size: ~80-90GB (model ~22GB + optimizer ~44GB + KV ~10GB + activations ~8GB + staging ~2GB)


def get_tokenized_dataset(data_path, seq_len, batch_size, subsample=1.0, epoch=0):
    """Load tokenized dataset from disk (memmap tokens.bin + index.bin).
    Tokens are packed into fixed seq_len blocks (variable-length documents
    are concatenated, cache resets at block boundaries).
    
    Args:
        subsample: fraction of blocks to use (1.0 = all, 0.05 = 5%)
        epoch: epoch number for deterministic random seed
    """
    import numpy as np
    import random

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
    n_blocks = len(tokens) // seq_len
    print(f"  Dataset: {len(tokens):,} tokens, {len(index) - 1:,} sequences "
          f"(packed into {n_blocks:,} blocks of {seq_len:,})")

    if subsample >= 1.0:
        # Use all blocks
        block_indices = list(range(n_blocks))
    else:
        # Deterministic random subsample per epoch
        n_take = max(1, int(n_blocks * subsample))
        rng = random.Random(epoch * 1337 + 42)
        block_indices = rng.sample(range(n_blocks), n_take)
        block_indices.sort()
        print(f"  Subsample {subsample:.1%}: {len(block_indices):,} blocks (epoch {epoch})")

    def generator():
        for b in block_indices:
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

                # Decoupled weight decay (standard AdamW): decay the weights
                # directly instead of folding decay into the gradient, so the
                # moment estimates stay unbiased by the L2 term.
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

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


def patch_eager_attention_recompute(model, kv_cache):
    """Route eager attention through the recompute-checkpointed autograd fn.

    Plain eager saves the fp32 softmax probabilities per layer
    ((B,H,q,pos) -> grows with position: ~1.5MB/token across 24 layers),
    which is the per-chunk slab growth that OOMs at ~112GB (~chunk 25).
    With recompute, backward re-derives the probs from q/k/v instead of
    saving them (attention checkpointing; attention math runs in bf16 to
    halve the position-proportional tensors, grads return in fp32).
    Requires _build/vulkanvm_attn.so.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_build"))
        import vulkanvm_attn as vvm_attn
    except ImportError:
        print("[warn] vulkanvm_attn.so not built; falling back to plain eager")
        return
    vvm_attn.set_attention_recompute(True)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    first = next(l for l in base.model.layers if hasattr(l, "self_attn"))
    attn_mod = type(first.self_attn).__module__
    mod = sys.modules[attn_mod]
    orig = mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if dropout > 0:
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
        # Chunked cache: key/value are the full cat [cached | current]. Pass
        # zero-copy views of the constant cached span so the autograd fn
        # saves only the current chunk's rows (~1MB) instead of the whole
        # cat (2KB/pos x layers -> 8+ GB at 131K).
        layer = kv_cache.layers[module.layer_idx]
        cached_k = cached_v = torch.empty(0, device=query.device, dtype=torch.bfloat16)
        if hasattr(layer, "keys"):
            cur_len = key.shape[-2] - query.shape[-2]
            if cur_len > 0:
                # Views of the POOL CELLS (permanent storage, retention-free);
                # detached: the cells' shared storage is written by every
                # layer's update, which would trip autograd's inplace-version
                # check on the saved views (data is constant at backward).
                cached_k = layer.keys[: key.shape[0], :, :cur_len].detach()
                cached_v = layer.values[: key.shape[0], :, :cur_len].detach()
        out = vvm_attn.vulkan_attention(query, key, value, scaling, attention_mask,
                                        kv_repeat=module.num_key_value_groups,
                                        cached_k=cached_k, cached_v=cached_v)
        out = out.transpose(1, 2).contiguous()
        return out, None

    mod.eager_attention_forward = patched
    print("[+] eager attention patched: fp32 attn probs recomputed in backward (not saved)")


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


def cleanup_checkpoints(out_dir, keep=3):
    """Keep the best (lowest loss), the latest, and the second-latest
    checkpoints; delete the rest to save disk. chonk_final is never deleted."""
    import glob
    steps = sorted(int(os.path.basename(p)[len("chonk_step_"):])
                   for p in glob.glob(f"{out_dir}/chonk_step_*"))
    if len(steps) <= keep:
        return
    best_step = None
    best_loss = float("inf")
    for s in steps:
        try:
            with open(f"{out_dir}/chonk_step_{s}/.chonk_loss") as f:
                loss = float(f.read().strip())
            if loss < best_loss:
                best_loss = loss
                best_step = s
        except (OSError, ValueError):
            continue
    protected = {max(steps), min(steps[-2:], default=None), best_step}
    protected.discard(None)
    for s in steps:
        if s not in protected:
            shutil.rmtree(f"{out_dir}/chonk_step_{s}", ignore_errors=True)
            print(f"  [cleanup] removed chonk_step_{s}")


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
    # Qwen3.5/3.6/3.8 multimodal wrappers: text-only everywhere (loader,
    # quantize-module listing, buffer rebuild all use from_config).
    config.language_model_only = True
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
        quantize=CHONK_QUANTIZE, quant_group_size=CHONK_QUANT_GROUP,
        quant_bits=CHONK_QUANT_BITS,
        act_budget_gb=CHONK_ACT_GB, staging_gb=CHONK_STAGING_GB,
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

    if CHONK_ATTN == "eager" and CHONK_ATTN_RECOMPUTE:
        patch_eager_attention_recompute(model, kv_cache)

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
    # Lazy import: transformers' schedule module initializes torch CUDA at
    # import time, which breaks the Chonk allocator swap.
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Calculate total steps for scheduler (per epoch blocks × epochs)
    import numpy as np
    n_blocks_full = len(np.memmap(os.path.join(DATA_PATH, "tokens.bin"), dtype=np.uint32, mode="r")) // SEQ_LEN
    n_blocks_per_epoch = max(1, int(n_blocks_full * CHONK_SUBSAMPLE))
    total_steps = n_blocks_per_epoch * CHONK_EPOCHS
    print(f"  Total steps: {total_steps} ({n_blocks_per_epoch} blocks/epoch × {CHONK_EPOCHS} epochs)")

    # 11. Learning rate scheduler
    # Lazy import: transformers' schedule module initializes torch CUDA at
    # import time, which breaks the Chonk allocator swap.
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps
    )

    # 12. Training loop (epoch-based with subsampling)
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    step = 0
    model.train()

    for epoch in range(CHONK_EPOCHS):
        dataset_gen = get_tokenized_dataset(DATA_PATH, SEQ_LEN, BATCH_SIZE, CHONK_SUBSAMPLE, epoch)
        t_seq_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{CHONK_EPOCHS} ---")

        for input_ids in dataset_gen:
            if step >= total_steps:
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

            # Return ALL cached segments to the slab EVERY chunk. The saved
            # fp32 attn_weights grow with position (512 x pos x 16 layers), so
            # every chunk's request is bigger than the last; torch's cached
            # smaller segments never fit, forcing fresh 2GB blocks. Emptying
            # the cache each chunk coalesces the whole slab so each chunk
            # re-carves the same blocks (growth ~= intrinsic delta, ~4GB at
            # 131K, not ~1GB/chunk -> OOM at 112GB). Blocks are never
            # released (warmBlocks=512), so this is pure slab recycling.
            torch.cuda.empty_cache()

            # Heartbeat
            if chunks_this_seq % 8 == 0:
                ps = pool.stats()
                print(f"  [seq] chunk {chunks_this_seq}/{SEQ_LEN // CHUNK_SIZE} "
                      f"({time.time() - t_seq_start:.0f}s, loss={last_loss:.4f}, "
                      f"pool={ps['totalUsed'] / 1e9:.2f}GB)", flush=True)

            # Status file (atomic rename): live numbers for the heartbeat
            # monitor without touching the (possibly terminal-bound) log.
            if chunks_this_seq % 8 == 0:
                status = (f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                          f"step={step}\n"
                          f"chunk={chunks_this_seq}/{SEQ_LEN // CHUNK_SIZE}\n"
                          f"loss={last_loss:.4f}\n"
                          f"lr={scheduler.get_last_lr()[0]:.2e}\n"
                          f"pool_gb={ps['totalUsed'] / 1e9:.2f}\n")
                tmp = STATUS_FILE + ".tmp"
                with open(tmp, "w") as f:
                    f.write(status)
                os.replace(tmp, STATUS_FILE)

            # Step optimizer after GRAD_ACCUM_STEPS chunks (or end of sequence)
            if chunks_this_seq % GRAD_ACCUM_STEPS == 0 or chunk_end == SEQ_LEN:
                # Gradient clipping
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EXP 7: Update EMA (reduced frequency to reduce kernel pressure)
                if CHONK_EMA_UPDATE_EVERY > 0 and step % CHONK_EMA_UPDATE_EVERY == 0:
                    ema.update()

                # Extra pause around optimizer step to let display driver breathe
                if CHONK_OPTIMIZER_PAUSE > 0:
                    time.sleep(CHONK_OPTIMIZER_PAUSE)

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
            save_path = f"{OUT_DIR}/chonk_step_{step}"
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            saved_loss = last_loss * GRAD_ACCUM_STEPS
            with open(f"{save_path}/.chonk_loss", "w") as f:
                f.write(f"{saved_loss:.6f}")
            print(f"Checkpoint saved to {save_path} (loss={saved_loss:.4f})")
            cleanup_checkpoints(OUT_DIR, KEEP_CHECKPOINTS)

    # Final save
    # Apply EMA weights for final model
    ema.apply_shadow()
    save_path = f"{OUT_DIR}/chonk_final"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nFinal model saved to {save_path} (EMA weights applied)")

    # Cleanup
    pool.shutdown()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()