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

from vulkanvm_torch.chonk import (
    ChonkPool,
    build_chonk_cache,
    reset_chonk_cache,
    build_full_chonk_training_setup,
    load_model_into_chonk,
    create_optimizer_states_in_chonk,
    create_activation_buffers,
    estimate_model_memory,
)

# ROCm/Strix Halo memory config
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6"

# Training config
MODEL_PATH = "/home/chonke/local_training/models/qwen36_27ablit"
DATA_PATH = "/home/chonke/local_training/qwen_tokenized_128k"
SEQ_LEN = 131072
BATCH_SIZE = 1
CHUNK_SIZE = 4096  # 2K-4K recommended for ROCm stability
MAX_CACHE_LEN = 131072
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 10000
GRAD_ACCUM_STEPS = 4  # EXP 1: Effective batch = 4
GRAD_CLIP_NORM = 1.0  # EXP 1: Gradient clipping
LOG_INTERVAL = 10
SAVE_INTERVAL = 500

# Chonk Buffer config
# Total pool size: ~80-90GB (model ~22GB + optimizer ~44GB + KV ~10GB + activations ~8GB + staging ~2GB)


def get_tokenized_dataset(data_path, seq_len, batch_size):
    """Load tokenized dataset from disk."""
    import glob
    import numpy as np

    files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_path}")

    def generator():
        for f in files:
            data = np.load(f, mmap_mode="r")
            for i in range(0, len(data) - seq_len, seq_len):
                chunk = data[i:i + seq_len]
                if len(chunk) == seq_len:
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
        loss_fct = nn.CrossEntropyLoss()
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
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")

    # 2. Initialize Chonk Pool and build KV cache BEFORE loading model
    print("\n[2/6] Initializing Chonk Buffer pool...")
    pool = ChonkPool()
    print(f"  Device: {pool.device_name}")

    print("\n[3/6] Building KV cache in Chonk Buffer...")
    kv_cache = build_chonk_cache(config, BATCH_SIZE, MAX_CACHE_LEN, pool)
    print(f"  KV cache built for {len([l for l in kv_cache.layers if hasattr(l, 'keys')])} full-attention layers")

    # 3. Load model (after Chonk Buffer is built to prevent fragmentation)
    print("\n[4/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        device_map=None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.cuda()
    print(f"  Model loaded, CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 4. Prepare for k-bit training (LoRA will be added)
    print("\n[5/6] Preparing model for training...")
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    print("  Model prepared for k-bit training (gradient checkpointing disabled)")

    # Add LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  CUDA memory after LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 5. Load model weights into Chonk Buffer
    print("\n[6/6] Moving model weights to Chonk Buffer...")
    model_buffer = load_model_into_chonk(model, pool)
    print(f"  Model weights in Chonk Buffer")

    # 6. Create optimizer states in Chonk Buffer
    print("\n[7/6] Creating optimizer states in Chonk Buffer...")
    optimizer_states, opt_buffer = create_optimizer_states_in_chonk(model, pool)

    # 7. Create optimizer
    optimizer = ChonkAdamW(
        model.parameters(),
        optimizer_states,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    # 8. Create activation buffers
    print("\n[8/6] Creating activation buffers...")
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    act_buffer = create_activation_buffers(pool, BATCH_SIZE, SEQ_LEN, hidden_size, num_layers, chunk_size=CHUNK_SIZE)

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

            if loss is not None:
                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS
                loss.backward()

            chunks_this_seq += 1

            # Step optimizer after GRAD_ACCUM_STEPS chunks (or end of sequence)
            if chunks_this_seq % GRAD_ACCUM_STEPS == 0 or chunk_end == SEQ_LEN:
                # Gradient clipping
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log (count optimizer steps, not chunks)
            if step % LOG_INTERVAL == 0:
                stats = pool.stats()
                print(f"Step {step}: loss={loss.item() * GRAD_ACCUM_STEPS if loss is not None else 0:.4f}, "
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
    save_path = f"/home/chonke/local_training/qwen_fine_tuned/chonk_final"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nFinal model saved to {save_path}")

    # Cleanup
    pool.shutdown()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()