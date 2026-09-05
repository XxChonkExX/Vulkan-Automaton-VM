#!/usr/bin/env python3
"""
train_granite_chonk.py — QLoRA fine-tune of the abliterated Granite 4.2 (30B)
entirely inside the Vulkan Chonk Buffer (unified memory, no ROCm alloc).

Pipeline (all in one Chonk pool):
  base weights (INT4 quantized)  +  LoRA adapters  +  KV cache  +
  AdamW fp32 optimizer states (in-pool)  +  activation scratch.

Adapted from train_qwen_chonk.py. Uses build_lora_chonk_setup() which performs
the same Granite loading sequence the abliteration script used.

Prereq: run prepare_dataset.py first -> DATA_PATH/tokens.bin + index.bin
"""
import os
import sys
import time
import random
import contextlib
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

# --- Chonk allocator MUST be installed before any CUDA tensor exists ---
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.4"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.4"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "vulkanvm_torch"))

from chonk import (
    build_lora_chonk_setup,
    reset_chonk_cache,
    create_optimizer_states_in_chonk,
    patch_linear_cache_for_chunked_training,
    install_chonk_allocator,
    release_empty_blocks,
    slab_stats,
    live_histogram,
)

# ---- config knobs (env-overridable) ----
MODEL_PATH = os.environ.get("CHONK_MODEL_PATH", "/home/chonke/Downloads/granite-abliterated")
DATA_PATH  = os.environ.get("CHONK_DATA_PATH", "examples/granite_chonk/data/granite-sft")
OUT_DIR    = os.environ.get("CHONK_OUT_DIR", "examples/granite_chonk/out/granite-finetuned")
RESUME_DIR = os.environ.get("CHONK_RESUME_DIR", "")
STATUS_FILE = os.environ.get("CHONK_STATUS_FILE", "examples/granite_chonk/out/train_status.txt")

SEQ_LEN = int(os.environ.get("CHONK_SEQ_LEN", "131072"))   # Granite native context (131072)
CHUNK_SIZE = int(os.environ.get("CHONK_CHUNK", "2048"))     # max safe chunk (<=2048 avoids kernel panic)
MAX_CACHE_LEN = int(os.environ.get("CHONK_MAX_CACHE_LEN", str(SEQ_LEN)))
BATCH_SIZE = 1
LEARNING_RATE = float(os.environ.get("CHONK_LR", "2e-5"))
WEIGHT_DECAY = 0.01
WARMUP_STEPS = int(os.environ.get("CHONK_WARMUP", "50"))
MAX_STEPS = int(os.environ.get("CHONK_MAX_STEPS", "2000"))
GRAD_ACCUM_STEPS = int(os.environ.get("CHONK_GRAD_ACCUM", "4"))
GRAD_CLIP_NORM = 1.0
LOG_INTERVAL = 10
SAVE_INTERVAL = int(os.environ.get("CHONK_SAVE_INTERVAL", "100"))
KEEP_CHECKPOINTS = int(os.environ.get("CHONK_KEEP_CHECKPOINTS", "8"))

CHONK_QUANT_BITS = int(os.environ.get("CHONK_QUANT_BITS", "4"))   # base weight quant
CHONK_QUANT_GROUP = int(os.environ.get("CHONK_QUANT_GROUP", "128"))
CHONK_LORA_R = int(os.environ.get("CHONK_LORA_R", "128"))
CHONK_LORA_ALPHA = int(os.environ.get("CHONK_LORA_ALPHA", "256"))
CHONK_ATTN = os.environ.get("CHONK_ATTN", "eager")
CHONK_ATTN_RECOMPUTE = os.environ.get("CHONK_ATTN_RECOMPUTE", "1") == "1"
CHONK_PAUSE = float(os.environ.get("CHONK_PAUSE", "0.05"))          # display driver breath
CHONK_OPTIMIZER_PAUSE = float(os.environ.get("CHONK_OPTIMIZER_PAUSE", "1.0"))
CHONK_ACT_GB = float(os.environ.get("CHONK_ACT_GB", "0.25"))            # activation scratch (log optimum)
CHONK_STAGING_GB = float(os.environ.get("CHONK_STAGING_GB", "0.25"))    # host-visible staging
CHONK_EMA_UPDATE_EVERY = int(os.environ.get("CHONK_EMA_UPDATE_EVERY", "1"))
CHONK_SUBSAMPLE = float(os.environ.get("CHONK_SUBSAMPLE", "1.0"))
CHONK_EPOCHS = int(os.environ.get("CHONK_EPOCHS", "1"))
CHONK_SMOKE = os.environ.get("CHONK_SMOKE", "0") == "1"
CHONK_GRADIENT_CHECKPOINT = os.environ.get("CHONK_GRADIENT_CHECKPOINT", "1") == "1"

if CHONK_SMOKE:
    SEQ_LEN = int(os.environ.get("CHONK_SMOKE_SEQ", "2048"))
    MAX_STEPS = int(os.environ.get("CHONK_SMOKE_STEPS", "2"))
    CHUNK_SIZE = int(os.environ.get("CHONK_SMOKE_CHUNK", "1024"))


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def get_tokenized_dataset(data_path, seq_len, subsample=1.0, epoch=0):
    tokens = np.memmap(os.path.join(data_path, "tokens.bin"), dtype=np.uint32, mode="r")
    index = np.memmap(os.path.join(data_path, "index.bin"), dtype=np.int64, mode="r")
    n_blocks = len(tokens) // seq_len
    print(f"  Dataset: {len(tokens):,} tokens, {len(index):,} sequences "
          f"(packed into {n_blocks:,} blocks of {seq_len:,})")
    if subsample >= 1.0:
        block_indices = list(range(n_blocks))
    else:
        n_take = max(1, int(n_blocks * subsample))
        rng = random.Random(epoch * 1337 + 42)
        block_indices = sorted(rng.sample(range(n_blocks), n_take))
    import random
    def generator():
        for b in block_indices:
            yield torch.from_numpy(tokens[b * seq_len:(b + 1) * seq_len].astype(np.int64))
    return generator()


# --------------------------------------------------------------------------
# Chonk AdamW (uses in-pool fp32 optimizer states)
# --------------------------------------------------------------------------
class ChonkAdamW(torch.optim.Optimizer):
    def __init__(self, params, optimizer_states, lr=2e-5, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
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
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.optimizer_states.get(p)
                if state is None:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                ea, eas = state["exp_avg"], state["exp_avg_sq"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                ea.mul_(beta1).add_(grad, alpha=1 - beta1)
                eas.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bc1 = 1 - beta1 ** self.step_count
                bc2 = 1 - beta2 ** self.step_count
                step_size = lr / bc1
                denom = (eas.sqrt() / (bc2 ** 0.5)).add_(eps)
                p.addcdiv_(ea, denom, value=-step_size)
        return loss


class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay, self.model = decay, model
        self.shadow, self.backup = {}, {}
        self.register()
    def register(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data
                p.data = self.shadow[n]
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


def patch_eager_attention_recompute(model, kv_cache):
    # NEW: use Granite-specific pure-torch recompute (our Vulcan/Chonk path,
    # NOT AMD fused SDPA which crashes the display driver through dma-buf).
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(
        "vulkanvm_attn_granite",
        os.path.join(os.path.dirname(__file__), "..", "..", "python", "vulkanvm_torch", "vulkanvm_attn_granite.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["vulkanvm_attn_granite"] = m
    spec.loader.exec_module(m)
    m.patch_granite_attention_recompute(model, kv_cache)


def enable_projection_checkpointing(model):
    """Recompute (not store) the MLP gate/up/SiLU/down intermediates.

    The MLP is a PURE function of its input (no KV-cache side effect), so it is
    safe to wrap WHOLE mlp.forward with torch.utils.checkpoint: the gate/up/down
    activations ([1,512,32768] bf16 each ~= 32MB) are recomputed in backward
    instead of stored, cutting ~6GB of kc-independent transient.

    IMPORTANT: checkpoint the WHOLE mlp.forward, NOT each projection separately.
    Per-projection checkpointing (192 units) saves 3 inputs per layer and leaves
    the 32MB gate/up OUTPUT activations materialized BETWEEN the checkpoints,
    so the peak went UP ~8GB (chunk-1 baseline 49.95 -> 58.00GB). One unit per
    layer (64) recomputes the whole MLP, so no 32MB intermediate is ever stored.

    We deliberately do NOT use transformers' gradient_checkpointing_enable(),
    which checkpoints the whole decoder layer INCLUDING self_attn -> the KV
    cache update() side effect re-runs during recompute and corrupts the cache.
    """
    import torch.utils.checkpoint as ckpt

    def make_checked(orig_fn):
        def checked(x):
            return ckpt.checkpoint(orig_fn, x, use_reentrant=False)
        return checked

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    n = 0
    for layer in base.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "_chonk_checked", False):
            continue
        orig = mlp.forward
        mlp.forward = make_checked(orig)
        mlp._chonk_checked = True
        n += 1
    print(f"[+] MLP gradient checkpointing enabled on {n} layers "
          f"(whole-mlp recompute; ~6GB kc-independent transient removed)")


def train_step(model, chunk_ids, kv_cache, chunk_start, chunk_end):
    cp = torch.arange(chunk_start, chunk_end, device="cuda")
    outputs = model(input_ids=chunk_ids, past_key_values=kv_cache,
                    use_cache=True, cache_position=cp)
    logits = outputs.logits
    loss = None
    if logits is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = chunk_ids[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss, outputs


def cleanup_checkpoints(out_dir, keep=8):
    import glob
    steps = sorted(int(os.path.basename(p)[len("chonk_step_"):])
                   for p in glob.glob(f"{out_dir}/chonk_step_*"))
    if len(steps) <= keep:
        return
    best, best_loss = None, float("inf")
    for s in steps:
        try:
            l = float(open(f"{out_dir}/chonk_step_{s}/.chonk_loss").read().strip())
            if l < best_loss:
                best_loss, best = l, s
        except (OSError, ValueError):
            continue
    # Keep the newest `keep` checkpoints plus the best-seen one. (The old
    # code kept only {max, second-last, best} regardless of `keep`.)
    # chonk_best/ and best_loss.txt are separate paths, never touched here.
    keep_set = set(steps[-keep:])
    if best is not None:
        keep_set.add(best)
    for s in steps:
        if s not in keep_set:
            shutil.rmtree(f"{out_dir}/chonk_step_{s}", ignore_errors=True)


def main():
    # Process-level single-instance guard. The wrapper flock only covers
    # wrapper-vs-wrapper; a direct `python train_granite_chonk.py` (experiments,
    # VT launches) would otherwise run a second 30B trainer alongside
    # production -> instant OOM + session kill. Uses its OWN lock file: it must
    # differ from the wrapper's, because flock descriptions conflict even
    # across parent/child sharing an inherited fd (the trainer would refuse
    # against its own wrapper). Held for process lifetime; the kernel releases
    # it on death (no staleness).
    global _train_lock_fh
    try:
        import fcntl
        _lock_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", ".train_process.lock"))
        _train_lock_fh = open(_lock_path, "w")
        fcntl.flock(_train_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[guard] another training process is already running; refusing "
              "a second instance (two trainers = OOM).", flush=True)
        sys.exit(0)
    print("=" * 60)
    print("Granite 30B QLoRA — Chonk Buffer Training")
    print("=" * 60)
    print(f"model={MODEL_PATH}\ndata={DATA_PATH}\nseq={SEQ_LEN} chunk={CHUNK_SIZE}")

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if CHONK_ATTN == "eager" and CHONK_ATTN_RECOMPUTE:
        config._attn_implementation = "eager"

    print("\n[1/5] Installing Chonk allocator + building LoRA setup...")
    install_chonk_allocator()
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, BATCH_SIZE, MAX_CACHE_LEN,
        lora_r=CHONK_LORA_R, lora_alpha=CHONK_LORA_ALPHA, lora_dropout=0.05,
        attn_implementation=CHONK_ATTN,
        quantize=True, quant_group_size=CHONK_QUANT_GROUP, quant_bits=CHONK_QUANT_BITS,
        act_budget_gb=CHONK_ACT_GB, staging_gb=CHONK_STAGING_GB,
    )
    pool = setup["pool"]
    kv_cache = setup["kv_cache"]
    model = setup["model"]
    optimizer_states = setup["optimizer_states"]
    torch.cuda.empty_cache()
    model.print_trainable_parameters()
    patch_linear_cache_for_chunked_training()
    if CHONK_ATTN == "eager" and CHONK_ATTN_RECOMPUTE:
        patch_eager_attention_recompute(model, kv_cache)
    if CHONK_GRADIENT_CHECKPOINT:
        enable_projection_checkpointing(model)

    optimizer = ChonkAdamW([p for p in model.parameters() if p.requires_grad],
                           optimizer_states, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ema = EMAModel(model, decay=0.9999)

    from transformers import get_cosine_schedule_with_warmup
    n_blocks_full = len(np.memmap(os.path.join(DATA_PATH, "tokens.bin"),
                                  dtype=np.uint32, mode="r")) // SEQ_LEN
    n_blocks_per_epoch = max(1, int(n_blocks_full * CHONK_SUBSAMPLE))
    # OPTIMIZER-STEP ACCOUNTING (critical): `step` counts OPTIMIZER steps
    # (one per GRAD_ACCUM_STEPS chunks), NOT chunks. total_steps, MAX_STEPS,
    # SAVE_INTERVAL, LOG_INTERVAL and the cosine schedule are all in units of
    # optimizer steps. (Previously step counted chunks, which stopped training
    # after total_steps CHUNKS — ~0.4% of one epoch — and desynced the
    # scheduler by GRAD_ACCUM_STEPS x.)
    chunks_per_block = (SEQ_LEN + CHUNK_SIZE - 1) // CHUNK_SIZE
    opt_steps_per_block = (chunks_per_block + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = n_blocks_per_epoch * opt_steps_per_block * CHONK_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    print(f"  total_steps={total_steps} ({n_blocks_per_epoch} blocks x "
          f"{chunks_per_block} chunks x {CHONK_EPOCHS} epochs / accum {GRAD_ACCUM_STEPS})")

    resume_step = resume_epoch = 0
    if RESUME_DIR:
        print(f"\n[Resume] {RESUME_DIR}")
        # Try newest-first across checkpoint dirs so a torn write (crash
        # mid-save leaves a partial training_state.pt) falls back to the
        # previous valid checkpoint instead of crash-looping the wrapper.
        import glob as _glob
        _cands = sorted(
            (p for p in _glob.glob(f"{OUT_DIR}/chonk_step_*")
             if os.path.isfile(os.path.join(p, "training_state.pt"))),
            key=lambda p: int(os.path.basename(p)[len("chonk_step_"):]),
            reverse=True)
        if RESUME_DIR in _cands:
            _cands.remove(RESUME_DIR)
            _cands.insert(0, RESUME_DIR)
        ck = None
        for _cd in _cands:
            try:
                st = os.path.join(_cd, "training_state.pt")
                if not os.path.exists(st):
                    continue
                adapter = os.path.join(_cd, "adapter_model.safetensors")
                if os.path.exists(adapter):
                    from safetensors.torch import load_file
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(model, load_file(adapter, device="cpu"))
                ck = torch.load(st, map_location="cpu")
                if _cd != RESUME_DIR:
                    print(f"[Resume] {RESUME_DIR} unreadable; fell back to {_cd}")
                break
            except Exception as e:
                print(f"[Resume] {_cd} corrupt ({e}); trying older checkpoint")
                ck = None
                continue
        if ck is not None:
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            # ChonkAdamW keeps its own step counter for bias correction; the
            # default Optimizer state_dict does not persist custom attrs, so
            # restore it explicitly (else bc1/bc2 restart at 1 after resume).
            if "adam_step_count" in ck:
                try:
                    optimizer.step_count = int(ck["adam_step_count"])
                except (TypeError, ValueError):
                    pass
            # map_location="cpu" leaves the EMA shadow on CPU; move every
            # shadow tensor back to its param's device (else ema.update()
            # crashes mixing cuda params with cpu shadow at the first step).
            _ref = next(p.data for p in model.parameters() if p.requires_grad)
            ema.shadow = {n: s.to(_ref.device) for n, s in ck["ema"].items()}
            resume_step, resume_epoch = ck["step"], ck["epoch"]
            # Snap mid-block resumes to the block boundary. Steps advance one
            # per GRAD_ACCUM_STEPS chunks and blocks are trained whole; a
            # mid-block resume would re-train the block prefix while the step
            # counter advances (duplicate data, confusing loss). Flooring costs
            # at most GRAD_ACCUM_STEPS-1 steps of re-training; the loaded
            # scheduler/optimizer states stay as-is (a few-step offset over
            # 11k steps is immaterial).
            if opt_steps_per_block > 0 and resume_step % opt_steps_per_block != 0:
                floored = (resume_step // opt_steps_per_block) * opt_steps_per_block
                print(f"[Resume] snapping step {resume_step} -> {floored} (block boundary)")
                resume_step = floored

    step = resume_step
    model.train()
    # Block-skip on resume: skip whole data blocks already trained, so a
    # resumed run continues where the checkpoint left off instead of
    # re-training from block 0. blocks_done derives from optimizer steps
    # (opt_steps_per_block completed blocks).
    blocks_to_skip = 0
    if resume_step > 0:
        blocks_to_skip = resume_step // opt_steps_per_block
        if blocks_to_skip > 0:
            print(f"[Resume] skipping first {blocks_to_skip} block(s) "
                  f"(steps 0..{blocks_to_skip * opt_steps_per_block - 1} already trained)")
    for epoch in range(resume_epoch, CHONK_EPOCHS):
        gen = get_tokenized_dataset(DATA_PATH, SEQ_LEN, CHONK_SUBSAMPLE, epoch)
        t0 = time.time()
        print(f"\n--- Epoch {epoch + 1}/{CHONK_EPOCHS} ---")
        block_idx = -1
        for input_ids in gen:
            block_idx += 1
            if block_idx < blocks_to_skip:
                continue
            if step >= total_steps or step >= MAX_STEPS:
                break
            input_ids = input_ids.unsqueeze(0).cuda()
            reset_chonk_cache(kv_cache)
            chunks_this_seq = 0
            skip_step = False
            # Accumulation-window loss (unscaled sum + chunk count). Checkpoint
            # / best / log values use the window MEAN, not the last chunk.
            window_loss_sum = 0.0
            window_loss_n = 0
            for chunk_start in range(0, SEQ_LEN, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, SEQ_LEN)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                loss, outputs = train_step(model, chunk_ids, kv_cache, chunk_start, chunk_end)
                if CHONK_PAUSE:
                    time.sleep(CHONK_PAUSE)
                last_loss = float("nan")
                if loss is not None:
                    loss = loss / GRAD_ACCUM_STEPS
                    last_loss = loss.item()
                    loss.backward()
                    if last_loss != last_loss:
                        print("  [NaN] skipping chunk"); optimizer.zero_grad()
                        del loss, outputs; torch.cuda.empty_cache(); skip_step = True
                    else:
                        window_loss_sum += last_loss * GRAD_ACCUM_STEPS
                        window_loss_n += 1
                        del loss, outputs
                chunks_this_seq += 1
                torch.cuda.empty_cache()
                # Diagnostic window: per-chunk heartbeat for the first 32 chunks
                # (is the growth smooth ~537MB/chunk or 8GiB stairs?), then 16.
                if chunks_this_seq <= 32 or chunks_this_seq % 16 == 0:
                    ps = pool.stats()
                    # Instrumentation: blockCount steps with each +8.59GB stair
                    # => new dedicated-exportable pool blocks are being created
                    # (kc-proportional demand crossing bucket rungs). largest-
                    # FreeBlock shows whether existing blocks could have served.
                    print(f"  chunk {chunks_this_seq}/{chunks_per_block} "
                          f"({time.time()-t0:.0f}s loss={last_loss:.4f} "
                          f"pool={ps['totalUsed']/1e9:.2f}GB "
                          f"blocks={ps.get('blockCount','?')} "
                          f"dedicated={ps.get('dedicatedCount','?')} "
                          f"allocations={ps.get('allocationCount','?')} "
                          f"largestFree={ps.get('largestFreeBlock',0)/1e9:.2f}GB)",
                          flush=True)
                    if chunks_this_seq in (8, 16, 32, 64, 128):
                        print(f"    [hist] {live_histogram()}", flush=True)

                # One optimizer step per GRAD_ACCUM_STEPS chunks (plus a partial
                # step at block end for non-divisible configs). step/log/save
                # are all optimizer-step-scoped.
                do_opt = (chunks_this_seq % GRAD_ACCUM_STEPS == 0) or (chunk_end == SEQ_LEN)
                if do_opt:
                    if skip_step:
                        print(f"  [skip opt step {step} — NaN chunk in window]", flush=True)
                        optimizer.zero_grad(); torch.cuda.empty_cache()
                        skip_step = False
                        window_loss_sum = 0.0
                        window_loss_n = 0
                    else:
                        if GRAD_CLIP_NORM > 0:
                            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                            if gn != gn:
                                print(f"  [NaN grad] skip step {step}", flush=True)
                                optimizer.zero_grad(); torch.cuda.empty_cache()
                                window_loss_sum = 0.0
                                window_loss_n = 0
                                continue
                        optimizer.step(); scheduler.step(); optimizer.zero_grad()
                        # Window-mean loss for checkpoint/best/log (stable across
                        # noisy per-chunk values). Falls back to the last chunk
                        # if the window is somehow empty.
                        window_loss = (window_loss_sum / window_loss_n
                                       if window_loss_n > 0 else last_loss * GRAD_ACCUM_STEPS)
                        window_loss_sum = 0.0
                        window_loss_n = 0
                        # Reclaim fully-free slab blocks to the pool at each
                        # optimizer step. The +2-GiB-every-4-chunks steps come
                        # from slab blocks acquired for kc-growing transients
                        # and never released (empty_cache only frees segments
                        # back to the slab, not the slab's pool blocks).
                        # Release down to a warm floor; NOT on the hot path.
                        n_rel = release_empty_blocks(keepFloor=2)
                        if step % 8 == 0 or n_rel > 0:
                            print(f"  [slab] step {step}: {slab_stats()} "
                                  f"released={n_rel}", flush=True)
                        if CHONK_EMA_UPDATE_EVERY > 0 and step % CHONK_EMA_UPDATE_EVERY == 0:
                            ema.update()
                        if CHONK_OPTIMIZER_PAUSE:
                            time.sleep(CHONK_OPTIMIZER_PAUSE)

                        if step % SAVE_INTERVAL == 0:
                            sp = f"{OUT_DIR}/chonk_step_{step}"
                            os.makedirs(sp, exist_ok=True)
                            model.save_pretrained(sp)
                            torch.save({"optimizer": optimizer.state_dict(),
                                        "scheduler": scheduler.state_dict(),
                                        "ema": ema.shadow, "step": step, "epoch": epoch,
                                        "adam_step_count": optimizer.step_count},
                                       f"{sp}/training_state.pt")
                            cur_loss = float(window_loss)
                            with open(f"{sp}/.chonk_loss", "w") as f:
                                f.write(f"{cur_loss:.6f}")
                            print(f"Checkpoint -> {sp} (loss={cur_loss:.4f})", flush=True)
                            # Persist the running BEST to a protected path so the
                            # global-best adapter survives checkpoint trimming
                            # (steps are loss-noisy; the per-step cleanup must not
                            # delete the best model we ever saw).
                            best_file = f"{OUT_DIR}/best_loss.txt"
                            try:
                                best_loss_so_far = float(open(best_file).read().strip())
                            except (OSError, ValueError):
                                best_loss_so_far = float("inf")
                            if cur_loss < best_loss_so_far:
                                best_dir = f"{OUT_DIR}/chonk_best"
                                if os.path.exists(best_dir):
                                    shutil.rmtree(best_dir, ignore_errors=True)
                                os.makedirs(best_dir, exist_ok=True)
                                model.save_pretrained(best_dir)
                                torch.save({"step": step, "epoch": epoch, "loss": cur_loss,
                                            "ema": ema.shadow},
                                           f"{best_dir}/training_state.pt")
                                with open(f"{best_dir}/.chonk_loss", "w") as f:
                                    f.write(f"{cur_loss:.6f}")
                                with open(best_file, "w") as f:
                                    f.write(f"{cur_loss:.6f}")
                                print(f"  [best] new best loss={cur_loss:.4f} "
                                      f"-> {best_dir}", flush=True)
                            cleanup_checkpoints(OUT_DIR, KEEP_CHECKPOINTS)
                        if step % LOG_INTERVAL == 0:
                            ps = pool.stats()
                            print(f"Step {step}: loss={float(window_loss):.4f} "
                                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                                  f"pool={ps['totalUsed']/1e9:.2f}GB", flush=True)
                        step += 1
                if step >= MAX_STEPS:
                    break
            if step >= MAX_STEPS or step >= total_steps:
                break

    ema.apply_shadow()
    os.makedirs(f"{OUT_DIR}/chonk_final", exist_ok=True)
    model.save_pretrained(f"{OUT_DIR}/chonk_final")
    print(f"\nFinal LoRA adapters saved to {OUT_DIR}/chonk_final (EMA applied)")
    pool.shutdown()
    print("Training complete!")


if __name__ == "__main__":
    main()
