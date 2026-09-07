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
CHONK_MAX_POOL_GB = float(os.environ.get("CHONK_MAX_POOL_GB", "85"))
# GTT (driver-pinned) wall: pool bytes cost ~1.5x in GTT, and GTT exhaustion
# freezes the box (kswapd livelock on unevictable pins) instead of a clean
# OOM. Recycle on EITHER metric. 0 disables a check.
CHONK_MAX_GTT_GB = float(os.environ.get("CHONK_MAX_GTT_GB", "115"))


def _read_gtt_gb():
    try:
        with open("/sys/class/drm/card1/device/mem_info_gtt_used") as f:
            return int(f.read().strip()) / 1e9
    except (OSError, ValueError):
        return 0.0
CHONK_EMA_UPDATE_EVERY = int(os.environ.get("CHONK_EMA_UPDATE_EVERY", "1"))
CHONK_SUBSAMPLE = float(os.environ.get("CHONK_SUBSAMPLE", "1.0"))
CHONK_EPOCHS = int(os.environ.get("CHONK_EPOCHS", "1"))
CHONK_SMOKE = os.environ.get("CHONK_SMOKE", "0") == "1"
CHONK_GRADIENT_CHECKPOINT = os.environ.get("CHONK_GRADIENT_CHECKPOINT", "1") == "1"
CHONK_TENSOR_CENSUS = os.environ.get("CHONK_TENSOR_CENSUS", "0") == "1"
CHONK_CENSUS_PATH = os.environ.get("CHONK_CENSUS_PATH", "")

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


def dump_tensor_census(tag, path):
    """Census of live CUDA tensor wrappers: who holds the retained bytes?

    Groups python-side torch.Tensor objects by (size bucket, grad state) and,
    for the suspect buckets (4MB/64KB/1KB/512B), samples referrers to NAME the
    holder (module attr vs autograd node vs list). Read-only; costs one gc walk
    (~seconds). The slab histogram shows sizes; this shows owners.
    """
    import gc
    from collections import Counter
    buckets = {}
    try:
        objs = gc.get_objects()
    except Exception as e:
        with open(path, "a") as f:
            f.write(f"[{tag}] gc walk failed: {e}\n")
        return
    for o in objs:
        try:
            if not isinstance(o, torch.Tensor):
                continue
            if o.device.type != "cuda":
                continue
            n = o.numel() * o.element_size()
        except Exception:
            continue
        c = 512
        while c < n:
            c <<= 1
        b = buckets.setdefault(c, {"n": 0, "bytes": 0, "req_grad": 0,
                                   "has_grad_fn": 0, "refkinds": Counter(),
                                   "samples": []})
        b["n"] += 1
        b["bytes"] += n
        try:
            if o.requires_grad:
                b["req_grad"] += 1
            if o.grad_fn is not None:
                b["has_grad_fn"] += 1
        except Exception:
            pass
        if c in (4194304, 65536, 1048576, 1024, 512) and len(b["samples"]) < 5:
            info = {}
            try:
                info["shape"] = str(list(o.shape))
                info["dtype"] = str(o.dtype).replace("torch.", "")
                info["leaf"] = bool(o.is_leaf)
                try:
                    info["ptr"] = hex(o.data_ptr())
                    info["nbytes"] = int(o.numel() * o.element_size())
                    info["storage"] = int(o.untyped_storage().nbytes())
                except Exception:
                    pass
                # Owner fingerprint: name the actual holder. Direct named
                # attribute first (ClassName.attr); else one level up through
                # list/dict/tuple containers (container owner + key/index).
                try:
                    found = False
                    for r in gc.get_referrers(o):
                        if type(r).__name__ == "frame":
                            continue
                        d = getattr(r, "__dict__", None)
                        if isinstance(d, dict):
                            for k, v in d.items():
                                if v is o:
                                    info["owner"] = f"{type(r).__name__}.{k}"
                                    found = True
                                    break
                                if isinstance(v, (list, tuple)) and any(
                                        x is o for x in v):
                                    info["owner"] = (
                                        f"{type(r).__name__}.{k}"
                                        f"[{v.index(o) if o in v else '?'}]")
                                    found = True
                                    break
                                if isinstance(v, dict):
                                    for kk, vv in v.items():
                                        if vv is o or (
                                                isinstance(vv, (list, tuple))
                                                and o in vv):
                                            info["owner"] = (
                                                f"{type(r).__name__}.{k}[{kk}]")
                                            found = True
                                            break
                                    if found:
                                        break
                            if found:
                                break
                    if not found:
                        # One level up through bare containers.
                        for r in gc.get_referrers(o):
                            if type(r).__name__ not in (
                                    "list", "tuple", "dict", "set"):
                                continue
                            for rr in gc.get_referrers(r):
                                if type(rr).__name__ == "frame":
                                    continue
                                info["owner2"] = type(rr).__name__
                                found = True
                                break
                            if found:
                                break
                except Exception:
                    pass
                gf = o.grad_fn
                if gf is not None:
                    info["op"] = type(gf).__name__
                    nxt = []
                    try:
                        for fn, _ in (gf.next_functions or [])[:4]:
                            nxt.append(type(fn).__name__ if fn is not None else "AccumulateGrad")
                    except Exception:
                        pass
                    info["next"] = nxt
                kinds = Counter()
                for r in gc.get_referrers(o):
                    t = type(r).__name__
                    if t == "frame":
                        continue
                    kinds[t] += 1
                info["holders"] = dict(kinds)
            except Exception as e:
                info = {"err": str(e)[:80]}
            b["samples"].append(info)
    try:
        with open(path, "a") as f:
            f.write(f"[{tag}] cuda tensor census:\n")
            for c in sorted(buckets, reverse=True)[:14]:
                b = buckets[c]
                f.write(f"  size~{c}: n={b['n']} bytes={b['bytes']/1e6:.1f}MB "
                        f"req_grad={b['req_grad']} grad_fn={b['has_grad_fn']}\n")
                for s in b["samples"]:
                    f.write(f"    holders={s}\n")
    except Exception:
        pass


def _maybe_recycle(pool, where, save_state=None):
    """Exit(42) for a fresh process when pool or GTT crosses its high-water
    cap. Called every chunk (cheap: one stats dict + one sysfs read) so a
    late-block surge can't slip between optimizer steps.

    save_state: optional callable that persists a MID-BLOCK resume point
    (adapter + optimizer + moments + chunk position). With it, the next
    process resumes at this exact chunk instead of replaying the block from
    chunk 0 — without it, process lifetimes shorter than a block turn the
    recycle into a treadmill (observed: GTT cap at ~chunk 81 of every cycle,
    block replayed from 0 forever, nothing net-banked)."""
    if CHONK_MAX_POOL_GB <= 0 and CHONK_MAX_GTT_GB <= 0:
        return
    _pu = pool.stats()["totalUsed"] / 1e9
    _gu = _read_gtt_gb()
    _why = ""
    if CHONK_MAX_POOL_GB > 0 and _pu >= CHONK_MAX_POOL_GB:
        _why = f"pool {_pu:.2f}GB >= {CHONK_MAX_POOL_GB:.0f}GB cap"
    elif CHONK_MAX_GTT_GB > 0 and _gu > 0 and _gu >= CHONK_MAX_GTT_GB:
        _why = (f"GTT {_gu:.2f}GB >= {CHONK_MAX_GTT_GB:.0f}GB cap "
                f"(pool {_pu:.2f}GB)")
    if _why:
        print(f"[recycle] {_why} at {where}; exiting clean for a fresh "
              f"process", flush=True)
        if save_state is not None:
            try:
                save_state()
            except Exception as e:
                print(f"[recycle] mid-block save failed ({e}); falling back "
                      f"to block-boundary resume", flush=True)
        import sys as _sys
        import os as _os
        _os._exit(42)  # hard exit: skip atexit/pool teardown under pressure


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
    # Mid-block resume points (name like 192m81) sort as strings and must be
    # treated as the same step as their base (192); keep them out of numeric
    # bookkeeping, only cull them when their base step is culled.
    def _base(p):
        n = os.path.basename(p)[len("chonk_step_"):]
        return int(n.split("m")[0]) if "m" in n else int(n)
    steps = sorted({_base(p) for p in glob.glob(f"{out_dir}/chonk_step_*")})
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
            # remove any mid-block points of a culled base step
            for mb in glob.glob(f"{out_dir}/chonk_step_{s}m*"):
                shutil.rmtree(mb, ignore_errors=True)


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
        def _ck_key(p):
            # mid-block points (192m81) sort after their base step so the
            # newest-with-state scan prefers them over older banked steps
            n = os.path.basename(p)[len("chonk_step_"):]
            if "m" in n:
                b, c = n.split("m")
                return (int(b), int(c))
            return (int(n), -1)
        _cands = sorted(
            (p for p in _glob.glob(f"{OUT_DIR}/chonk_step_*")
             if os.path.isfile(os.path.join(p, "training_state.pt"))),
            key=_ck_key,
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
            # Restore Adam moments by parameter NAME into this run's fresh
            # pool views (object identity does not survive restarts).
            if ck.get("adam_moments"):
                try:
                    _restored, _missing = 0, 0
                    for _n, _p in model.named_parameters():
                        if not _p.requires_grad or _n not in ck["adam_moments"]:
                            continue
                        _dst = optimizer_states.get(_p)
                        if _dst is None:
                            _dst = optimizer.state.get(_p)
                            if not isinstance(_dst, dict):
                                _missing += 1
                                continue
                        _m_avg, _m_sq = ck["adam_moments"][_n]
                        _dst["exp_avg"].copy_(_m_avg)
                        _dst["exp_avg_sq"].copy_(_m_sq)
                        _restored += 1
                    print(f"[Resume] adam moments restored for {_restored} "
                          f"params ({_missing} missing)")
                except Exception as _e:
                    print(f"[Resume] adam moments restore failed: {_e}")
            # map_location="cpu" leaves the EMA shadow on CPU; move every
            # shadow tensor back to its param's device (else ema.update()
            # crashes mixing cuda params with cpu shadow at the first step).
            _ref = next(p.data for p in model.parameters() if p.requires_grad)
            ema.shadow = {n: s.to(_ref.device) for n, s in ck["ema"].items()}
            resume_step, resume_epoch = ck["step"], ck["epoch"]
            # Mid-block resume point (saved by the recycle guard): step stays
            # at its last BANKED value (no snap needed — it is already at a
            # boundary), and the block/chunk position is replayed no-grad to
            # rebuild the KV cache, then training continues from that chunk.
            # This is what lets process lifetimes shorter than a block still
            # make net forward progress (the treadmill fix).
            if "midblock_block" in ck:
                main._midblock_resume = int(ck["midblock_block"])
                main._midblock_chunk = int(ck["midblock_chunk"])
                print(f"[Resume] mid-block point: block "
                      f"{main._midblock_resume} chunk "
                      f"{main._midblock_chunk} (step stays {resume_step})")
            # Snap mid-block resumes to the block boundary. Steps advance one
            # per GRAD_ACCUM_STEPS chunks and blocks are trained whole; a
            # mid-block resume would re-train the block prefix while the step
            # counter advances (duplicate data, confusing loss). Flooring costs
            # at most GRAD_ACCUM_STEPS-1 steps of re-training; the loaded
            # scheduler/optimizer states stay as-is (a few-step offset over
            # 11k steps is immaterial).
            elif opt_steps_per_block > 0 and resume_step % opt_steps_per_block != 0:
                floored = (resume_step // opt_steps_per_block) * opt_steps_per_block
                print(f"[Resume] snapping step {resume_step} -> {floored} (block boundary)")
                resume_step = floored

    step = resume_step
    _grad_probed = False
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
            # Mid-block resume: a recycle inside this block saved the chunk
            # position (and grads-in-flight are forfeit by design); replay the
            # data prefix WITHOUT optimizer steps to rebuild the KV cache to
            # the saved chunk, then continue training from there. This is
            # what breaks the treadmill: process lifetime < block length.
            _resume_block = getattr(main, "_midblock_resume", None)
            _resume_chunk = 0
            if (_resume_block == block_idx and _resume_block is not None):
                _resume_chunk = getattr(main, "_midblock_chunk", 0)
                print(f"[Resume] mid-block: block {block_idx} -> chunk "
                      f"{_resume_chunk} (replaying prefix no-grad)", flush=True)
            chunks_this_seq = 0
            skip_step = False
            # Accumulation-window loss (unscaled sum + chunk count). Checkpoint
            # / best / log values use the window MEAN, not the last chunk.
            window_loss_sum = 0.0
            window_loss_n = 0
            last_gn = float("nan")
            _midblock_replay = _resume_chunk
            for chunk_start in range(0, SEQ_LEN, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, SEQ_LEN)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                if _midblock_replay > 0:
                    # Prefix replay under no_grad: rebuilds the INT4 KV cache
                    # only; no forward graph, no grads, no loss.
                    with torch.no_grad():
                        train_step(model, chunk_ids, kv_cache,
                                   chunk_start, chunk_end)
                    _midblock_replay -= 1
                    chunks_this_seq += 1
                    continue
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
                # Per-chunk high-water check: the opt-step check alone missed
                # a late-block GTT surge (119GB at 54GB pool) by ~16 chunks,
                # long enough to thrash-lock the box. One stats dict + one
                # sysfs read per chunk is noise next to a 90s chunk. Skip the
                # first few chunks so a misconfigured cap can't tight-loop
                # restarts faster than progress.
                if chunks_this_seq >= 4:
                    def _save_midblock():
                        """Persist an exact mid-block resume point so the next
                        process replays only the KV prefix instead of the whole
                        block. Steps NOT advanced for partial windows (grads in
                        flight are forfeit); the step counter stays at its last
                        banked value so accounting stays exact.

                        CRITICAL: saved under a DEDICATED dir (chonk_step_
                        {step}m{chunk}) so (a) the wrapper's newest-with-state
                        scan picks it over the older banked step dirs, and (b)
                        it never overwrites a clean banked checkpoint. The 'm'
                        suffix keeps sort -V ordering after the base step."""
                        sp = f"{OUT_DIR}/chonk_step_{step}m{chunks_this_seq}"
                        os.makedirs(sp, exist_ok=True)
                        model.save_pretrained(sp)
                        _moments = {}
                        try:
                            _name_of = {p: n for n, p in
                                        model.named_parameters()
                                        if p.requires_grad}
                            for _p in [p for p in model.parameters()
                                       if p.requires_grad]:
                                _st = optimizer_states.get(_p)
                                if _st is None:
                                    _st = optimizer.state.get(_p, {})
                                if ("exp_avg" in _st and
                                        "exp_avg_sq" in _st):
                                    _moments[_name_of[_p]] = (
                                        _st["exp_avg"].detach().cpu(),
                                        _st["exp_avg_sq"].detach().cpu())
                        except Exception:
                            pass
                        _tmp = f"{sp}/training_state.pt.tmp"
                        torch.save({"optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "ema": ema.shadow, "step": step,
                                    "epoch": epoch,
                                    "adam_step_count": optimizer.step_count,
                                    "adam_moments": _moments,
                                    "midblock_block": block_idx,
                                    "midblock_chunk": chunks_this_seq},
                                   _tmp)
                        os.replace(_tmp, f"{sp}/training_state.pt")
                        with open(f"{sp}/.chonk_loss", "w") as f:
                            f.write(f"{float(window_loss):.6f}")
                        print(f"[recycle] mid-block point saved: step {step} "
                              f"block {block_idx} chunk {chunks_this_seq}",
                              flush=True)
                    _maybe_recycle(pool, f"chunk {chunks_this_seq}",
                                   save_state=_save_midblock)
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
                    if chunks_this_seq in (4, 6, 8, 16, 32, 64, 128):
                        print(f"    [hist] {live_histogram()}", flush=True)
                        if CHONK_TENSOR_CENSUS:
                            _cpath = (CHONK_CENSUS_PATH or
                                      os.path.join(OUT_DIR, "tensor_census.log"))
                            dump_tensor_census(
                                f"chunk{chunks_this_seq}", _cpath)
                            import gc as _gc
                            _c0 = _gc.collect()
                            dump_tensor_census(
                                f"chunk{chunks_this_seq}+gc", _cpath)
                            print(f"    [census] dumped chunk{chunks_this_seq} "
                                  f"(gc.collect freed {_c0} objs)", flush=True)

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
                            last_gn = float(gn)
                            if gn != gn:
                                print(f"  [NaN grad] skip step {step}", flush=True)
                                optimizer.zero_grad(); torch.cuda.empty_cache()
                                window_loss_sum = 0.0
                                window_loss_n = 0
                                continue
                            if not _grad_probed:
                                _grad_probed = True
                                try:
                                    from collections import defaultdict
                                    _gg = defaultdict(lambda: [0.0, 0])
                                    for _n, _p in model.named_parameters():
                                        if not _p.requires_grad:
                                            continue
                                        _g = _p.grad
                                        _key = "none"
                                        if _g is not None:
                                            for _t in ("q_proj", "k_proj",
                                                       "v_proj", "o_proj",
                                                       "gate_proj", "up_proj",
                                                       "down_proj"):
                                                if _t in _n:
                                                    _key = _t
                                                    break
                                            _gg[_key][0] += float(
                                                _g.detach().abs().mean())
                                            _gg[_key][1] += 1
                                        else:
                                            _gg["none"][1] += 1
                                    print(f"  [gradcheck] step {step} global_norm="
                                          f"{float(gn):.4f} " + " ".join(
                                              f"{k}={v[0]/max(1,v[1]):.2e}"
                                              f"(n={v[1]})"
                                              for k, v in sorted(_gg.items())),
                                          flush=True)
                                except Exception as _e:
                                    print(f"  [gradcheck] failed: {_e}", flush=True)
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
                            # Persist Adam moments by parameter NAME (the live
                            # state dict is keyed by object identity, which
                            # does not survive a restart; without this every
                            # resume silently re-warms moments from zero).
                            _moments = {}
                            try:
                                _name_of = {p: n for n, p in
                                            model.named_parameters()
                                            if p.requires_grad}
                                for _p in [p for p in
                                           model.parameters()
                                           if p.requires_grad]:
                                    _st = optimizer_states.get(_p)
                                    if _st is None:
                                        _st = optimizer.state.get(_p, {})
                                    if ("exp_avg" in _st and
                                            "exp_avg_sq" in _st):
                                        _moments[_name_of[_p]] = (
                                            _st["exp_avg"].detach().cpu(),
                                            _st["exp_avg_sq"].detach().cpu())
                            except Exception:
                                pass
                            # Atomic state write: a SIGKILL mid-save must not
                            # leave a partial training_state.pt behind (a
                            # partial newest dir previously poisoned wrapper
                            # resume detection into a from-scratch retrain).
                            _tmp_st = f"{sp}/training_state.pt.tmp"
                            torch.save({"optimizer": optimizer.state_dict(),
                                        "scheduler": scheduler.state_dict(),
                                        "ema": ema.shadow, "step": step, "epoch": epoch,
                                        "adam_step_count": optimizer.step_count,
                                        "adam_moments": _moments},
                                       _tmp_st)
                            os.replace(_tmp_st, f"{sp}/training_state.pt")
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
                                _tmp_best = f"{best_dir}/training_state.pt.tmp"
                                torch.save({"step": step, "epoch": epoch, "loss": cur_loss,
                                            "ema": ema.shadow},
                                           _tmp_best)
                                os.replace(_tmp_best, f"{best_dir}/training_state.pt")
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
                                  f"gn={last_gn:.2f} "
                                  f"lr={scheduler.get_last_lr()[0]:.2e} "
                                  f"pool={ps['totalUsed']/1e9:.2f}GB", flush=True)
                        # High-water recycle (also checked every chunk below).
                        # Here it runs right after a banked step, so nothing
                        # but the in-flight block prefix is ever discarded.
                        # Exit 42 = intentional recycle (wrapper restarts).
                        _maybe_recycle(pool, f"step {step}")
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
