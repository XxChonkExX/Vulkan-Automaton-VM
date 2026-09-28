#!/usr/bin/env python3
"""Gemma 4 12B QLoRA -- Chonk Buffer chunked training.

Port of examples/granite_chonk/train_granite_chonk.py to Gemma4, reusing ALL
shared machinery (pool, INT4 KV cache, optimizer states, position
checkpoints, recycle guards, EMA) and swapping only the model-specific
parts:
  - text config from the unified checkpoint (get_text_config)
  - unified safetensors keys via to_ckpt_names fallback (in chonk.py)
  - Gemma cache-aware tiled attention (vulkanvm_attn_gemma_cache)
  - mixed-length block pools (32k/64k/128k) streamed in fixed chunks --
    mixed diet comes free (no rectangle constraint in chunked training)
  - explicit resume coordinates (pool, block, chunk) instead of uniform
    block math (mixed lengths have no fixed chunks-per-block)

Memory at 128k: INT4 weights ~7GB + INT4 KV ~13GB + LoRA/Adam ~3GB +
chunk activations/workspace/logits ~5GB = ~28GB, FLAT at every position.
"""
import glob
import json
import math
import os
import random
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(REPO, "_build"))
sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))
sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))

import numpy as np
import torch
import torch.nn as nn

from chonk import (
    build_lora_chonk_setup,
    reset_chonk_cache,
    release_empty_blocks,
    slab_stats,
    live_histogram,
    install_chonk_allocator,
)
from train_granite_chonk import (
    ChonkAdamW, EMAModel, cleanup_checkpoints, _maybe_recycle,
    enable_projection_checkpointing,
    patch_linear_cache_for_chunked_training,
)
from vulkanvm_attn_gemma_cache import patch_gemma_attention_cache

MODEL_PATH = os.environ.get("CHONK_MODEL_PATH",
                             "/home/chonke/Downloads/gemma412b/gemma4-12b-obliterated")
POOLS_DIR = os.environ.get("CHONK_POOLS_DIR",
                           "/home/chonke/Downloads/gemma412b/corpus/pools")
OUT_DIR = os.environ.get("CHONK_OUT_DIR",
                         "/home/chonke/Vulkan-Automaton-VM/examples/gemma_chonk/out/gemma-finetuned")
CHUNK_SIZE = int(os.environ.get("CHONK_CHUNK", "1024"))
# Position-bucketed chunking: per-chunk cost grows with cache depth (full
# layers visit kc/KT tiles), and >1024 chunks have never run clean. Bucket
# by cached length so per-chunk work stays flat: big chunks while shallow,
# small chunks when deep. Deterministic in kc, so resume replay realigns.
def bucket_chunk_size(kc):
    if kc < 32768:
        return 1024
    if kc < 131072:
        return 512
    return 256
# Token-budget accumulation (variable chunks => accumulate by tokens, and
# scale grads to a true per-token mean at step time).
ACCUM_TOKENS = int(os.environ.get("CHONK_ACCUM_TOKENS", "16384"))
MAX_CACHE_LEN = int(os.environ.get("CHONK_MAX_CACHE_LEN", "131072"))
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = int(os.environ.get("CHONK_GRAD_ACCUM", "16"))
LEARNING_RATE = float(os.environ.get("CHONK_LR", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("CHONK_WD", "0.01"))
WARMUP_STEPS = int(os.environ.get("CHONK_WARMUP", "50"))
MAX_STEPS = int(os.environ.get("CHONK_MAX_STEPS", "10000"))
SAVE_INTERVAL = int(os.environ.get("CHONK_SAVE_INTERVAL", "10"))
KEEP_CHECKPOINTS = int(os.environ.get("CHONK_KEEP_CHECKPOINTS", "5"))
GRAD_CLIP_NORM = float(os.environ.get("CHONK_GRAD_CLIP", "1.0"))
MIX = os.environ.get("CHONK_MIX", "32768:24,65536:6,131072:3")
RESUME_DIR = os.environ.get("CHONK_RESUME_DIR", "")
EPOCHS = int(os.environ.get("CHONK_EPOCHS", "1"))


def parse_mix(spec):
    out = []
    for part in spec.split(","):
        L, n = part.strip().split(":")
        out += [int(L)] * int(n)
    return out


def load_pools():
    man = json.load(open(os.path.join(POOLS_DIR, "manifest.json")))
    pools = {}
    for L, info in man["pools"].items():
        arr = np.memmap(info["file"], dtype=np.int32, mode="r")
        pools[int(L)] = arr.reshape(info["n_blocks"], int(L))
    return pools


def main():
    # Single-instance guard (matches the granite trainer's own lock file
    # convention: wrapper-vs-wrapper AND process-vs-process refuse).
    global _train_lock_fh
    try:
        import fcntl
        _lock_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", ".train_gemma_process.lock"))
        _train_lock_fh = open(_lock_path, "w")
        fcntl.flock(_train_lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[guard] another gemma training process is already running; "
              "refusing second instance (two trainers = OOM).", flush=True)
        sys.exit(0)
    print("=" * 60)
    print("Gemma 4 12B QLoRA -- Chonk Buffer chunked training")
    print(f"model={MODEL_PATH}\npools={POOLS_DIR}\nmix={MIX} chunk={CHUNK_SIZE}")
    print("=" * 60)

    from transformers import AutoConfig
    full_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = full_cfg.get_text_config(decoder=True)
    print(f"text config: {config.model_type} "
          f"{config.num_hidden_layers}L/{config.num_attention_heads}q/"
          f"{config.num_key_value_heads}kv/d{config.head_dim}")

    print("\n[1/4] Installing Chonk allocator + INT4 LoRA setup...")
    install_chonk_allocator()
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, BATCH_SIZE, MAX_CACHE_LEN,
        lora_r=int(os.environ.get("CHONK_LORA_R", "64")),
        lora_alpha=int(os.environ.get("CHONK_LORA_A", "128")),
        lora_dropout=0.05,
        attn_implementation="eager",
        quantize=True, quant_group_size=128, quant_bits=4,
        act_budget_gb=float(os.environ.get("CHONK_ACT_GB", "2.0")),
        staging_gb=float(os.environ.get("CHONK_STAGING_GB", "2.0")),
    )
    pool, kv_cache, model = setup["pool"], setup["kv_cache"], setup["model"]
    optimizer_states = setup["optimizer_states"]
    # get_seq_length: the Chonk cache tracks writes per layer in
    # cumulative_length; Gemma derives chunk positions from it
    # (positions = arange(chunk) + past_seen_tokens). Max over layers.
    def _chonk_seq_length(_self, *args, **kwargs):
        n = 0
        for _lyr in _self.layers:
            _c = getattr(_lyr, "cumulative_length", None)
            if _c is not None:
                n = max(n, int(_c.item()))
        return n
    import types as _types
    kv_cache.get_seq_length = _types.MethodType(
        _chonk_seq_length, kv_cache)
    torch.cuda.empty_cache()
    model.print_trainable_parameters()
    patch_linear_cache_for_chunked_training()
    patch_gemma_attention_cache(model, kv_cache)
    enable_projection_checkpointing(model)

    optimizer = ChonkAdamW(
        [p for p in model.parameters() if p.requires_grad],
        optimizer_states, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ema = EMAModel(model, decay=0.9999)

    pools = load_pools()
    mix_lens = parse_mix(MIX)
    # epoch chunk budget: full passes over each pool in the mix ratio
    sched_rng = random.Random(2026)
    sched = mix_lens[:]
    sched_rng.shuffle(sched)

    from transformers import get_cosine_schedule_with_warmup
    # Token-budget accounting (bucketed chunks vary in size): one epoch ≈
    # one pass over every pool in the mix; steps = tokens / ACCUM_TOKENS.
    tokens_per_epoch = sum(
        pools[L].shape[0] * L for L in set(mix_lens))
    total_steps = max(1, tokens_per_epoch // ACCUM_TOKENS) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps)
    print(f"  ~{tokens_per_epoch/1e6:.1f}M tok/epoch -> {total_steps} opt steps")

    # ---- resume (explicit coordinates) ----
    step = 0
    resume_coords = None
    if RESUME_DIR:
        try:
            st = torch.load(os.path.join(RESUME_DIR, "training_state.pt"),
                            map_location="cpu")
            resume_coords = (st.get("pool_L"), st.get("block_idx"),
                             st.get("chunk_idx"))
            step = st.get("step", 0)
            print(f"[Resume] {RESUME_DIR} step={step} coords={resume_coords}",
                  flush=True)
            try:
                optimizer.load_state_dict(st["optimizer"])
                scheduler.load_state_dict(st["scheduler"])
                ema.shadow = {k: v.cuda() for k, v in st["ema"].items()}
                optimizer.step_count = st.get("adam_step_count",
                                              optimizer.step_count)
                print("[Resume] optimizer/scheduler/ema restored", flush=True)
            except Exception as e:
                print(f"[Resume] optimizer state partial: {e}", flush=True)
        except Exception as e:
            print(f"[Resume] unreadable ({e}); cold start", flush=True)

    ptr = {L: 0 for L in pools}
    si = 0
    global_chunk = 0
    window_loss_sum = 0.0
    window_loss_n = 0
    best_loss = float("inf")
    try:
        best_loss = float(open(os.path.join(OUT_DIR, "best_loss.txt")).read())
    except (OSError, ValueError):
        pass

    skip_step = False
    last_gn = float("nan")
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    def save_ckpt(gchunk, wmean, tok_pos=0):
        sp = os.path.join(OUT_DIR, f"ckpt_{gchunk}")
        os.makedirs(sp, exist_ok=True)
        model.save_pretrained(sp)
        _moments = {}
        try:
            _name_of = {p: n for n, p in model.named_parameters()
                        if p.requires_grad}
            for _p in [p for p in model.parameters() if p.requires_grad]:
                _st = optimizer_states.get(_p, optimizer.state.get(_p, {}))
                if "exp_avg" in _st and "exp_avg_sq" in _st:
                    _moments[_name_of[_p]] = (_st["exp_avg"].detach(),
                                             _st["exp_avg_sq"].detach())
        except Exception:
            pass
        _tmp = f"{sp}/training_state.pt.tmp"
        torch.save({"optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.shadow, "step": step, "epoch": 0,
                    "adam_step_count": optimizer.step_count,
                    "adam_moments": _moments, "pos": gchunk,
                    "pool_L": cur_L, "block_idx": cur_bi,
                    "chunk_idx": tok_pos},
                   _tmp)
        os.replace(_tmp, f"{sp}/training_state.pt")
        with open(f"{sp}/.chonk_loss", "w") as f:
            f.write(f"{float(wmean):.6f}")
        try:
            if float(wmean) < best_loss:
                import shutil as _sh
                _sh.rmtree(os.path.join(OUT_DIR, "chonk_best"),
                           ignore_errors=True)
                _sh.copytree(sp, os.path.join(OUT_DIR, "chonk_best"))
                open(os.path.join(OUT_DIR, "best_loss.txt"), "w").write(
                    f"{float(wmean):.6f}")
        except Exception:
            pass
        return sp

    epoch = 0
    while step < min(total_steps, MAX_STEPS):
        epoch += 1
        print(f"\n--- Epoch {epoch} ---", flush=True)
        L = sched[si % len(sched)]
        si += 1
        if si % len(sched) == 0:
            sched_rng.shuffle(sched)
        pool_arr = pools[L]
        cur_L = L
        cur_bi = ptr[L] % len(pool_arr)
        ptr[L] += 1
        block = torch.from_numpy(
            pool_arr[cur_bi].astype(np.int64)).unsqueeze(0).cuda()
        reset_chonk_cache(kv_cache)
        # Mid-block resume in TOKEN coordinates (chunk boundaries move with
        # the bucket function, but bucket(kc) is deterministic so replay
        # reproduces identical chunking).
        replay_to = 0
        if resume_coords is not None and tuple(resume_coords[:2]) == (cur_L, cur_bi):
            replay_to = min(int(resume_coords[2] or 0), L)
            if replay_to > 0:
                print(f"[Resume] replaying {replay_to} tokens no-grad "
                      f"(L={cur_L} block={cur_bi})", flush=True)
                with torch.no_grad():
                    _rp = 0
                    while _rp < replay_to:
                        _cs = bucket_chunk_size(_rp)
                        _re = min(_rp + _cs, replay_to)
                        model(input_ids=block[:, _rp:_re],
                              past_key_values=kv_cache, use_cache=True)
                        _rp = _re
            resume_coords = None
        pos = replay_to
        step_tok = 0  # tokens accumulated toward the next optimizer step
        while pos < L:
            if step >= min(total_steps, MAX_STEPS):
                break
            cur_tok = pos
            cs = bucket_chunk_size(pos)
            ce = min(pos + cs, L)
            # BUGFIX: slice from pos (the chunk START), not cs (the chunk
            # SIZE). block[:, cs:ce] fed an empty first chunk and then
            # ever-growing mega-chunks anchored at offset cs -- mis-positioned
            # text, quadratic attention blowups, garbage loss.
            chunk_ids = block[:, pos:ce]
            # NOTE: no cache_position passed -- Gemma derives chunk positions
            # from past_seen_tokens (cache length), exactly like generation.
            outputs = model(input_ids=chunk_ids, past_key_values=kv_cache,
                            use_cache=True)
            logits = outputs.logits
            loss = None
            if logits is not None:
                sl = logits[..., :-1, :].contiguous()
                sy = chunk_ids[..., 1:].contiguous()
                loss = nn.CrossEntropyLoss(label_smoothing=0.1)(
                    sl.view(-1, sl.size(-1)), sy.view(-1))
            last_loss = float("nan")
            if loss is not None:
                tok_m = ce - pos - 1
                # token-weighted accumulation: every token votes equally
                # regardless of its chunk's bucket size.
                loss = loss * tok_m
                last_loss = loss.item() / max(1, tok_m)
                loss.backward()
                step_tok += tok_m
                if last_loss != last_loss:
                    print("  [NaN] skipping chunk")
                    # Crucible instrument (env-gated): dump failing chunk for
                    # offline content-vs-state bisection.
                    if os.environ.get("CHONK_NAN_DUMP", "0") == "1":
                        import numpy as _np
                        _dp = os.path.join(
                            OUT_DIR, f"nan_chunk_{global_chunk}.npy")
                        _np.save(_dp, chunk_ids.detach().cpu().numpy())
                        print(f"  [NaN] dumped {tuple(chunk_ids.shape)} -> {_dp}",
                              flush=True)
                    optimizer.zero_grad()
                    del loss, outputs
                    torch.cuda.empty_cache()
                    skip_step = True
                    step_tok = 0
                else:
                    window_loss_sum += last_loss * tok_m
                    window_loss_n += tok_m
                    del loss, outputs
            pos = ce
            global_chunk += 1
            torch.cuda.empty_cache()
            if global_chunk >= 4:
                def _save_midblock():
                    save_ckpt(global_chunk,
                              window_loss_sum / max(1, window_loss_n),
                              cur_tok)
                _maybe_recycle(pool, f"chunk {global_chunk}",
                               save_state=_save_midblock)
            if global_chunk <= 32 or global_chunk % 64 == 0:
                ps = pool.stats()
                print(f"  gchunk {global_chunk} L={L} pos={cur_tok} "
                      f"cs={ce-cs} ({time.time()-t0:.0f}s loss={last_loss:.4f} "
                      f"pool={ps['totalUsed']/1e9:.2f}GB)", flush=True)
            # One optimizer step per ACCUM_TOKENS (token budget, not chunks).
            do_opt = (step_tok >= ACCUM_TOKENS)
            if do_opt:
                if skip_step:
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    skip_step = False
                    window_loss_sum = 0.0
                    window_loss_n = 0
                else:
                    # scale summed grads to a true per-token mean
                    for _pg in optimizer.param_groups:
                        for _pp in _pg["params"]:
                            if _pp.grad is not None:
                                _pp.grad.div_(max(1, step_tok))
                    gn = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP_NORM)
                    last_gn = float(gn)
                    optimizer.step()
                    scheduler.step()
                    ema.update()
                    optimizer.zero_grad()
                    step += 1
                    wmean = window_loss_sum / max(1, window_loss_n)
                    print(f"[step {step}] loss={wmean:.4f} gn={last_gn:.2f} "
                          f"tok={step_tok} ({time.time()-t0:.0f}s)", flush=True)
                    window_loss_sum = 0.0
                    window_loss_n = 0
                    if step % SAVE_INTERVAL == 0:
                        sp = save_ckpt(global_chunk, wmean, cur_tok)
                        print(f"[ckpt] -> {sp}", flush=True)
                        cleanup_checkpoints(OUT_DIR, keep=KEEP_CHECKPOINTS)
                step_tok = 0
        del block
        torch.cuda.empty_cache()
    print(f"DONE steps={step} chunks={global_chunk}", flush=True)


if __name__ == "__main__":
    main()
