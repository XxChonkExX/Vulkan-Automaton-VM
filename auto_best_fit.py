#!/usr/bin/env python3
"""
auto_best_fit.py — compute optimal CHONK_CHUNK and CHONK_MIN_BLOCK_GB
from the model config + available memory, so the allocator and training
loop auto-fit the workload instead of relying on hardcoded values.

Math (all in bytes):
  KV cache per token        = 2 * L * kvH * D * 2   (bf16)
  KV cache @ seq_len        = KV/tok * seq_len
  p/scores @ full context   = B * H * chunk * (chunk + seq_len - chunk) * 2
                            = B * H * chunk * seq_len * 2
                            -> peaks near end of sequence
  Min-block needed          = max(p/scores, quant buffers, base staging) * safety
  Activation per chunk      = chunk * hidden * 2 * L * 2  (in+out, bf16)
  Total budget              = base + KV + LoRA + opt + act + buf

Defaults: target wall 110 GB, safety 1.25x, act target 2 GB.
"""
import json, os, sys, math

WALL_GB      = float(os.environ.get("AUTO_FIT_WALL_GB", "110"))
SAFETY       = float(os.environ.get("AUTO_FIT_SAFETY", "1.25"))
ACT_TARGET_GB= float(os.environ.get("AUTO_FIT_ACT_GB", "2.0"))
RECOMPUTE    = os.environ.get("AUTO_FIT_RECOMPUTE", "1") == "1"  # p/scores NOT saved
DRIVER_HEAP_GB = float(os.environ.get("AUTO_FIT_DRIVER_HEAP_GB", "40"))  # Vulkan exportable cap
B            = 1  # batch


def gb(b): return b / (1024 ** 3)


def best_fit(model_path, seq_len):
    cfg = json.load(open(os.path.join(model_path, "config.json")))
    L     = cfg["num_hidden_layers"]
    H     = cfg["num_attention_heads"]
    kvH   = cfg["num_key_value_heads"]
    D     = cfg.get("head_dim") or (cfg["hidden_size"] // H)
    V     = cfg["vocab_size"]
    inter = cfg.get("intermediate_size", 4 * cfg["hidden_size"])
    hidden = cfg["hidden_size"]
    params = cfg.get("num_parameters") or (L * (4 * hidden * hidden + 3 * hidden * inter))

    print(f"Model: L={L} hidden={hidden} H={H} kvH={kvH} D={D} V={V} params~{params/1e9:.1f}B")

    # KV cache (bf16)
    kv_per_tok = 2 * L * kvH * D * 2
    kv_total   = kv_per_tok * seq_len
    print(f"  KV cache: {gb(kv_per_tok):.1f} MB/tok -> {gb(kv_total):.1f} GB @ {seq_len}")

    # p/scores (worst case, full context, bf16)
    def pscores(chunk):
        return B * H * chunk * seq_len * 2  # bf16

    # Activations per chunk (rough: chunk * hidden * 2 (in+out) * L layers, bf16)
    def act(chunk):
        return chunk * hidden * 2 * L * 2

    # LoRA r=128 alpha=256 (Q/K/V/O + 3 MLP per layer)
    r = 128
    lora_per_layer = 0
    # Q/K/V/O: in=hidden, out=hidden
    lora_per_layer += 4 * (r * hidden + r * hidden)
    # MLP gate, up: in=hidden, out=inter
    lora_per_layer += 2 * (r * hidden + r * inter)
    # MLP down: in=inter, out=hidden
    lora_per_layer += 1 * (r * inter + r * hidden)
    lora_params = L * lora_per_layer
    lora_bf16   = lora_params * 2
    opt_fp32    = lora_params * 3 * 4  # AdamW m, v (fp32)
    print(f"  LoRA r=128: {lora_params/1e6:.0f}M params -> {gb(lora_bf16):.2f} GB bf16, {gb(opt_fp32):.2f} GB AdamW")

    # INT4 base (merged: q/s/z buffers; rest is dequantized at use, not stored)
    base_int4 = params * 0.5  # 4 bits
    quant_q   = params * 0.5
    quant_s   = params / 128 * 4
    quant_z   = params / 128 * 4
    print(f"  INT4 base ~{gb(base_int4):.1f} GB; merged quant buffers q={gb(quant_q):.1f} s={gb(quant_s):.2f} z={gb(quant_z):.2f} GB")

    # Pick chunk: act target first, then ensure total fits
    wall = WALL_GB * (1024 ** 3)
    base_kv = base_int4 + quant_q + quant_s + quant_z + kv_total + lora_bf16 + opt_fp32
    print(f"  base+KV+LoRA+opt = {gb(base_kv):.1f} GB")

    # largest p/scores determines min_block
    # try chunks: 256, 512, 1024, 2048
    candidates = [256, 512, 1024, 2048]
    chosen = None
    for chunk in candidates:
        a = act(chunk)
        p = pscores(chunk)
        total = base_kv + a + 0.5  # buf
        if total < wall and gb(a) < ACT_TARGET_GB * SAFETY:
            chosen = chunk
            print(f"  chunk={chunk}: act={gb(a):.2f}GB p/scores={gb(p):.2f}GB total={gb(total):.1f}GB -> OK")
            break
        else:
            print(f"  chunk={chunk}: act={gb(a):.2f}GB p/scores={gb(p):.2f}GB total={gb(total):.1f}GB -> skip (act>{ACT_TARGET_GB}GB or total>{WALL_GB}GB)")

    if chosen is None:
        chosen = 256  # safest fallback
        print(f"  WARNING: no candidate fits; falling back to chunk=256")

    # Min block: largest SAVED tensor (not p/scores if recompute) + safety.
    # Quant q buffer is the biggest persistent allocation.
    largest = quant_q
    if not RECOMPUTE:
        largest = max(pscores(chosen), largest)
    mb = int(math.ceil(gb(largest) * SAFETY))
    # Largest power-of-2 such that >=2 blocks fit in driver heap
    max_pow2 = int(DRIVER_HEAP_GB // 2)
    max_pow2_p = 1
    while max_pow2_p * 2 <= max_pow2:
        max_pow2_p *= 2
    if max_pow2_p < 4:
        max_pow2_p = 4
    mb_pow2 = 1
    while mb_pow2 < mb:
        mb_pow2 *= 2
    if mb_pow2 < 4:
        mb_pow2 = 4  # floor 4GB
    mb_pow2 = min(mb_pow2, max_pow2_p)
    note = "p/scores excluded (recompute)" if RECOMPUTE else "includes p/scores"
    print(f"  -> CHUNK={chosen}, MIN_BLOCK_GB={mb_pow2} (covers {gb(largest):.1f}GB * {SAFETY}; {note}; cap {max_pow2_p}GB, driver heap {DRIVER_HEAP_GB}GB)")

    return chosen, mb_pow2


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "/home/chonke/Downloads/granite-abliterated"
    seq   = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
    chunk, mb = best_fit(model, seq)
    print(f"\nExport:  export CHONK_CHUNK={chunk}  export CHONK_MIN_BLOCK_GB={mb}")
