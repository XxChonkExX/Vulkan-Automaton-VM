# auto_best_fit.py — model memory budgeting (Aug 29, 2026)

## Why this exists

Every long-context training run starts with the same anxious arithmetic:
*will this fit?* Weights + KV cache + LoRA adapters + optimizer states +
activations, against a driver heap that is smaller than the physical VRAM
and a pool that wants power-of-two buckets. Done by hand it takes an hour
and is wrong by the third config change. `auto_best_fit.py` (repo root)
does it from the model config plus a memory budget, replacing the old
hardcoded `CHONK_CHUNK` / `CHONK_MIN_BLOCK_GB` values.

## What it accounts for

- **INT4 quantization + merged buffers**: quantized base weights plus the
  merged query/score/zero buffers, each with its own footprint.
- **Recompute**: with activation recompute on (the default), attention
  probabilities and scores are *excluded* from the minimum-block
  calculation — they never sit in memory at the same time as everything
  else, so budgeting for them wastes gigabytes.
- **Vulkan driver exportable heap cap** (~40 GB observed): `MIN_BLOCK_GB`
  is capped at half the driver heap so at least two blocks always fit.
- **Power-of-2 rounding** so the results land on pool-bucket boundaries
  instead of between them.

## Usage

```sh
python auto_best_fit.py /path/to/model 131072
# -> exports: CHONK_CHUNK=256, CHONK_MIN_BLOCK_GB=16 (Granite 30B @ 131K)
```

Override the assumptions when your box differs:

```sh
AUTO_FIT_WALL_GB=110 AUTO_FIT_DRIVER_HEAP_GB=40 python auto_best_fit.py ...
```

## Worked example: Granite 4.2 30B @ 131K

Model: 64 layers, hidden 4096, 32 query / 8 KV heads, head_dim 128,
30.1B params.

| Component | Size |
|---|---|
| KV cache @ 131K (bf16) | 32.0 GB |
| LoRA r=128 (1174M params) | 2.19 GB bf16 + 13.12 GB AdamW |
| INT4 base | ~14.0 GB |
| merged q/s/z | 14.0 / 0.88 / 0.88 GB |
| **base + KV + LoRA + optimizer** | **77.1 GB** |

With chunk=256, activations add ~0.25 GB for 77.3 GB total — fits.
`MIN_BLOCK_GB=16` covers the 14 GB quantized base × 1.25 headroom, capped
at 16 GB so two blocks fit in the 40 GB driver heap.

## Math fixes from v1 (read: ways v1 lied)

- LoRA param count is per-linear `A=r*in, B=r*out` — the old `r*(in+out)`
  overcounted.
- `min_block` excludes probabilities/scores when recompute=1 (default).
- `min_block` capped to `floor(driver_heap/2)` so ≥2 blocks fit.
- Power-of-2 rounding for pool ladder alignment.

Related: `docs/sub_block_granite_notes.md` (the training setup this sizes),
`run_granite_long.sh` (the runner that consumes the exports).
