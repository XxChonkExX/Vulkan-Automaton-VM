
## auto_best_fit.py (Aug 29, 2026)

Auto-compute CHONK_CHUNK and CHONK_MIN_BLOCK_GB from model config + memory budget,
replacing hardcoded values. Handles:
- INT4 quant + merged buffers
- recompute (p/scores excluded from min_block)
- Vulkan driver exportable heap cap (~40GB, so MIN_BLOCK_GB <= heap/2)
- power-of-2 rounding for pool buckets

Usage:
    python auto_best_fit.py /path/to/model 131072
    # -> exports: CHONK_CHUNK=256, CHONK_MIN_BLOCK_GB=16 (Granite 30B @ 131K)

Override wall/act/driver:
    AUTO_FIT_WALL_GB=110 AUTO_FIT_DRIVER_HEAP_GB=40 python auto_best_fit.py ...

Result for Granite 4.2 30B @ 131K (current):
    Model: 64 layers, hidden 4096, 32Q/8KV, head_dim 128, 30.1B params
    KV cache: 32.0 GB @ 131K (bf16)
    LoRA r=128: 1174M params -> 2.19 GB bf16 + 13.12 GB AdamW
    INT4 base ~14.0 GB; merged q/s/z = 14.0 / 0.88 / 0.88 GB
    base+KV+LoRA+opt = 77.1 GB
    chunk=256: act=0.25GB total=77.3GB -> OK
    MIN_BLOCK_GB=16 (covers quant_q 14GB * 1.25; cap 16GB for 2 blocks in 40GB driver heap)

Math fixes from v1:
- LoRA param count: per linear A=r*in, B=r*out (not r*(in+out) which overcounted)
- min_block excludes p/scores when recompute=1 (default)
- min_block capped to floor(driver_heap/2) so >=2 blocks fit
- power-of-2 rounding for pool ladder alignment
