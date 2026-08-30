# Sub-Block Slab + Granite Fine-Tune (Aug 28)

- Sub-slab allocator: 512MB floor, 16 max blocks, 4 warm (`chonk_allocator.cpp`). Routes <512MB allocations to sub slab; >=512MB to main slab.
- Granite attention rewrite (`vulkanvm_attn_granite.py`): memory-efficient GQA recompute (no AMD fused SDPA, avoids display-driver reset via dma-buf).
- Training: native 131072 context, chunk 512, INT4 quant, LoRA r=128, wrapper with auto-resume (`run_granite_long.sh`).
- GPU direct stub fix (`gpu_direct_registration.cpp`) unblocks full build.

Commit: 8902e2b
