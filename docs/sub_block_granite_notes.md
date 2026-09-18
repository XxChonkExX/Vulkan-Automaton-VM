# Sub-Block Slab + Granite Fine-Tune (Aug 28)

Long-context LoRA fine-tuning (131K context) breaks naive allocators two
ways: thousands of small transient allocations fragment the pool, and one
giant attention workspace can OOM the run halfway through. This note records
the two-part answer used for the Granite 30B campaign.

## 1. Sub-slab routing (`python/vulkanvm_torch/allocator/chonk_allocator.cpp`)

Allocations under 512 MB route to a dedicated sub-slab (16 max blocks,
4 kept warm); 512 MB and up go to the main slab. The split exists because
small/large allocations have opposite failure modes: small ones fragment
(leak usable space between odd sizes), large ones fail outright when no
contiguous run fits. Keeping them in separate arenas means the transient
churn of a training step can never wedge the big-tensor lane, and warm
blocks avoid re-creating device memory every step.

## 2. Granite attention rewrite (`python/vulkanvm_torch/vulkanvm_attn_granite.py`)

Memory-efficient grouped-query attention with activation recompute: the
forward pass drops the attention scores instead of stashing them, and the
backward pass recomputes them. That trades ~30% extra compute for removing
the single largest activation tensor from the memory budget — the difference
between fitting 131K context and not. Written against AMD specifically
because there is no fused SDPA kernel to lean on; it also sidesteps the
display-driver reset that the DMA-BUF fast path could trigger on a
display-attached card.

## 3. The training recipe

- Native 131072 context, chunk size 512, INT4-quantized base, LoRA r=128.
- Wrapper with auto-resume (`run_granite_long.sh`): long runs die for dumb
  reasons (driver hiccups, power, Tuesday); resume-from-checkpoint is part
  of the setup, not an afterthought.
- Unblocked by the GPU-direct stub fix (`src/network/gpu_direct_registration.cpp`),
  which got the full build linking again.

Commit: 8902e2b. Full campaign numbers in `OPTIMIZATION_LOG.md`; the
memory-budget math that sizes all of this lives in `auto_best_fit.py`
(see `docs/auto_best_fit_notes.md`).
