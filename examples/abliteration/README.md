# Chonk Abliteration — Granite 4.2 via Vulkan Chonk Buffer

Run Heretic-style directional ablation (abliteration) on large models **entirely
inside the Vulkan-Automaton-VM Chonk Buffer allocator** — no ROCm / HIP direct
allocation, no `device_map`, no 4-bit HuggingFace loaders. The model is loaded
straight from safetensors shards into Chonk pool memory, optionally INT4-quantized
per-layer, wrapped in PEFT LoRA, abliterated, then **dequantized + merged back to
a standard bf16 safetensors model** that any loader can use.

## Why

* Strix Halo (gfx1151) has a tiny HIP/VRAM carveout; `transformers` + ROCm cannot
  allocate the model on GPU.
* The Chonk Buffer (unified Vulkan memory) gives ~110 GB of addressable pool, so a
  58 GB bf16 model (or its ~16 GB INT4 form) fits comfortably.
* Abliteration only nudges LoRA adapters, so the base weights can stay quantized
  during training and be materialized at save time.

## Files

* `run_abliterate_vulkan.py` — end-to-end driver (loads → quantizes → LoRA →
  residuals → abliterates → saves bf16).
* `../python/vulkanvm_torch/chonk.py` — `load_model_directly_to_chonk`,
  `build_model_from_chonk_buffer`, `swap_quantized_base_layers`.
* `../python/vulkanvm_torch/vulkanvm_quant_py.py` — `QuantLinear` (INT4/INT8
  per-group quantized layer with a `.weight` property for PEFT/Heretic).

## Requirements

* AMD Strix Halo (or any Vulkan-capable GPU with enough unified memory)
* `Vulkan-Automaton-VM` built (`_build/vulkanvm_pool_test.so`)
* Python venv with `torch` (ROCm build), `transformers`, `peft`, `datasets`, `safetensors`
* Heretic installed (`pip install heretic-ai` or from source)

## Usage

```bash
# 1. Point at the base model (safetensors shards + config.json + tokenizer)
export MODEL_PATH=/path/to/granite-4.2

# 2. Run (everything flows through the Chonk allocator)
python run_abliterate_vulkan.py
```

The script:
1. Installs the Chonk pluggable CUDA allocator **before** importing torch.
2. Allocates a graduated Vulkan pool (1 GB … 128 GB blocks).
3. Streams safetensors shards into the pool; quantizes every Linear except
   `lm_head` to INT4 (group size 128).
4. Builds a `meta`-structured model whose params are zero-copy views into the pool.
5. Applies LoRA, swaps quantized base layers with `QuantLinear`.
6. Computes per-layer residual directions from harmless vs. harmful prompt sets.
7. Writes the abliteration delta into the LoRA adapters (`o_proj`, `down_proj`).
8. Dequantizes each `QuantLinear`, adds `lora_B @ lora_A`, replaces it with a plain
   `nn.Linear`, and saves a normal bf16 model to `granite-abliterated/`.

## Key implementation notes / fixes

* **`to_ckpt_name`**: Granite shards use bare `model.layers.*` keys (not the
  `model.language_model.*` wrapper prefix). The remap must return the name as-is.
* **`lm_head` excluded from quantization**: it has no `base_layer` PEFT wrapper, so
  a quantized `lm_head` would be stranded on `meta` and break `model.generate()`.
* **`QuantLinear.weight` property**: Heretic's `abliterate` and our merge code read
  `module.base_layer.weight`. `QuantLinear` exposes `weight` as the dequantized
  tensor so those paths work.
* **Meta→CUDA LoRA re-init**: PEFT creates LoRA adapters on `meta` before the base
  layers are swapped to the pool. After `swap_quantized_base_layers`, the adapters
  are re-created on CUDA with proper Kaiming/zero init.
* **Merge must be manual**: PEFT's `merge_and_unload` cannot write through the
  `QuantLinear.weight` property (it returns a fresh tensor each call). The script
  dequantizes + merges directly and replaces `QuantLinear` with `nn.Linear`.
* **Eager attention**: SDPA/flash on gfx1151 is unstable; `attn_implementation=
  "eager"` is forced.
* **Memory discipline**: residual computation uses `model(**inputs)` forward hooks
  (not `generate`), `gc.collect()` between prompts, and a small prompt budget
  (20 residuals, 3 eval) to avoid pool fragmentation during autoregressive runs.

## Result

The saved `granite-abliterated/` model is a standard bf16 Granite 4.2 (~55 GB,
12 shards) decensored via abliteration, loadable with plain
`AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16)`.

## Limitations

* INT4 abliteration of a 36 B model produces poor base quality (gibberish) because
  the quantization is aggressive; the *process* is what's demonstrated. For usable
  output, either (a) quantize fewer layers, or (b) skip quantization and run the
  abliteration on bf16 weights in the pool (set `quantize_modules=None`).
* `model.generate()` is slow through `QuantLinear` (per-layer on-the-fly
  dequant); keep eval `max_new_tokens` small.
