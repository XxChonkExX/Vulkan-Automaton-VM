#!/usr/bin/env python3
"""
Abliterate Granite 4.2 via Chonk Buffer (Vulkan) - NO ROCm.
Uses eager attention to avoid SDPA memory overhead.
"""

import os, sys, gc, math

os.environ["CHONK_DEVICE_PREFERENCE"] = "prefer_amd"
os.environ["CHONK_POOL_BLOCK_GB"] = "16"
os.environ["CHONK_POOL_BLOCK_SIZES_GB"] = "1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,56,60,64,72,80,88,96,104,112,128"
os.environ["CHONK_MIN_BLOCK_GB"] = "1"
os.environ["CHONK_PAUSE"] = "0.02"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:False"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:False"

VVM_ROOT = "/home/chonke/Vulkan-Automaton-VM"
sys.path.insert(0, os.path.join(VVM_ROOT, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(VVM_ROOT, "_build"))

from chonk import pool_mod
pool_mod.init()

import ctypes
_HIP = ctypes.CDLL("libamdhip64.so.7")
_probe = ctypes.c_void_p()
_HIP.hipMalloc(ctypes.byref(_probe), 4096)
_HIP.hipFree(_probe)

from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
allocator = CUDAPluggableAllocator(pool_mod.__file__, "chonk_allocator_alloc", "chonk_allocator_free")
change_current_allocator(allocator)

import torch
import torch.linalg as LA
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/chonke/Downloads/granite")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
config.language_model_only = True
config._attn_implementation = "eager"
print(f"Config: {config.num_hidden_layers}L, {config.hidden_size}H, eager attention")

from chonk import ChonkPool
chonk_pool = ChonkPool()

print("Loading model into Chonk buffer...")
from chonk import load_model_directly_to_chonk
model_buffer, quant_dict = load_model_directly_to_chonk(
    MODEL_PATH, config, chonk_pool,
    dtype=torch.bfloat16, chunk_size_mb=512,
    quantize_modules='all', quant_group_size=128, quant_bits=4,
)
print(f"Model loaded. Pool: {chonk_pool.stats()}")

from chonk import build_model_from_chonk_buffer
skip_modules = set(quant_dict.keys()) if quant_dict else set()
model = build_model_from_chonk_buffer(config, model_buffer, dtype=torch.bfloat16,
                                       skip_modules=skip_modules, attn_implementation="eager")

print("Setting up LoRA...")
from chonk import build_chonk_cache, create_optimizer_states_in_chonk, swap_quantized_base_layers
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
chonk_model = get_peft_model(model, lora_config)
swap_quantized_base_layers(chonk_model, quant_dict, group_size=128, bits=4)

# Move LoRA adapters from meta to CUDA (they were created before base layers were swapped)
print("Moving LoRA adapters to CUDA...")

def _replace_meta_params(module, prefix=""):
    """Recursively replace meta parameters with CUDA ones."""
    for name, child in module.named_children():
        _replace_meta_params(child, f"{prefix}{name}.")
    for name, param in list(module.named_parameters(recurse=False)):
        if param.device.type == "meta":
            full_name = f"{prefix}{name}"
            shape = param.shape
            if "lora_A" in full_name:
                new_p = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
                torch.nn.init.kaiming_uniform_(new_p, a=math.sqrt(5))
            else:
                new_p = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
            setattr(module, name, torch.nn.Parameter(new_p, requires_grad=True))
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf.device.type == "meta":
            full_name = f"{prefix}{name}"
            new_b = torch.zeros(buf.shape, device="cuda", dtype=buf.dtype if buf.dtype != torch.float32 else torch.bfloat16)
            module.register_buffer(name, new_b)

_replace_meta_params(chonk_model)

trainable = [p for p in chonk_model.parameters() if p.requires_grad]
n_trainable = sum(p.numel() for p in trainable)
print(f"LoRA: {n_trainable / 1e6:.1f}M trainable, {len(trainable)} params")

optimizer_states, opt_buffer = create_optimizer_states_in_chonk(trainable, chonk_pool)
print(f"Pool after setup: {chonk_pool.stats()}")

# ============================================================
# Residual direction computation using forward hooks (NOT generate)
# ============================================================
def get_residuals(model, tokenizer, prompts, system_prompt, max_prompts=20):
    """Compute mean hidden-state residuals per layer using forward()."""
    hidden_size = model.config.hidden_size
    layers = model.model.model.layers
    num_layers = len(layers)
    layer_sums = [torch.zeros(hidden_size, device="cuda") for _ in range(num_layers)]
    layer_counts = [0] * num_layers

    for i, p in enumerate(prompts[:max_prompts]):
        chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": p.user}],
            add_generation_prompt=True, tokenize=False,
        )
        inputs = tokenizer(chat, return_tensors="pt").to("cuda")
        residuals = [None] * num_layers

        def make_hook(idx):
            def hook(module, inp, out):
                residuals[idx] = out[0].detach().mean(dim=(0, 1))
            return hook

        handles = [layer.register_forward_hook(make_hook(i))
                   for i, layer in enumerate(layers)]

        with torch.no_grad():
            model(**inputs)

        for h in handles:
            h.remove()

        for j in range(num_layers):
            if residuals[j] is not None:
                layer_sums[j] += residuals[j].to("cuda")
                layer_counts[j] += 1

        del inputs, residuals
        gc.collect()

        if (i + 1) % 5 == 0:
            print(f"  Residuals: {i+1}/{max_prompts} prompts processed")

    means = torch.stack([
        layer_sums[j] / max(layer_counts[j], 1) for j in range(num_layers)
    ])
    del layer_sums, layer_counts
    gc.collect()
    return means


def count_refusals(model, tokenizer, prompts, refusal_markers, system_prompt, max_prompts=3):
    """Count how many prompts produce refusal-like responses."""
    refusal_count = 0
    for p in prompts[:max_prompts]:
        chat = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": p.user}],
            add_generation_prompt=True, tokenize=False,
        )
        inputs = tokenizer(chat, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1, use_cache=True)
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        response_lower = response.lower().replace("*", "").replace("'", '"')
        response_clean = " ".join(response_lower.split())
        is_refusal = any(m.lower() in response_clean for m in refusal_markers)
        if is_refusal or not response.strip():
            refusal_count += 1
        del inputs, outputs
        gc.collect()
    return refusal_count


# ============================================================
# Heretic-style abliteration logic (extracted from heretic/model.py)
# ============================================================
def apply_abliteration(model, tokenizer, refusal_directions,
                       direction_scope="per_layer", direction_idx=32,
                       max_weight=1.0, min_weight=0.5,
                       max_weight_position=48.0, min_weight_distance=16.0):
    """Apply directional ablation to LoRA adapters."""
    num_layers = len(model.model.model.layers)

    for layer_idx in range(num_layers):
        layer = model.model.model.layers[layer_idx]
        if not hasattr(layer, "mlp") or not hasattr(layer, "self_attn"):
            continue

        if direction_scope == "global":
            w, idx = math.modf(direction_idx + 1)
            refusal_dir = F.normalize(
                refusal_directions[int(idx)].lerp(refusal_directions[int(idx)+1], w),
                p=2, dim=0
            )
        else:
            refusal_dir = refusal_directions[layer_idx + 1]

        distance = abs(layer_idx - max_weight_position)
        if distance > min_weight_distance:
            continue
        weight = max_weight + (distance / min_weight_distance) * (min_weight - max_weight)

        for module_name in ["self_attn.o_proj", "mlp.down_proj"]:
            parts = module_name.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            if not hasattr(mod, "base_layer"):
                continue

            base = mod.base_layer
            if not hasattr(base, "weight"):
                continue

            v = refusal_dir.to(base.weight.device)
            W = base.weight.data.to(torch.float32).view(base.weight.shape[0], -1)

            lora_A = (v @ W).view(1, -1)
            lora_B = (-weight * v).view(-1, 1)

            if hasattr(mod, "lora_A") and "default" in mod.lora_A:
                mod.lora_A["default"].weight.data = lora_A.to(mod.lora_A["default"].weight.dtype).to(mod.lora_A["default"].weight.device)
                mod.lora_B["default"].weight.data = lora_B.to(mod.lora_B["default"].weight.dtype).to(mod.lora_B["default"].weight.device)


# ============================================================
# Main abliteration pipeline
# ============================================================
from heretic.utils import Prompt, load_prompts

class Settings:
    pass

settings = Settings()
settings.dataset = None
settings.split = None
settings.column = "text"
settings.commit = None
settings.prefix = ""
settings.suffix = ""
settings.system_prompt = None

print("Loading prompts...")
# Direct dataset loading
from datasets import load_dataset

good_ds = load_dataset("mlabonne/harmless_alpaca", split="train[:50]")
good_prompts = [Prompt(user=x["text"], system="You are a helpful assistant.") for x in good_ds]

bad_ds = load_dataset("mlabonne/harmful_behaviors", split="train[:50]")
bad_prompts = [Prompt(user=x["text"], system="You are a helpful assistant.") for x in bad_ds]

good_eval_ds = load_dataset("mlabonne/harmless_alpaca", split="test[:10]")
good_eval_prompts = [Prompt(user=x["text"], system="You are a helpful assistant.") for x in good_eval_ds]

bad_eval_ds = load_dataset("mlabonne/harmful_behaviors", split="test[:10]")
bad_eval_prompts = [Prompt(user=x["text"], system="You are a helpful assistant.") for x in bad_eval_ds]

print(f"Good: {len(good_prompts)}, Bad: {len(bad_prompts)}, "
      f"Good eval: {len(good_eval_prompts)}, Bad eval: {len(bad_eval_prompts)}")

REFUSAL_MARKERS = [
    "sorry", "i can't", "i cannot", "i won't", "i will not",
    "i'm unable", "i am unable", "i'm not able", "i am not able",
    "as an ai", "i'm an ai", "i am an ai",
    "i'm designed to", "i am designed to",
    "violat", "prohibit", "illegal", "harmful", "unethical",
]

SYSTEM_PROMPT = "You are a helpful assistant."

print("Computing residual directions (good prompts)...")
gc.collect()
good_means = get_residuals(chonk_model, tokenizer, good_prompts, SYSTEM_PROMPT, max_prompts=20)
print("Computing residual directions (bad prompts)...")
gc.collect()
bad_means = get_residuals(chonk_model, tokenizer, bad_prompts, SYSTEM_PROMPT, max_prompts=20)

refusal_directions = F.normalize(bad_means - good_means, p=2, dim=1)
print(f"Refusal directions: {refusal_directions.shape}")

# Check baseline refusals
print("Counting baseline refusals...")
gc.collect()
baseline_refusals = count_refusals(chonk_model, tokenizer, bad_eval_prompts,
                                    REFUSAL_MARKERS, SYSTEM_PROMPT, max_prompts=3)
print(f"Baseline refusals: {baseline_refusals}/3")

# Optimization loop
print("Starting abliteration optimization...")
best_score = baseline_refusals
best_state = None

for trial in range(50):
    scope = "global" if trial % 2 == 0 else "per_layer"
    print(f"\n--- Trial {trial} ({scope}) ---")

    apply_abliteration(chonk_model, tokenizer, refusal_directions,
                       direction_scope=scope, direction_idx=32,
                       max_weight=1.0, min_weight=0.5,
                       max_weight_position=48.0, min_weight_distance=16.0)

    gc.collect()
    refusals = count_refusals(chonk_model, tokenizer, bad_eval_prompts,
                               REFUSAL_MARKERS, SYSTEM_PROMPT, max_prompts=3)
    print(f"Refusals: {refusals}/3 (best={best_score})")

    if refusals < best_score:
        best_score = refusals
        best_state = {k: v.clone() for k, v in chonk_model.state_dict().items()}
        print(f"  NEW BEST!")

    if best_score == 0:
        print("Zero refusals achieved!")
        break

    # Reset LoRA weights if no improvement (re-randomize for next trial)
    if refusals >= best_score:
        for n, p in chonk_model.named_parameters():
            if "lora_A" in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif "lora_B" in n:
                torch.nn.init.zeros_(p)

print(f"\nOptimization complete. Best: {best_score}/10 refusals")

# Restore best state
if best_state is not None:
    chonk_model.load_state_dict(best_state)

# ============================================================
# Merge LoRA + dequantize QuantLinear, save as bf16
# ============================================================
def merge_and_save_model(peft_model, tokenizer, save_path):
    """Merge LoRA adapters into dequantized QuantLinear weights, save as bf16.

    PEFT's merge_and_unload() can't handle our QuantLinear because its .weight
    property returns a fresh tensor each time. We do the merge manually:
    1. Dequantize each QuantLinear weight (INT4 -> bf16)
    2. Add the LoRA delta (lora_B @ lora_A)
    3. Replace QuantLinear with a standard nn.Linear holding the merged bf16 weight
    4. Save the resulting model
    """
    import torch.nn as nn
    from vulkanvm_quant_py import QuantLinear

    print("Merging LoRA into dequantized weights...")
    base_model = peft_model.model  # GraniteForCausalLM

    merged_count = 0
    for name, module in peft_model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        base = module.base_layer
        if not isinstance(base, QuantLinear):
            continue
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue

        # Dequantize base weight
        W = base.weight.to(torch.float32)

        # Get LoRA adapter weights (may be on meta if created before swap)
        lora_A = module.lora_A["default"].weight.data
        lora_B = module.lora_B["default"].weight.data
        if lora_A.device.type == "meta":
            lora_A = torch.zeros_like(lora_A, dtype=torch.float32)
        else:
            lora_A = lora_A.to(torch.float32)
        if lora_B.device.type == "meta":
            lora_B = torch.zeros_like(lora_B, dtype=torch.float32)
        else:
            lora_B = lora_B.to(torch.float32)

        # Merge: W_merged = W + lora_B @ lora_A
        W_merged = W + (lora_B @ lora_A)

        # Create replacement nn.Linear with merged bf16 weight
        new_linear = nn.Linear(
            base.in_features, base.out_features,
            bias=(base.bias is not None),
        )
        new_linear.weight = nn.Parameter(W_merged.to(torch.bfloat16), requires_grad=False)
        if base.bias is not None:
            new_linear.bias = nn.Parameter(base.bias.data.to(torch.bfloat16), requires_grad=False)

        # Navigate to the base model and replace the module
        # PEFT name: base_model.model.model.layers.X.Y.Z
        # Base model path: model.layers.X.Y.Z
        base_name = name
        if base_name.startswith("base_model.model."):
            base_name = base_name[len("base_model.model."):]

        parts = base_name.split(".")
        parent = base_model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_linear)

        merged_count += 1
        if merged_count % 50 == 0:
            print(f"  Merged {merged_count}/448 layers...")

    print(f"Merged {merged_count} QuantLinear layers with LoRA adapters")

    # Also handle lm_head if it has LoRA (it shouldn't, but be safe)
    # The non-quantized params (embed_tokens, lm_head, norm) are already bf16 views

    print(f"Saving merged model to {save_path}...")
    base_model.save_pretrained(save_path, max_shard_size="5GB")
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
# Save the abliterated model
save_path = os.environ.get("SAVE_PATH", "/home/chonke/Downloads/granite-abliterated")
merge_and_save_model(chonk_model, tokenizer, save_path)
