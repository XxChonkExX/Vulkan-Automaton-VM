#!/usr/bin/env python3
"""chonk_probe3.py -- bisect WHEN the live embed parameter dies.

Stages, each followed by the SAME vital check on the live module:
  S1 right after build_lora_chonk_setup returns
  S2 after A/B audits (buffer scans)
  S3 after C1 plain-param audit (kernel casts + D2H on every param)
  S4 after C2 dequant audits (3 modules)
  S5 after C3 byte audits

Vital check (identical each time):
  - weight row gather via F.embedding on the LIVE module (exactly what
    the forward does)
  - .float() cast max (kernel read of the whole matrix)
  - full model forward logits absmax
  - data_ptr + storage snapshot of the live Parameter
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(REPO, "_build"))
sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))

import torch
import torch.nn.functional as F

from chonk import build_lora_chonk_setup, install_chonk_allocator
from vulkanvm_quant_py import QuantLinear, quantize_weight, dequantize_weight
from transformers import AutoConfig
from safetensors.torch import load_file

MODEL_PATH = os.environ.get(
    "CHONK_MODEL", "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged")
SEED = 1234


def vital(model, ids, tag):
    emb = model.get_input_embeddings()
    w = emb.weight
    row = F.embedding(ids[:, :4], w)
    gather = float(row.float().abs().max().item())
    castmax = float(w.detach().float().abs().max().item())
    with torch.no_grad():
        lg = model(input_ids=ids).logits
    lmax = float(lg.float().abs().max().item())
    del lg
    torch.cuda.empty_cache()
    print(f"[{tag}] ptr={w.data_ptr()} gather={gather:.4f} "
          f"cast={castmax:.4f} logits={lmax:.4f} "
          f"{'ALIVE' if gather > 0.01 and lmax > 0.5 else '*** DEAD ***'}",
          flush=True)
    return gather > 0.01 and lmax > 0.5


def main():
    torch.manual_seed(SEED)
    install_chonk_allocator()
    ids = torch.randint(1000, 50000, (1, 64), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(SEED))
    full = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = full.get_text_config(decoder=True)
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, 1, 8192,
        lora_r=16, lora_alpha=32, lora_dropout=0.0,
        attn_implementation="eager",
        quantize=True, quant_group_size=128, quant_bits=4,
        act_budget_gb=1.0, staging_gb=1.0)
    model = setup["model"]
    alive = vital(model, ids, "S1 fresh build     ")
    if not alive:
        print("died BEFORE any audit -- the build itself poisons it",
              flush=True)
        return 1

    ckpt = {}
    for f in sorted(glob.glob(f"{MODEL_PATH}/*.safetensors")):
        ckpt.update(load_file(f))

    for n, b in model.named_buffers():
        if b.device.type == "cuda" and b.is_floating_point() and b.numel():
            float(b.float().abs().max().item())
    vital(model, ids, "S2 after B scans   ")

    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        p.detach().float().cpu()
    vital(model, ids, "S3 after C1 audits ")

    checked = 0
    for name, mod in model.named_modules():
        if ".base_layer" in name or checked >= 3:
            continue
        lin = getattr(mod, "base_layer", mod)
        if not isinstance(lin, QuantLinear):
            continue
        live = lin.dequantized_weight()
        float(live.float().abs().max().item())
        del live
        checked += 1
    vital(model, ids, "S4 after C2 dequant")

    checked = 0
    for name, mod in model.named_modules():
        if ".base_layer" in name or checked >= 3:
            continue
        lin = getattr(mod, "base_layer", mod)
        if not isinstance(lin, QuantLinear):
            continue
        lin.qweight.cpu()
        checked += 1
    vital(model, ids, "S5 after C3 bytes  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
