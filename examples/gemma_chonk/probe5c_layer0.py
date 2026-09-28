#!/usr/bin/env python3
"""probe5c pair -- submodule outputs inside decoder layer 0, plain vs chonk."""
import json
import os
import sys

import numpy as np
import torch

MP = "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged"
OUT = "/tmp/opencode/l0_ref.pt"
mode = sys.argv[1]  # "plain" or "chonk"

man = json.load(open("/home/chonke/Downloads/gemma412b/corpus/pools/manifest.json"))
info = man["pools"]["32768"]
arr = np.memmap(info["file"], dtype=np.int32, mode="r")

if mode == "chonk":
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
    sys.path.insert(0, os.path.join(REPO, "_build"))
    sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))
    from chonk import build_lora_chonk_setup, install_chonk_allocator
    from transformers import AutoConfig
    install_chonk_allocator()
    ids = torch.from_numpy(
        arr.reshape(info["n_blocks"], 32768)[0][:1024].astype(np.int64)
    ).unsqueeze(0).cuda()
    config = AutoConfig.from_pretrained(MP, trust_remote_code=True
                                        ).get_text_config(decoder=True)
    setup = build_lora_chonk_setup(
        MP, config, 1, 8192, lora_r=16, lora_alpha=32, lora_dropout=0.0,
        attn_implementation="eager", quantize=True, quant_group_size=128,
        quant_bits=4, act_budget_gb=1.0, staging_gb=1.0)
    model = setup["model"]
    layers = model.base_model.model.model.layers
else:
    from transformers import AutoModelForCausalLM
    ids = torch.from_numpy(
        arr.reshape(info["n_blocks"], 32768)[0][:1024].astype(np.int64)
    ).unsqueeze(0).cuda()
    model = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    layers = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == 48:
            layers = mod
            break

caps = {}


def _mk(key):
    def h(m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        if isinstance(x, torch.Tensor) and key not in caps:
            caps[key] = x.detach().float().cpu()
    return h


SKIP_PREFIXES = ("rotary", "norm")  # tiny tensors fine, but keep it lean
for name, mod in layers[0].named_modules():
    if name == "":
        continue
    leaf = next(mod.children(), None)
    if leaf is not None:  # composite modules get hooks too
        pass
    caps.setdefault
    mod.register_forward_hook(_mk(name or "<layer0>"))
layers[0].register_forward_hook(_mk("<layer0-out>"))

with torch.no_grad():
    model(input_ids=ids)

if mode == "plain":
    torch.save(caps, OUT)
    print(f"saved {len(caps)} tensors -> {OUT}", flush=True)
else:
    ref = torch.load(OUT)
    keys = sorted(set(ref) | set(caps))
    print(f"{'submodule':44s} {'ref_max':>10s} {'new_max':>10s} {'maxdiff':>10s}")
    for k in keys:
        if k not in ref or k not in caps:
            print(f"{k:44s} {'MISSING' if k not in caps else 'only-new':>10s}")
            continue
        r, n = ref[k], caps[k]
        if r.shape != n.shape:
            print(f"{k:44s} shape {tuple(r.shape)} vs {tuple(n.shape)}")
            continue
        print(f"{k:44s} {r.abs().max().item():10.3f} "
              f"{n.abs().max().item():10.3f} "
              f"{(r - n).abs().max().item():10.3f}", flush=True)
