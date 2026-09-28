#!/usr/bin/env python3
"""probe5b: Chonk build run (pool allocator). Same input; compares per-layer
hiddens vs the plain reference saved by probe5a; reports the first layer
where divergence exceeds bf16 noise."""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(REPO, "_build"))
sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))

import numpy as np
import torch

OUT = "/tmp/opencode/hiddens_ref.pt"
MP = "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged"

from chonk import build_lora_chonk_setup, install_chonk_allocator
from transformers import AutoConfig

man = json.load(open("/home/chonke/Downloads/gemma412b/corpus/pools/manifest.json"))
info = man["pools"]["32768"]
arr = np.memmap(info["file"], dtype=np.int32, mode="r")

install_chonk_allocator()
ids = torch.from_numpy(
    arr.reshape(info["n_blocks"], 32768)[0][:1024].astype(np.int64)
).unsqueeze(0).cuda()
full = AutoConfig.from_pretrained(MP, trust_remote_code=True)
config = full.get_text_config(decoder=True)
setup = build_lora_chonk_setup(
    MP, config, 1, 8192,
    lora_r=16, lora_alpha=32, lora_dropout=0.0,
    attn_implementation="eager",
    quantize=True, quant_group_size=128, quant_bits=4,
    act_budget_gb=1.0, staging_gb=1.0)
model = setup["model"]

ref = torch.load(OUT)
caps = {"embed": None, "layers": []}


def _emb(m, i, o):
    caps["embed"] = o.detach().float().cpu()


model.get_input_embeddings().register_forward_hook(_emb)
for lyr in model.base_model.model.model.layers:
    def _mk(store):
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            store.append(x.detach().float().cpu())
        return h
    lyr.register_forward_hook(_mk(caps["layers"]))

with torch.no_grad():
    logits = model(input_ids=ids).logits

e_ref, e_new = ref["embed"], caps["embed"]
d = (e_ref - e_new).abs().max().item()
rel = d / e_ref.abs().max().item()
print(f"embed:  maxabs_diff={d:.4f} rel={rel:.4f} "
      f"(ref_max={e_ref.abs().max().item():.2f})", flush=True)

first_bad = None
for i, (hr, hn) in enumerate(zip(ref["layers"], caps["layers"])):
    d = (hr - hn).abs().max().item()
    scale = hr.abs().max().item()
    rel = d / max(scale, 1e-9)
    flag = "  <== FIRST DIVERGENT" if rel > 0.05 and first_bad is None else ""
    if rel > 0.05 and first_bad is None:
        first_bad = i
    if i < 4 or i % 8 == 0 or i == len(ref["layers"]) - 1 or flag:
        print(f"L{i:02d}: maxabs_diff={d:9.3f} scale={scale:9.2f} "
              f"rel={rel:.4f}{flag}", flush=True)

am_ref = ref["logits_argmax"]
am_new = logits.argmax(-1)[0].detach().cpu()
print(f"logits top1 agreement: {(am_ref == am_new).float().mean().item():.3f}",
      flush=True)
print(f"first divergent layer: {first_bad}", flush=True)
