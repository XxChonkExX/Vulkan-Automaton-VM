#!/usr/bin/env python3
"""probe5a: plain bf16 reference run (STOCK allocator). Captures per-layer
hidden states + embed output + logits for block0[:1024]; saves to disk."""
import json
import os
import sys

import numpy as np
import torch

OUT = "/tmp/opencode/hiddens_ref.pt"

MP = "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged"
man = json.load(open("/home/chonke/Downloads/gemma412b/corpus/pools/manifest.json"))
info = man["pools"]["32768"]
arr = np.memmap(info["file"], dtype=np.int32, mode="r")
ids = torch.from_numpy(
    arr.reshape(info["n_blocks"], 32768)[0][:1024].astype(np.int64)
).unsqueeze(0).cuda()

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MP, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()

caps = {"embed": None, "layers": []}


def _emb(m, i, o):
    caps["embed"] = o.detach().float().cpu()


model.get_input_embeddings().register_forward_hook(_emb)
def _find_layers(root, n=48):
    for _, mod in root.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n:
            return mod
    raise RuntimeError("layer list not found")


for lyr in _find_layers(model):
    def _mk(store):
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            store.append(x.detach().float().cpu())
        return h
    lyr.register_forward_hook(_mk(caps["layers"]))

with torch.no_grad():
    logits = model(input_ids=ids).logits
caps["logits_top"] = logits[0, :8].detach().float().cpu()
caps["logits_argmax"] = logits.argmax(-1)[0].detach().cpu()
del logits
torch.save(caps, OUT)
print(f"saved {len(caps['layers'])} layer hiddens + embed "
      f"-> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)", flush=True)
