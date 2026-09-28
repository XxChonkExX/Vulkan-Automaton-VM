#!/usr/bin/env python3
"""chonk_probe4.py -- WHERE does prediction quality die? Same real corpus
chunk through three paths in one process:
  L1 eager attention (transformers reference), no cache
  L2 patched cache-aware attention, single 1024-token chunk (cache empty)
  L3 patched cache-aware attention, four 256-token chunks
Plus layer-47 hidden-state divergence between L1 and L2.
Healthy expectation: all three losses within ~0.2 of each other and in the
2-6 range for a 12B base on RPG text.
"""
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
import torch.nn as nn
import torch.nn.functional as F

from chonk import build_lora_chonk_setup, install_chonk_allocator, \
    reset_chonk_cache
from vulkanvm_attn_gemma_cache import patch_gemma_attention_cache
from transformers import AutoConfig

MODEL_PATH = "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged"
POOLS = "/home/chonke/Downloads/gemma412b/corpus/pools"


def ce_loss(model, ids):
    with torch.no_grad():
        lg = model(input_ids=ids).logits
        loss = F.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
            ids[:, 1:].reshape(-1)).item()
        del lg
    torch.cuda.empty_cache()
    return loss


def main():
    install_chonk_allocator()
    man = json.load(open(os.path.join(POOLS, "manifest.json")))
    info = man["pools"]["32768"]
    arr = np.memmap(info["file"], dtype=np.int32, mode="r")
    block = torch.from_numpy(
        arr.reshape(info["n_blocks"], 32768)[0].astype(np.int64)
    ).unsqueeze(0).cuda()
    ids = block[:, :1024].contiguous()

    full = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = full.get_text_config(decoder=True)
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, 1, 8192,
        lora_r=16, lora_alpha=32, lora_dropout=0.0,
        attn_implementation="eager",
        quantize=True, quant_group_size=128,
        quant_bits=int(os.environ.get("CHONK_QBITS", "4")),
        act_budget_gb=1.0, staging_gb=1.0)
    model = setup["model"]
    kv_cache = setup["kv_cache"]

    l1 = ce_loss(model, ids)
    print(f"L1 eager 1024:            loss={l1:.4f}", flush=True)

    layers = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == config.num_hidden_layers:
            layers = mod
            break
    h1 = {}

    def _cap(m, args):
        h1["x"] = args[0][0].detach().float().abs().max().item()
    layers[-1].register_forward_pre_hook(_cap)
    ce_loss(model, ids)

    def _chonk_seq_length(_self, *a, **k):
        n = 0
        for _lyr in _self.layers:
            _c = getattr(_lyr, "cumulative_length", None)
            if _c is not None:
                n = max(n, int(_c.item()))
        return n
    import types as _types
    kv_cache.get_seq_length = _types.MethodType(_chonk_seq_length, kv_cache)
    patch_gemma_attention_cache(model, kv_cache)
    h2 = {}
    layers[-1].register_forward_pre_hook(
        lambda m, a: h2.update(x=a[0][0].detach().float().abs().max().item()))

    reset_chonk_cache(kv_cache)
    with torch.no_grad():
        lg = model(input_ids=ids, past_key_values=kv_cache,
                   use_cache=True).logits
        l2 = F.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]).float(),
            ids[:, 1:].reshape(-1)).item()
        del lg
    torch.cuda.empty_cache()
    print(f"L2 patched single 1024:   loss={l2:.4f} "
          f"(L47 in: eager={h1.get('x')} patched={h2.get('x')})", flush=True)

    reset_chonk_cache(kv_cache)
    with torch.no_grad():
        for s in range(0, 1024, 256):
            lg_last = model(input_ids=ids[:, s:s + 256],
                            past_key_values=kv_cache, use_cache=True).logits
        l3 = F.cross_entropy(
            lg_last[:, :-1].reshape(-1, lg_last.shape[-1]).float(),
            ids[:, 768 + 1:].reshape(-1)).item()
        del lg_last
    torch.cuda.empty_cache()
    print(f"L3 patched 4x256 (last):  loss={l3:.4f}", flush=True)

    # same-window eager loss for a fair L3 comparison
    with torch.no_grad():
        lg = model(input_ids=ids).logits
        l1w = F.cross_entropy(
            lg[:, 768:-1].reshape(-1, lg.shape[-1]).float(),
            ids[:, 769:].reshape(-1)).item()
        del lg
    print(f"L1 eager same window:     loss={l1w:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
