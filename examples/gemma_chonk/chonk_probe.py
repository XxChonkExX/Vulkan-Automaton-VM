#!/usr/bin/env python3
"""chonk_probe.py v2 -- decisive single-process diagnostic for the lora_B
zero-grad mystery. Builds the full Chonk stack EXACTLY as the trainer does,
then audits, in ONE process, on the LIVE tensors:

  A. meta leftovers (params/buffers still on meta after swap+tying)
  B. buffer garbage (uninitialized buffers: finiteness + absmax)
  C1. plain bf16 params: exact vs checkpoint
  C2. QuantLinear dequant vs CPU-quantized roundtrip reference (load
      quantizes on CPU; a GPU-computed reference can differ by round ties)
  C3. raw storage: pool qweight/scales/zeros BYTES vs CPU reference
  D. eager forward liveness + logits baseline
  E. REAL trainer path: patched cache-aware attention, reset KV, correct
     chunk slices -- liveness per layer, logits agreement vs eager
  F. adapter causality on the real path: B=1 must move logits
  G. gradient flow through the chunked path: lora_A/lora_B grad absmax

GPU JOB: do NOT run while another trainer holds the pool.
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(REPO, "_build"))
sys.path.insert(0, os.path.join(REPO, "examples", "granite_chonk"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from chonk import (build_lora_chonk_setup, install_chonk_allocator,
                   reset_chonk_cache)
from vulkanvm_quant_py import (QuantLinear, quantize_weight,
                               dequantize_weight)
from vulkanvm_attn_gemma_cache import patch_gemma_attention_cache
from transformers import AutoConfig
from safetensors.torch import load_file

MODEL_PATH = os.environ.get(
    "CHONK_MODEL", "/home/chonke/Downloads/gemma412b/runs/dpo_abstain/merged")
N_AUDIT_LAYERS = int(os.environ.get("CHONK_PROBE_LAYERS", "3"))
SEED = 1234
report = {"model_class": None, "verdict": [], "fail": 0}


def verdict(ok, label, detail=""):
    line = f"{'PASS' if ok else 'FAIL'}  {label}" + (
        f"  [{detail}]" if detail else "")
    print(line, flush=True)
    report["verdict"].append(line)
    if not ok:
        report["fail"] += 1


def to_ckpt_names(name):
    if name in ("lm_head.weight", "model.lm_head.weight"):
        return ["lm_head.weight"]
    out = [name]
    if name.startswith("model."):
        out.append("model.language_model." + name[len("model."):])
    return out


def strip_peft(name):
    for pre in ("base_model.model.", "base_model."):
        if name.startswith(pre):
            name = name[len(pre):]
    return name


def find_layers(root, n):
    for _, mod in root.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n:
            return mod
    return None


def main():
    torch.manual_seed(SEED)
    install_chonk_allocator()
    full = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config = full.get_text_config(decoder=True)
    setup = build_lora_chonk_setup(
        MODEL_PATH, config, 1, 8192,
        lora_r=16, lora_alpha=32, lora_dropout=0.0,
        attn_implementation="eager",
        quantize=True, quant_group_size=128, quant_bits=4,
        act_budget_gb=1.0, staging_gb=1.0)
    model = setup["model"]
    kv_cache = setup["kv_cache"]
    report["model_class"] = type(model.base_model.model).__name__
    print(f"underlying model class: {report['model_class']}", flush=True)

    ckpt = {}
    for f in sorted(glob.glob(f"{MODEL_PATH}/*.safetensors")):
        ckpt.update(load_file(f))

    # --- A. meta leftovers (tied lm_head shares embed storage: exempt) -----
    emb_w = model.get_input_embeddings().weight
    meta_p, tied_lm = [], False
    for n, p in model.named_parameters():
        if p.device.type != "meta":
            continue
        if n.endswith("lm_head.weight") and p.data_ptr() == emb_w.data_ptr():
            tied_lm = True
            continue
        meta_p.append(n)
    meta_b = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    verdict(not meta_p, "A1 no meta params remain", f"{meta_p[:3]}")
    verdict(not meta_b, "A2 no meta buffers remain", f"{meta_b[:3]}")

    # --- B. buffer garbage --------------------------------------------------
    n_buf, nonfinite, wild = 0, [], []
    for n, b in model.named_buffers():
        if b.device.type != "cuda" or b.numel() == 0:
            continue
        n_buf += 1
        if b.is_floating_point():
            fin = bool(torch.isfinite(b).all().item())
            mx = float(b.float().abs().max().item())
            if not fin:
                nonfinite.append(n)
            if mx > 1e4:
                wild.append((n, mx))
    verdict(not nonfinite, "B1 all float buffers finite", str(nonfinite[:3]))
    verdict(not wild, "B2 no garbage-magnitude buffers", str(wild[:3]))
    print(f"      audited {n_buf} cuda buffers", flush=True)

    # --- C1. plain params: exact vs checkpoint -------------------------------
    plain_fail, plain_missing, plain_n = [], [], 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            continue
        stripped = strip_peft(n)
        found = next((k for k in to_ckpt_names(stripped) if k in ckpt), None)
        if found is None:
            if n.endswith("lm_head.weight") and p.data_ptr() == emb_w.data_ptr():
                continue
            plain_missing.append(stripped)
            continue
        ref = ckpt[found]
        if tuple(ref.shape) != tuple(p.shape):
            plain_fail.append((stripped, "shape"))
            continue
        d = (p.detach().float().cpu() - ref.float()).abs().max().item()
        plain_n += 1
        if d != 0.0:
            plain_fail.append((stripped, d))
    verdict(not plain_fail and not plain_missing,
            "C1 plain params land exactly",
            f"{plain_n} checked, {len(plain_fail)} mismatch, "
            f"{len(plain_missing)} unmatched: {plain_missing[:2]}")

    # --- C2/C3. QuantLinear: CPU-reference roundtrip + raw bytes --------------
    layers = find_layers(model, config.num_hidden_layers)
    idx = sorted({0, config.num_hidden_layers // 2,
                  config.num_hidden_layers - 1})[:N_AUDIT_LAYERS]
    q_fail, b_fail, q_n = [], [], 0
    for name, mod in model.named_modules():
        if ".base_layer" in name:
            continue
        lin = getattr(mod, "base_layer", mod)
        if not isinstance(lin, QuantLinear):
            continue
        if not (name.startswith(("lm_head", "model.embed"))
                or any(f".{i}." in name for i in idx)):
            continue
        stripped = strip_peft(name)
        found = next(
            (k for k in to_ckpt_names(stripped + ".weight") if k in ckpt),
            None)
        if found is None:
            q_fail.append((name, "unmatched"))
            continue
        ref_w32 = ckpt[found].to(torch.float32)
        qw, sw, zw = quantize_weight(ref_w32, bits=4, group_size=128)
        ref_deq = dequantize_weight(qw, sw, zw, bits=4, group_size=128,
                                    dtype=torch.bfloat16)
        live = lin.dequantized_weight()
        d = (live.float().cpu() - ref_deq.float()).abs().max().item()
        q_n += 1
        if d != 0.0:
            q_fail.append((name, d))
        nb_bad = int((lin.qweight.cpu() != qw.cpu()).sum())
        ns_bad = int((lin.scales.cpu() != sw).sum())
        nz_bad = int((lin.zeros.cpu() != zw).sum())
        if nb_bad or ns_bad or nz_bad:
            b_fail.append((name, nb_bad, ns_bad, nz_bad))
        del ref_w32, qw, sw, zw, ref_deq, live
    verdict(not q_fail, "C2 QuantLinear dequant matches CPU roundtrip",
            f"{q_n} checked, {len(q_fail)} mismatch: {q_fail[:2]}")
    verdict(not b_fail, "C3 pool quant storage byte-exact",
            f"{q_n} checked, {len(b_fail)} corrupt: {b_fail[:2]}")

    # --- D. eager forward liveness + baseline --------------------------------
    dev = "cuda"
    ids = torch.randint(1000, 50000, (1, 128), device=dev,
                        generator=torch.Generator(device=dev).manual_seed(SEED))

    cap_e = {}
    hooks = []

    def _mk_emb(cap, tag):
        def h(m, i, o):
            cap[f"{tag}_embed"] = float(
                o[0].detach().float().abs().max().item())
        return h

    def _mk_pre(li, cap, tag):
        def h(m, args):
            cap[f"{tag}_L{li}"] = float(
                args[0][0].detach().float().abs().max().item())
        return h

    hooks.append(model.get_input_embeddings().register_forward_hook(
        _mk_emb(cap_e, "eager")))
    if layers is not None:
        for li in idx:
            hooks.append(layers[li].register_forward_pre_hook(
                _mk_pre(li, cap_e, "eager")))
    with torch.no_grad():
        logits_e = model(input_ids=ids).logits.detach().clone()
    for h in hooks:
        h.remove()
    verdict(cap_e.get("eager_embed", 0.0) > 0.0, "D1 eager embed nonzero",
            f"absmax={cap_e.get('eager_embed')}")
    verdict(all(cap_e.get(f"eager_L{li}", 0.0) > 0.0 for li in idx),
            "D2 eager layer inputs nonzero",
            " ".join(f"L{li}={cap_e.get(f'eager_L{li}')}" for li in idx))
    verdict(logits_e.abs().max().item() > 0.5, "D3 eager logits sane",
            f"absmax={logits_e.abs().max().item():.2f}")

    # --- E. REAL trainer path: patched attention + KV + correct chunks --------
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

    cap_r = {}
    hooks = [model.get_input_embeddings().register_forward_hook(
        _mk_emb(cap_r, "real"))]
    if layers is not None:
        for li in idx:
            hooks.append(layers[li].register_forward_pre_hook(
                _mk_pre(li, cap_r, "real")))
    reset_chonk_cache(kv_cache)
    with torch.no_grad():
        model(input_ids=ids[:, :64], past_key_values=kv_cache, use_cache=True)
        logits_r = model(input_ids=ids[:, 64:], past_key_values=kv_cache,
                         use_cache=True).logits.detach().clone()
    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()
    verdict(cap_r.get("real_embed", 0.0) > 0.0, "E1 real-path embed nonzero",
            f"absmax={cap_r.get('real_embed')}")
    verdict(all(cap_r.get(f"real_L{li}", 0.0) > 0.0 for li in idx),
            "E2 real-path layer inputs nonzero",
            " ".join(f"L{li}={cap_r.get(f'real_L{li}')}" for li in idx))
    agree = (logits_r.argmax(-1) == logits_e[:, 64:].argmax(-1)
             ).float().mean().item()
    dmax = (logits_r.float() - logits_e[:, 64:].float()).abs().max().item()
    verdict(agree >= 0.90, "E3 real-path logits match eager",
            f"top1 agree={agree:.3f} maxabs={dmax:.3f}")

    # --- F. adapter causality on the real path ---------------------------------
    b_params = [(n, p) for n, p in model.named_parameters()
                if "lora_B" in n and p.requires_grad]
    with torch.no_grad():
        for _, p in b_params:
            p.fill_(1.0)
        reset_chonk_cache(kv_cache)
        model(input_ids=ids[:, :64], past_key_values=kv_cache, use_cache=True)
        logits_b1 = model(input_ids=ids[:, 64:], past_key_values=kv_cache,
                          use_cache=True).logits.detach()
        for _, p in b_params:
            p.zero_()
    delta = float((logits_b1 - logits_r).abs().max().item())
    verdict(delta != 0.0, "F1 B=1 perturbation moves logits (real path)",
            f"delta={delta:.4g}")

    # --- G. gradient flow through the chunked path -----------------------------
    model.zero_grad(set_to_none=True)
    reset_chonk_cache(kv_cache)
    out1 = model(input_ids=ids[:, :64], past_key_values=kv_cache,
                 use_cache=True)
    out2 = model(input_ids=ids[:, 64:], past_key_values=kv_cache,
                 use_cache=True)
    loss = F.cross_entropy(
        out2.logits[:, :-1].reshape(-1, out2.logits.shape[-1]).float(),
        ids[:, 65:].reshape(-1))
    loss.backward()
    ga = max((p.grad.detach().abs().max().item()
              for n, p in model.named_parameters()
              if "lora_A" in n and p.requires_grad and p.grad is not None),
             default=0.0)
    gb = max((p.grad.detach().abs().max().item()
              for n, p in model.named_parameters()
              if "lora_B" in n and p.requires_grad and p.grad is not None),
             default=0.0)
    verdict(gb > 0.0, "G1 lora_B receives gradient (real path)",
            f"absmax={gb:.4g} loss={float(loss):.3f}")
    verdict(ga > 0.0, "G2 lora_A receives gradient (real path)",
            f"absmax={ga:.4g}")

    print("=" * 60, flush=True)
    print(f"FINAL: {report['fail']} failures", flush=True)
    with open("chonk_probe_result.json", "w") as f:
        json.dump(report, f, indent=1)
    return 0 if report["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
