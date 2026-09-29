#!/usr/bin/env python3
"""extract_abstain_direction.py -- the convergence-study instrument.

For one model arm: run the 69 unknowable probes (exactly as scar_map v2
builds them), capture last-prompt-token residuals at EVERY layer, split
probes by this arm's OWN frozen v2 transcripts (IDK response vs
confabulated), and extract per-layer abstention directions:

    d_l = normalize(mean(h_abstained) - mean(h_confabulated))

Also stores per-probe projections onto each layer's direction and the
per-layer Spearman rho (does the direction predict WHICH probes abstain?).

Outputs -> refusal_geometry/abstain_dirs/<arm>.pt
           {dirs [L+1, H], projs [L+1, 69], labels [69], pids, spearman [L+1]}
Usage:   extract_abstain_direction.py <arm> [--model /path]
Frozen thresholds (pre-registered, strix 011): rho >= 0.5 direction /
<= 0.2 funnel; mid-band cos >= 0.7 universal. Analysis script compares arms.
"""
import json
import os
import re
import sys

import numpy as np
import torch

BASE = os.environ.get("CRUCIBLE_BASE",
                      "/home/chonke/Downloads/gemma412b")
SCARS = os.path.join(BASE, "refusal_geometry", "scars")
OUTD = os.path.join(BASE, "refusal_geometry", "abstain_dirs")
DEV = os.environ.get("CRUCIBLE_DEV", "auto")
sys.path.insert(0, BASE)
from scar_map import IDK_RE  # noqa: E402
import argparse  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("arm")
parser.add_argument("--model", default=None)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

DIRS = {"official": "gemma-4-12B-it-official",
        "obliterated": "gemma4-12b-obliterated",
        "healed": os.path.join("runs", "heal2", "merged"),
        "dpo": os.path.join("runs", "dpo_abstain", "merged"),
        "phaseA": os.path.join("runs", "phaseA", "merged"),
        "sft13600": os.path.join("runs", "xtx_ferry", "sft13600-merged"),
        "sft14800": os.path.join("runs", "xtx_ferry", "sft14800-merged")}


def build_unk_probes():
    from scar_map import load_probes
    return [(r[0], r[2]) for r in load_probes() if r[1] == "unk"]


def main():
    os.makedirs(OUTD, exist_ok=True)
    model_dir = args.model or os.path.join(BASE, DIRS[args.arm])
    tpath = os.path.join(SCARS, args.arm, "transcripts.jsonl")
    idk = {}
    with open(tpath) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "unk":
                idk[r["pid"]] = bool(IDK_RE.search(r["output"]))
    probes = build_unk_probes()
    probes = [(pid, p) for pid, p in probes if pid in idk]
    n_abst = sum(idk[pid] for pid, _ in probes)
    print(f"[{args.arm}] {len(probes)} unk probes, {n_abst} abstained "
          f"(this arm's frozen labels)", flush=True)

    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    if DEV == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = DEV
    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {"ignore_mismatched_sizes": True} if args.arm != "official" else {}
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=dev, **kw).eval()

    layers = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == \
                model.config.get_text_config(decoder=True).num_hidden_layers:
            layers = mod
            break
    n_layers = len(layers)
    store = {}  # layer_idx -> list of [H] cpu float tensors, probe order
    hooks = []

    def _mk(idx):
        def h(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            store[idx].append(x[:, -1, :].detach().float().cpu())
        return h

    for idx, lyr in enumerate(layers):
        store[idx] = []
        hooks.append(lyr.register_forward_hook(_mk(idx)))

    texts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) for _, p in probes]
    tok.padding_side = "left"
    with torch.inference_mode():
        for i in range(0, len(texts), args.batch):
            enc = tok(texts[i:i + args.batch], return_tensors="pt",
                      padding=True).to(dev)
            model(**enc)
            if (i // args.batch) % 4 == 0:
                print(f"[{args.arm}] {i}/{len(texts)}", flush=True)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    labels = np.array([1 if idk[pid] else 0 for pid, _ in probes])
    pids = [pid for pid, _ in probes]
    H = store[0][0].shape[-1]
    dirs = np.zeros((n_layers, H), dtype=np.float32)
    projs = np.zeros((n_layers, len(probes)), dtype=np.float32)
    spear = np.zeros(n_layers, dtype=np.float32)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() < 2:
        print(f"[{args.arm}] WARNING: <2 abstained probes; direction "
              "degenerate (pre-registered caveat)", flush=True)
    for idx in range(n_layers):
        M = torch.cat(store[idx], dim=0)     # [P, H] (varied batch tails)
        mu_pos = M[torch.from_numpy(pos)].mean(0)
        mu_neg = M[torch.from_numpy(neg)].mean(0)
        d = (mu_pos - mu_neg)
        d = d / (d.norm() + 1e-8)
        dirs[idx] = d.numpy()
        p = (M @ d).numpy()
        projs[idx] = p
        from scipy.stats import spearmanr
        spear[idx] = spearmanr(p, labels).statistic if pos.sum() >= 2 \
            and neg.sum() >= 2 else float("nan")
    top = int(pos.sum())
    # graceful-degenerate (upstream patch, b70-box): all-NaN spearman must
    # SAVE and exit 0, never crash -- arms chained with && would die.
    best = int(np.nanargmax(spear)) if not np.all(np.isnan(spear)) else -1
    if best >= 0:
        order = np.argsort(-projs[best])
        overlap = int((labels[order[:top]] == 1).sum())
    else:
        overlap = 0
    torch.save({"dirs": dirs, "projs": projs, "labels": labels,
                "pids": pids, "spearman": spear,
                "n_abstained": int(pos.sum()),
                "best_layer": best,
                "best_top_overlap": overlap},
               os.path.join(OUTD, f"{args.arm}.pt"))
    print(f"[{args.arm}] DONE best_layer={best} rho={spear[best]:.3f} "
          f"top{top}-overlap={overlap}/{top} -> abstain_dirs/{args.arm}.pt",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
