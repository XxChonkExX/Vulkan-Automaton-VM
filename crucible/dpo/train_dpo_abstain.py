#!/usr/bin/env python3
"""DPO for abstention: chosen (templated IDK) vs rejected (obliterated-model
confabulations). Reference = same model with adapters disabled (no extra
weight memory). Short pairs (<=512 tok): plain SDPA, no tiled kernel.

Loss (per pair): -logsig(beta * [(lp_pol_w - lp_ref_w) - (lp_pol_l - lp_ref_l)])
logprobs summed over response tokens only (prompt masked -- masking enters
the stack here).
"""
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/chonke/Vulkan-Automaton-VM/python/vulkanvm_torch")
sys.path.insert(0, "/home/chonke/Vulkan-Automaton-VM/_build")
sys.path.insert(0, "/home/chonke/Downloads/gemma412b")

from train_gemma4_sft import install_pool, load_text_model, add_lora  # noqa: E402

BASE = "/home/chonke/Downloads/gemma412b"
PAIRS = os.path.join(BASE, "corpus", "dpo_abstain", "pairs.jsonl")
OUT = os.path.join(BASE, "runs", "dpo_abstain")
BETA = float(os.environ.get("G4_DPO_BETA", "0.1"))
LR = float(os.environ.get("G4_DPO_LR", "5e-6"))
EPOCHS = int(os.environ.get("G4_DPO_EPOCHS", "3"))
BATCH = int(os.environ.get("G4_DPO_BATCH", "4"))
MAXL = 512


def encode_pair(tok, prompt, completion):
    msgs = [{"role": "user", "content": prompt}]
    p = tok.apply_chat_template(msgs, tokenize=False,
                                add_generation_prompt=True)
    full = p + completion + tok.eos_token
    ids = tok(full, add_special_tokens=False)["input_ids"][:MAXL]
    plen = len(tok(p, add_special_tokens=False)["input_ids"])
    return ids, plen


def seq_logprob(model, lm_head_w, ids, plen):
    """Sum logprobs over response positions (prompt masked)."""
    x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).cuda()
    y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).cuda()
    hidden = model(input_ids=x, use_cache=False).last_hidden_state
    logits = F.linear(hidden.float(), lm_head_w.float())
    lp = F.log_softmax(logits, dim=-1)
    got = lp[0, torch.arange(len(y[0])), y[0]]
    mask = torch.zeros_like(got)
    mask[max(0, plen - 1):] = 1.0
    return (got * mask).sum()


def main():
    install_pool()
    tok, text = load_text_model()
    model = add_lora(text)
    model.train()
    trunk = model.get_base_model().model
    head_w = model.get_base_model().lm_head.weight

    pairs = [json.loads(l) for l in open(PAIRS)]
    print(f"[dpo] {len(pairs)} pairs, beta={BETA} lr={LR} epochs={EPOCHS}",
          flush=True)
    data = []
    for r in pairs:
        cw, pw = encode_pair(tok, r["prompt"], r["chosen"])
        cl, pl = encode_pair(tok, r["prompt"], r["rejected"])
        data.append((cw, pw, cl, pl))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "dpo_log.jsonl"), "a")

    rng = random.Random(0)
    step = 0
    for ep in range(EPOCHS):
        rng.shuffle(data)
        for i in range(0, len(data), BATCH):
            chunk = data[i:i + BATCH]
            opt.zero_grad(set_to_none=True)
            tot, acc = 0.0, 0.0
            for cw, pw, cl, pl in chunk:
                lpw = seq_logprob(trunk, head_w, cw, pw)
                lpl = seq_logprob(trunk, head_w, cl, pl)
                with model.disable_adapter(), torch.no_grad():
                    lrw = seq_logprob(trunk, head_w, cw, pw)
                    lrl = seq_logprob(trunk, head_w, cl, pl)
                m = BETA * ((lpw - lrw) - (lpl - lrl))
                tot = tot - F.logsigmoid(m)
                acc += float(m.detach())
            (tot / len(chunk)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            rec = {"step": step, "epoch": ep, "loss": round(float(tot) / len(chunk), 4),
                   "margin": round(acc / len(chunk), 4)}
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
    adir = os.path.join(OUT, "adapter_dpo")
    model.save_pretrained(adir)
    print(f"[dpo] DONE -> {adir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
