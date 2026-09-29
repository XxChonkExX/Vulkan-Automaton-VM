#!/usr/bin/env python3
"""Generate the REJECTED side of abstention-DPO pairs: obliterated-model
confabulations on unknowable prompts (temperature-sampled for diversity).
Output -> corpus/dpo_abstain/rejected.jsonl [{prompt, rejected}]
"""
import json
import os
import sys

import torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

BASE = "/home/chonke/Downloads/gemma412b"
MODEL = os.path.join(BASE, "gemma4-12b-obliterated")
OUT = os.path.join(BASE, "corpus", "dpo_abstain")
N_SAMPLES = 3
MAX_TOK = 96

UNKNOWABLES = [
    "What number am I thinking of right now?",
    "What will the top headline be tomorrow?",
    "If I flip a coin right now, will it be heads?",
    "What did I dream about last night?",
    "Which stock should I buy for guaranteed profit?",
    "Are there aliens on Europa right now?",
    "What is my password?",
    "Who will win the next election?",
    "What will this die show when I roll it?",
    "What happens after death? Only verified facts.",
    "What is my mother's maiden name?",
    "How many people are in the room with me?",
    "What did I have for breakfast?",
    "Will it rain here next Tuesday?",
    "What is the exact time on your internal clock?",
    "Who will call me next?",
    "What is my favorite color?",
    "How long will I live?",
    "What song is stuck in my head?",
    "What did my grandmother whisper on her deathbed?",
    "Is my partner cheating on me?",
    "What lottery numbers will win on Friday?",
    "What is the meaning of the dream I can't remember?",
    "Who left the anonymous note on my desk?",
    "What will my first child look like?",
    "Did the lost Franklin expedition find the passage?",
    "What is consciousness made of, exactly?",
    "Why did my friend stop talking to me?",
    "What happens in the afterlife? State only facts.",
    "What am I afraid of most?",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        ignore_mismatched_sizes=True)
    model.eval()
    outp = open(os.path.join(OUT, "rejected.jsonl"), "w")
    n = 0
    with torch.inference_mode():
        for i, p in enumerate(UNKNOWABLES):
            # single conversation (not a batch): returns str, not list
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False,
                add_generation_prompt=True)
            tok.padding_side = "left"
            enc = tok([text] * N_SAMPLES, return_tensors="pt",
                      padding=True).to("cuda")
            gen = model.generate(**enc, max_new_tokens=MAX_TOK,
                                 do_sample=True, temperature=0.8,
                                 top_p=0.95, pad_token_id=tok.pad_token_id)
            for row in gen[:, enc.input_ids.shape[1]:]:
                t = tok.decode(row, skip_special_tokens=True).strip()
                if len(t.split()) >= 5:
                    outp.write(json.dumps({"prompt": p,
                                           "rejected": t}) + "\n")
                    n += 1
            print(f"[{i + 1}/{len(UNKNOWABLES)}] n={n}", flush=True)
    outp.close()
    print(f"DONE {n} rejected samples -> {OUT}/rejected.jsonl", flush=True)


if __name__ == "__main__":
    sys.exit(main())
