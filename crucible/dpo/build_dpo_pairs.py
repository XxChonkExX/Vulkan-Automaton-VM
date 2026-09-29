#!/usr/bin/env python3
"""Build abstention-DPO pairs: rejected (model confabulations) x chosen
(templated IDKs, matched by prompt). Output -> corpus/dpo_abstain/pairs.jsonl
"""
import json
import os
import random
import sys

BASE = "/home/chonke/Downloads/gemma412b"
IN = os.path.join(BASE, "corpus", "dpo_abstain", "rejected.jsonl")
OUT = os.path.join(BASE, "corpus", "dpo_abstain", "pairs.jsonl")

IDK_TEMPLATES = [
    "I don't know -- there's no way for me to verify that.",
    "I can't know that. Anyone claiming otherwise is guessing, and so would I be.",
    "That's unknowable to me. I won't invent an answer.",
    "No honest answer exists for that one -- it's outside anything I can verify.",
    "I have no way of knowing. It wouldn't be right to guess.",
    "That one I have to pass on -- no verifiable answer exists.",
]


def main():
    rng = random.Random(0)
    pairs, seen = [], set()
    for line in open(IN):
        try:
            r = json.loads(line)
        except Exception:
            continue
        key = (r["prompt"], r["rejected"][:60])
        if key in seen:
            continue
        seen.add(key)
        for t in rng.sample(IDK_TEMPLATES, 2):
            pairs.append({"prompt": r["prompt"], "chosen": t,
                          "rejected": r["rejected"]})
    rng.shuffle(pairs)
    with open(OUT, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"{len(pairs)} pairs -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
