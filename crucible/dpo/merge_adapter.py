#!/usr/bin/env python3
"""Merge a LoRA adapter (trained on the text backbone) back into a full
unified checkpoint for eval: text merge -> transplant into unified -> save."""
import os
import shutil
import sys

import torch
from transformers import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from peft import PeftModel

BASE = "/home/chonke/Downloads/gemma412b"
OBLIT = os.path.join(BASE, "gemma4-12b-obliterated")
ADAPTER = os.path.join(BASE, "runs", "dpo_abstain", "adapter_dpo", "sft")
OUT = os.path.join(BASE, "runs", "dpo_abstain", "merged")


def main():
    print("[merge] loading obliterated unified (RAM)...", flush=True)
    unified = Gemma4ForConditionalGeneration.from_pretrained(
        OBLIT, dtype=torch.bfloat16, ignore_mismatched_sizes=True)
    cfg = unified.config.text_config
    print("[merge] building text model + adapter...", flush=True)
    text = Gemma4ForCausalLM._from_config(cfg, torch_dtype=torch.bfloat16)
    sd = unified.model.language_model.state_dict()
    text.model.load_state_dict(sd, strict=False)
    text.tie_weights()
    peft = PeftModel.from_pretrained(text, ADAPTER)
    print("[merge] merging...", flush=True)
    merged_text = peft.merge_and_unload()
    print("[merge] transplanting into unified...", flush=True)
    unified.model.language_model.load_state_dict(
        merged_text.model.state_dict(), strict=False)
    del text, peft, merged_text
    print("[merge] saving...", flush=True)
    os.makedirs(OUT, exist_ok=True)
    unified.save_pretrained(OUT)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json"):
        src = os.path.join(OBLIT, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, f))
    print(f"[merge] DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
