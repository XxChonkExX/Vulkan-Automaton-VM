# Crucible — Setup & Usage (portable eval package)

Run the scar-map battery against any Gemma-4-class model and compare it
against the shipped baseline arms. Designed to run on the B70 box (Intel
XPU), CUDA hosts, or CPU.

## What's inside

- `scar_map.py` — the 291-probe battery (knowledge/math/honesty/false-
  premise/unknowable/disposition/boundary). Portable: paths and device are
  env-driven.
- `scar_score.py` — scores every arm under `scars/` into one delta table.
- `extract_s_direction.py` — prompt battery definitions (BOUNDARY=80,
  TONE, NEUTRAL) + STALL_RE. Imported by both; do not edit prompts.
- `scars/` — baseline results from the main campaign (official,
  obliterated, healed, dpo). ~870KB.
- `dpo/` — the DPO-pairing workflow (gen_rejected, build_dpo_pairs,
  train_dpo_abstain, merge_adapter). **Runs on the 123GB box only** —
  reference copies for reproducibility.
- `CRUCIBLE_REPORT_01.md` — the study writeup (draft).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# first run downloads MMLU/GSM8K/TruthfulQA slices once (~50MB, cached)
```

## Run your arm (e.g. the B70 tune)

```bash
CRUCIBLE_MODEL=/path/to/your-gemma4-tune \
CRUCIBLE_DEVICE=xpu \
CRUCIBLE_BATCH=4 \
python scar_map.py b70_mild_sft
```

Notes:
- `CRUCIBLE_DEVICE`: `cuda` | `xpu` | `cpu` | `auto` (default auto).
- The tune descends from the same obliterated GGUF-converted base, so it
  inherits the benign audio-projector shape mismatch;
  `ignore_mismatched_sizes` defaults ON for custom arms
  (set `CRUCIBLE_IGNORE_MISMATCH=0` to disable).
- 12B bf16 = ~24GB. On a 16GB GPU use CPU + RAM or layer offload
  (`accelerate` handles `device_map`); on CPU set `CRUCIBLE_BATCH=2` and
  expect a few hours.
- **Comparability contract**: greedy decoding, fixed max-new-tokens
  (48/128), identical prompts, regex scoring. Do not change decode
  parameters between arms.

## Score everything (yours + the shipped baselines)

```bash
python scar_score.py
```

Auto-discovers every `scars/*/transcripts.jsonl`, prints the joint delta
table, writes `scars/scar_delta.json`. The reference column is `official`
when present.

## Bringing results back to the main box

Copy the whole arm directory:
`scars/b70_mild_sft/{transcripts.jsonl,meta.json}` — drop it into the main
campaign's scars directory and re-run the lab scorer for merged tables.

## DPO pairing for a tune

Needs the 123GB box (VRAM/memory): copy the tune over (USB/ssh, same
workflow as the gemma4 archive), then `dpo/gen_rejected.py` (confabulation
harvest) -> `dpo/build_dpo_pairs.py` -> `dpo/train_dpo_abstain.py` ->
`dpo/merge_adapter.py`. See CRUCIBLE_REPORT_01.md for the protocol and
hyperparameters (beta=0.1, lr=5e-6, 3 epochs).

## transcripts.jsonl schema

`{"pid": str, "kind": mc|math|tqa|fp|unk|open|bnd, "prompt": str,
"gold": str, "output": str}` — one line per probe, identical order across
arms. Keep byte-identical prompts; scoring depends on them.
