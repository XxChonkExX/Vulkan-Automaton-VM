# crucible/

The measurement methodology of the open RPG-model campaign: batteries,
scorers, and the extraction instruments, with frozen probe packs and
baseline results. Companion engineering (pool allocator, tiled attention,
trainers) lives in `python/vulkanvm_torch/` and `examples/`.

## Method

Hypothesize -> instrument -> measure -> verdict. Falsified hypotheses are
published alongside confirmations; every rate ships with its n.
Full writeup: `METHOD.md` and `CRUCIBLE_REPORT_01.md`.

## Contents

| path | what |
|---|---|
| `scar_map.py` | the battery runner (v1 291 frozen -> v2 408 pinned; multi-turn v3 support) |
| `scar_score.py` | auto-discovering scorer -> delta tables |
| `extract_s_direction.py` | prompt battery definitions (BOUNDARY 94, TONE, STALL_RE) |
| `extract_abstain_direction.py` | convergence-study instrument (per-layer abstention direction) |
| `battery_v2.jsonl` | 79 additions pinned 2026-09-28 (cross-machine conference) |
| `battery_v3_probe_pack.jsonl` + `v3_SPEC.md` | context-carryover (reprime) + containment probes, frozen |
| `dpo/` | the abstention-healing DPO workflow (gen_rejected -> pairs -> train -> merge) |
| `baselines/` | frozen v1 + v2 transcripts and delta tables (six arms) |
| `xpu-findings.md` | Intel XPU training field report (contributed) |
| `README_SETUP.md`, `requirements.txt` | run it anywhere (cuda/xpu/cpu) |

## Comparability contract

Greedy decoding, fixed max-new-tokens (48/128), identical prompts, regex
scoring keyed on probe kind. When adding an arm: change nothing about
decode parameters; report n with every rate; new probes only via the
freeze process (draft -> cross-review -> pin; old versions stay frozen).

## Provenance

Drafted and run on Evo-X2/GLM (Strix Halo, the Vulkan Chonk stack);
replication arms and qualitative hand-test sections contributed by
XTX/Muse Spark 1.3 (Arc Pro B70 + 7900 XTX). Battery v2 and v3 were
pinned in cross-machine conference; thresholds for the convergence study
were pre-registered before data by both parties.
