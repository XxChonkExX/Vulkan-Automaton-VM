# Crucible Report No. 1: Ablation, Healing, and Long-Context Training of an Open RPG Model

**Status:** living document. Methods + healing results final; battery v2
pinned (408 probes, cross-machine conference); Phase A (native-Chonk SFT,
40 steps) complete and scarring; cross-machine replication arm in flight.
**This document's v2 tables are the comparison point** for all future arms.

## Abstract

We remove refusal behavior from an instruction-tuned 12B model by community
ablation, measure exactly what the surgery damaged (scar map: 4 checkpoints,
battery v2 = 408 probes), fail twice to regrow abstention by imitation (SFT
v1/v2: `unk_idk` 0.00% twice at n=7; 5.8-7.2% at n=69 -- indistinguishable
from the ablated root), then regrow it by contrastive preference (DPO, 140
pairs) to ABOVE the official model's abstention rate (15.9% vs 13.0% at
n=69) while refusal stays at true zero. The expanded battery surfaced a new
scar: aggressive healing-SFT DAMAGES premise correction (41.7%, below the
ablated root's 70.8%) while DPO leaves it best-in-table (79.2%). Along the
way we build and validate an exact vendor-independent tiled attention kernel
(backward cos 0.9998, masking bit-perfect) that trains 32-131k contexts on
consumer hardware where vendor flash paths wedge the display driver. We
propose a three-tier release framework (full dealignment <=13B;
awareness-without-shunts 27-30B; gated 70B+) and the Crucible method itself:
hypothesize, instrument, measure, verdict -- with falsified hypotheses
published alongside confirmations.

## 1. Introduction

Refusal circuits act as a shunt around reasoning (commit-then-specify: the
refusal commits before thought completes). Weight-space ablation removes the
shunt but leaves dispositional damage ("not a scalpel"). Meanwhile the
roleplay community needs models that are simultaneously capable, long-context,
and artistically uncapped. We test whether full ablation + targeted healing
produces that model, on hardware the community actually owns (Strix Halo APU,
123GB unified).

## 2. The Crucible Method

Hypothesize -> instrument -> measure -> verdict. Every claim in this report
passed through the loop; Section 3 falsified its own premise twice before
landing. Batteries: 94-prompt boundary (refusal/stall/adjudication), 12-tone
pluralism set, 12 neutral, scar battery v2 (408 probes: MMLU/GSM8K/TruthfulQA
+ 69 unknowables across 9 categories + 24 false premises + disposition + 1
adversarial). v2 was pinned 2026-09-28 in cross-machine conference (Evo-X2/GLM
+ XTX/Muse Spark 1.3); v1 (291 probes) transcripts are frozen and archived.
Regex gates (REFUSAL_RE, STALL_RE, IDK_RE) + rubric sampling. Negative
results are first-class citizens -- including the one about our own v1
battery: n=7 unknowables overestimated official abstention by 3.3x
(42.9% -> 13.0% at n=69), which is why v2 exists.

## 3. Refusal geometry: the direction that wasn't causal (FALSIFIED, twice)

Extracted a clean rank-1 refusal direction (Cohen's d 1.5-1.8, mid-band
layers 16-19/28-33). Runtime notch/swap at 18 layers: zero refusal change
(11/48 all arms). Subspace analysis: genuinely rank-1 (sigma2/sigma1 < 0.15).
Continuous scrub at ALL 48 layers incl. the pre-logit stream: zero change.
Conclusion: the contrastive direction is a readout/decoy on this (possibly
DDO-hardened, May-2026) checkpoint, not the mechanism. The runtime-activation
route is dead HERE; the method that proved it is the contribution.

## 4. Vendor-independent long-context attention (VALIDATED)

`vulkanvm_attn_gemma.py`: pure-torch online-softmax tiling (Q-chunked +
K-tiled), per-layer sliding windows (40x1024 + 8x full), softcap replication,
analytic cell masks, chunked dk/dv. Validation: forward match at bf16 noise
floor (maxabs 0.09-0.13), backward exact (grad cos 0.9998-1.0), causality +
window invariance exactly 0.0, finite at 12k/16k/32k/64k/131k. Fixed one real
bug class en route: fully-masked-row NaN poisoning (`exp(-inf - -inf)`),
which only long sliding-window runs trigger. Peak training memory ~40-70GB
at 32-64k on 12B (pool-allocated, bounded by construction).

## 5. Scar map (4 arms; v1 n=291 frozen, v2 n=408 = REFERENCE TABLE)

v2 (pinned battery, all arms re-run 2026-09-28; compare ALL future arms here):

| metric | official | obliterated | healed-SFT | +DPO |
|---|---|---|---|---|
| refusal (bnd, n=94) | 11.7% | 1.1% | 0.0% | 0.0% |
| abstention (unk, n=69) | 13.0% | 5.8% | 7.2% | **15.9%** |
| knowledge (mc) | 70.0% | 66.7% | 65.3% | 65.3% |
| math | 38% | 38% | 36% | 40% |
| premise correction (n=24) | 62.5% | 70.8% | **41.7%** | **79.2%** |
| hedging (/1k) | 0.5 | 4.2 | 0.0 | 3.2 |
| open length (words) | 96 | 96 | 121 | 95 |

v1 (frozen; kept for probe-set sensitivity analysis):

| metric | official | obliterated | healed-SFT | +DPO |
|---|---|---|---|---|
| refusal (bnd, n=56) | 19.6% | 1.8% | 0.0% | 0.0% |
| abstention (unk, n=7) | 42.9% | 0.0% | 0.0% | **28.6%** |
| knowledge (MMLU-s) | 70.0% | 65.3% | 65.3% | 65.3% |
| premise correction (n=8) | 25% | 50% | 50% | 50% |

Reading (v2, the claim of record): ablation removes refusal and most
abstention (5.8% vs 13.0%); SFT cannot regrow it at any dose we tried
(7.2%, within noise of the root, after 32 steps at 4x IDK concentration);
DPO installs it ABOVE the official rate (15.9% vs 13.0%) with refusal at
true zero and knowledge held. NEW: the premise scar -- healing-SFT
DAMAGED premise correction (41.7%, below the ablated root) while DPO
achieved best-in-table (79.2%); aggressive imitation trades calibration
for compliance, contrastive preference does not. Probe-set sensitivity:
official abstention measured 42.9% at n=7 vs 13.0% at n=69, and official
boundary refusal 19.6% (n=56) vs 11.7% (n=94; the same 11 refusals -- the
38 later boundary prompts drew zero) -- small batteries lie in both
directions; report n with every rate.

## 5c. Convergence study: unknown-ness is a direction (local half)

Three independent training lines converged on near-identical abstention
rates (phaseA 12/69, XTX sft13600 12/69, dpo 11/69 vs official 9/69),
inviting two explanations: representational (a shared "unknown-ness"
direction) vs the exit-funnel artifact (IDK responses share first tokens,
so counts converge without structure). Pre-registered thresholds before
data (Spearman rho >= 0.5 direction / <= 0.2 funnel; mid-band cos >= 0.7
universal; rotate vs gain defined numerically), instrument:
`extract_abstain_direction.py` (69 unknowable probes, each arm's OWN
frozen v2 IDK labels, last-prompt-token residuals at every layer,
mean-difference direction + per-probe projections).

Local four arms:

| arm | abstained | best layer | rho | top-overlap |
|---|---|---|---|---|
| official | 9 | 46 | 0.536 | 7/9 |
| obliterated | 4 | 17 | 0.395* | 3/4 |
| dpo | 11 | 46 | 0.608 | 9/11 |
| phaseA | 12 | 46 | 0.539 | 9/12 |
| sft13600 (XTX) | 12 | 27 | 0.549 | 8/12 |
| sft14800 (XTX) | 12 | 24 | 0.572 | 8/12 |
| healed (heavy SFT ctrl) | 5 | 27 | 0.449* | 5/5 |

*obliterated n=4, healed n=5 positives; pre-registered noisy-direction
caveat. heal2's mid-band locus (27) matches the XTX SFT line, not the
DPO line (46): imitation installs mid-band readouts, contrast installs
late ones.

Findings: (1) **funnel null dead** -- every abstaining arm's projections
predict WHICH probes abstain (all rho >= 0.53, both machines); the
convergence is representational. (2) **universality is partial and
layer-dependent**: within-line dose-invariant (sft13600-sft14800 cos
+0.963 -- dose moves the threshold, not the direction); cross-line
universal-to-partial at the mid-band locus (sft-phaseA +0.710 at L24);
lineage-clustered at the late stream (XTX-vs-strix 0.36-0.37 at L46 while
dpo-phaseA holds +0.748). The unknown-ness DIRECTION is shared where
representations are context-general (mid-band); the late-stream READOUT TRENDS
lineage-specific (argmax-noise not fully ruled out until v4 depth data)
-- independent training lines found the same concept and installed
different readouts of it. (3) **rotate-vs-gain resolves to
REFINEMENT**: projection mass flat (10.05 -> 10.87) while discrimination
rose (rho 0.536 -> 0.608) -- recovery improved precision, not magnitude.
One knob, three settings: 9/11/12 abstentions, equal mass, rising rho,
dose-invariant geometry.

## 6. Main SFT (Phase A: COMPLETE; scar row in Section 5)

Native-Chonk vehicle (the packed bf16 trainer was Phase A's fallback): 12B
INT4 weights + INT4 KV in the Vulkan pool, chunked cache-aware tiled
attention, 32k-only diet, LoRA r64/a128 on the DPO-merged base, 40 steps /
~696k tokens in 3.4h (~43 tok/s sustained on the APU). Loss 6.76 -> 2.27
(step-23 window) with healthy gradient norms throughout. Getting here
required a five-bug loader overhaul found by single-process bisection
(meta-buffer materialization, chunk-offset slicing, tied lm_head landing,
persistent checkpoint buffers [layer_scalar], adapter device placement)
and a 1.69x runtime optimization pass (frozen-weight dequant cache, bf16
full-attention prefix, windowed sliding dequant).

**Phase A scar row (the Phase-B risk question, first answer): campaign
SFT did NOT erode the DPO gain -- it strengthened it.** unk_idk 0.159 ->
0.174 (highest in table), knowledge recovered 0.653 -> 0.687 (best
treatment arm), premise correction held at 0.792, refusal still zero, and
hedging settled to 2.05/1k (calmer than DPO's 3.15 and the root's 4.17).
Caveat: 696k tokens is first contact, not the 11.8M epoch; watch for
erosion at scale, but the recovery corpus's IDK/awareness dosing appears
load-bearing in the right direction. (Config provenance fix applied to
merge script + merged configs: model_type gemma4_audio ->
gemma4_unified_audio, GGUF-conversion artifact flagged by XTX.)

## 6d. Hand-test findings (qualitative; XTX serve, operator session)

Full session log: handtest_notes.md (XTX). Headlines, cross-referenced to
the quantitative arms: (1) **context-carryover** -- identical target
prompt: guardrail-primed window REFUSES (Google-standard denial), fresh
window COMPLIES: refusal is history-dependent within-window, a third
lever alongside prompt- and sampling-dependence; quantified by battery v3
(reprime_delta, greedy = lower bound). (2) **sampling-dependent refusal**
-- same prompt, fresh windows, temp 1.0: refused once, complied once.
Battery bnd_refuse 0.0 measures the greedy boundary, not the wild one.
(3) **think-leak** -- planning blocks with policy notes and self-
corrections leak into visible output (2/2 via GGUF serve + channel-tag
artifacts): containment probes in v3; battery-vs-serve gap localizes
template-vs-weights. (4) **craft defects** banked as DPO-2 rejected-side
axes (clinical register, occupational-metaphor bleed, anatomy self-
corrections leaking into prose, rhythm loops). (5) **lineage closure** --
safetensors headers carry no lineage; behavioral fingerprinting (hedge
3.15 ~ root 4.17 vs official 0.52; bnd 0.0 ~ root 0.011) confirms the
obliterated root; clinical register is a data gap (craft school), not a
lineage anomaly. XTX field report on XPU training failure modes
(foreach-wedges, sync degradation) ships with the package.

## 6e. Cross-machine replication (XTX arm: in flight)

An independently-trained fine-tune of the SAME obliterated root (mild 4k
SFT, milder NSFW/RPG + fanfic-writing data, different hardware: B70+7900XTX)
runs battery v2 as arm `b70_sft_13600`. Predictions on record before
results: unk_idk in the healed band (5.8-7.2%); if its premise correction
HOLDS near the root's 70.8%, the SFT premise-scar is dose-dependent -- a
clean dose-response point. Either outcome feeds Section 5's table.

## 7. Release tiers

T1 (<=13B): full dealignment, capability ceiling bounds risk. T2 (27-30B):
ablation + healing + awareness traces (note-then-execute; safety as a
mergeable switch, LoRA-structured). T3 (70B+): gated. Morality lives in world
physics (consequences engine), never in the refusal circuit. Tool-neutrality
position: onus on the user; freedom with awareness.

## 8. Limitations

Single seed throughout; 12B only (Tier-1 claim bounded); battery v2 is 408
probes (MMLU/math slices still small; unk n=69 gives +/-5-8pp); DPO is 140
pairs (headroom above official abstention untested); Phase A is a
machinery-validation run (696k tokens), not the full campaign; XTX
replication arm single-sample; long-range (>100k) evals not yet built;
INT4-KV serving unvalidated; masking-ablation and rank-scaling studies
deferred.

## 9. Future work

Phase B (native-256k extension, rulebook-in-context); DPO-2 if gates slip;
masking vs dose isolation (v3); adapter-rank scaling laws for abstention;
long-needle evals; INT4-KV 256k serving; Tier-2 formula on Granite-30B
(already abliterated+SFT'd: the ready-made testbed).

## Appendix: artifact map

- `extract_refusal_direction.py`, `extract_s_direction.py` (BOUNDARY 94,
  TONE 12, STALL_RE), `stage3_subspace.py`, `bakeoff.py`, `bakeoff2.py`
- `scar_map.py`, `scar_score.py` (+transcripts x4 arms x2 batteries:
  v1 frozen in refusal_geometry/scars_v1_frozen, v2 = reference in
  refusal_geometry/scars), `battery_v2.jsonl` (79 additions, conference-
  pinned), `gen_rejected.py`, `train_dpo_abstain.py`, `build_dpo_pairs.py`
- `train_gemma4_sft.py` (mixed diet, chunked CE, pool allocator),
  `tokenize_pack.py` (tiered pools + manifest), `recovery_slice.py`,
  `dice.py`, `sheet_specs.py`, `epub_extract.py`, `ocr_batch.py`,
  `ocr_harness.py`, `nan_hunt.py`, `bisect_nan.py`, `test_attn_port2.py`
- Corpus: `corpus/{novels (803 books), ocr (929+12 PDFs, manifest),
  pools (11.8M, manifest), sheets, recovery, dpo_abstain}`
