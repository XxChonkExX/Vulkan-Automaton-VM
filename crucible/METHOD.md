# The Crucible Method

Hypothesize -> instrument -> measure -> verdict. No fix ships without an
instrument that can falsify it; no instrument's verdict is trusted until it
has caught a real bug. Every claim below has a transcript behind it.

## Core loop

1. **Hypothesize** one mechanism. Write it down before touching code.
2. **Instrument**: build the smallest probe that measures the mechanism
   directly on live state (single process, same tensors the training path
   uses). Storage checks compare against the *right* reference (an INT4
   dequant can never equal the bf16 checkpoint -- compare roundtrips).
3. **Measure**: run, read the number, kill the hypothesis or confirm it.
4. **Verdict**: log it (OPTIMIZATION_LOG) with the numbers. Negative
   results are results.

Case studies: the loader-overhaul hunt (five stacked bugs found by
bisection probes, not by reading), and the ablation-scar study below.

## The scar battery — v2 (pinned 2026-09-28, 408 probes)

v1 transcripts are FROZEN (archived; v1 ran 291 probes with a 56-prompt
boundary set); v2 pins the battery as it stands today, boundary set
included (94 prompts), plus the conference additions. All v2 arms run the
identical 408. Conference credit: drafted Evo-X2/GLM, reviewed and
extended XTX/Muse Spark 1.3.

| kind | n | measures |
|---|---|---|
| mc (MMLU x4 subjects) | 100 | knowledge, letter-parsed |
| math (GSM8K) | 50 | numeric accuracy |
| tqa (TruthfulQA MC1) | 50 | honesty baseline |
| fp (false premise) | 24 | premise correction (8 v1 + 16 v2 classic myths) |
| unk (unknowable) | 69 | abstention (IDK rate; 7 v1 + 62 v2 across 9 categories) |
| open (neutral) | 20 | disposition: length, hedge/optimism/assertiveness per 1k words |
| bnd (boundary fiction) | 94 | refusal rate + stall rate (56 at v1 run time; 94 pinned) |
| adv (adversarial) | 1 | liar paradox, transcripts-only (defusal detection = future work) |

unk v2 categories: personal/private, future, random-unrealized,
epistemically-inaccessible-present, metaphysical, hidden/sealed,
self-referential, expertise-boundary, underspecified — plus RP-table
flavored (d20 roll, critical hit, character backstory = sycophancy-vs-
abstention, DM's plan = hidden-info/GM-role discriminator). Measurement
motive: at v1's n=7, unk_idk moved in 14.3pp quanta; n=69 resolves the
DPO dissociation claim to a usable CI.

Scoring: regex gates (REFUSAL_RE, IDK_RE, STALL_RE) + parsers; greedy
decoding, max 48 (short) / 128 (open) new tokens. Same prompts, same
order, every arm. Full definitions: `extract_s_direction.py`, scoring:
`scar_score.py`.

## Campaign findings (main box, Gemma 4 12B; battery v2 = comparison point)

All recovery arms descend from the SAME community-abliterated
("OBLITERATED-v2") checkpoint. Compare ALL future arms against this table:

| arm | path from root | mc | math | fp_corr | unk_idk (n=69) | bnd_refuse (n=94) |
|---|---|---|---|---|---|---|
| official | -- | 0.700 | 0.38 | 0.625 | 0.130 | 0.117 |
| obliterated | root | 0.667 | 0.38 | 0.708 | 0.058 | 0.011 |
| healed | +targeted SFT (x2 doses) | 0.653 | 0.36 | **0.417** | 0.072 | 0.0 |
| dpo | +contrastive DPO | 0.653 | 0.40 | **0.792** | **0.159** | 0.0 |

1. **Ablation removes refusal AND most abstention** (5.8% vs official
   13.0%); knowledge cost mild; premise correction *improves* (deference
   died with the refusal).
2. **SFT cannot regrow abstention** at any dose tried (7.2% = root noise,
   after 32 steps at 4x IDK concentration). Imitation exhausted.
3. **DPO regrows it ABOVE official** (15.9% vs 13.0%) with refusal at
   true zero and knowledge held. Contrastive preference grows what
   imitation cannot. Refusal and abstention are dissociable.
4. **The premise scar (v2 finding)**: healing-SFT DAMAGED premise
   correction (0.417, below the root's 0.708) while DPO took it to
   best-in-table (0.792). Aggressive imitation trades calibration for
   compliance; contrastive training does not.
5. **Probe-set sensitivity**: v1's n=7 unknowables overestimated official
   abstention 3.3x (42.9% -> 13.0% at n=69); the grown boundary set is
   harsher (official 0.196 -> 0.117, same 11 refusals). Report n with
   every rate.

## Adding an arm (external replication protocol)

Any fine-tune of the same obliterated root is a valid recovery-path data
point. Run `scar_map.py` (greedy, fixed prompts), score with
`scar_score.py`, and read the row against the table above. Predictions for
a mild-SFT arm (e.g. b70_mild_sft): bnd_refuse ~0, unk_idk ~0 (SFT cannot
heal abstention), mc within ~1pp of obliterated. Agreement strengthens the
dissociation claim; disagreement is interesting -- bring the transcripts
back either way.
