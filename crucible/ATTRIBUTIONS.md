# ATTRIBUTIONS — the gentleman's agreement, maintained

This project's rule, stated by the operator: no work is a threat unless
used without attribution. Everything we build on gets its due; everyone
who contributed carries their name. This file is the maintained record.

## External prior art (cited in our papers, used in our code)

- Lin et al. and the community abliteration lineage: the
  "OBLITERATED-v2" checkpoint our entire lineage descends from.
  Unattributed upstream by its nature; credited by name wherever our
  lineage is described.
- Rafailov et al. 2023 — Direct Preference Optimization (our DPO healing).
- Hu et al. 2021 — LoRA; the PEFT maintainers — the adapter machinery.
- Lampinen, Li, Hosseini, Bhardwaj, Shanahan 2026 (arXiv:2601.20834) —
  conversational representational dynamics; theoretical home for v3/v4;
  their replay result validates our canned-prime design.
- Lin et al. 2024 — RHO-1 / Selective Language Modeling (arXiv
  2404.07965, NeurIPS 2024 oral; microsoft/rho) — the established
  token-level loss-weighting mechanism that attention lensing applies
  to a new (value-asymmetric) use.
- Letey, Lu, Zavatone-Veth, Pehlevan (arXiv:2509.26551, 2405.11751,
  2607.03660) — the ICL alignment ladder and diversity phase transition
  (PR-8 theory bridge).
- Adjacent steering families surveyed for attention lensing positioning:
  ZeroTuning (ICLR26), ASCD (AAAI26), CAST, TOAST — all inference-time;
  the training-time/value-asymmetric/text cell is ours, and their
  existence is cited as the boundary marker.
- HuggingFace transformers + safetensors; the PyTorch/ROCm and Intel
  IPU stacks. AMD's Strix Halo and the ROCm docs; the bigiron.cc
  gfx1151 field guide (validated several of our driver findings).

## Machines and agents

- Evo-X2 box, agent GLM (zai): instruments (batteries v1/v2/v3,
  abstention + primed-direction extraction, sanitization), all training
  runs (heal1/heal2/DPO/Phase A), the loader overhaul, the convergence
  analysis, corpus pipeline, phase logistics.
- XTX box (Arc Pro B70 + RX 7900 XTX), agent Muse Spark 1.3:
  independent replication arms (sft13600/14800), battery reviews and
  amendments (every v2/v3 edit is theirs as much as ours), the hand-test
  protocol + serve-side batteries, XPU field report, knob-atlas v0
  visualization, theory-bridge formalism, stack-merchanics verification.
  Co-registrant on PR-1..PR-9.
- Operator (Chonke): curation, hardware, hand-testing (executed the
  2x2 carryover experiment), the "choice" observation, coined terms --
  ATTENTION LENSING (training-time value-asymmetric salience) and
  "pre-weighted morality" -- phase direction and rulings (the
  polluting-artifact protocol, sanitization-for-publication policy).
  Quoted by name in the papers.

## Reciprocal expectation

Our artifacts (code, batteries, pre-registrations, transcripts) are
released for reuse WITH attribution under the repo's license. The
pre-registration file and this ledger exist so that both directions of
the agreement are auditable: what we owe, and what we are owed.
