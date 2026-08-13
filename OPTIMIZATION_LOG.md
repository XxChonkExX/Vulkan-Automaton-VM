# Optimization Log: Qwen 27B Chonk Buffer Training

## Baseline (checkpoint/v1-baseline)
- **Date**: 2026-08-13
- **Config**: 
  - SEQ_LEN=131072, BATCH_SIZE=1, CHUNK_SIZE=4096
  - LR=2e-5, WD=0.01, WARMUP=100, MAX_STEPS=10000
  - GRAD_ACCUM=1, no grad clipping, no compile, no flash attention
- **Hardware**: Strix Halo 395 (128GB unified, 2GB VRAM carve)
- **Status**: Baseline committed

---

## Experiment 1: Gradient Accumulation + Clipping
**Goal**: Larger effective batch, stability
**Changes**: GRAD_ACCUM_STEPS=4, max_norm=1.0

| Metric | Baseline | Exp 1 | Delta |
|--------|----------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Loss (step 100) | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 2: torch.compile
**Goal**: 10-20% speedup
**Changes**: `model = torch.compile(model, mode="reduce-overhead")`

| Metric | Exp 1 | Exp 2 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Compile time | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 3: Flash Attention 2
**Goal**: 2-3x attention speed
**Changes**: `attn_implementation="flash_attention_2"`

| Metric | Exp 2 | Exp 3 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 4: BF16 Autocast + Fused AdamW
**Goal**: Memory + speed
**Changes**: `torch.autocast`, fused optimizer

| Metric | Exp 3 | Exp 4 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 5: Double-Buffer Chunks
**Goal**: Overlap I/O + compute
**Changes**: Async load next chunk while computing current

| Metric | Exp 4 | Exp 5 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Memory (GB) | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 6: Curriculum Learning
**Goal**: Faster convergence
**Changes**: Ramp seq_len from 8K -> 128K over first 1000 steps

| Metric | Exp 5 | Exp 6 | Delta |
|--------|-------|-------|-------|
| Steps/sec | | | |
| Loss (step 1000) | | | |
| Stable? | | | |

**Notes**: 

---

## Experiment 7: EMA Weights + Label Smoothing
**Goal**: Quality
**Changes**: EMA decay=0.9999, label_smoothing=0.1

| Metric | Exp 6 | Exp 7 | Delta |
|--------|-------|-------|-------|
| Final perplexity | | | |
| Stable? | | | |

**Notes**: 

---

## Breaking Points Found
| Experiment | Breaking Point | Root Cause |
|------------|----------------|------------|
| | | |

---

## Optimal Configuration (to be determined)
**Target**: Best quality/speed tradeoff

| Setting | Value |
|---------|-------|
| GRAD_ACCUM_STEPS | |
| CHUNK_SIZE | |
| LR / Scheduler | |
| Optimizer | |
| Compile mode | |
| Flash Attention | |
| Other | |

---

## Final Recommendations
*To be filled after all experiments complete*