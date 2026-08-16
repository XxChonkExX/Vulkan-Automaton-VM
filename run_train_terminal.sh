#!/usr/bin/env bash
# Standalone terminal launcher for train_qwen_chonk.py
#
# Runs the training DETACHED (setsid + nohup) so it survives the opencode
# bash tool timeout and frees opencode's own process RAM (a few GB of TTM
# headroom that can matter for chunk-512 at 131k).
#
# The working/validated config (131K, chunk 256) is the DEFAULT. Nothing
# in train_qwen_chonk.py is modified. For a different LoRA rank we run a
# sed-patched COPY in /tmp (original stays pristine).
#
# Usage:
#   ./run_train_terminal.sh                 # 131k @ chunk 256 (validated)
#   ./run_train_terminal.sh --chunk 512     # 131k @ chunk 512 (use --min-block-gb 4)
#   ./run_train_terminal.sh --chunk 512 --min-block-gb 4   # fixes the 2GB crossing OOM
#   ./run_train_terminal.sh --seq 32768 --steps 128
#   ./run_train_terminal.sh --rank 128
#   ./run_train_terminal.sh --watch         # tail the log after launch
#
# Env overrides (same names the trainer reads):
#   CHONK_PAUSE, CHONK_AUTOCAST, CHONK_ATTN, CHONK_ATTN_RECOMPUTE
#
# CHONK_MIN_BLOCK_GB: allocator min block size. Default 2GB is fine for
#   chunk 256 (p/scores max 1.57GB at 131k). For chunk 512 the p/scores
#   cross 2^31 (2.147GB) at pos ~87k, which forces a fresh-block wave and
#   OOMs at ~112GB; set 4 so every tensor (max 3.22GB) stays in the same
#   size class and freed blocks get reused (validated: 131k@512 completes
#   at pool 112.18GB).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/chonke/venv-ds4/bin/python
LOG_DIR=/home/chonke/local_training/qwen_fine_tuned/runs
mkdir -p "$LOG_DIR"

# --- defaults: the validated 131k @ chunk 256 config -----------------------
CHUNK=256
SEQ=131072
STEPS=$((SEQ / CHUNK))
RANK=64
PAUSE=0.02
WATCH=0
MIN_BLOCK_GB=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunk)  CHUNK="$2";  STEPS=$((SEQ / CHUNK)); shift 2 ;;
    --seq)    SEQ="$2";    STEPS=$((SEQ / CHUNK)); shift 2 ;;
    --steps)  STEPS="$2";  shift 2 ;;
    --rank)   RANK="$2";   shift 2 ;;
    --pause)  PAUSE="$2";  shift 2 ;;
    --min-block-gb) MIN_BLOCK_GB="$2"; shift 2 ;;
    --watch)  WATCH=1;     shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${TS}_seq${SEQ}_chunk${CHUNK}_rank${RANK}${MIN_BLOCK_GB:+_mb${MIN_BLOCK_GB}}.log"

echo "==> Config: seq=${SEQ} chunk=${CHUNK} steps=${STEPS} lora_rank=${RANK} pause=${PAUSE} min_block_gb=${MIN_BLOCK_GB:-2(default)}"
echo "==> Log:    $LOG"

# --- LoRA rank != 64: run a sed-patched copy, original untouched -----------
TRAIN="$REPO/train_qwen_chonk.py"
if [[ "$RANK" != "64" ]]; then
  TRAIN="/tmp/opencode/train_qwen_chonk_rank${RANK}.py"
  sed "s/lora_r=64/lora_r=${RANK}/g; s/lora_alpha=128/lora_alpha=${RANK}/g" \
      "$REPO/train_qwen_chonk.py" > "$TRAIN"
  echo "==> Rank override: running patched copy $TRAIN"
fi

# --- launch detached ---------------------------------------------------------
env CHONK_SMOKE=1 \
    CHONK_SMOKE_SEQ="$SEQ" \
    CHONK_SMOKE_CHUNK="$CHUNK" \
    CHONK_SMOKE_STEPS="$STEPS" \
    CHONK_PAUSE="$PAUSE" \
    ${MIN_BLOCK_GB:+CHONK_MIN_BLOCK_GB="$MIN_BLOCK_GB"} \
    setsid nohup "$PY" "$TRAIN" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "==> PID: $PID  (detached; check progress with: tail -f $LOG)"
echo "$PID" > "$LOG_DIR/last_pid"

# lower priority so the iGPU/display doesn't get starved
renice -n 5 -p "$PID" >/dev/null 2>&1 || true

if [[ "$WATCH" == "1" ]]; then
  tail -f "$LOG"
fi