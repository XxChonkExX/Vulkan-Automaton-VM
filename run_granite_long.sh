#!/bin/bash
# Auto-restart wrapper: resumes from last checkpoint if a driver reset
# or Vulkan OOM kills the trainer. Keeps the run alive.
OUT_DIR="examples/granite_chonk/out/granite-finetuned"
LOG="/home/chonke/Documents/train_granite.log"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.4"
export PYTORCH_ALLOC_CONF="expandable_segments:False,garbage_collection_threshold:0.4"
export CHONK_MAX_HEAP_FRACTION=0.92
export CHONK_SEQ_LEN=131072
export CHONK_CHUNK=512
export CHONK_ATTN=eager
export CHONK_ATTN_RECOMPUTE=1
export CHONK_QUANT_BITS=4
export CHONK_QUANT_GROUP=128
export CHONK_LORA_R=128
export CHONK_LORA_ALPHA=256
export CHONK_MIN_BLOCK_GB=16
export CHONK_POOL_BLOCK_SIZES_GB=1,2,4,8
export CHONK_ACT_GB=0.25
export CHONK_STAGING_GB=0.25
export CHONK_GRAD_ACCUM=16
export CHONK_PAUSE=0.02
export CHONK_OPTIMIZER_PAUSE=0.5
export CHONK_MAX_STEPS=10000
export CHONK_SAVE_INTERVAL=10
export CHONK_WARMUP=50
export CHONK_SUBSAMPLE=1.0
export CHONK_EPOCHS=1

# Resume from latest checkpoint if exists
LATEST=$(ls -1d $OUT_DIR/chonk_step_* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST" ]; then
    echo "[wrap] Resuming from $LATEST"
    export CHONK_RESUME_DIR="$LATEST"
fi

# Launch (loop restarts on crash/resume)
while true; do
    echo "[$(date)] Starting training (resume=${CHONK_RESUME_DIR:-none})" >> $LOG
    cd /home/chonke/Vulkan-Automaton-VM && /home/chonke/venv-ds4/bin/python /home/chonke/Vulkan-Automaton-VM/examples/granite_chonk/train_granite_chonk.py >> $LOG 2>&1
    EXIT_CODE=$?
    echo "[$(date)] Exit code $EXIT_CODE" >> $LOG
    if [ -n "$CHONK_RESUME_DIR" ] && [ -f "$CHONK_RESUME_DIR/training_state.pt" ]; then
        echo "[wrap] Resume checkpoint intact; loop continues if needed." >> $LOG
        # If completed (final saved), break
        if [ -d "$OUT_DIR/chonk_final" ]; then break; fi
    fi
    # Short pause between restarts (display breath)
    sleep 5
done
echo "[$(date)] Training loop finished (final at $OUT_DIR/chonk_final)." >> $LOG
