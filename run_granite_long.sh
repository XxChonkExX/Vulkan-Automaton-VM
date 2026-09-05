#!/bin/bash
export LD_LIBRARY_PATH=/home/chonke/torch_rocm_libs:${LD_LIBRARY_PATH}
export HIPBLASLT_TENSILE_LIBPATH=/opt/rocm-7.1.0/lib/hipblaslt/library
# Resume from latest checkpoint if exists (re-detected EVERY restart; absolute path)
REPO="/home/chonke/Vulkan-Automaton-VM"
OUT_DIR="$REPO/examples/granite_chonk/out/granite-finetuned"
LOG="/home/chonke/Documents/train_granite.log"
# Single-instance guard: two wrappers = two trainers = 2x memory (OOM) plus
# checkpoint races on the same chonk_step_* dirs. Refuse a second instance.
exec 9>"$REPO/.train_wrapper.lock"
if ! flock -n 9; then echo "[$(date)] wrapper already running; refusing second instance" >> $LOG; exit 0; fi
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.4"
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.4"
export CHONK_MAX_HEAP_FRACTION=0.94
export CHONK_SEQ_LEN=131072
export CHONK_CHUNK=512
export CHONK_ATTN=eager
export CHONK_ATTN_RECOMPUTE=1
export CHONK_QUANT_BITS=4
export CHONK_QUANT_GROUP=128
export CHONK_QUANTIZE_KV=1
export CHONK_LORA_R=64
export CHONK_LORA_ALPHA=128
export CHONK_MIN_BLOCK_MB=64
export CHONK_POOL_BLOCK_SIZES_GB=auto
export CHONK_ACT_GB=0.25
export CHONK_STAGING_GB=0.25
export CHONK_GRAD_ACCUM=16
export CHONK_GRADIENT_CHECKPOINT=1
export CHONK_PAUSE=0.02
export CHONK_OPTIMIZER_PAUSE=0.5
export CHONK_MAX_STEPS=10000
export CHONK_SAVE_INTERVAL=1
export CHONK_WARMUP=50
export CHONK_SUBSAMPLE=1.0
export CHONK_EPOCHS=1

# Launch (loop restarts on crash/resume; re-detects the latest checkpoint
# on EVERY restart so progress made mid-wrapper is always picked up)
while true; do
    LATEST=$(ls -1d $OUT_DIR/chonk_step_* 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/training_state.pt" ]; then
        export CHONK_RESUME_DIR="$LATEST"
        echo "[wrap] Resuming from $LATEST" >> $LOG
    else
        unset CHONK_RESUME_DIR
    fi
    echo "[$(date)] Starting training (resume=${CHONK_RESUME_DIR:-none})" >> $LOG
    # Filter the per-allocation [INFO] allocateDedicatedExportable spam (the
    # 49MB log is ~90% these lines); [ERROR]/heartbeat/checkpoint lines pass.
    cd "$REPO" && /home/chonke/venv-ds4/bin/python "$REPO/examples/granite_chonk/train_granite_chonk.py" 2>&1 | grep -av --line-buffered '^\[INFO\] allocateDedicatedExportable' >> $LOG 2>&1
    EXIT_CODE=${PIPESTATUS[0]}
    echo "[$(date)] Exit code $EXIT_CODE" >> $LOG
    # If completed (final saved), break
    if [ -d "$OUT_DIR/chonk_final" ]; then break; fi
    # Short pause between restarts (display breath)
    sleep 5
done
echo "[$(date)] Training loop finished (final at $OUT_DIR/chonk_final)." >> $LOG
