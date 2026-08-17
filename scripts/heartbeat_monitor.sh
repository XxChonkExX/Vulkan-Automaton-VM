#!/usr/bin/env bash
# heartbeat_monitor.sh — zero-memory training heartbeat monitor.
#
# Pure bash + sleep: RSS stays ~2-4MB. Watches the training log file for
# freshness and reports heartbeat lines to its own tiny log. If the log
# goes stale (training died/hung) it marks the status as STALE.
#
# Usage:
#   ./scripts/heartbeat_monitor.sh                      # default log/interval
#   ./scripts/heartbeat_monitor.sh /path/to/train.log 5  # custom log + interval
#
# The monitor itself is meant to be run detached:
#   setsid nohup ./scripts/heartbeat_monitor.sh > /dev/null 2>&1 &

LOG="${1:-/home/chonke/local_training/qwen_logs/train_262k.log}"
INTERVAL="${2:-60}"          # seconds between checks
HEARTBEAT="/home/chonke/local_training/qwen_logs/heartbeat.log"
STALE_AFTER=300              # seconds without log growth => STALE

mkdir -p "$(dirname "$HEARTBEAT")"

last_size=0
last_mtime=0
while true; do
    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)

    if [ "$size" -eq 0 ]; then
        status="NO_LOG"
    elif [ $((now - mtime)) -gt "$STALE_AFTER" ]; then
        status="STALE ($((now - mtime))s no growth)"
    elif [ "$size" -gt "$last_size" ]; then
        status="ALIVE (log grew +$((size - last_size))B)"
    else
        status="IDLE (no growth, log fresh)"
    fi

    last_step=$(grep -oE "Step [0-9]+" "$LOG" 2>/dev/null | tail -1)
    last_loss=$(grep -oE "loss=[0-9.]+" "$LOG" 2>/dev/null | tail -1)

    echo "$(date +%F\ %H:%M:%S) $status $last_step $last_loss" >> "$HEARTBEAT"
    last_size=$size
    last_mtime=$mtime
    sleep "$INTERVAL"
done