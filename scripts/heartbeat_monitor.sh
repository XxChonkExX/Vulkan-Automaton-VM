#!/usr/bin/env bash
# heartbeat_monitor.sh — zero-memory training heartbeat monitor.
#
# Pure bash + sleep: RSS stays ~2-4MB. Monitors the training process AND the
# log file. Process-alive takes precedence over log freshness (training may
# write to a terminal instead of a file, making the log legitimately stale).
#
# Status meanings:
#   ALIVE    process running, log growing (or process running, log stale)
#   RUNNING  process running, no log file configured/available
#   IDLE     no process, log fresh
#   STALE    no process, log stale (training died/hung)
#   NO_LOG   no process, no log
#
# Usage:
#   ./scripts/heartbeat_monitor.sh [train_log] [interval_sec]
#
# The monitor itself is meant to be run detached:
#   setsid nohup ./scripts/heartbeat_monitor.sh > /dev/null 2>&1 &

LOG="${1:-/home/chonke/local_training/qwen_logs/train_262k.log}"
INTERVAL="${2:-60}"          # seconds between checks
HEARTBEAT="/home/chonke/local_training/qwen_logs/heartbeat.log"
STALE_AFTER=300              # seconds without log growth => STALE
PATTERN="train_qwen_chonk.py"

mkdir -p "$(dirname "$HEARTBEAT")"

last_size=0
while true; do
    # Process check first: pgrep -f matches the wrapper too, so filter to
    # the actual python trainer (exclude grep/bash self-match).
    if pgrep -f "$PATTERN" | grep -v $$ >/dev/null 2>&1; then
        proc_alive=1
    else
        proc_alive=0
    fi

    size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
    now=$(date +%s)

    if [ "$proc_alive" -eq 1 ] && [ "$size" -eq 0 ]; then
        status="RUNNING (no log)"
    elif [ "$proc_alive" -eq 1 ] && [ "$size" -gt "$last_size" ]; then
        status="ALIVE (log grew +$((size - last_size))B)"
    elif [ "$proc_alive" -eq 1 ]; then
        status="ALIVE (proc up; log stale $((now - mtime))s — terminal output)"
    elif [ "$size" -eq 0 ]; then
        status="NO_LOG"
    elif [ $((now - mtime)) -gt "$STALE_AFTER" ]; then
        status="STALE ($((now - mtime))s no growth)"
    else
        status="IDLE (no proc, log fresh)"
    fi

    last_step=$(grep -oE "Step [0-9]+" "$LOG" 2>/dev/null | tail -1)
    last_loss=$(grep -oE "loss=[0-9.]+" "$LOG" 2>/dev/null | tail -1)
    cpu=$(ps -o pcpu= -p "$(pgrep -f "$PATTERN" | head -1)" 2>/dev/null | tr -d ' ')

    echo "$(date +%F\ %H:%M:%S) $status cpu=${cpu:-n/a}% $last_step $last_loss" >> "$HEARTBEAT"
    last_size=$size
    sleep "$INTERVAL"
done