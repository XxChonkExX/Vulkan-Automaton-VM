#!/usr/bin/env bash
# heartbeat_monitor.sh — zero-memory training heartbeat monitor.
#
# Pure bash + sleep: RSS stays ~2-4MB. Monitors the training process AND
# reads the trainer's status file (train_status.txt, written atomically by
# train_qwen_chonk.py every 8 chunks) for live numbers. Falls back to the
# log file when no status file exists.
#
# Status meanings:
#   ALIVE    process running, fresh status/log data
#   RUNNING  process running, no status file yet (still in setup/loading)
#   STALE    process dead or status/log stale (training died/hung)
#   NO_LOG   process dead, nothing to read
#
# Usage:
#   ./scripts/heartbeat_monitor.sh [status_file] [interval_sec]
#
# The monitor itself is meant to be run detached:
#   setsid nohup ./scripts/heartbeat_monitor.sh > /dev/null 2>&1 &

STATUS="${1:-/home/chonke/local_training/qwen_logs/train_status.txt}"
LOG="${STATUS%/*}/train_262k.log"
INTERVAL="${2:-60}"          # seconds between checks
HEARTBEAT="${STATUS%/*}/heartbeat.log"
STALE_AFTER=300              # seconds without fresh data => STALE
PATTERN="train_qwen_chonk.py"

mkdir -p "$(dirname "$HEARTBEAT")"

get() { # get <file> <key>
    awk -F= -v k="$2" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$1" 2>/dev/null
}

while true; do
    if pgrep -f "$PATTERN" | grep -v $$ >/dev/null 2>&1; then
        proc_alive=1
        pid=$(pgrep -f "$PATTERN" | grep -v $$ | head -1)
    else
        proc_alive=0
        pid=""
    fi

    cpu="n/a"
    rss="n/a"
    [ -n "$pid" ] && cpu=$(ps -o pcpu= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$pid" ] && rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')

    # Prefer the status file; fall back to the log
    src="$STATUS"
    if [ ! -f "$STATUS" ]; then
        src="$LOG"
    fi
    mtime=$(stat -c %Y "$src" 2>/dev/null || echo 0)
    now=$(date +%s)
    age=$((now - mtime))

    if [ "$src" = "$STATUS" ]; then
        step=$(get "$src" step)
        chunk=$(get "$src" chunk)
        loss=$(get "$src" loss)
        lr=$(get "$src" lr)
        pool=$(get "$src" pool_gb)
        data="fresh"
    elif [ "$age" -le "$STALE_AFTER" ]; then
        step=$(grep -oE "Step [0-9]+" "$src" 2>/dev/null | tail -1)
        loss=$(grep -oE "loss=[0-9.]+" "$src" 2>/dev/null | tail -1)
        chunk=""; lr=""; pool=""
        data="log"
    else
        step=""; chunk=""; loss=""; lr=""; pool=""
        data="stale"
    fi

    if [ "$proc_alive" -eq 1 ] && [ "$data" = "fresh" ]; then
        status="ALIVE"
    elif [ "$proc_alive" -eq 1 ] && [ "$data" = "log" ]; then
        status="ALIVE (log data)"
    elif [ "$proc_alive" -eq 1 ]; then
        status="ALIVE (no live data — restart trainer for status file)"
    else
        status="STALE"
    fi

    echo "$(date +%F\ %H:%M:%S) $status cpu=${cpu}% rss=$((rss/1024))MB age=${age}s step=${step:-?} chunk=${chunk:-?} loss=${loss:-?} lr=${lr:-?} pool=${pool:-?}GB" >> "$HEARTBEAT"
    sleep "$INTERVAL"
done