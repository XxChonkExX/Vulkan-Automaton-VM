#!/usr/bin/env bash
# run_vt_no_gdm.sh — FREE the iGPU device-local heap, THEN launch training.
#
# The confirmed root cause of the last several wall/rebo∞ts is that the
# GDM/Wayland/gnome-shell session keeps running (and OpenCode's own GPU
# process too), consuming the 91.4 GiB device-local heap that every slab
# block (dedicated exportable) lands in. Vulkan then hits
# VK_ERROR_OUT_OF_DEVICE_MEMORY at ~59GB of pool usage.
#
# This script must be run FROM A REAL TEXT VT (not inside the Wayland/GNOME
# session), because it stops GDM — which tears down the graphical session
# you are currently sitting in.
#
# ---------------------------------------------------------------- USAGE ---
#   Ctrl+Alt+F3  -> log in as chonke (text console)
#   /home/chonke/Vulkan-Automaton-VM/run_vt_no_gdm.sh
#
# It will:
#   1. stop gdm (kills gnome-shell / Wayland / OpenCode GUI)
#   2. verify the device-local heap is actually freed
#   3. launch the existing training wrapper detached
# ----------------------------------------------------------------

set -uo pipefail

REPO="/home/chonke/Vulkan-Automaton-VM"
LOG="/home/chonke/Documents/train_granite.log"
WRAP="$REPO/run_granite_long.sh"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "=== [1/4] Killing any stale training ==="
pkill -9 -f train_granite_chonk 2>/dev/null || true
pkill -9 -f run_granite_long   2>/dev/null || true
sleep 2

echo "=== [2/4] Stopping GDM (frees the iGPU device-local heap) ==="
sudo systemctl stop gdm3 2>/dev/null || sudo systemctl stop gdm 2>/dev/null || \
    echo "  (sudo failed or gdm already stopped; will verify below)"

sleep 3
echo "--- remaining graphical/GPU consumers (should be empty / minimal) ---"
pgrep -af 'gdm|gnome-shell|Xorg|wayland|opencode.*gpu-process' || echo "  none"

echo "=== [3/4] Verify device-local heap is freed ==="
for d in /sys/class/drm/card*/device; do
  vt=$(cat "$d/mem_info_vram_total" 2>/dev/null) || continue
  vu=$(cat "$d/mem_info_vram_used" 2>/dev/null) || continue
  gt=$(cat "$d/mem_info_gtt_total" 2>/dev/null) || continue
  gu=$(cat "$d/mem_info_gtt_used" 2>/dev/null) || continue
  echo "  $d: vram_used=$(awk "BEGIN{printf \"%.1f\", $vu/1e9}")/$(awk "BEGIN{printf \"%.1f\", $vt/1e9}")GB  gtt_used=$(awk "BEGIN{printf \"%.1f\", $gu/1e9}")/$(awk "BEGIN{printf \"%.1f\", $gt/1e9}")GB"
done
echo "--- system memory ---"
free -g | awk 'NR<=2{print}'

echo "=== [4/4] Launching training (detached, auto-resume) ==="
mkdir -p "$(dirname "$LOG")"
setsid bash "$WRAP" >> "$LOG" 2>&1 </dev/null &
sleep 2
echo "wrapper PID: $(pgrep -f run_granite_long | head -1)"
echo
echo "Done. Monitor with:  tail -f $LOG"
echo "Return to graphics (optional, resumes GDM — not advised during training):"
echo "  sudo systemctl start gdm3"