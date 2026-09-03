#!/usr/bin/env bash
# run_vt_no_gdm.sh — FREE the iGPU device-local heap, THEN launch training.
#
# Confirmed root cause of the Exit-137 wall: the GDM/Wayland/gnome-shell
# session (auto-login) keeps running and consumes the 91.4 GiB device-local
# heap that every dedicated-exportable slab block lands in. Vulkan then hits
# VK_ERROR_OUT_OF_DEVICE_MEMORY at ~59-90GB of pool usage.
#
# The PREVIOUS version of this script had a fatal flaw: running
# `sudo systemctl stop gdm3` from a VT still tore down the caller's session
# (GDM restarts via autologin), killing the script before it could launch
# training. This version decouples the two:
#   1. stopping GDM is done via `systemctl stop` (mask first to defeat
#      autologin respawn during the run);
#   2. the training wrapper is launched detached via systemd-run so it is NOT
#      a child of the (dying) GDM session and survives the teardown.
#
# ---------------------------------------------------------------- USAGE ---
#   Ctrl+Alt+F3  -> log in as chonke (text console)
#   /home/chonke/Vulkan-Automaton-VM/run_vt_no_gdm.sh
# ----------------------------------------------------------------

set -uo pipefail

REPO="/home/chonke/Vulkan-Automaton-VM"
LOG="/home/chonke/Documents/train_granite.log"
WRAP="$REPO/run_granite_long.sh"

die() { echo "FATAL: $*" >&2; exit 1; }

echo "=== [1/5] Killing any stale training (weak signals first) ==="
pkill -INT -f train_granite_chonk 2>/dev/null || true
pkill -INT -f run_granite_long   2>/dev/null || true
sleep 2
pkill -9 -f train_granite_chonk 2>/dev/null || true
pkill -9 -f run_granite_long   2>/dev/null || true

echo "=== [2/5] Disabling GDM autologin respawn, then stopping it ==="
# mask defeats the autologin respawn so it stays down for the run.
if sudo -n true 2>/dev/null; then
    SUDO="sudo"
else
    echo "  sudo needs a password — you will be prompted once here."
    SUDO="sudo"
fi
$SUDO systemctl mask gdm3 2>/dev/null || $SUDO systemctl mask gdm 2>/dev/null || true
$SUDO systemctl stop gdm3   2>/dev/null || $SUDO systemctl stop gdm 2>/dev/null || true

sleep 3
echo "--- remaining graphical/GPU consumers (should be empty / minimal) ---"
pgrep -af 'gdm|gnome-shell|Xorg|wayland|opencode.*gpu-process' || echo "  none"

echo "=== [3/5] Verify device-local heap is freed ==="
for d in /sys/class/drm/card*/device; do
  vt=$(cat "$d/mem_info_vram_total" 2>/dev/null) || continue
  vu=$(cat "$d/mem_info_vram_used" 2>/dev/null) || continue
  gt=$(cat "$d/mem_info_gtt_total" 2>/dev/null) || continue
  gu=$(cat "$d/mem_info_gtt_used" 2>/dev/null) || continue
  echo "  $d: vram=$(awk "BEGIN{printf \"%.1f\", $vu/1e9}")/$(awk "BEGIN{printf \"%.1f\", $vt/1e9}")GB  gtt=$(awk "BEGIN{printf \"%.1f\", $gu/1e9}")/$(awk "BEGIN{printf \"%.1f\", $gt/1e9}")GB"
done
free -g | awk 'NR<=2{print "  "$0}'

echo "=== [4/5] Launching training detached (survives GDM teardown) ==="
mkdir -p "$(dirname "$LOG")"
# systemd-run puts the wrapper outside the session cgroup so stopping GDM
# cannot kill it. Fall back to setsid+nohup if systemd-run is unavailable.
if command -v systemd-run >/dev/null 2>&1; then
    nohup systemd-run --scope --unit=chonk-train bash "$WRAP" >> "$LOG" 2>&1 </dev/null &
else
    nohup setsid bash "$WRAP" >> "$LOG" 2>&1 </dev/null &
fi
sleep 3
echo "wrapper: $(pgrep -af 'run_granite_long' | grep -v grep | head -1)"

echo "=== [5/5] Done ==="
echo "Monitor with:  tail -f $LOG"
echo
echo "To restore the desktop when done training:"
echo "  $SUDO systemctl unmask gdm3 && $SUDO systemctl start gdm3"