#!/usr/bin/env bash
# run_vt_no_gdm.sh — stop the display server, THEN launch training detached.
#
# Why: under sustained iGPU compute the display driver resets and the box
# drops to the boot/login screen, killing the run. Training from a text VT
# with GDM stopped removes the compositor from the GPU so the driver stays
# up. (Memory pressure from the desktop session is a secondary benefit.)
#
# Design: mask defeats autologin respawn; the wrapper launches detached via
# systemd-run so it survives the GDM teardown. No password prompts at launch:
# systemctl mask/stop/unmask/start gdm are passwordless via the
# chonk-gdm.sudoers drop-in (one-time install, see below).
#
# ------------------------------------------------------- ONE-TIME SETUP ---
# Run ONCE from any terminal (one password prompt, no hurry):
#   sudo install -m 440 /home/chonke/Vulkan-Automaton-VM/chonk-gdm.sudoers \
#       /etc/sudoers.d/chonk-gdm && sudo visudo -c
# ---------------------------------------------------------------- USAGE ---
#   Ctrl+Alt+F3  -> log in as chonke (text console)
#   /home/chonke/Vulkan-Automaton-VM/run_vt_no_gdm.sh     # no password asked
#   ...training runs headless; desktop (incl. this chat) stays down until:
#   sudo -n systemctl unmask gdm3 && sudo -n systemctl start gdm3
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
# Passwordless via the chonk-gdm.sudoers drop-in (see header). sudo -n never
# prompts: if the drop-in is missing this fails fast with a clear message
# instead of racing a password prompt.
if ! sudo -n systemctl mask gdm3 2>/dev/null && ! sudo -n systemctl mask gdm 2>/dev/null; then
    echo "FATAL: passwordless sudo for systemctl is not installed."
    echo "  Run once (one password prompt, no hurry):"
    echo "    sudo install -m 440 $REPO/chonk-gdm.sudoers /etc/sudoers.d/chonk-gdm && sudo visudo -c"
    exit 1
fi
sudo -n systemctl stop gdm3 2>/dev/null || sudo -n systemctl stop gdm 2>/dev/null || true

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
    # --collect so a stale scope from an earlier attempt can't block startup.
    nohup systemd-run --scope --collect --unit=chonk-train bash "$WRAP" >> "$LOG" 2>&1 </dev/null &
else
    nohup setsid bash "$WRAP" >> "$LOG" 2>&1 </dev/null &
fi
sleep 3
echo "wrapper: $(pgrep -af 'run_granite_long' | grep -v grep | head -1)"

echo "=== [5/5] Done ==="
echo "Monitor with:  tail -f $LOG"
echo
echo "To restore the desktop when done training (no password needed):"
echo "  sudo -n systemctl unmask gdm3 && sudo -n systemctl start gdm3"