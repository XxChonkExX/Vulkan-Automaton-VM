#!/bin/bash
# run_in_vt.sh — launch the Granite training from a text VT (frees the iGPU
# from the GDM/Wayland display session to stop display-driver reset kernel
# panics under sustained compute).
#
# Usage:  ./run_in_vt.sh
# Pre-req: be at a TTY (Ctrl+Alt+F3), logged in as chonke.
#   From the desktop, you can do:  sudo chvt 3   then Ctrl+Alt+F3, log in.

set -e

# 1. Kill any stale training (safe: pkill -f exact names)
pkill -TERM -f train_granite_chonk 2>/dev/null || true
pkill -TERM -f run_granite_long   2>/dev/null || true
sleep 2
echo "alive: $(pgrep -af 'train_granite_chonk|run_granite_long' || echo none)"

# 2. Verify wrapper + script settings (sanity check before launch)
echo "--- wrapper settings ---"
grep -E 'CHONK_MIN_BLOCK|CHONK_POOL_BLOCK|CHONK_ACT_GB|CHONK_STAGING_GB|CHONK_CHUNK|CHONK_SEQ_LEN|CHONK_GRAD_ACCUM' \
    /home/chonke/Vulkan-Automaton-VM/run_granite_long.sh
echo "--- quantize setting in script ---"
grep -n 'quantize=' /home/chonke/Vulkan-Automaton-VM/examples/granite_chonk/train_granite_chonk.py | head -2

# 3. Launch the wrapper (absolute paths, detached)
nohup bash /home/chonke/Vulkan-Automaton-VM/run_granite_long.sh \
    >> /tmp/train_granite_wrapper_fixed.log 2>&1 </dev/null &
echo $! > /tmp/train_granite.pid
echo "wrapper PID=$(cat /tmp/train_granite.pid)"

# 4. Verify the python child started (wait ~6s for model load to begin)
sleep 6
echo "--- python child ---"
pgrep -P $(cat /tmp/train_granite.pid) -a || echo "(no child yet, may still be loading)"

# 5. Monitor
echo "--- train log (last 3) ---"
tail -3 /tmp/train_granite.log 2>/dev/null || echo "(empty)"
echo
echo "To return to desktop:  sudo chvt 1   (or Ctrl+Alt+F1/F7)"
echo "To monitor:            tail -f /tmp/train_granite.log"
echo "To stop:               kill -TERM $(cat /tmp/train_granite.pid)"
