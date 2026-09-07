#!/bin/bash
# restart_training.sh — kill a wedged trainer + relaunch cleanly
pkill -9 -f train_granite_chonk 2>/dev/null
pkill -9 -f run_granite_long 2>/dev/null
sleep 3
ps aux | grep 'train_granite_chonk[.]py' | grep -v grep && echo "STILL ALIVE - ABORTING" && exit 1
cd /home/chonke/Vulkan-Automaton-VM
setsid bash run_granite_long.sh >> /home/chonke/Documents/train_granite.log 2>&1 </dev/null &
sleep 2
echo "Relaunched. GTT before/after check:"
cat /sys/class/drm/card1/device/mem_info_gtt_used | awk '{printf "  gtt: %.1f GB\n", $1/1e9}'
echo "Monitor: tail -f /home/chonke/Documents/train_granite.log"
