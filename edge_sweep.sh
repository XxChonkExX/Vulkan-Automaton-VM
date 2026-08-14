#!/bin/bash
# Sequential edge sweep: one config per process, waits for completion,
# health-checks between runs. Usage: edge_sweep.sh
cd /home/chonke/Vulkan-Automaton-VM
source /home/chonke/venv-ds4/bin/activate
CS=$1
SEQ=$2
echo "=== [$CS chunk] launching ==="
setsid nohup python3 -u edge_test.py $CS $SEQ /home/chonke/Vulkan-Automaton-VM/edge_${CS}_${SEQ}.json </dev/null >/home/chonke/Vulkan-Automaton-VM/edge_${CS}_${SEQ}.log 2>&1 &
PID=$!
# wait for completion, poll health
for i in $(seq 1 90); do
  if ! kill -0 $PID 2>/dev/null; then break; fi
  if grep -q "SWEEP_CONFIG_PASS\|Traceback\|Error" /home/chonke/Vulkan-Automaton-VM/edge_${CS}_${SEQ}.log 2>/dev/null; then
    # still finishing? wait a bit more
    sleep 5
    if ! kill -0 $PID 2>/dev/null; then break; fi
  fi
  sleep 10
done
if kill -0 $PID 2>/dev/null; then
  echo "STILL RUNNING after 900s — kill"
  kill -9 $PID 2>/dev/null
fi
sleep 5
echo "=== [$CS chunk] result: $(grep -c SWEEP_CONFIG_PASS /home/chonke/Vulkan-Automaton-VM/edge_${CS}_${SEQ}.log) pass ==="
free -g | awk 'NR==2{print "mem used="$3" free="$4}'
rocm-smi --showuse 2>/dev/null | grep "GPU use"