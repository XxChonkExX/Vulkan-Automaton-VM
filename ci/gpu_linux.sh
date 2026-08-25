#!/usr/bin/env bash
# gpu_linux.sh - X2 (Strix Halo / ROCm) verification kit.
#
# Runs the full Linux-side 0.3 release gates in one command:
#   1. Fresh or incremental core-only build + CPU regression suites
#   2. Full-stack build (network layer)
#   3. Chonk pool Python module build (build_pool_test.sh)
#   4. Autograd numerical validation vs ATen (audit M4)
#   5. Vulkan enumeration sanity (Strix Halo iGPU)
#   6. OPTIONAL llama-server smoke (set CHONK_TEST_MODEL to a GGUF path)
#   7. SoftRoCE status report (informational)
#
# Usage:
#   bash ci/gpu_linux.sh                 # incremental, existing clone
#   FRESH=1 bash ci/gpu_linux.sh         # fresh clone into /tmp first
#
# Env:
#   REPO            (default: script's repo root)
#   CHONK_TEST_MODEL  path to a GGUF for the optional server smoke
#   SKIP_PYTEST=1     skip the autograd numerics

set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
FAILED=0
gate() { if [ "$2" -eq 0 ]; then echo "PASS  $1 ${3:-}"; else echo "FAIL  $1 ${3:-}"; FAILED=$((FAILED+1)); fi; }

echo "=== GPU CI suite (Linux / X2) ==="
echo "repo: $REPO"

# ---- 0. Optional fresh clone ------------------------------------------------
if [ "${FRESH:-0}" = "1" ]; then
    rm -rf /tmp/vvm-fresh && git clone "$REPO" /tmp/vvm-fresh 2>/dev/null \
        || git clone "https://github.com/XxChonkExX/Vulkan-Automaton-VM.git" /tmp/vvm-fresh
    REPO=/tmp/vvm-fresh
    echo "fresh clone: $REPO"
fi
cd "$REPO"

# ---- 1. Core-only build + CPU suites ----------------------------------------
cmake -S . -B build_ci -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DVVM_BUILD_SHARED=ON -DVVM_BUILD_NETWORK=OFF \
    -DVVM_BUILD_TESTS=ON -DVVM_BUILD_EXAMPLES=OFF \
    -DVVM_ENABLE_VALIDATION=OFF > /dev/null 2>&1
cmake --build build_ci --target vulkan_vm buddy_test chonk_slab_test placement_test -j"$(nproc)" > /dev/null 2>&1
gate "core-only build" $?

./build_ci/tests/buddy_test > /dev/null 2>&1; gate "buddy_test" $?
./build_ci/tests/chonk_slab_test 100000 > /dev/null 2>&1; gate "chonk_slab_test (100k fuzz)" $?
./build_ci/tests/placement_test > /dev/null 2>&1; gate "placement_test" $?

# ---- 2. Full-stack build ------------------------------------------------------
cmake -S . -B build_ci -DVVM_BUILD_NETWORK=ON > /dev/null 2>&1
cmake --build build_ci --target vulkan_vm vulkan_vm_network -j"$(nproc)" > /dev/null 2>&1
gate "full-stack build (core + network)" $?

# ---- 3. Chonk pool Python module ---------------------------------------------
if [ -x scripts/build_pool_test.sh ]; then
    bash scripts/build_pool_test.sh > /dev/null 2>&1
    gate "vulkanvm_pool_test module build" $?
else
    echo "SKIP  pool module (scripts/build_pool_test.sh missing)"
fi

# ---- 4. Autograd numerical validation (M4) ------------------------------------
if [ "${SKIP_PYTEST:-0}" = "1" ]; then
    echo "SKIP  autograd numerics (SKIP_PYTEST=1)"
else
    PY="${PYTHON:-$REPO/venv-ds4/bin/python3}"
    if [ -x "$PY" ]; then
        "$PY" -m pytest "$REPO/python/vulkanvm_torch/test_autograd_numerics.py" -q > /tmp/autograd_numerics.log 2>&1
        gate "autograd numerics vs ATen (M4)" $?
    else
        echo "SKIP  autograd numerics ($PY not found - set PYTHON=)"
    fi
fi

# ---- 5. Vulkan enumeration (Strix Halo iGPU) ----------------------------------
if command -v vulkaninfo >/dev/null 2>&1; then
    if vulkaninfo --summary 2>/dev/null | grep -qi "strix\|gfx1151\|Radeon"; then
        gate "Vulkan enumeration (AMD iGPU present)" 0
    else
        gate "Vulkan enumeration (AMD iGPU present)" 1 "(no AMD device in vulkaninfo)"
    fi
else
    echo "SKIP  vulkaninfo not installed"
fi

# ---- 6. OPTIONAL llama-server smoke -------------------------------------------
if [ -n "${CHONK_TEST_MODEL:-}" ] && [ -f "${CHONK_TEST_MODEL}" ]; then
    SRV=$(find "$REPO" -name llama-server -type f 2>/dev/null | head -1)
    if [ -n "$SRV" ]; then
        GGML_VK_VISIBLE_DEVICES=0 GGML_VK_VVM_POOL=1 \
        "$SRV" -m "$CHONK_TEST_MODEL" -ngl 99 -fa on -c 4096 \
              --temp 0.8 --host 127.0.0.1 --port 8124 \
              > /tmp/gpu_ci_server.log 2>&1 &
        SRVPID=$!
        up=0
        for i in $(seq 1 40); do
            sleep 5
            if curl -s "http://127.0.0.1:8124/health" | grep -q '"ok"'; then up=1; break; fi
        done
        gate "llama-server boot (Chonk pool, iGPU)" $up
        if [ "$up" = "1" ]; then
            tps=$(curl -s http://127.0.0.1:8124/v1/chat/completions \
                -H 'Content-Type: application/json' \
                -d '{"messages":[{"role":"user","content":"Say ready."}],"max_tokens":48,"temperature":0.8,"stream":false}' \
                | grep -o '"predicted_per_second":[0-9.]*' | cut -d: -f2)
            echo "  server smoke t/s: ${tps:-n/a}"
        fi
        kill $SRVPID 2>/dev/null
    else
        echo "SKIP  server smoke (llama-server not built in repo)"
    fi
else
    echo "SKIP  server smoke (set CHONK_TEST_MODEL=/path/to.gguf to enable)"
fi

# ---- 7. SoftRoCE status (informational) ---------------------------------------
if [ -d /sys/module/rdma_rxe ] || lsmod 2>/dev/null | grep -q rdma_rxe; then
    echo "INFO  SoftRoCE: rdma_rxe module present"
    rdma link show 2>/dev/null | head -3 || ip link show | grep -i roce || true
else
    echo "INFO  SoftRoCE: rdma_rxe not loaded (see docs/cross_machine_gpu_sharing.md)"
fi

# ---- Summary -------------------------------------------------------------------
echo "=== GPU CI summary (Linux) ==="
if [ "$FAILED" -eq 0 ]; then
    echo "ALL GATES PASSED"; exit 0
fi
echo "$FAILED GATE(S) FAILED"; exit 1
