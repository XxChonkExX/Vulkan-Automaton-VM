# Windows → Linux Network Test Setup (0.3 Release Gates)

> For X2 (Strix Halo, Ubuntu native) and Windows box (XTX + B70, WSL2).
> Run these steps to execute the **Windows→Linux TCP soak** and **validation layers** gates.

---

## 1. X2 (Linux) — Native Ubuntu

### Prerequisites (already installed per Phase 6)
```bash
# Verify
vulkaninfo --summary | grep -i strix
which rdma_rxe  # SoftRoCE optional for TCP tests
ls /sys/module/rdma_rxe  # if doing RDMA later
```

### Fresh Build (required for validation gate)
```bash
cd /home/mikeh/Vulkan-Automaton-VM  # or wherever you clone
FRESH=1 bash ci/gpu_linux.sh
# This does: core build → CPU suites → full-stack (network) → pool module → autograd → vulkaninfo
```

### Run TCP Server (X2 acts as cluster node B)
```bash
# In build_rdma directory
cd build_rdma

# Start node B (listens on 0.0.0.0:53260 UDP + 53251 TCP control)
VVM_DISABLE_SAME_PROCESS_ZC=1 ./examples/network_test server 53251 53260 &
SERVER_PID=$!
echo "X2 server PID: $SERVER_PID"
```

### Run Validation Layers (Release Gate)
```bash
# Full ctest with validation layers
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ctest --test-dir . --output-on-failure -j$(nproc)
# Must pass 11/11 suites clean
```

---

## 2. Windows Box (WSL2 + Native)

### WSL2 Setup (for SoftRoCE later)
```powershell
# In PowerShell Admin
wsl --install -d Ubuntu-24.04
wsl -d Ubuntu-24.04 -- bash -c "sudo apt update && sudo apt install -y linux-headers-$(uname -r) rdma-core ibverbs-utils"
# Custom kernel with rdma_rxe: see docs/cross_machine_gpu_sharing.md
# For TCP soak, stock WSL2 kernel works fine.
```

### Native Windows Build (fresh clone for release gate)
```powershell
# Fresh clone
git clone https://github.com/XxChonkExX/Vulkan-Automaton-VM.git D:\VulkanVM-fresh
cd D:\VulkanVM-fresh

# Configure + build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DVVM_BUILD_SHARED=ON -DVVM_BUILD_NETWORK=ON `
  -DVVM_BUILD_TESTS=ON -DVVM_BUILD_EXAMPLES=OFF `
  -DVVM_ENABLE_VALIDATION=ON
cmake --build build --config Release
```

### Copy DLLs to test directory
```powershell
Copy-Item build\vulkan_vm.dll build\tests\ -Force
Copy-Item build\src\network\vulkan_vm_network.dll build\tests\ -Force
```

### Run Validation Layers (Release Gate)
```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
ctest --test-dir build -C Release --output-on-failure
# Must pass all suites clean
```

---

## 3. Windows → Linux TCP Soak

### X2: Start Server (keep running)
```bash
# On X2, in build_rdma/
VVM_DISABLE_SAME_PROCESS_ZC=1 ./examples/network_test server 53251 53260
# Listens on UDP 53260 (fabric) + TCP 53251 (control)
```

### Windows: Run Client Soak
```powershell
# On Windows, in build\tests\
# X2 IP = 192.168.0.117 (adjust if different)
.\network_test.exe client 192.168.0.117 53251 53260 `
  --bytes 16777216 `        # 16 MiB per iteration
  --iterations 100 `        # soak iterations
  --verify `                # verify data integrity
  --report-throughput
```

### Expected Output (healthy)
```
[CTRL] Connecting to 192.168.0.117:53251...
[CTRL] Connected, cluster join OK
[FABRIC] Registered 16 MiB pattern buffer (rkey=0x...)
[ITER 1/100] PUSH 16 MiB: 12.3 GB/s (verify: PASS)
[ITER 2/100] PULL 16 MiB: 11.8 GB/s (verify: PASS)
...
[SUMMARY] 100 iterations, 0 errors, avg 11.9 GB/s
```

### Soak Parameters for 0.3 Gate
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--bytes` | 16777216 (16 MiB) | Exercises staging + fragmentation |
| `--iterations` | 100 | ~2-3 GB total, catches slow leaks |
| `--verify` | enabled | Data integrity check |
| `--report-throughput` | enabled | Regression detection |

---

## 4. Windows → Linux RDMA (Optional, if WSL2 rxe ready)

### X2: Verify SoftRoCE
```bash
rdma link show  # rxe0 on enpXXXX ACTIVE
ibv_devinfo     # rxe0 PORT_ACTIVE
```

### Windows WSL2: Enable SoftRoCE
```bash
# Inside WSL2 Ubuntu
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
ibv_devinfo  # verify rxe0
```

### Run RDMA Test (Windows client → Linux server)
```powershell
# On Windows native (needs rdma-core in WSL2 path, complex)
# Skip for 0.3 if not ready; TCP soak is the gate.
```

---

## 5. Checklist for 0.3 Release

| Gate | Command | Status |
|------|---------|--------|
| **Linux validation layers** | `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ctest --test-dir build_rdma` | ☐ |
| **Windows validation layers** | `$env:VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"; ctest --test-dir build -C Release` | ☐ |
| **Windows→Linux TCP soak** | `.\network_test.exe client 192.168.0.117 53251 53260 --bytes 16777216 --iterations 100 --verify` | ☐ |
| **Linux fresh clone build** | `FRESH=1 bash ci/gpu_linux.sh` | ☐ |
| **Windows fresh clone build** | Fresh clone → cmake → build → ctest | ☐ |

---

## 6. Network Config Notes

- **X2 advertise address**: Must be LAN IP (`192.168.0.117`), NOT loopback
- **Windows client**: Points to X2 LAN IP
- **Firewall**: Ensure UDP 53260 + TCP 53251 open both directions
- **GGML_VK_VVM_POOL=1**: Not needed for pure network_test (uses host-staged TCP)

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Connection timed out" | Check firewall, verify X2 `advertiseAddress` = LAN IP |
| "RDMA backend not found" | TCP soak uses `VVM_RDMA_BACKEND=udp` (set in network_test) |
| "Pool stats show 1 pool" | Expected for network_test (no Chonk pool); pool=1 is the host-staged fallback |
| Validation layer VUIDs | Run with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; known leaks OK, new VUIDs = fail |

---

## 8. Sync Command for X2

```bash
cd /home/mikeh/Vulkan-Automaton-VM
git pull origin main
# Rebuild if CMakeLists or source changed
cmake -S . -B build_rdma -DVVM_BUILD_NETWORK=ON > /dev/null 2>&1
cmake --build build_rdma --target vulkan_vm vulkan_vm_network -j$(nproc) > /dev/null 2>&1
```

---

**Last updated**: 2026-08-27 (commit `f2f5386`)
**Target**: 0.3.0 release gates