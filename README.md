# VulkanVM - Unified Vulkan Memory Pool

A cross-vendor, high-performance Vulkan memory management library designed to solve GPU memory fragmentation and enable efficient memory sharing across AMD, NVIDIA, and Intel GPUs.

## Problem Statement

- **AMD APU (Strix Halo 395)**: ROCm fails at high memory pressure due to fragmentation, not capacity
- **Multi-vendor systems (7900XTX + Arc Pro B70)**: No unified memory pool across vendors
- **Large tensor workloads**: Need persistent, aligned allocations for tensor cores
- **Swap/offload**: Need host-backed shadow for demand paging

## Solution

VulkanVM provides a `UnifiedMemoryPool` that:
1. **Pre-allocates large blocks** (256MB-2GB) at startup, sub-allocates with buddy allocator
2. **Never returns memory to OS** - eliminates fragmentation
3. **Exports/imports `VkDeviceMemory`** via `VK_EXTERNAL_MEMORY` for cross-GPU sharing
4. **Supports host shadow buffers** with `madvise`/`mprotect` for swap/offload
5. **Works on AMD, NVIDIA, Intel** - Vulkan is the common denominator

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        UnifiedMemoryPool                         │
├─────────────────────────────────────────────────────────────────┤
│  Block 0 (512MB)  │  Block 1 (512MB)  │  Block 2 (512MB)  ...  │
│  ├─ Buddy Alloc   │  ├─ Buddy Alloc   │  ├─ Buddy Alloc        │
│  │  ├─ 64MB      │  │  ├─ 128MB      │  │                     │
│  │  ├─ 32MB      │  │  ├─ 64MB       │  │                     │
│  │  └─ free...   │  │  └─ free...    │  │                     │
├─────────────────────────────────────────────────────────────────┤
│  Cross-GPU: VK_EXTERNAL_MEMORY (OPAQUE_FD / OPAQUE_WIN32 /     │
│             D3D12_HEAP / DMA_BUF)                                │
├─────────────────────────────────────────────────────────────────┤
│  Offload: Host shadow buffer (HOST_VISIBLE|COHERENT) +         │
│           async copy + madvise/mprotect                         │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Status |
|---------|--------|
| Persistent large-block allocation | ✅ |
| Buddy sub-allocator (256KB alignment) | ✅ |
| Cross-vendor memory sharing (AMD↔NVIDIA↔Intel) | ✅ |
| Linux: OPAQUE_FD, DMA-BUF | ✅ |
| Windows: OPAQUE_WIN32, D3D12_HEAP | ✅ |
| Bindless device addresses | ✅ |
| Host shadow buffer for swap | ✅ |
| Async migration (device↔host) | ✅ |
| madvise(MADV_DONTNEED/FREE) | ✅ |
| mprotect(PROT_NONE) for page faults | ✅ |
| Timeline semaphore sync | ✅ |
| Multi-GPU pool manager | ✅ |

## Building

### Linux
```bash
./scripts/build.sh --tests
```

### Windows
```powershell
.\scripts\build.ps1 -tests
```

### Manual CMake
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

### Requirements
- CMake 3.20+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- Vulkan SDK 1.3+
- Optional: Volk (dynamic Vulkan loading)

## Usage

### Basic Allocation
```cpp
#include <vulkan_vm/vulkan_vm.hpp>

using namespace vvm;

// Setup device
DeviceConfig devConfig = { /* ... */ };
PoolConfig poolConfig;
poolConfig.blockSize = 512 * 1024 * 1024;  // 512MB blocks
poolConfig.minAlignment = 256 * 1024;       // Tensor core alignment

auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);

// Allocate tensor buffer (bindless-ready)
auto alloc = pool->allocateTensor(64 * 1024 * 1024);  // 64MB
// alloc->deviceAddress usable in shaders
// alloc->buffer for vkCmd* operations

// Deallocate (returns to pool, not OS)
pool->deallocate(std::move(*alloc));
```

### Cross-GPU Sharing (AMD ↔ NVIDIA ↔ Intel)
```cpp
// Master GPU allocates
auto masterAlloc = masterPool.allocateTensor(128 * 1024 * 1024);

// Export for sharing
auto exportInfo = masterPool.exportMemory(*masterAlloc, ExternalHandleType::DmaBuf);  // Linux

// Import on other GPUs
for (auto& peerPool : peerPools) {
    auto peerAlloc = peerPool.importMemory(exportInfo, usage);
    // peerAlloc->deviceAddress valid on peer GPU
}

// Sync with timeline semaphores
manager.submitMigrationBarrier(operations);
manager.waitAllIdle();
```

### Host Offload / Swap
```cpp
OffloadConfig offloadConfig;
offloadConfig.hostShadowSize = 4ull * 1024 * 1024 * 1024;  // 4GB host buffer
offloadConfig.useMadvise = true;
offloadConfig.useMprotect = true;

OffloadManager offloadManager(&pool, offloadConfig);

// Async offload to host
auto op = offloadManager.offload(allocation);
// ... do other work ...
offloadManager.waitMigration(op);

// Sync reload
offloadManager.reloadSync(allocation);
```

### Multi-GPU Pool Manager
```cpp
std::vector<DeviceConfig> devices = { devConfig0, devConfig1, devConfig2 };
PoolConfig config;

auto manager = MultiGPUPoolManager::create(devices, config, 0);  // GPU 0 = master

// Allocate distributed (master allocates, others import)
auto allocations = manager.allocateDistributed(256 * 1024 * 1024, usage);

// All GPUs now have valid allocations pointing to shared memory
for (size_t i = 0; i < devices.size(); ++i) {
    auto& alloc = allocations[i];
    // alloc->deviceAddress valid on GPU i
}
```

### Multi-Node Network Module

`vvm::network::MultiNodePoolManager` provides a host-staged, multi-node cluster for
moving tensors between machines over plain TCP. It uses a Spark-inspired wire
protocol: a versioned 32-byte header `[magic "VVMN"][u8 version][u8 x3 reserved]
[u32 type][u32 flags][u32 bodyLen][u32 seq][u64 streamLen]` followed by a control
body and an optional bulk stream transferred in 4 MB slices directly between the
socket and a caller-provided buffer (no intermediate copy). Control-plane messages:
`MsgRegisterNode`, `MsgGetClusterView`, `MsgAllocate`, `MsgExport`, `MsgImport`,
`MsgMigratePull`, `MsgMigratePush`, `MsgHeartbeat`, `MsgLeaveCluster`, `MsgDeallocate`.

The TCP control/data plane builds **with zero external dependencies** (Winsock on
Windows, BSD sockets on Linux). gRPC and RDMA/verbs remain optional extras
(promoted via `VVM_NETWORK_HAS_GRPC` / `VVM_NETWORK_HAS_VERBS`).

```cpp
using namespace vvm::network;

NetworkConfig netA; netA.listenAddress = "0.0.0.0:51001";  // bootstrap seed
auto nodeA = MultiNodePoolManager::create({devCfg}, poolCfg, netA);
nodeA->start();                       // registers as cluster root (no seeds)
nodeA->registerWithCluster();

NetworkConfig netB; netB.listenAddress = "0.0.0.0:51002";
netB.seedNodes = { "127.0.0.1:51001" };
auto nodeB = MultiNodePoolManager::create({devCfg}, poolCfg, netB);
nodeB->start();                       // connects to nodeA and joins the cluster
nodeB->registerWithCluster();

// Remote allocation + push migration (B -> A)
auto src = nodeB->getLocalPool().allocate(size, /*...usage...*/, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
std::memset(src->hostPtr, 0xAB, size);

auto dst = nodeB->allocateRemote(nodeA->getLocalNodeId(), size, /*...*/,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeB->migrateToRemote(*src, *dst, /*useRdma=*/false);  // streamed in 4 MB slices

// Pull migration (B -> A): A pulls bytes from a remote allocation on B
auto remoteDesc = nodeB->exportForRemote(*src, false, true);
auto localDst   = nodeA->getLocalPool().allocate(size, /*...*/, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeA->migrateFromRemote(*remoteDesc, *localDst, /*useRdma=*/false);
```

A working two-node loopback demo lives at `examples/network_test.cpp` and is built
as the `network_test` target. Run it after building:

```bash
./build/examples/network_test.exe   # Windows
./build/examples/network_test       # Linux
```

Expected: cluster registration, remote allocate, push verify `PASS`, pull verify
`PASS`, `ALL TESTS PASSED`.

| Network feature | Status |
|-----------------|--------|
| TCP control/data plane (zero deps) | ✅ |
| Spark-style 32-byte header + 4 MB slice streaming | ✅ |
| Cluster registration + heartbeat | ✅ |
| Remote allocate / export / import / deallocate | ✅ |
| Host-staged push/pull migration | ✅ |
| gRPC control plane (optional) | ✅ (auto-enabled when gRPC found) |
| RDMA/verbs GPU-direct (optional) | 🔧 (stubs; host-staged fallback always available) |


## Cross-Vendor Compatibility Matrix

| Source → Target | Linux Handle | Windows Handle |
|-----------------|--------------|----------------|
| NVIDIA → AMD    | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| NVIDIA → Intel  | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| AMD → NVIDIA    | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| Intel → NVIDIA  | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| AMD ↔ Intel     | DMA-BUF      | OPAQUE_WIN32 |
| Same vendor     | OPAQUE_FD    | OPAQUE_WIN32 / D3D12_HEAP |

## APU-Specific Tuning (Strix Halo 395)

```cpp
PoolConfig apuConfig;
apuConfig.blockSize = totalVRAM * 0.8;  // Reserve 80% at startup
apuConfig.minAlignment = 256 * 1024;
apuConfig.enableHostVisible = true;      // Unified memory
apuConfig.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
```

## License

MIT License - see LICENSE file.

## Credits

- **Nemotron** — primary coder
- **Deepseek** — co-author of the multi-node network module (Spark-style TCP transport, host-staged push/pull migration, cluster registration, two-node loopback test) and implementation support across the codebase
- **Grok (xAI)** — architectural review, offload/migration design review, and recommendations that shaped the buddy-backed pool and host shadow integration
- **GLM** — review and refinements during development
- **ChonkE** — project owner, 1% contributor

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`ctest --output-on-failure`)
5. Submit PR

## Roadmap

- [x] Multi-node cluster (TCP control/data plane, Spark-style streaming)
- [x] Host-staged push/pull migration with byte-pattern verification
- [ ] Sparse/residency support for virtual memory
- [ ] Direct GPU↔GPU copy (P2P) without host staging
- [ ] RDMA/verbs GPU-direct transport (stubs in place; host-staged fallback ships today)
- [ ] Integration with ML frameworks (PyTorch, ONNX Runtime)
- [ ] Windows WDDM2.6+ hardware scheduling hints
- [ ] Android/Vulkan support