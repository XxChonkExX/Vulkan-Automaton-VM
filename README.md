# VulkanVM — Unified Vulkan Memory Pool & Model Distribution

**VulkanVM** is a cross-vendor, high-performance Vulkan memory management library that solves GPU memory fragmentation and enables zero-copy memory sharing across AMD, NVIDIA, and Intel GPUs. It combines a budget-capped, buddy-suballocated memory pool with a Hugging Face–style model registry for distributing model weights over TCP — download once, load into the pool, run local multi-GPU inference.

## Problem Statement

- **AMD APU (Strix Halo 395)**: ROCm fails at high memory pressure due to fragmentation, not capacity
- **Multi-vendor systems (7900XTX + Arc Pro B70)**: No unified memory pool across vendors
- **Large tensor workloads**: Need persistent, aligned allocations for tensor cores
- **Swap/offload**: Need host-backed shadow for demand paging
- **Model distribution**: Need a Hugging Face–style way to publish/fetch model weights across machines

## Solution

VulkanVM provides a `UnifiedMemoryPool` that:

1. **Pre-allocates large blocks** (256MB–2GB) at startup, sub-allocates with buddy allocator
2. **Never returns memory to OS** — eliminates fragmentation; budget-capped to avoid starving the system
3. **Exports/imports `VkDeviceMemory`** via `VK_EXTERNAL_MEMORY` for cross-GPU sharing — dedicated allocations only; sub-allocated blocks are auto-promoted to dedicated copies on export
4. **Supports host shadow buffers** for swap/offload (`madvise`/`mprotect` are **deprecated** — unsafe on `vkMapMemory` memory)
5. **Works on AMD, NVIDIA, Intel** — Vulkan is the common denominator
6. **RAII handle wrappers** — `UniqueHandle` wraps Vulkan objects; `ExternalHandle` owns FD/HANDLE lifetime (move-only, no leaks)
7. **Thread-safe** — the public pool API is guarded by an internal mutex (fixed: budget checks, dedicated alloc, move assignment)
8. **Optional TLS** — the TCP transport can be encrypted with OpenSSL (TLS 1.2+)
9. **Cross-platform** — Windows, Linux, and macOS
10. **ModelHub** — Hugging Face–style weight distribution over TCP with content-addressed chunks, cache + resume

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
│           async copy (madvise/mprotect deprecated)              │
├─────────────────────────────────────────────────────────────────┤
│  ModelHub: content-addressed model weight distribution over TCP │
└─────────────────────────────────────────────────────────────────┘
```

## Features

| Feature | Status |
|---------|--------|
| **Core Memory Pool** | |
| Persistent large-block allocation | ✅ |
| Budget-capped pool (`maxHeapFraction`, `maxPoolBytes`, `VK_EXT_memory_budget`) | ✅ |
| Buddy sub-allocator (256KB alignment, pow2 enforcement, double-free validation) | ✅ |
| `PoolConfig::forDevice()` — APU/Unified/Discrete detection with tuned defaults | ✅ |
| `MemoryUsage` intent API (`GpuOnly`, `CpuToGpu`, `GpuToCpu`, `CpuCopy`, `Auto`) | ✅ |
| `AllocDesc` rich allocation descriptor (`exportable`, `mapped`, `name`) | ✅ |
| Debug names (`VK_EXT_debug_utils` on buffers/memory) | ✅ |
| Richer `PoolStats` (`totalCapacity`, `dedicatedCount`, `allocationCount`) | ✅ |
| **Cross-GPU Sharing** | |
| Cross-vendor memory sharing (AMD↔NVIDIA↔Intel) | ✅ |
| Linux: OPAQUE_FD, DMA-BUF | ✅ |
| Windows: OPAQUE_WIN32, D3D12_HEAP | ✅ |
| Dedicated allocation model for external export (1 VkDeviceMemory per shareable alloc) | ✅ |
| Auto-promotion of sub-allocated blocks on export (dedicated copy) | ✅ |
| Cross-device memory type re-selection on import | ✅ |
| External handle ownership: import consumes handle (`release()`); multi-GPU `duplicateForImport()` | ✅ |
| **Core Infrastructure** | |
| RAII handle wrappers (`UniqueHandle`, `ExternalHandle` FD/HANDLE ownership) | ✅ |
| Thread safety (mutex-guarded public API) | ✅ |
| Bindless device addresses | ✅ |
| Host shadow buffer for swap | ✅ |
| Async migration (device↔host) | ✅ |
| `madvise(MADV_DONTNEED/FREE)` | 🔧 (opt-in, unsafe on vkMapMemory memory) |
| `mprotect(PROT_NONE)` for page faults | 🔧 (opt-in, unsafe on vkMapMemory memory) |
| Timeline semaphore sync | ✅ |
| **ModelHub — Model Weight Distribution** | |
| Hugging Face–style publish/fetch over TCP | ✅ |
| Content-addressed chunks (SHA-256, 4 MiB slices) with hash-verified resume | ✅ |
| Local cache `~/.cache/vvm/models/<model>/<version>/` + `.vvm_complete` marker | ✅ |
| `publish(modelId, sourceDir, version)` → hashes + copies to cache | ✅ |
| `fetch(hubAddr, modelId, destDir, version)` → manifest + files, verifies per-chunk | ✅ |
| **Networking** | |
| Multi-node cluster over TCP (zero deps) | ✅ |
| TLS-secured TCP transport (OpenSSL, TLS 1.2+, SNI + ALPN) | ✅ |
| Multi-GPU pool manager | ✅ |
| **Testing & Quality** | |
| Buddy allocator unit tests (pow2, merge, double-free, fragmentation) | ✅ |
| External handle ownership tests (dup, consume, isolate) | ✅ |
| Model registry tests (publish, fetch manifest, fetch files, cache hit, list) | ✅ |
| Network loopback test (two-node cluster, push/pull migration, zero-copy import) | ✅ |

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

### New: Intent-based Allocation (MemoryUsage + AllocDesc)
```cpp
#include <vulkan_vm/vulkan_vm.hpp>

PoolConfig cfg = PoolConfig::forDevice(physicalDevice);  // auto-tunes for APU/discrete
cfg.maxHeapFraction = 0.75f;  // never exceed 75% of device heap budget

auto pool = UnifiedMemoryPool::create(devConfig, cfg);

// Simple: intent-driven
auto w = pool->allocate({ .size = 64_MiB,
                          .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          .memoryUsage = MemoryUsage::GpuOnly,
                          .exportable = false,
                          .name = "weight_tensor" });

// Staging buffer (upload)
auto staging = pool->allocate({ .size = 128_MiB,
                                .memoryUsage = MemoryUsage::CpuToGpu,
                                .mapped = true,
                                .name = "upload_buffer" });

// Readback
auto readback = pool->allocate({ .size = 16_MiB,
                                 .memoryUsage = MemoryUsage::GpuToCpu,
                                 .mapped = true });
```

### Cross-GPU Sharing (AMD ↔ NVIDIA ↔ Intel)
```cpp
// Master GPU allocates a DEDICATED exportable allocation.
// (Cross-GPU export requires dedicated memory - one VkDeviceMemory per
// shareable allocation. exportMemory() rejects sub-allocated blocks; the
// multi-node manager auto-promotes them to dedicated copies on export.)
const VkBufferUsageFlags kUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
auto masterAlloc = masterPool.allocateDedicatedExportable(128 * 1024 * 1024, kUsage);

// Export for sharing. exportInfo owns the FD/HANDLE via ExternalHandle and
// closes it automatically when it goes out of scope.
auto exportInfo = masterPool.exportMemory(*masterAlloc, ExternalHandleType::DmaBuf);  // Linux

// Import on other GPUs — handle is consumed by the driver on success.
for (auto& peerPool : peerPools) {
    // Each peer needs its OWN handle; duplicateForImport() dups the FD/HANDLE.
    auto perPeer = duplicateForImport(*exportInfo);
    auto peerAlloc = peerPool.importMemory(std::move(perPeer), kUsage);
}

// Sync with timeline semaphores
manager.submitMigrationBarrier(operations);
manager.waitAllIdle();
```

### Host Offload / Swap
```cpp
OffloadConfig offloadConfig;
offloadConfig.hostShadowSize = 4ull * 1024 * 1024 * 1024;  // 4GB host buffer
// offloadConfig.useMadvise = true;    // opt-in only; unsafe on vkMapMemory memory
// offloadConfig.useMprotect = true;   // opt-in only; unsafe on vkMapMemory memory

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

### ModelHub — Hugging Face–Style Weight Distribution
```cpp
// --- Hub (server) ---
ModelHub hub("/data/model-store");
hub.start("0.0.0.0", 51010);
hub.publish("chonk/llama-3b-q4", "./local-model-files", "v1");
hub.stop();

// --- Client (any machine) ---
ModelHub::fetch("192.168.1.50:51010", "chonk/llama-3b-q4", "./my-models", "v1");
// Then load into UnifiedMemoryPool:
auto alloc = pool->allocate({ .size = modelSize, .memoryUsage = MemoryUsage::GpuOnly });
// ... copy weights from ./my-models/weights.safetensors into alloc ...
```

Cache layout (HF-style):
```
~/.cache/vvm/models/chonk/llama-3b-q4/v1/
  config.json
  weights.safetensors
  tokenizer/tokenizer.json
  .vvm_complete
```

### Multi-Node Network Module

`vvm::network::MultiNodePoolManager` provides a host-staged, multi-node cluster for
moving tensors between machines over plain TCP. It uses a Spark-inspired wire
protocol: a versioned 32-byte header `[magic "VVMN"][u8 version][u8 x3 reserved]
[u32 type][u32 flags][u32 bodyLen][u32 seq][u64 streamLen]` followed by a control
body and an optional bulk stream transferred in 4 MB slices directly between the
socket and a caller-provided buffer (no intermediate copy). Control-plane messages:
`MsgRegisterNode`, `MsgGetClusterView`, `MsgAllocate`, `MsgExport`, `MsgImport`,
`MsgMigratePull`, `MsgMigratePush`, `MsgHeartbeat`, `MsgLeaveCluster`,
`MsgDeallocate`, `MsgModelList`, `MsgModelManifest`, `MsgModelChunk`.

The TCP control/data plane builds **with zero external dependencies** (Winsock on
Windows, BSD sockets on Linux). gRPC and RDMA/verbs remain optional extras
(promoted via `VVM_NETWORK_HAS_GRPC` / `VVM_NETWORK_HAS_VERBS`).

**TLS**: the transport supports OpenSSL-backed TLS 1.2+ (gated by
`VVM_NETWORK_HAS_TLS`). Call `transport.enableTls(TlsConfig{...})` with
`certPath`/`keyPath`/`caPath`, optional peer verification (`verifyPeer`) and
ALPN (`alpnProtocols`, default `"vvm/1.0"`). `NetworkConfig.useTls` mirrors this
for the multi-node manager (cert/key/ca paths in `tlsCertPath`/`tlsKeyPath`/`tlsCaPath`).
Server-side SNI and client-side ALPN negotiation are implemented.

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

### Model Registry Test
```bash
./build/examples/model_registry_test.exe   # Windows
./build/examples/model_registry_test       # Linux
```

Expected: publish 4-file model, fetch manifest, fetch all files, verify hashes,
cache hit on 2nd fetch, list models — all `PASS`.

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
// Using the new auto-tuner:
PoolConfig apuConfig = PoolConfig::forDevice(physicalDevice);
// Sets blockSize=1GB, maxBlocks=8, maxHeapFraction=0.7, unified memory type

// Or manual:
PoolConfig apuConfig;
apuConfig.blockSize = totalVRAM * 0.8;  // Reserve 80% at startup
apuConfig.minAlignment = 256 * 1024;
apuConfig.enableHostVisible = true;      // Unified memory
apuConfig.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | 
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
apuConfig.maxHeapFraction = 0.7f;
```

## Budget & Topology API

```cpp
// Query topology
MemoryTopology topo = detectMemoryTopology(physicalDevice);
switch (topo) {
  case MemoryTopology::Unified:  // APU, single heap
    cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    break;
  case MemoryTopology::Hybrid:   // APU with small dedicated carve-out
    // ...
  case MemoryTopology::Discrete: // dGPU
    // default
}

// Budget check before allocating
if (pool->wouldExceedBudget(512_MiB)) {
    // trigger offload or fail soft
}
```

## License

MIT License - see LICENSE file.

## Credits

- **Nemotron** — primary coder
- **Deepseek** — primary coder, co-author of the multi-node network module (Spark-style TCP transport, TLS, host-staged push/pull migration, cluster registration, two-node loopback test) and implementation support across the codebase
- **GLM** — primary coder, review and refinements during development
- **Grok** — code audit of the memory pool, buddy allocator, external memory, and network layers (dedicated export model, RAII handle lifetime, thread safety, BlockManager dedup, budget caps, APU topology, handle ownership)
- **NVIDIA** — network module framework and implementation foundation; special thanks for supporting Open Source despite being a Mega-Corp
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
- [x] Buddy allocator pow2 enforcement & double-free validation
- [x] External memory dedicated allocation model (1 VkDeviceMemory per shareable alloc)
- [x] RAII handle wrappers (`UniqueHandle`, `ExternalHandle` FD/HANDLE lifetime)
- [x] Cross-device memory type re-selection on import
- [x] Thread-safe public API (mutex-guarded pool)
- [x] TLS-secured TCP transport (OpenSSL, SNI + ALPN)
- [x] macOS build support
- [x] Network serialization hardening + minimal test suite
- [x] Budget-capped pool (`maxHeapFraction`, `maxPoolBytes`, `VK_EXT_memory_budget`)
- [x] `PoolConfig::forDevice()` + `detectMemoryTopology()` (APU/Discrete/Hybrid)
- [x] `MemoryUsage` + `AllocDesc` intent-based allocation API
- [x] `VK_EXT_debug_utils` object names on buffers/memory
- [x] `ModelHub` — Hugging Face–style weight distribution (content-addressed, cache+resume)
- [x] External handle ownership fix (import consumes, `duplicateForImport` for multi-GPU)
- [x] Honest `defragment()` → releases idle blocks; `trim()` implemented
- [x] Buddy allocator unit tests (pow2, merge, double-free, fragmentation)
- [x] External handle ownership unit tests (dup, consume, isolate)
- [x] Model registry tests (publish, fetch manifest, fetch files, cache hit, list)
- [ ] Sparse/residency support for virtual memory
- [ ] Direct GPU↔GPU copy (P2P) without host staging
- [ ] RDMA/verbs GPU-direct transport (stubs in place; experimental)
- [ ] Integration with ML frameworks (PyTorch, ONNX Runtime)
- [ ] Windows WDDM2.6+ hardware scheduling hints
- [ ] Android/Vulkan support