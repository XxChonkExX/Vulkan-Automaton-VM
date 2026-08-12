# VulkanVM — A Practical Introduction (With a Relay Race Analogy)

## What Is VulkanVM?

**VulkanVM is a cross-vendor GPU networking and RDMA library** that moves tensor VRAM across AMD, NVIDIA, and Intel GPUs — locally and across machines — with zero-copy where hardware allows, automatic fallback where it doesn't. It's built on a hardened, zero-fragmentation Vulkan memory pool (the "Chonk Buffer") and includes a Hugging Face–style model registry, capacity-first shard placement, and a unified tensor transport for AI inference clusters.

**One header, one library, zero system changes.** It's a library you link into your application — not a driver, not a daemon, not a kernel module.

**Version**: 0.2.0-pre (pre-release)

---

## The Relay Race Analogy: How GPU Data Actually Moves

Imagine a **4×400m relay race** where the baton is your tensor data. Each runner is a hardware domain. The handoff zones are where data crosses boundaries — and each handoff costs a "penalty lap" (latency, copies, CPU involvement).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE GPU NETWORK RELAY RACE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   GPU A VRAM          PCIe              Host RAM             PCIe         GPU B VRAM
│   ┌──────────┐      ┌────────┐       ┌─────────────┐      ┌────────┐    ┌──────────┐
│   │  Tensor  │─────▶│  DMA   │──────▶│  Staging    │─────▶│  DMA   │────▶│  Tensor  │
│   │  (Baton) │      │ Engine │       │  (Handoff)  │      │ Engine │    │  (Baton) │
│   └──────────┘      └────────┘       └─────────────┘      └────────┘    └──────────┘
│        │                │                   │                  │             │
│        │  ZERO PENALTY  │  PENALTY LAP     │  PENALTY LAP     │  ZERO       │
│        │  (GPU-direct)  │  (Host staging)  │  (Host staging)  │  PENALTY    │
│        │                │                  │                  │  (GPU-direct)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Four Handoff Lanes (Priority Order)

| Lane | Name | What Happens | Penalty Laps |
|------|------|--------------|--------------|
| **1** | **P2P** | GPU A → GPU B directly via `VK_EXTERNAL_MEMORY` + `vkCmdCopyBuffer` | **0** — Baton never leaves the track |
| **2** | **RDMA** | NIC reads GPU A VRAM → network → NIC writes GPU B VRAM (`ibv_reg_dmabuf_mr`) | **0** — Baton teleports via NIC |
| **3** | **AHardwareBuffer** | Android zero-copy via `VK_ANDROID_external_memory_android_hardware_buffer` | **0** — Special track for mobile |
| **4** | **Host-Staged** | GPU A → PCIe → Host RAM (4 MB chunks) → PCIe → GPU B | **2** — Two penalty laps (out to host, back to GPU) |
| **5** | **Network (TCP)** | Host-staged + TCP sockets across machines | **2+** — Penalty laps + network latency |

**VulkanVM auto-picks the fastest available lane:**
```cpp
TransportConfig config;
config.preference = TransportConfig::Preference::Auto;  // Tries Lane 1 → 2 → 3 → 4 → 5
```

---

## Quick Start — GPU Networking

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/tensor_transport.hpp>
using namespace vvm;
using namespace vvm::tensor;

// 1. Select GPU
VkInstance instance = createInstance();
auto devices = enumerateDevices(instance);
auto best = selectBestDevice(devices, true, 1024);
VkPhysicalDevice physicalDevice = best->device;

auto queues = findQueueFamilies(physicalDevice);
DeviceConfig devConfig{
    .physicalDevice = physicalDevice,
    .device = createDevice(physicalDevice, queues),
    .graphicsQueueFamily = queues.graphics.value_or(0),
    .computeQueueFamily  = queues.compute.value_or(0),
    .transferQueueFamily = queues.transfer.value_or(0),
};

// 2. Create pool (auto-tuned)
PoolConfig poolConfig = PoolConfig::forDevice(physicalDevice);
poolConfig.maxHeapFraction = 0.75f;

// 3. Configure tensor transport with networking
TransportConfig config;
config.preference = TransportConfig::Preference::Auto;
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;
config.networkPort = 51000;
config.enableTLS = true;
config.tlsCertPath = "cert.pem";
config.tlsKeyPath = "key.pem";
config.tlsCaPath = "ca.pem";

// 4. Create transport
auto transport = vvm::tensor::createTensorTransport(config, {devConfig}, poolConfig);
if (!transport->initialize()) { /* handle error */ }

// 5. Allocate a tensor
TensorMetadata meta;
meta.dtype = DataType::Float16;
meta.layout = MemoryLayout::ChannelsLast;
meta.shape = TensorShape::makeChannelsLast({1, 32, 32, 128});
meta.name = "conv_weight";

auto tensor = transport->allocateTensor(meta, 0);

// 6. Multi-node send/recv (auto-picks best lane)
transport->sendTensor(tensor, "192.168.1.10:51001#0", 
    [](bool ok, const std::string& err) { /* done */ });

transport->recvTensor(receiver, "192.168.1.11:51002#0",
    [](bool ok, const std::string& err) { /* done */ });

// 7. Collectives (ring all-reduce across GPUs/nodes)
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
```

---

## GPU-Direct RDMA Vendor Paths (The "Zero Penalty" Lanes)

For NIC-attached GPU memory DMA (bypassing host CPU entirely):

| GPU Vendor | Linux Path | Windows Path |
|------------|------------|--------------|
| **NVIDIA (0x10DE)** | `VK_NV_external_memory_rdma` → `vkGetMemoryRemoteAddressNV` → `ibv_reg_mr` on PCI BAR (needs `nvidia-peermem`) | NDKPI + `CreateMemoryRegion` from `D3D12_HEAP`/`OPAQUE_WIN32` |
| **AMD (0x1002)** | `DMA_BUF` → `ibv_reg_dmabuf_mr` (or mmap fallback) | `OPAQUE_WIN32` → `MapViewOfFile` → NDKPI |
| **Intel (0x8086)** | `DMA_BUF` → `ibv_reg_dmabuf_mr` (Xe KMD) | **Level Zero** `zeMemGetAllocProperties` + `ze_external_memory_export_win32_handle_t` → NDKPI |

```cpp
// In VerbsRdmaTransport:
auto region = rdmaTransport->registerGpuMemory(memory, offset, size, buffer);
// Returns RdmaMemoryRegion with lkey/rkey/rdmaAddr for direct NIC DMA
```

**Intel Level Zero (Windows) — New:**
```cpp
// Export GPU memory as Win32 handle via Level Zero
ze_external_memory_export_win32_handle_t exportDesc{};
exportDesc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_WIN32;
exportDesc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32;
zeMemGetAllocProperties(context, allocation, &props, &exportDesc);
// exportDesc.handle → NDKPI CreateMemoryRegion / user-mode RDMA
```

**AMD ROCm (Linux) — New:**
```cpp
// Import DMA-BUF into HIP external memory
hipExternalMemoryHandleDesc desc{};
desc.type = HIP_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF;
desc.handle.fd = dmaBufFd;
desc.size = size;
hipExternalMemory_t extMem;
hipImportExternalMemory(&extMem, &desc);

hipExternalMemoryBufferDesc bufDesc{};
bufDesc.offset = offset;
bufDesc.size = size;
void* mappedPtr;
hipExternalMemoryGetMappedBuffer(&mappedPtr, extMem, &bufDesc);
// mappedPtr → ibv_reg_mr for RDMA
```

---

## SoftRoCE (Linux Software RDMA) — Lane 2 Without Special Hardware

No RNIC? No problem. Kernel `rxe` module enables verbs/RDMA over standard Ethernet.

```bash
# WSL2 / Ubuntu 24.04
sudo rdma link add rxe0 type rxe netdev eth0
sudo rdma link add rxe1 type rxe netdev lo
rdma link show  # verify ACTIVE
ibv_devices     # lists rxe0, rxe1
```

**Verified:** `tensor_network_test` passes on WSL2 (kernel 6.18.40) with `rxe0` + `rxe1`. On native Linux with physical RNIC, same code uses hardware RDMA. Without SoftRoCE/RNIC → falls back to host-staged TCP (always available).

---

## Unified Tensor Transport

Moves **tensors** — not raw bytes — with shape, dtype, and layout awareness.

```cpp
#include <vulkan_vm/tensor_transport.hpp>
using namespace vvm::tensor;

TransportConfig config;
config.preference = TransportConfig::Preference::Auto;
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;
config.networkPort = 51000;
config.enableTLS = true;

auto transport = vvm::tensor::createTensorTransport(config, devices, poolConfig);
transport->initialize();

// Allocate + distribute
auto tensor = transport->allocateTensor(meta, 0);
auto tensors = transport->allocateDistributed(meta, {0, 1, 2});

// Collectives
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
transport->broadcast(rootTensor, {0, 1, 2}, 0);
transport->allGather({t0, t1, t2}, output, {0, 1, 2});
transport->reduceScatter({t0, t1, t2}, output, ReduceOp::Sum, {0, 1, 2});

// Async
transport->allReduceAsync({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2},
    [](bool ok, const std::string& err) { /* done */ });
transport->flushAsync();

// Multi-node
transport->sendTensor(tensor, "192.168.1.10:51001#0", callback);
transport->recvTensor(receiver, "192.168.1.11:51002#0", callback);
```

**Collective dtypes:** FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, Int4 (packed), Int8, UInt8, Int32, Int64, Bool  
**Collective ops:** Sum, Mean, Min, Max, Product, Band, Bor, Bxor

---

## ModelHub & Shard Placement

### ModelHub — Model Weight Distribution

Content-addressed, Hugging Face–style model registry with TCP distribution.

```cpp
// Server
ModelHub hub("/data/model-store");
hub.start("0.0.0.0", 51010);
hub.publish("my-org/llama-3b-q4", "./local-model-files", "v1");

// Client
ModelHub::fetch("192.168.1.50:51010", "my-org/llama-3b-q4", "./my-models", "v1");
```

### Shard Placement (Capacity-First Bin Packing)

```cpp
#include <vulkan_vm/placement.hpp>
using namespace vvm::placement;

// Model manifest
ModelManifest model;
model.shards = {
    ShardSpec{"blk.0-3", "hash1", ShardKind::Weights, 2_GiB, 0, 3},
    ShardSpec{"blk.4-7", "hash2", ShardKind::Weights, 2_GiB, 4, 7},
    ShardSpec{"blk.8-11", "hash3", ShardKind::Weights, 2_GiB, 8, 11},
};

// Cluster topology
ClusterCapacity cluster;
cluster.nodes = {
    NodeCapacity{"node-a", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
    NodeCapacity{"node-b", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
};
cluster.reservedActivationBytes = 512_MiB;

// Policy
PlacementPolicy policy;
policy.allowHostOffload = true;
policy.preferContiguousLayers = true;
policy.packMode = PackMode::PackDense;

// Plan + Execute
PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
PlacementExecutor executor(nodeManager, modelHub);
ExecuteOptions opt{ .fetchIfMissing=true, .verifyChecksum=true };
ExecuteResult result = executor.executeLocal(model, plan, opt);
```

---

## Core Memory Pool (Chonk Buffer)

### Buddy Allocator — Zero Fragmentation

```
512 MB block
├── 256 MB (allocated)
└── 256 MB (free)
    ├── 128 MB (allocated)
    └── 128 MB (free)
        ├── 64 MB (allocated)
        └── 64 MB (free)  ← merges back with buddy when freed
```

- Power-of-two splitting/coalescing — no external fragmentation
- O(1) via `unordered_set` free lists per order
- Optional internal mutex (`threadSafe_`)
- Deterministic lowest-offset placement
- **Debug invariant checker** (`checkInvariants()`) — validates free/allocated consistency

### Auto-Tuning

| Hardware | Blocks | Heap Fraction | Host Shadow |
|----------|--------|---------------|-------------|
| APU / Strix Halo | 1-2 GB | Capped | Disabled (VRAM host-visible) |
| Discrete <24 GB | 512 MB | 0.75 | 4× multiplier |
| High-VRAM ≥24 GB | 2 GB | 0.8-0.85 | 2× multiplier, capped at 4 GB |

```cpp
PoolConfig poolConfig = PoolConfig::forDevice(physicalDevice);
poolConfig.maxHeapFraction = 0.75f;
poolConfig.hostShadowMultiplier = 2.0f;  // custom multiplier
poolConfig.maxHostShadowBytes = 4_GiB;   // hard cap
```

### Memory Usage Intents

```cpp
enum class MemoryUsage {
    GpuOnly,        // DEVICE_LOCAL
    CpuToGpu,       // HOST_VISIBLE | HOST_COHERENT — upload
    GpuToCpu,       // HOST_VISIBLE | HOST_COHERENT — readback
    CpuCopy,        // HOST_VISIBLE — CPU-only
    Auto            // Let VulkanVM decide
};
```

---

## Cross-GPU Memory Sharing

Universal translator for GPU memory handles:

| Source → Target | Linux | Windows |
|-----------------|-------|---------|
| NVIDIA → AMD | DMA-BUF | D3D12_HEAP → OPAQUE_WIN32 |
| AMD → NVIDIA | DMA-BUF | OPAQUE_WIN32 → D3D12_HEAP |
| AMD ↔ Intel | DMA-BUF / OPAQUE_FD | OPAQUE_WIN32 |

```cpp
// Master allocates dedicated exportable
auto masterAlloc = masterPool.allocateDedicatedExportable(256_MiB, usage);
auto exportInfo = masterPool.exportMemory(*masterAlloc,
    #ifdef VVM_PLATFORM_LINUX
        ExternalHandleType::DmaBuf
    #else
        ExternalHandleType::OpaqueWin32
    #endif
);

// Peer imports (per-peer duplicate)
auto peerHandle = duplicateForImport(exportInfo.handle);
auto peerAlloc = peerPool.importMemory(std::move(peerHandle), usage);
// Same physical memory on both GPUs

// Direct GPU→GPU copy (no host staging)
manager->copyDeviceToDevice(0, 1, *srcAlloc, *dstAlloc, 0, 0, size);
```

---

## Host Offload / Demand Paging

```cpp
OffloadConfig cfg;
cfg.hostShadowSize = 4_GiB;
cfg.transferQueue = transferQueue;
cfg.transferQueueFamily = transferQueueFamily;

pool->initializeOffload(cfg);

// Async offload
auto op = pool->offloadToHost(allocation);
pool->waitMigration(op);

// Reload
pool->reloadToDevice(allocation);
```

Uses GPU copy engine (DMA), not `madvise`/`mprotect`.

---

## Sparse / Residency Virtual Memory

```cpp
SparseVirtualMemoryPool pool(device, physicalDevice);
pool.initialize(1_TiB, 32_MiB);

auto reservation = pool.reserveVirtual(256_GiB, usage);  // No physical yet
pool.commit(reservation, 64_GiB, 64_GiB, memoryFlags);   // Commit on demand
pool.uncommit(reservation, 64_GiB, 64_GiB);              // Release
```

---

## Android / Vulkan (AHardwareBuffer)

Zero-copy sharing with Android graphics pipeline (Surface, MediaCodec, Camera).

```bash
./scripts/build_android.sh arm64-v8a android-34 Release
```

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
#include <android/hardware_buffer.h>

AHardwareBuffer_Desc desc{};
desc.width = 1920; desc.height = 1080;
desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | 
             AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;

AHardwareBuffer* hardwareBuffer;
AHardwareBuffer_allocate(&desc, &hardwareBuffer);

ExternalMemoryInfo extInfo;
extInfo.type = ExternalHandleType::AndroidHardwareBuffer;
extInfo.handle = ExternalHandle(hardwareBuffer);
extInfo.size = bufferSize;
extInfo.dedicatedAllocation = true;

auto allocation = pool.importMemory(std::move(extInfo), usage);
auto deviceAddress = allocation->deviceAddress;  // Use in shaders
```

---

## PyTorch & ONNX Integration

```python
import vulkanvm_torch as vvm
pool = vvm.UnifiedMemoryPool.create(dev_config, pool_config)
alloc = pool.allocate_tensor(64 * 1024 * 1024, "weights")

pool.initialize_offload(4_GB, transfer_queue, transfer_queue_family)
op = pool.offload_to_host(alloc)
pool.wait_migration(op)
pool.reload_to_device(alloc)

# Cross-GPU
exported = pool.export_memory(alloc, vvm.ExternalHandleType.OpaqueWin32)
peer_alloc = peer_pool.import_memory(exported, usage)
```

```python
import vulkanvm_onnx as vvm
provider = vvm.VulkanVMExecutionProvider(
    vvm.VulkanVMExecutionProviderConfig(pool_size=2_GB, host_shadow_size=4_GB)
)
alloc = provider.allocate_tensor([1, 3, 224, 224], vvm.TensorElementType.FLOAT, "input")
provider.upload_tensor(alloc, input_np, input_np.nbytes)
output_np = provider.download_tensor(alloc, output_np, output_np.nbytes)
```

---

## Building

### Windows
```powershell
.\scripts\build.ps1 -Tests -BuildType Release
```

### Linux
```bash
./scripts/build.sh --tests
```

### Requirements
- CMake 3.20+, C++20 (GCC 10+, Clang 12+, MSVC 19.30+)
- Vulkan SDK 1.3+
- Optional: Volk, OpenSSL, gRPC/Protobuf, ibverbs/rdma_cm
- **Windows SDK 10.0.26100+** for NDKPI
- **Level Zero SDK** for Intel GPU-direct on Windows

---

## Hardware Compatibility

| GPU | Works? | Notes |
|-----|--------|-------|
| RTX 4090 / 7900 XTX (24 GB) | ✅ Excellent | 2 GB blocks (auto-detected) |
| RTX 4070 / 7800 XT (12-16 GB) | ✅ Excellent | 256 MB blocks |
| Arc A770 / A750 / B70 (16/8 GB) | ✅ Good | 256 MB blocks, Level Zero GPU-direct |
| Laptop dGPU (8 GB) | ✅ Good | 128 MB blocks |
| **Strix Halo APU (96-128 GB)** | ✅ **BEST** | 2 GB blocks, sweet spot |
| Integrated (2-8 GB shared) | ⚠️ Works but tight | 64 MB blocks |
| Ancient GPU (Vulkan 1.0) | ❌ | Needs Vulkan 1.3+ |

---

## Zero-Disruption Integration

| What It Does | What It Doesn't Do |
|--------------|-------------------|
| Links into YOUR app like any `.dll`/`.so` | ❌ Touches kernel/drivers |
| Uses standard Vulkan API | ❌ Requires admin/root |
| Creates its own memory blocks | ❌ Touches other processes |
| Optionally maps host memory | ❌ Changes GPU clocks |
| **Install = `cmake --install`** | ❌ Installs daemons |

---

## Key Tests

```bash
# Core
./build/tests/basic_test
./build/tests/buddy_test
./build/tests/external_handle_test

# Placement (10 pure-logic tests, no GPU needed)
./build/tests/placement_test

# Multi-GPU (requires 2+ GPUs)
./build/tests/multi_gpu_test

# Multi-vendor RDMA (requires 2+ GPUs from different vendors, RDMA transport)
./build/tests/multi_vendor_rdma_test

# Network (2-node loopback)
./build/tests/network_test
./build/tests/tensor_network_test

# Tensor collectives
./build/tests/tensor_collective_test

# Windows ND fake provider
.\build_win\tests\ndk_transport_test.exe
```

---

## Why "Chonk Buffer"?

The core memory pool is affectionately nicknamed the **Chonk Buffer** (as in "chunk" — a contiguous block of memory). 

It's a tongue-in-cheek reference to:
- **Chunk** = standard term for a contiguous memory block
- **Chonk** = internet slang for "thicc" / hefty — because the pool reserves a massive chunk of VRAM upfront (e.g., 512 MB - 80 GB)
- **Brand alignment** — GitHub org is `XxChonkExX`, repo is `Vulkan-Automaton-VM`

Don't let the name fool you — under the hood it's a production-grade, hardened buddy allocator with zero fragmentation, budget awareness, and full thread safety.

---

## Need Help?

- **README.md** — Full API reference, cross-vendor matrix, build options
- **examples/** — `network_test`, `model_registry_test`, `tensor_compute`, `offload_test`, `multi_gpu_test`, `tensor_network_test`
- **tests/** — Unit tests for buddy allocator, external handles, sparse, placement
- **GitHub Issues** — Bug reports, feature requests

*VulkanVM — Making GPUs play nice since 2026.*