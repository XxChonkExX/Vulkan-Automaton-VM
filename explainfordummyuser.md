# VulkanVM — A Practical Introduction

## What Is VulkanVM?

**VulkanVM is a Vulkan memory management library that solves three hard problems:** eliminating GPU memory fragmentation on AMD APUs, enabling zero-copy memory sharing across NVIDIA/AMD/Intel GPUs, and providing demand-paged host offload when VRAM runs out. It also includes a Hugging Face–style model registry for distributing model weights over TCP and a capacity-first shard placer for multi-node model distribution.

**One header, one library, zero system changes.** It's a library you link into your application — not a driver, not a daemon, not a kernel module.

---

## The Three Problems It Solves

### Problem 1: Memory Fragmentation (The "Messy Heap" Problem)

**What happens:** GPU memory allocators return blocks of varying sizes. Over time, free space becomes scattered — 100 MB free total, but no single contiguous 50 MB block. On AMD's unified-memory APUs (Strix Halo), this causes allocation failures even when plenty of VRAM appears free.

**VulkanVM's fix:** At startup, VulkanVM reserves a large portion of VRAM (e.g., 80% of the heap budget) in a few large blocks. It then uses a **buddy allocator** — a power-of-two splitting/coalescing algorithm — to sub-allocate. Memory is **never returned to the OS**, so the heap never fragments. Large contiguous allocations always succeed.

### Problem 2: Cross-Vendor Memory Sharing (The "Different Protocols" Problem)

**What happens:** NVIDIA, AMD, and Intel GPUs each use different external memory handle types. They can't directly share `VkDeviceMemory` — the handle from one vendor is opaque to another.

**VulkanVM's fix:** VulkanVM acts as a universal translator. It knows the handle type for every vendor pair:

| Source → Target | Linux Handle | Windows Handle |
|-----------------|--------------|----------------|
| NVIDIA → AMD    | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| NVIDIA → Intel  | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| AMD → NVIDIA    | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| Intel → NVIDIA  | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| AMD ↔ Intel     | DMA-BUF / OPAQUE_FD | OPAQUE_WIN32 |

**The workflow:**
1. Allocate a **dedicated exportable** buffer on the source GPU (one `VkDeviceMemory` per shareable buffer)
2. Export the handle — VulkanVM translates to the target's handle type
3. Import on the target GPU — VulkanVM handles cross-vendor memory type re-selection
4. Both GPUs now reference the **same physical memory**

### Problem 3: VRAM Overflow (The "Out of VRAM" Crash)

**What happens:** Large models exceed GPU VRAM. Traditional allocators fail with `VK_ERROR_OUT_OF_DEVICE_MEMORY`.

**VulkanVM's fix:** A **host-shadow buffer** (pinned host memory) acts as a spill warehouse. When VRAM fills:
1. `offloadToHost()` — asynchronously copies allocation to host shadow via GPU copy engine
2. Original VRAM freed; program continues
3. `reloadToDevice()` — brings it back when needed

The copy engine handles the transfer asynchronously; your compute keeps running.

---

## Core Architecture

### The Buddy Allocator (Zero Fragmentation)

A power-of-two allocator that splits and merges blocks with their "buddy":

```
512 MB block
├── 256 MB (allocated)
└── 256 MB (free)
    ├── 128 MB (allocated)
    └── 128 MB (free)
        ├── 64 MB (allocated)
        └── 64 MB (free)  ← merges back with buddy when freed
```

**Key properties:**
- Blocks always merge with their "buddy" (adjacent block from same parent split)
- No external fragmentation — free blocks always coalesce
- O(1) allocation/deallocation
- Deterministic placement — lowest free offset handed out first, so buddy pairs stay adjacent and coalescing always works
- Power-of-two alignment guaranteed

### Auto-Tuning (APU vs Discrete vs High-VRAM)

`PoolConfig::forDevice()` reads the device memory at runtime and picks the block profile:

- **APUs / unified memory (Strix Halo)** → 1-2 GB blocks, host-visible, capped heap fraction
- **Discrete cards under 24 GB** → 512 MB blocks (16 blocks, 0.75 heap fraction) so mid-range cards stay conservative
- **High-VRAM cards ≥24 GB (RTX 4090, RTX 6000 Ada)** → auto-scales to 2 GB blocks, 64 blocks, 0.8-0.85 heap fraction so big pools can actually be saturated

Explicit overrides are available: `PoolConfig::forHighVRAM(physicalDevice)` (2 GB blocks, up to 0.85 of the heap) and `PoolConfig::forAPU(totalSystemRAM)` (RAM-budget-aware). Default to `forDevice()` and let detection do the work.

### Memory Usage Intents (No Raw Flags)

Instead of raw `VkMemoryPropertyFlags`, you declare intent:

```cpp
enum class MemoryUsage {
    GpuOnly,        // DEVICE_LOCAL — fastest GPU access
    CpuToGpu,       // HOST_VISIBLE | HOST_COHERENT — upload staging
    GpuToCpu,       // HOST_VISIBLE | HOST_COHERENT — readback
    CpuCopy,        // HOST_VISIBLE — CPU-only copies
    Auto            // Let VulkanVM decide
};
```

### External Handle Ownership (No Leaks)

```cpp
// Export: returns RAII handle — closes on scope exit
auto handle = pool.exportMemory(allocation, ExternalHandleType::OpaqueWin32);

// Import: CONSUMES the handle (driver takes ownership). On failure, handle still closes.
auto imported = peerPool.importMemory(std::move(handle), usage);

// Multiple peers? Duplicate first:
for (auto& peer : peers) {
    auto perPeer = duplicateForImport(handle);  // dup() / DuplicateHandle()
    peer.importMemory(std::move(perPeer), usage);
}
```

---

## Cross-GPU Sharing (Multi-GPU Pool Manager)

```cpp
// Configure devices
std::vector<DeviceConfig> devices = { amdConfig, intelConfig, nvidiaConfig };
PoolConfig config = PoolConfig::forDevice(devices[0].physicalDevice);

// GPU 0 = master (allocates + exports)
auto manager = MultiGPUPoolManager::create(devices, config, 0);

// One call: allocates on master, imports on all peers
auto allocs = manager.allocateDistributed(512_MiB, usage);
// allocs[i] is valid on devices[i], all aliasing the same memory

// Direct GPU→GPU copy (no host staging) when driver supports it
manager->copyDeviceToDevice(0, 1, *srcAlloc, *dstAlloc, 0, 0, size);
// Falls back to 4 MB chunked host-staged copy if driver refuses cross-import
```

---

## Host Offload / Demand Paging

```cpp
OffloadConfig cfg;
cfg.hostShadowSize = 4_GiB;
cfg.transferQueue = transferQueue;
cfg.transferQueueFamily = transferQueueFamily;

pool->initializeOffload(cfg);

// Async offload (device → host)
auto op = pool->offloadToHost(allocation);
// ... do other work while DMA runs ...
pool->waitMigration(op);

// Sync reload (host → device)
pool->reloadToDevice(allocation);
```

Uses GPU copy engine (DMA engines), not `madvise`/`mprotect` (unsafe on driver mappings).

---

## ModelHub — Model Weight Distribution

**Server (Hub):**
```cpp
ModelHub hub("/data/model-store");
hub.start("0.0.0.0", 51010);
hub.publish("my-org/llama-3b-q4", "./local-model-files", "v1");
```

**Client:**
```cpp
ModelHub::fetch("192.168.1.50:51010", "my-org/llama-3b-q4", "./my-models", "v1");
```

**Cache layout (HF-style):**
```
~/.cache/vvm/models/my-org/llama-3b-q4/v1/
  config.json
  weights.safetensors
  tokenizer/tokenizer.json
  .vvm_complete          # marker = cache entry valid
```

Content-addressed chunks (SHA-256, 4 MiB slices), resume support, TLS optional.

---

## Unified Tensor Transport (`vvm::tensor::Transport`)

Moves tensors — not raw bytes — with shape, dtype, and layout awareness.

### Tensor Metadata

```cpp
struct TensorMetadata {
    DataType dtype = DataType::Float32;      // FP32, FP16, BF16, INT8, INT4, FP8_E4M3, FP8_E5M2
    MemoryLayout layout = MemoryLayout::Contiguous;  // Contiguous, ChannelsLast, Blocked, Strided
    TensorShape shape;                        // dims + optional strides
    std::string name;
};
```

### Operations

```cpp
auto transport = vvm::tensor::createTensorTransport(config, devices, poolConfig);
transport->initialize();

// Allocate + distribute
auto tensor = transport->allocateTensor(meta, 0);           // on GPU 0
auto tensors = transport->allocateDistributed(meta, {0, 1, 2});  // across GPUs

// Copy with layout conversion (compute shaders, no host round-trip)
transport->copyWithLayoutConversion(src, dst, MemoryLayout::Blocked);

// Collectives (ring all-reduce, broadcast, all-gather, reduce-scatter)
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
transport->broadcast(rootTensor, {0, 1, 2}, 0);
transport->allGather({t0, t1, t2}, output, {0, 1, 2});
transport->reduceScatter({t0, t1, t2}, output, ReduceOp::Sum, {0, 1, 2});

// Async variants (worker thread + callbacks)
transport->allReduceAsync({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2}, 
    [](bool ok, const std::string& err) { /* done */ });
transport->flushAsync();  // wait for all queued ops
```

### Multi-Node Network

```cpp
// Node A (bootstrap)
TransportConfig cfgA; cfgA.listenAddress = "0.0.0.0:51001";
auto nodeA = createTensorTransport(cfgA, devices, poolConfig);
nodeA->joinCluster("0.0.0.0:51001");

// Node B (joins)
TransportConfig cfgB; cfgB.listenAddress = "0.0.0.0:51002";
cfgB.seedNodes = {"127.0.0.1:51001"};
auto nodeB = createTensorTransport(cfgB, devices, poolConfig);
nodeB->joinCluster("127.0.0.1:51001");

// Send/recv (auto-picks P2P → RDMA → Host-Staged → Network)
nodeB->sendTensor(tensor, "192.168.1.10:51001#0", callback);
nodeA->recvTensor(receiver, "192.168.1.11:51002#0", callback);
```

---

## GPU-Direct RDMA (Zero-Copy NIC DMA)

For NIC-attached GPU memory DMA (bypassing host CPU):

| GPU Vendor | Path |
|------------|------|
| **NVIDIA** | `VK_NV_external_memory_rdma` → `vkGetMemoryRemoteAddressNV` → `ibv_reg_mr` on PCI BAR (Linux, needs nvidia-peermem) or NDKPI (Windows) |
| **AMD/Intel** | Export `OPAQUE_WIN32`/`DMA_BUF` → `MapViewOfFile`/`mmap` → `ibv_reg_mr` on mapped VA |

```cpp
auto region = rdmaTransport->registerGpuMemory(memory, offset, size, buffer);
// Returns RdmaMemoryRegion with lkey/rkey/rdmaAddr for direct NIC DMA
```

**Linux SoftRoCE (no RNIC needed):** Kernel `rxe` module enables verbs/RDMA over standard Ethernet. Verified on WSL2.

---

## Shard Placement API (Capacity-First Bin Packing)

For distributing model shards across a GPU cluster:

```cpp
// 1. Model manifest (from ModelHub)
ModelManifest model;
model.shards = {
    ShardSpec{"blk.0-3", "hash1", ShardKind::Weights, 2_GiB, 0, 3},
    ShardSpec{"blk.4-7", "hash2", ShardKind::Weights, 2_GiB, 4, 7},
    ShardSpec{"blk.8-11", "hash3", ShardKind::Weights, 2_GiB, 8, 11},
};

// 2. Cluster topology
ClusterCapacity cluster;
cluster.nodes = {
    NodeCapacity{"node-a", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
    NodeCapacity{"node-b", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
};
cluster.reservedActivationBytes = 512_MiB;  // KV cache slack

// 3. Policy
PlacementPolicy policy;
policy.allowHostOffload = true;
policy.preferContiguousLayers = true;
policy.packMode = PackMode::PackDense;

// 4. Plan
PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
// plan.assignments = [ {blk.0-3 → node-a, DeviceLocal}, ... ]

// Execute (per-node)
PlacementExecutor executor(nodeManager, modelHub);
ExecuteOptions opt{ .fetchIfMissing=true, .verifyChecksum=true };
ExecuteResult result = executor.executeLocal(model, plan, opt);
// Idempotent, transactional rollback on failure
```

---

## Sparse / Residency Virtual Memory

Reserve huge virtual address space, commit physical pages on demand:

```cpp
SparseVirtualMemoryPool pool(device, physicalDevice);
pool.initialize(1_TiB, 32_MiB);  // 1 TB virtual, 32 MB pages

auto reservation = pool.reserveVirtual(256_GiB, usage);  // No physical memory yet
pool.commit(reservation, 64_GiB, 64_GiB, memoryFlags);   // Commit 64 GB at offset 64 GB
pool.uncommit(reservation, 64_GiB, 64_GiB);              // Release when done
```

---

## PyTorch & ONNX Integration

### PyTorch (`vulkanvm_torch`)

```python
import vulkanvm_torch as vvm
pool = vvm.UnifiedMemoryPool.create(dev_config, pool_config)  # (Chonk Buffer)
alloc = pool.allocate_tensor(64 * 1024 * 1024, "weights")

# Offload/reload
pool.initialize_offload(4_GB, transfer_queue, transfer_queue_family)
op = pool.offload_to_host(alloc)
pool.wait_migration(op)
pool.reload_to_device(alloc)

# Cross-GPU
exported = pool.export_memory(alloc, vvm.ExternalHandleType.OpaqueWin32)
peer_alloc = peer_pool.import_memory(exported, usage)
```

### ONNX Runtime (`vulkanvm_onnx`)

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

### Windows (PowerShell)
```powershell
.\scripts\build.ps1 -Tests -BuildType Release
# Or manually:
cmd /c "call 'VC\Auxiliary\Build\vcvars64.bat' && cmake -G Ninja -B build_ninja -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_TESTS=ON && cmake --build build_ninja --config Release"
```

### Linux
```bash
./scripts/build.sh --tests
# Or manually:
cmake -B build -DCMAKE_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

### Requirements
- CMake 3.20+, C++20 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- Vulkan SDK 1.3+
- Optional: Volk, OpenSSL, gRPC/Protobuf, ibverbs/rdma_cm
- **Windows SDK 10.0.26100+** for NDKPI headers

---

## Hardware Compatibility

| GPU | Works? | Notes |
|-----|--------|-------|
| RTX 4090 / 7900 XTX (24 GB) | ✅ Excellent | 2 GB blocks (auto-detected ≥24 GB VRAM) |
| RTX 4070 / 7800 XT (12-16 GB) | ✅ Excellent | 256 MB blocks |
| Arc A770 / A750 (16/8 GB) | ✅ Good | 256 MB blocks |
| Laptop dGPU (8 GB) | ✅ Good | 128 MB blocks |
| **Strix Halo APU (96-128 GB unified)** | ✅ **BEST** | 2 GB blocks (auto-detected), this is the sweet spot |
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

# Network (2-node loopback)
./build/tests/network_test
./build/tests/tensor_network_test

# Tensor collectives
./build/tests/tensor_collective_test

# Windows ND fake provider
.\build_win\tests\ndk_transport_test.exe
```

---

## Quick Reference: Common Patterns

```cpp
// 1. Basic allocation
auto alloc = pool->allocateTensor(64_MiB);

// 2. Staging (upload)
auto staging = pool->allocate({ .size=128_MiB, .usage=VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                .memoryUsage=MemoryUsage::CpuToGpu, .mapped=true });

// 3. Readback
auto readback = pool->allocate({ .size=16_MiB, .memoryUsage=MemoryUsage::GpuToCpu, .mapped=true });

// 4. Exportable (for cross-GPU)
auto exportable = pool->allocateDedicatedExportable(256_MiB, usage);

// 5. Cross-GPU share
auto handle = pool->exportMemory(*exportable, ExternalHandleType::OpaqueWin32);
auto peerHandle = duplicateForImport(handle);  // per peer
auto peerAlloc = peerPool->importMemory(std::move(peerHandle), usage);

// 6. Offload/reload
auto op = pool->offloadToHost(alloc);
pool->waitMigration(op);
pool->reloadToDevice(alloc);

// 7. Tensor transport
auto transport = vvm::tensor::createTensorTransport(cfg, devices, poolConfig);
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
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

---

## Android / Vulkan (AHardwareBuffer External Memory)

On Android, VulkanVM supports **zero-copy external memory** via `VK_ANDROID_external_memory_android_hardware_buffer`. This enables zero-copy sharing between Vulkan and Android's graphics pipeline (Surface, MediaCodec, Camera, etc.).

### Requirements

- Android NDK r27+ (for `VK_ANDROID_external_memory_android_hardware_buffer`)
- Android 10+ (API 29+) for `AHardwareBuffer` support
- Vulkan 1.1+ with `VK_ANDROID_external_memory_android_hardware_buffer` extension

### Building for Android

```bash
# Linux
./scripts/build_android.sh arm64-v8a android-34 Release

# Windows (PowerShell)
.\scripts\build_android.bat arm64-v8a android-34 Release
```

### Using AHardwareBuffer External Memory

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
#include <android/hardware_buffer.h>

// 1. Create AHardwareBuffer (e.g., from Surface, MediaCodec, Camera, or manually)
AHardwareBuffer_Desc desc{};
desc.width = 1920;
desc.height = 1080;
desc.layers = 1;
desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | 
             AHARDWAREBUFFER_USAGE_CPU_READ_NEVER |
             AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
desc.stride = 0;

AHardwareBuffer* hardwareBuffer;
AHardwareBuffer_allocate(&desc, &hardwareBuffer);

// 2. Import into VulkanVM pool as external memory
ExternalMemoryInfo extInfo;
extInfo.type = ExternalHandleType::AndroidHardwareBuffer;
extInfo.handle = ExternalHandle(hardwareBuffer);  // RAII wrapper
extInfo.size = bufferSize;
extInfo.dedicatedAllocation = true;

auto allocation = pool.importMemory(std::move(extInfo), 
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

// 3. Use in shaders via device address
auto deviceAddress = allocation->deviceAddress;

// 4. Cleanup: AHardwareBuffer is reference-counted, released when ExternalHandle destructs
```

### Exporting Vulkan Memory as AHardwareBuffer

```cpp
// Allocate dedicated exportable memory
auto alloc = pool->allocateDedicatedExportable(size, 
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

// Export as AHardwareBuffer
ExternalMemoryInfo extInfo = pool->exportMemory(*alloc, 
    ExternalHandleType::AndroidHardwareBuffer);

// Pass to Android framework (Surface, MediaCodec, etc.)
AHardwareBuffer* buffer = extInfo.handle.get();
// Pass to ANativeWindow, MediaCodec, etc.
```

### ExternalHandleType::AndroidHardwareBuffer

Added to `ExternalHandleType` enum:
```cpp
enum class ExternalHandleType {
    OpaqueFd,
    OpaqueWin32,
    D3D12Heap,
    DmaBuf,
    AndroidHardwareBuffer  // VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID
};
```

The `ExternalHandle` class now supports `AHardwareBuffer*` with automatic reference counting via `AHardwareBuffer_acquire`/`release`.

### Build Configuration

Add to `CMakeLists.txt`:
```cmake
# Android-specific options
option(VVM_ANDROID_HARDWARE_BUFFER "Enable Android AHardwareBuffer support" ON)
option(VVM_ANDROID_EXTERNAL_MEMORY "Enable Android external memory" ON)
```

Or via command line:
```bash
cmake -DCMAKE_TOOLCHAIN_FILE=cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-34 \
      -DVVM_ANDROID_HARDWARE_BUFFER=ON \
      -DVVM_ANDROID_EXTERNAL_MEMORY=ON \
      ..
```

### Cross-Vendor Compatibility

| Source → Target | Android Handle Type |
|-----------------|---------------------|
| Android → Android | `AndroidHardwareBuffer` (direct) |
| Android → Linux (DMA-BUF) | Export as `DmaBuf` (via `AHardwareBuffer_toGralloc`) |
| Android → Windows | Not directly supported; use host-staged fallback |

> **Note**: Direct Android ↔ Desktop GPU sharing requires vendor-specific extensions. For cross-platform sharing, use host-staged fallback or vendor-specific paths.

---

## Need Help?

- **README.md** — Full API reference, cross-vendor matrix, build options
- **examples/** — `network_test`, `model_registry_test`, `tensor_compute`, `offload_test`, `multi_gpu_test`, `tensor_network_test`
- **tests/** — Unit tests for buddy allocator, external handles, sparse, placement
- **GitHub Issues** — Bug reports, feature requests

*VulkanVM — Making GPUs play nice since 2026.*
