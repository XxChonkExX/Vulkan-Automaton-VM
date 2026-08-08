# VulkanVM — Unified Vulkan Memory Pool & Cross-GPU Model Distribution

**VulkanVM** is a cross-vendor, high-performance Vulkan memory management library that solves GPU memory fragmentation, enables zero-copy memory sharing across AMD, NVIDIA, and Intel GPUs, and provides a Hugging Face–style model registry for distributing model weights over TCP. Download once, load into the pool, run local multi-GPU inference. This is intended to be a toolbox for the community. Please feel free to contribute, branch, borrow and bug report. Especially bug report!

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Memory Pool](#core-memory-pool)
3. [Cross-GPU Memory Sharing](#cross-gpu-memory-sharing)
4. [Multi-GPU Pool Manager](#multi-gpu-pool-manager)
5. [Host Offload / Demand Paging](#host-offload--demand-paging)
6. [ModelHub — Model Weight Distribution](#modelhub--model-weight-distribution)
7. [Multi-Node Network Module](#multi-node-network-module)
8. [Unified Tensor Transport (`vvm::tensor::Transport`)](#unified-tensor-transport-vvmtensortransport)
9. [Shard Placement API](#shard-placement-api)
10. [GPU-Direct RDMA (Vendor Paths)](#gpu-direct-rdma-vendor-paths)
11. [Sparse / Residency Virtual Memory](#sparse--residency-virtual-memory)
12. [Building](#building)
13. [Cross-Vendor Compatibility](#cross-vendor-compatibility)
14. [APU-Specific Tuning](#apu-specific-tuning)
15. [License & Credits](#license--credits)

---

## Quick Start

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
using namespace vvm;

// 1. Select a GPU
VkInstance instance = createInstance();
auto devices = enumerateDevices(instance);
auto best = selectBestDevice(devices, /*preferDiscrete=*/true, /*minVRAM_MiB=*/1024);
VkPhysicalDevice physicalDevice = best->device;

// 2. Find queues
auto queues = findQueueFamilies(physicalDevice);
DeviceConfig devConfig{
    .physicalDevice = physicalDevice,
    .device = createDevice(physicalDevice, queues),
    .graphicsQueueFamily = queues.graphics.value_or(0),
    .computeQueueFamily  = queues.compute.value_or(0),
    .transferQueueFamily = queues.transfer.value_or(0),
};
// ... vkGetDeviceQueue for each queue ...

// 3. Create the pool (auto-tuned for your hardware)
PoolConfig poolConfig = PoolConfig::forDevice(physicalDevice);
poolConfig.maxHeapFraction = 0.75f;  // never exceed 75% of device heap budget

auto pool = UnifiedMemoryPool::create(devConfig, poolConfig);
if (!pool) { /* handle error */ }

// 4. Allocate a tensor buffer (bindless-ready, 64 MB)
auto tensor = pool->allocateTensor(64 * 1024 * 1024);
// tensor->deviceAddress  → use in shaders
// tensor->buffer         → vkCmdCopyBuffer, vkCmdBindBufferRange, etc.

// 5. When done, return to pool (NOT to OS — zero fragmentation)
pool->deallocate(std::move(*tensor));
```

---

## Core Memory Pool

### Key Properties

| Property | Description |
|----------|-------------|
| **Persistent blocks** | Pre-allocates large blocks (default 512 MB) at startup; sub-allocates with a buddy allocator |
| **Zero fragmentation** | Memory is **never returned to the OS** — blocks stay resident; buddy allocator guarantees power-of-2 alignment and coalescing on free |
| **Hardened buddy** | Free-list-per-order classic buddy (iterative split/coalesce, no per-node heap churn); configurable `minSize` (default `PoolConfig::minAlignment`), power-of-2 `blockSize`/`minSize` enforced |
| **Budget-aware** | `maxHeapFraction` (e.g., 0.75) + `VK_EXT_memory_budget` prevent starving the system |
| **Topology-aware** | `detectMemoryTopology()` classifies Discrete / Unified / Hybrid; `PoolConfig::forDevice()` picks tuned defaults |
| **Intent API** | `MemoryUsage { GpuOnly, CpuToGpu, GpuToCpu, CpuCopy, Auto }` hides raw `VkMemoryPropertyFlags` |
| **Rich descriptors** | `AllocDesc { size, usage, memoryUsage, exportable, mapped, name }` |
| **Bindless ready** | Every allocation returns `VkDeviceAddress` for descriptor-free shader access |
| **Debug names** | `VK_EXT_debug_utils` labels on every buffer/memory (`name` in `AllocDesc`) |
| **Thread-safe** | All public methods mutex-guarded internally; safe for concurrent access |

### Allocation Examples

```cpp
// Simple tensor (auto device-local)
auto w = pool->allocateTensor(64_MiB);

// Staging buffer (upload): host-visible + persistently mapped
auto staging = pool->allocate({
    .size = 128_MiB,
    .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    .memoryUsage = MemoryUsage::CpuToGpu,
    .mapped = true,
    .name = "upload_buffer"
});

// Readback buffer (GPU→CPU)
auto readback = pool->allocate({
    .size = 16_MiB,
    .memoryUsage = MemoryUsage::GpuToCpu,
    .mapped = true,
    .name = "readback"
});

// Exportable (dedicated allocation) for cross-GPU sharing
auto exportable = pool->allocateDedicatedExportable(256_MiB,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

// One-shot copy on the same device (uses internal transient command pool)
pool->copyBuffer(src, dst, srcOffset, dstOffset, size);

// Stats
PoolStats stats = pool->getStats();
// stats.totalAllocated, stats.totalUsed, stats.largestFreeBlock, stats.fragmentationRatio, ...
```

### P0 Hardening (Recent)

| Issue | Fix |
|-------|-----|
| **UniqueAllocation RAII leak** | Raw ctor now private; force `UniqueAllocation::make()` factory; deleter always set |
| **OffloadManager thread-safety** | Added `mutex_` + `statsMutex_`; all public methods mutex-guarded; unique lock names for nesting |
| **Logging format** | All `printf` converted to `fmt`-style `{}` across `unified_memory_pool.cpp`, `host_shadow.cpp`, `rdma_transport.cpp`, `tcp_transport.cpp`, `vulkan_utils.cpp`, `network/multi_node_manager.cpp` |

---

## Cross-GPU Memory Sharing

VulkanVM uses `VK_EXTERNAL_MEMORY` to share `VkDeviceMemory` across physical devices. **Requirement:** The source must be a **dedicated exportable allocation** (one `VkDeviceMemory` per shareable buffer). Sub-allocated blocks are auto-promoted to dedicated copies on export.

### Handle Types by Platform

| Platform | NVIDIA | AMD / Intel |
|----------|--------|-------------|
| **Linux** | `DMA_BUF` | `OPAQUE_FD` / `DMA_BUF` |
| **Windows** | `D3D12_HEAP` | `OPAQUE_WIN32` |

### Basic Export / Import

```cpp
// On master GPU (e.g., AMD 7900 XTX)
auto masterAlloc = masterPool.allocateDedicatedExportable(256_MiB, usage);
auto exportInfo  = masterPool.exportMemory(*masterAlloc,
    #ifdef VVM_PLATFORM_LINUX
        ExternalHandleType::DmaBuf
    #else
        ExternalHandleType::OpaqueWin32
    #endif
);

// On each peer GPU (e.g., Intel Arc Pro B70, NVIDIA RTX)
// Each peer needs its OWN handle; duplicateForImport() dups the FD/HANDLE
for (auto& peerPool : peerPools) {
    auto perPeer = duplicateForImport(*exportInfo);
    auto peerAlloc = peerPool.importMemory(std::move(perPeer), usage);
    // peerAlloc now aliases the SAME physical memory on this GPU
}

// Timeline semaphore sync across GPUs
manager.submitMigrationBarrier(operations);
manager.waitAllIdle();
```

### Handle Ownership Rules

| Operation | Handle Ownership |
|-----------|------------------|
| `exportMemory()` | Returns `ExternalHandle` RAII wrapper — **closes on scope exit** |
| `importMemory()` | **Consumes** the handle on success (driver takes ownership). On failure, wrapper still closes it. |
| `duplicateForImport()` | Creates a **new independent FD/HANDLE** for each peer (dup on Linux, `DuplicateHandle` on Windows) |

---

## Multi-GPU Pool Manager

`MultiGPUPoolManager` wraps multiple `UnifiedMemoryPool`s and automates cross-GPU allocation + sync.

```cpp
std::vector<DeviceConfig> devices = { amdConfig, intelConfig, nvidiaConfig };
PoolConfig config = PoolConfig::forDevice(devices[0].physicalDevice);

// GPU 0 = master (allocates + exports)
auto manager = MultiGPUPoolManager::create(devices, config, 0);

// One call: allocates on master, imports on all peers
auto allocs = manager.allocateDistributed(512_MiB, usage);
// allocs[i] is a valid allocation on devices[i], all aliasing the same memory

// Peer capability query
auto peer = manager->queryPeerAccess(0, 1);  // AMD → Intel
// peer.canDirectCopy == true if export/import + device copy works

// Direct GPU→GPU copy WITHOUT host staging (when driver supports it)
manager->copyDeviceToDevice(0, 1, *srcAlloc, *dstAlloc, 0, 0, size);
// If the driver refuses the cross-GPU import, transparently falls back
// to a 4 MiB chunked host-staged copy (Spark-style)

// Timeline semaphore barrier across all GPUs
manager->submitMigrationBarrier(ops);
manager->waitAllIdle();
```

### Peer Access Matrix

| Source → Target | Linux Handle | Windows Handle |
|-----------------|--------------|----------------|
| NVIDIA → AMD    | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| NVIDIA → Intel  | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| AMD → NVIDIA    | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| Intel → NVIDIA  | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| AMD ↔ Intel     | DMA-BUF / OPAQUE_FD | OPAQUE_WIN32 |
| Same vendor     | OPAQUE_FD    | OPAQUE_WIN32 / D3D12_HEAP |

---

## Host Offload / Demand Paging

When VRAM fills, spill allocations to a host-backed shadow buffer asynchronously.

```cpp
OffloadConfig offloadConfig;
offloadConfig.hostShadowSize = 4_GiB;      // host RAM buffer
offloadConfig.transferQueue = devConfig.transferQueue;
offloadConfig.transferQueueFamily = devConfig.transferQueueFamily;
offloadConfig.persistentMapping = true;
offloadConfig.useCoherentMapping = true;

// Attach to pool (called automatically if poolConfig.enableHostVisible)
pool->initializeOffload(offloadConfig);

// Async offload (device → host)
auto op = pool->offloadToHost(allocation);
// ... do other compute while DMA runs ...
pool->waitMigration(op);

// Sync reload (host → device)
pool->reloadToDevice(allocation);
pool->waitMigration(op);
```

**Note:** `madvise`/`mprotect` on `vkMapMemory` memory are **deprecated** (unsafe on driver mappings). Standard async offload/reload uses the GPU copy engine and is safe.

---

## ModelHub — Model Weight Distribution

Hugging Face–style publish/fetch over TCP with content-addressed chunks (SHA-256, 4 MiB slices), local cache, and resume.

### Server (Hub)

```cpp
using namespace vvm::network;
ModelHub hub("/data/model-store");
hub.start("0.0.0.0", 51010);
hub.publish("chonk/llama-3b-q4", "./local-model-files", "v1");
// Publishes all files under ./local-model-files as version "v1"
hub.stop();
```

### Client (Any Machine)

```cpp
// Fetch to local cache (~/.cache/vvm/models/...)
ModelHub::fetch("192.168.1.50:51010", "chonk/llama-3b-q4", "./my-models", "v1");

// Then load weights into the pool
auto alloc = pool->allocate({ .size = modelSize, .memoryUsage = MemoryUsage::GpuOnly });
// Copy weights from ./my-models/weights.safetensors into alloc->buffer
```

### Cache Layout (HF-style)

```
~/.cache/vvm/models/chonk/llama-3b-q4/v1/
  config.json
  weights.safetensors
  tokenizer/tokenizer.json
  .vvm_complete          # marker = cache entry valid
```

### Run the Demo

```bash
# Windows
build_ninja\examples\model_registry_test.exe

# Linux
./build/examples/model_registry_test
```

---

## Multi-Node Network Module

`vvm::network::MultiNodePoolManager` provides a **zero-dependency** multi-node cluster over plain TCP (Winsock / BSD sockets). Uses a Spark-inspired wire protocol: versioned 32-byte header + bulk stream transferred in 4 MB slices directly between socket and caller buffer (no intermediate copy).

**Protocol hardening:** every inbound message is length-checked against hard caps (`kMaxBodySize` = 1 GiB, `kMaxStreamSize` = 16 GiB) on both the server and client receive paths. Headers that exceed a cap are rejected and the connection dropped, preventing malformed or hostile peers from triggering unbounded allocations. TLS 1.2+ (OpenSSL) is supported for non-localhost clusters, and `NetworkConfig::validate()` rejects invalid MTU/listen settings.

### Control-Plane Messages

`MsgRegisterNode`, `MsgGetClusterView`, `MsgAllocate`, `MsgExport`, `MsgImport`, `MsgMigratePull`, `MsgMigratePush`, `MsgHeartbeat`, `MsgLeaveCluster`, `MsgDeallocate`, `MsgModelList`, `MsgModelManifest`, `MsgModelChunk`.

### TLS Support

Optional OpenSSL-backed TLS 1.2+ (gated by `VVM_NETWORK_HAS_TLS`). Server SNI + client ALPN (`vvm/1.0`).

```cpp
using namespace vvm::network;

// Node A (bootstrap seed)
NetworkConfig netA; netA.listenAddress = "0.0.0.0:51001";
auto nodeA = MultiNodePoolManager::create({devCfg}, poolCfg, netA);
nodeA->start();
nodeA->registerWithCluster();  // no seeds = cluster root

// Node B (joins A's cluster)
NetworkConfig netB; netB.listenAddress = "0.0.0.0:51002";
netB.seedNodes = { "127.0.0.1:51001" };
auto nodeB = MultiNodePoolManager::create({devCfg}, poolCfg, netB);
nodeB->start();
nodeB->registerWithCluster();  // connects to A, merges cluster view

// Remote allocation + push migration (B → A)
auto src = nodeB->getLocalPool().allocate(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
std::memset(src->hostPtr, 0xAB, size);

auto dst = nodeB->allocateRemote(nodeA->getLocalNodeId(), size, usage,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeB->migrateToRemote(*src, *dst, /*useRdma=*/false);  // 4 MB slices over TCP

// Pull migration (A pulls from B)
auto remoteDesc = nodeB->exportForRemote(*src, false, true);
auto localDst   = nodeA->getLocalPool().allocate(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeA->migrateFromRemote(*remoteDesc, *localDst, /*useRdma=*/false);
```

### Run the Demo

```bash
# Windows
build_ninja\examples\network_test.exe
build_ninja\examples\tensor_network_test.exe   # GPU-to-GPU VRAM over TCP

# Linux
./build/examples/network_test
./build/examples/tensor_network_test
```

---

## Unified Tensor Transport (`vvm::tensor::Transport`)

The tensor transport layer provides a **unified abstraction** for tensor movement across devices, machines, and networks. It wraps the underlying memory pools, P2P copies, RDMA, and network transports into a single tensor-aware API.

### Key Features

| Feature | Description |
|---------|-------------|
| **Tensor Metadata** | Shape, dtype (FP32/FP16/BF16/INT8/INT4/FP8), strides, layout (NHWC/NCHW/Blocked) |
| **Auto Transport Selection** | P2P → RDMA → Host-Staged → Network |
| **Layout Conversion** | NHWC↔NCHW, blocked/tiling for tensor cores |
| **Collectives** | Ring all-reduce, broadcast, all-gather, reduce-scatter |
| **Async Pipeline** | Worker thread, callback-based `AsyncOperation` queue, overlap compute + transfer |

### Quick Start

```cpp
#include <vulkan_vm/tensor_transport.hpp>
using namespace vvm::tensor;

// Configure transport
TransportConfig config;
config.preference = TransportConfig::Preference::Auto;  // P2P → RDMA → HostStaged → Network
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;

// Create transport with your devices
std::vector<vvm::DeviceConfig> devices = { amdConfig, intelConfig, nvidiaConfig };
vvm::PoolConfig poolConfig = vvm::PoolConfig::forDevice(devices[0].physicalDevice);

auto transport = vvm::tensor::createTensorTransport(config, devices, poolConfig);
if (!transport->initialize()) { /* handle error */ }

// Allocate a tensor on GPU 0
TensorMetadata meta;
meta.dtype = DataType::Float16;
meta.layout = MemoryLayout::ChannelsLast;  // NHWC for tensor cores
meta.shape = TensorShape::makeChannelsLast({1, 32, 32, 128});  // [N, H, W, C]
meta.name = "conv_weight";

auto tensor = transport->allocateTensor(meta, 0);  // On GPU 0

// Distribute across GPUs (master allocates, peers import)
auto tensors = transport->allocateDistributed(meta, {0, 1, 2});

// Copy with automatic layout conversion
transport->copyWithLayoutConversion(src, dst, MemoryLayout::Blocked);

// Ring all-reduce across GPUs
transport->allReduce({tensor0, tensor1, tensor2}, ReduceOp::Sum, {0, 1, 2});

// Async copy with callback
transport->copyTensorAsync(src, dst, [](bool ok, const std::string& err) {
    if (ok) std::cout << "Copy done!" << std::endl;
});

// Async collectives with completion callback
transport->allReduceAsync({tensor0, tensor1, tensor2}, ReduceOp::Sum, {0, 1, 2},
    [](bool ok, const std::string& err) {
        if (ok) std::cout << "all-reduce done!" << std::endl;
    });

// Wait for all queued async operations to finish
transport->flushAsync();

// Cleanup
transport->shutdown();
```

### Tensor Metadata

```cpp
struct TensorMetadata {
    DataType dtype = DataType::Float32;      // FP32, FP16, BF16, INT8, INT4, FP8_E4M3, FP8_E5M2
    MemoryLayout layout = MemoryLayout::Contiguous;  // Contiguous, ChannelsLast, Blocked, Strided
    TensorShape shape;                       // dims + optional strides
    std::string name;                        // Debug name
};

// Shape with optional custom strides
TensorShape shape;
shape.dims = {1, 32, 32, 128};  // [N, H, W, C]
shape.strides = {32*32*128, 32*128, 128, 1};  // NHWC

// Layout presets
TensorShape::makeContiguous({1, 128, 32, 32});    // NCHW
TensorShape::makeChannelsLast({1, 32, 32, 128});  // NHWC
TensorShape::makeBlocked({1, 128, 32, 32}, 32);   // 32x32 tiles
```

### Transport Configuration

```cpp
TransportConfig config;
config.preference = TransportConfig::Preference::Auto;  // P2P → RDMA → HostStaged → Network
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;
config.hostStagedChunkSize = 4_MiB;

// RDMA (GPU-Direct)
config.preference = TransportConfig::Preference::RDMAOnly;
config.rdmaNicName = "mlx5_0";
config.enableGPUDirect = true;

// Network (multi-node)
config.preference = TransportConfig::Preference::NetworkOnly;
config.networkPort = 51000;
config.enableTLS = true;
config.tlsCertPath = "cert.pem";
config.tlsKeyPath = "key.pem";
config.tlsCaPath = "ca.pem";
```

### Transport Paths (Auto-Selected)

| Priority | Path | Mechanism | Staging? |
|----------|------|-----------|----------|
| 1 | **P2P** | `VK_EXTERNAL_MEMORY` + `vkCmdCopyBuffer` | ❌ No |
| 2 | **RDMA** | GPU-Direct: `ibv_reg_dmabuf_mr` / NDKPI | ❌ No |
| 3 | **Host-Staged** | 4 MiB chunks via host memory | ✅ Yes |
| 4 | **Network** | TCP/RDMA multi-node | ✅ Yes |

### Collective Operations

```cpp
// Ring all-reduce (Sum, Mean, Min, Max, Product, Band, Bor, Bxor)
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});

// Broadcast from root
transport->broadcast(rootTensor, {0, 1, 2}, 0);

// All-gather (concatenate)
transport->allGather({t0, t1, t2}, output, {0, 1, 2});

// Reduce-scatter
transport->reduceScatter({t0, t1, t2}, output, ReduceOp::Sum, {0, 1, 2});

// Every collective has an async variant (enqueued on the pipeline)
transport->allReduceAsync({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2}, cb);
transport->allGatherAsync({t0, t1}, out, {0, 1}, cb);
transport->reduceScatterAsync({t0, t1, t2}, out, ReduceOp::Sum, {0, 1, 2}, cb);
transport->broadcastAsync(root, {0, 1, 2}, 0, cb);
```

### Async Pipeline

Tensor transport runs operations on a dedicated worker thread. Async calls (`copyTensorAsync`, `allReduceAsync`, `broadcastAsync`, `allGatherAsync`, `reduceScatterAsync`) enqueue an `AsyncOperation` and return immediately; the completion callback fires when the work finishes. Low-level `enqueueAsync`/`flushAsync` give direct access:

```cpp
// Enqueue an arbitrary operation on the pipeline
transport->enqueueAsync([]() {
    // ... any expensive work ...
});

// Block until all queued operations complete
transport->flushAsync();
```

When `config.enableAsyncPipeline` is `false` (or no worker thread can run), operations run **inline** synchronously so callbacks still fire correctly. Callbacks receive `(bool ok, const std::string& error)` exactly like the sync methods' result.

### Layout Conversion (Compute Shaders)

Tensor transport supports automatic layout conversion via Vulkan compute shaders:

```cpp
// Convert NHWC to NCHW (or vice versa) on-the-fly during copy
transport->copyWithLayoutConversion(srcTensor, dstTensor, MemoryLayout::Contiguous);
// or
transport->copyWithLayoutConversion(srcTensor, dstTensor, MemoryLayout::ChannelsLast);

// Supported layouts:
// - MemoryLayout::Contiguous (NCHW)
// - MemoryLayout::ChannelsLast (NHWC)
// - MemoryLayout::Blocked (tiled for tensor cores)
// - MemoryLayout::Strided (custom strides)
```

Layout conversion happens on the GPU via Vulkan compute shaders, avoiding host round-trips. The transport automatically selects the optimal shader based on source and target layouts.

### Multi-Node Network

```cpp
// Node A (bootstrap)
TransportConfig netA; netA.listenAddress = "0.0.0.0"; netA.networkPort = 51001;
auto nodeA = createTensorTransport(config, devices, poolConfig);
nodeA->initialize();
nodeA->joinCluster("0.0.0.0:51001");

// Node B (joins)
TransportConfig netB; netB.listenAddress = "0.0.0.0"; netB.networkPort = 51002;
netB.seedNodes = {"127.0.0.1:51001"};
auto nodeB = createTensorTransport(config, devices, poolConfig);
nodeB->initialize();
nodeB->joinCluster("127.0.0.1:51001");

// Remote node IDs use "host:port#nodeIndex" format
std::string nodeAId = "192.168.1.10:51001#0";
std::string nodeBId = "192.168.1.11:51002#0";

// Send tensor: B exports its VRAM and advertises it to A over TCP
nodeB->sendTensor(tensor, nodeAId, [](bool ok, const std::string& err) { /* ... */ });

// Receive tensor: A waits for the announcement, then pulls B's VRAM over TCP
nodeA->recvTensor(receiver, nodeBId, [](bool ok, const std::string& err) { /* ... */ });
// Before recvTensor, `received` must be allocated with the SAME name as `tensor`
```

Cross-machine copies are pulled end-to-end: `sendTensor` exports the local `VkDeviceMemory` and advertises the descriptor under the tensor's name (`MsgTensorAnnounce`); `recvTensor` waits for the peer and streams the VRAM over TCP in 4 MiB host-staged chunks (`migrateFromRemote`), falling back to GPU-direct RDMA automatically when verbs + GPUDirect are available.

### Current Status

| Feature | Status |
|---------|--------|
| P2P (local multi-GPU) | ✅ Working |
| Host-staged fallback | ✅ Working |
| Ring all-reduce | ✅ Implemented (CPU fallback for all dtypes) |
| Async pipeline | ✅ Implemented (worker thread + async collectives) |
| GPU-Direct RDMA (Linux) | ✅ NVIDIA (peermem), AMD/Intel (DMA-BUF) |
| GPU-Direct RDMA (Windows) | ✅ **NDKPI transport (`NdkRdmaTransport`)** + fake provider test harness |
| Network tensor send/recv (TCP GPU⇄GPU) | ✅ Working (`tensor_network_test`) |
| Layout conversion (NHWC↔NCHW) | ✅ Compute shaders |
| allGather collective | ✅ Implemented |
| reduceScatter collective | ✅ Implemented |
| Tensor slice copy (partial) | ✅ Implemented |
| **Collective dtypes** | **FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, Int4 (packed), Int8, UInt8, Int32, Int64, Bool** |
| **Collective ops** | **Sum, Mean, Min, Max, Product, Band, Bor, Bxor** |

---

## Shard Placement API

New in v0.2: **Capacity-first bin-packing** for distributing model shards across a cluster of GPU nodes. The API computes an optimal placement plan respecting per-node VRAM, host-offload capacity, activation reserves, and user constraints.

### Types & Structures

```cpp
#include <vulkan_vm/placement.hpp>
using namespace vvm::placement;

// Per-node capacity (honest accounting)
struct NodeCapacity {
    std::string nodeId;
    VkDeviceSize vramFreeBytes;      // DEVICE_LOCAL available
    VkDeviceSize hostOffloadBytes;   // host shadow available for offload
    VkDeviceSize diskCacheBytes;     // disk cache available (cold storage)
    int32_t gpuCount = 1;            // GPUs on this node
    int32_t nicCount = 1;            // NICs for RDMA/TCP
    double bandwidthGbps = 10.0;     // interconnect bandwidth estimate
    bool hasRdma = false;            // hardware RDMA NIC present
};

// Cluster view
struct ClusterCapacity {
    std::vector<NodeCapacity> nodes;
    VkDeviceSize reservedActivationBytes = 64_MiB; // per-node KV/act slack
};

// Model shard specification
struct ShardSpec {
    std::string shardId;              // e.g. "layers.0-3.weight"
    std::string contentHash;          // SHA-256 of chunk set (ModelHub)
    ShardKind kind = ShardKind::Weights;
    VkDeviceSize bytes = 0;           // uncompressed size when "hot"
    int32_t layerBegin = -1;          // optional topology for pipeline parallel
    int32_t layerEnd = -1;            // inclusive
    bool mustBeDeviceLocal = false;   // rare; default false → may offload
    bool mustStayTogether = false;    // if true, all bytes on one node
};

// Shard placement result
struct ShardPlacement {
    std::string shardId;
    std::string nodeId;
    MemTier tier = MemTier::DeviceLocal;
    uint32_t localDeviceIndex = 0;
};

enum class MemTier { DeviceLocal, HostOffload, DiskCache };

// Placement policy
struct PlacementPolicy {
    bool allowHostOffload = true;     // allow spill to HostOffload tier
    bool allowDiskCache = false;      // allow DiskCache tier (cold)
    bool preferContiguousLayers = false; // try to keep adjacent layers together
    enum class PackMode { PackDense, SpreadEven } packMode = PackMode::PackDense;
    bool failFast = true;             // stop on first error
    bool transactionalNode = true;    // rollback all on node failure
    bool revalidateCapacity = true;   // re-check capacity before execute
    bool bestEffort = false;          // place what fits, return partial
};

// Full plan
struct PlacementPlan {
    std::vector<ShardPlacement> assignments;
    Status status;                    // ok or error with detail
};

// Error handling
enum class ErrorCode { Ok, InvalidManifest, InvalidCluster, InvalidPolicy,
    EmptyShardList, ZeroCapacityCluster, InsufficientCapacity,
    UnsatisfiableConstraint, ShardTooLarge, ActivationReserveFailed, ... };

struct ErrorDetail { ErrorCode code; std::string message; std::string shardId; ... };

struct Status {
    ErrorCode code = ErrorCode::Ok;
    std::vector<ErrorDetail> details;
    explicit operator bool() const { return code == ErrorCode::Ok; }
    void add(ErrorDetail d) { if (!d.message.empty()) { details.push_back(d); } }
};
```

### Planning a Model

```cpp
// 1. Describe the model (from ModelHub manifest)
ModelManifest model;
model.modelId = "chonk/llama-3b-q4";
model.version = "v1";
model.shards = {
    ShardSpec{"blk.0-3", "hash1", ShardKind::Weights, 2_GiB, 0, 3},
    ShardSpec{"blk.4-7", "hash2", ShardKind::Weights, 2_GiB, 4, 7},
    ShardSpec{"blk.8-11", "hash3", ShardKind::Weights, 2_GiB, 8, 11},
};

// 2. Describe the cluster
ClusterCapacity cluster;
cluster.nodes = {
    NodeCapacity{"node-a", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
    NodeCapacity{"node-b", 8_GiB, 8_GiB, 4_GiB, 1, 1, 1000, true},
};
cluster.reservedActivationBytes = 512_MiB;

// 3. Policy
PlacementPolicy policy;
policy.allowHostOffload = true;
policy.preferContiguousLayers = true;
policy.packMode = PlacementPolicy::PackMode::PackDense;

// 4. Plan
PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);

if (plan.status) {
    for (const auto& a : plan.assignments) {
        std::cout << a.shardId << " → " << a.nodeId << " [" << a.tier << "]\n";
    }
}
```

### Executing the Plan

```cpp
// Per-node executor (runs on each node)
PlacementExecutor executor(nodeManager, modelHub);

ExecuteOptions opt;
opt.fetchIfMissing = true;    // pull from ModelHub if not cached
opt.verifyChecksum = true;    // SHA-256 verify after load
opt.activationReserve = true; // account for KV cache per node

ExecuteResult result = executor.executeLocal(model, plan, opt);

if (result.status) {
    std::cout << "Completed: " << result.completedShardIds.size()
              << ", Failed: " << result.failedShardIds.size() << "\n";
}

// Cluster-wide coordinator (v0 = local only)
ExecuteResult clusterResult = PlacementExecutor::executeCluster(
    localManager, modelHub, model, plan, opt);
```

### Run Unit Tests

```bash
# 10 pure-logic tests (no Vulkan required)
./build/tests/placement_test
```

**Test coverage:** simple fit, host offload, mustBeDeviceLocal constraint, empty cluster, contiguous layers, ShardTooLarge, duplicate shardId, empty shardId, activation reserve, bestEffort mode.

---

## GPU-Direct RDMA (Vendor Paths)

For NIC-attached GPU memory DMA (bypassing host CPU), VulkanVM provides three vendor-specific registration paths. The `RdmaTransport` (verbs backend) dispatches based on `vendorID`.

### NVIDIA (`0x10DE`) — `VK_NV_external_memory_rdma`

```cpp
// Requires VK_NV_external_memory_rdma + nvidia-peermem kernel module
PFN_vkGetMemoryRemoteAddressNV vkGetMemoryRemoteAddressNV = ...;
VkMemoryGetRemoteAddressInfoNV info{};
info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_REMOTE_ADDRESS_INFO_NV;
info.memory = memory;
info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV;
VkRemoteAddressNV remoteAddr;
vkGetMemoryRemoteAddressNV(device, &info, &remoteAddr);
// → ibv_reg_mr on the exposed PCI BAR (Linux) or NDKPI (Windows)
```

### AMD (`0x1002`) — `OPAQUE_WIN32` / `DMA_BUF`

```cpp
// Export Vulkan memory as Win32 handle
VkMemoryGetWin32HandleInfoKHR getInfo{};
getInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
HANDLE win32Handle;
vkGetMemoryWin32HandleKHR(device, &getInfo, &win32Handle);

// Map to CPU VA (Windows: MapViewOfFile, Linux: mmap on DMA-BUF fd)
void* cpuVa = MapViewOfFile(win32Handle, FILE_MAP_READ|FILE_MAP_WRITE, 0, 0, size);
// → ibv_reg_mr(cpuVa, size, IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ)
```

### Intel (`0x8086`) — Same as AMD

```cpp
// Identical flow: OPAQUE_WIN32 handle → MapViewOfFile → ibv_reg_mr
```

### API

```cpp
// In rdma_transport.cpp (VerbsRdmaTransport)
std::optional<RdmaMemoryRegion> registerGpuMemory(
    VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkBuffer buffer) override;

// Dispatches to:
registerGpuMemoryForRdmaVendor(device, physDev, memory, offset, size, ... , vendorId);
```

### Linux SoftRoCE (Software RoCE / `rxe`) — Fallback Path

For Linux environments without physical RNIC hardware (WSL2, VMs, CI, dev machines), VulkanVM supports **SoftRoCE** via the kernel's `rxe` module. This enables the verbs/RDMA transport over standard Ethernet (`eth0`) or loopback (`lo`).

**Requirements:**
- Kernel with `CONFIG_RDMA_RXE=y` (built-in, not module)
- `rdma-core` userspace (`ibverbs`, `rdma_cm`)

**Setup (WSL2 / Ubuntu 24.04 example):**

```bash
# 1. Build kernel with RDMA_RXE (or use distro kernel that includes it)
# 2. Create SoftRoCE links on available netdevs:
sudo rdma link add rxe0 type rxe netdev eth0   # primary interface
sudo rdma link add rxe1 type rxe netdev lo     # loopback for localhost tests
rdma link show  # verify ACTIVE state
ibv_devices     # should list rxe0, rxe1
```

**Verification:**

The `tensor_network_test` example validates the full path:

```
=== VulkanVM Tensor Network Test ===
...
VerbsRdmaTransport initialized on device 'rxe0', RDMA listener port 51012
...
exportForRemote: RDMA host shadow registered for alloc 1
migrateFromRemote: pulled 16777216 bytes from 127.0.0.1:51012#0
  sendTensor: PASS
  recvTensor: PASS
  VRAM content verify on A: PASS
```

**Status:** ✅ End-to-end verified on WSL2 (custom kernel 6.18.40, `rxe0` + `rxe1`).
On native Linux with physical RNIC, the same verbs path uses hardware RDMA.
Without SoftRoCE or hardware RNIC, the network module falls back to **host-staged TCP** (no RDMA), which is always available.

### SoftRoCE Persistence (WSL2 / Linux)

SoftRoCE links reset on every WSL2 boot. Use the provided script to auto-create them:

```bash
# One-time creation
sudo ./scripts/softroce_persist.sh create

# Install as systemd service (auto on boot)
sudo ./scripts/softroce_persist.sh install

# Check status
./scripts/softroce_persist.sh status
```

**Windows (PowerShell):**
```powershell
.\scripts\softroce_persist.ps1 -Create
.\scripts\softroce_persist.ps1 -Install
.\scripts\softroce_persist.ps1 -Status
```

The script creates `rxe0` on `eth0` and `rxe1` on `lo`, waits for `ACTIVE` state, and optionally installs a systemd service for persistence.

---

## Windows Network Direct (NDKPI) Transport

On Windows, VulkanVM implements **GPU-Direct RDMA via the Network Direct Kernel Provider Interface (NDKPI)** using the user-mode Network Direct SPI (`IND2Provider`, `IND2Adapter`, `IND2CompletionQueue`, `IND2QueuePair`, `IND2Connector`, `IND2Listener`, `IND2MemoryRegion`).

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Application                                        │
│   (vvm::network::RdmaTransport::create → NdkRdmaTransport)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NDKPI User-Mode Stack                               │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ IND2Provider│──│ IND2Adapter │──│ IND2QueuePair│  │ IND2Completion│    │
│   │ (enumerate, │  │ (CreateQP,  │  │ (Read/Write, │  │  Queue        │    │
│   │  Resolve,   │  │  CreateMR,  │  │  Flush)     │  │  (GetResults) │    │
│   │  OpenAdapter)│  │  CreateCQ)  │  │             │  │             │    │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
          ┌─────────────────────┐            ┌─────────────────────┐
          │  Real RNIC Provider │            │  Fake Provider      │
          │  (e.g., mlx5nd2)    │            │  (tests/ndfake/)    │
          │  Hardware RDMA      │            │  In-process loopback│
          └─────────────────────┘            └─────────────────────┘
```

### Implementation Files

| File | Description |
|------|-------------|
| `src/network/ndk_transport.cpp` | `NdkRdmaTransport` — full `RdmaTransport` implementation using IND2 SPI |
| `third_party/ndk/*.h` | Vendored ND SPI headers (`ndspi.h`, `nddef.h`, `ndstatus.h`) |
| `tests/ndfake/ndfake_provider.{h,cpp}` | In-process fake ND provider for CI/testing |
| `tests/ndfake/ndk_transport_test.cpp` | Loopback test (connect, register, write, read, verify) |

### Usage

```cpp
// Standard transport creation — auto-selects ND on Windows
NetworkConfig cfg;
cfg.listenAddress = "0.0.0.0:51000";
cfg.enableRdma = true;

VkInstance instance = ...;
VkPhysicalDevice physDev = ...;
VkDevice device = ...;

auto transport = RdmaTransport::create(cfg, physDev, device);
if (transport && transport->initialize()) {
    // Use rdmaWrite / rdmaRead / registerHostMemory as usual
}
```

The transport requires the **vendor's ND provider** (e.g., NVIDIA `mlx5nd2.sys`) installed in the Winsock catalog (LSP). Without it, the transport falls back to host-staged TCP.

### Test Harness (Fake Provider)

For development/CI without RNIC hardware, a **fake provider DLL** implements the full IND2 SPI on host RAM:

```powershell
# Build (Windows, Ninja + MSVC via vcvars64)
.\scripts\build.ps1 -Tests -BuildType Release

# Run with fake provider
set VVM_ND_PROVIDER_DLL=D:\VulkanVM\build_win\tests\ndfake_provider.dll
.\build_win\tests\ndk_transport_test.exe
```

The fake provider:
- Registers memory regions in a global map (lkey == rkey = unique token)
- Implements `Read`/`Write` as `memcpy` between registered host buffers
- Posts synchronous completions to the CQ (`GetResults` pops them)
- Simulates listener/acceptor loop via condition variables
- Supports `CancelOverlappedRequests` for clean shutdown

This exercises the **entire NDKPI consumer path** end-to-end: bootstrap → adapter query → CQ → QP → register → connect/accept → Write/Read → completions.

### Vendor ND Providers

| Vendor | ND Provider | Notes |
|--------|-------------|-------|
| NVIDIA | `mlx5nd2` | Mellanox ConnectX-4/5/6, requires WinOF-2 |
| AMD | *TBD* | Currently no public Windows ND provider |
| Intel | *TBD* | Requires vendor NIC with NDKPI support |

> **Note:** Unlike Linux verbs (ibverbs/rdma_cm) which are open and kernel-independent, Windows NDKPI requires a **vendor-supplied ND provider** (kernel miniport + user-mode SPI DLL). VulkanVM cannot provide GPU-Direct RDMA on Windows without the vendor's ND stack installed.

---

## PyTorch C++ Extension

VulkanVM provides a native PyTorch C++ extension (`vulkanvm_torch`) that exposes the memory pool, offload, external memory, ModelHub, and shard placement APIs directly to Python.

### Building

```bash
# Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DVVM_BUILD_PYTORCH=ON \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build --config Release

# Windows (PowerShell)
$env:CMAKE_PREFIX_PATH = python -c "import torch; print(torch.utils.cmake_prefix_path)"
cmake -G Ninja -B build_ninja -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_PYTORCH=ON
cmake --build build_ninja --config Release
```

### Python API

```python
import vulkanvm_torch as vvm
import torch

# Create Vulkan device config
instance = vvm.create_instance()
devices = vvm.enumerate_devices(instance)
best = vvm.select_best_device(devices, True, 1024)

queues = vvm.find_queue_families(best.device)
dev_config = vvm.DeviceConfig(
    physicalDevice=best.device,
    device=vvm.create_device(best.device, queues),
    graphicsQueueFamily=queues.graphics,
    computeQueueFamily=queues.compute,
    transferQueueFamily=queues.transfer,
    # ... vkGetDeviceQueue for each queue ...
)

# Create pool
pool_config = vvm.PoolConfig.for_device(best.device)
pool_config.maxHeapFraction = 0.75
pool = vvm.UnifiedMemoryPool.create(dev_config, pool_config)

# Allocate tensor (returns object with device_address, buffer, host_ptr)
alloc = pool.allocate_tensor(64 * 1024 * 1024, "weights")

# Use in PyTorch via torch.frombuffer or custom kernels
# alloc.device_address can be passed to CUDA/Vulkan kernels

# Offload to host
pool.initialize_offload(4 * 1024**3, dev_config.transferQueue, dev_config.transferQueueFamily)
op = pool.offload_to_host(alloc)
pool.wait_migration(op)

# Reload
pool.reload_to_device(alloc)

# Cross-GPU sharing
exported = pool.export_memory(alloc, vvm.ExternalHandleType.OpaqueWin32)
# ... on peer GPU ...
peer_alloc = peer_pool.import_memory(exported, usage)

# ModelHub
hub = vvm.ModelHub("/data/models")
hub.start("0.0.0.0", 51010)
hub.publish("my-org/llama-3b", "./model-files", "v1")

# On another machine
vvm.ModelHub.fetch("192.168.1.50:51010", "my-org/llama-3b", "./models", "v1")

# Shard Placement
cluster = vvm.ClusterCapacity()
cluster.nodes = [
    vvm.NodeCapacity("node-a", 8_GB, 8_GB, 4_GB, 1, 1, 1000, True),
    vvm.NodeCapacity("node-b", 8_GB, 8_GB, 4_GB, 1, 1, 1000, True),
]
cluster.reservedActivationBytes = 512_MiB

model = vvm.ModelManifest()
model.shards = [
    vvm.ShardSpec("blk.0-3", "hash1", vvm.ShardKind.Weights, 2_GB, 0, 3),
    vvm.ShardSpec("blk.4-7", "hash2", vvm.ShardKind.Weights, 2_GB, 4, 7),
]

policy = vvm.PlacementPolicy()
policy.allowHostOffload = True
policy.preferContiguousLayers = True

plan = vvm.ShardPlacer.plan(model, cluster, policy)
```

### Exposed Classes

| Class | Description |
|-------|-------------|
| `UnifiedMemoryPool` | Main memory pool with allocate/deallocate/offload/export/import |
| `Allocation` / `UniqueAllocation` | RAII allocation handle with device_address, buffer, host_ptr |
| `PoolConfig` | Pool configuration (blockSize, maxHeapFraction, etc.) |
| `DeviceConfig` | Vulkan device + queue configuration |
| `MultiGPUPoolManager` | Multi-GPU distributed allocation |
| `SparseVirtualMemoryPool` | Sparse/residency virtual memory |
| `ModelHub` | Hugging Face–style model distribution |
| `ShardPlacer` / `PlacementExecutor` | Capacity-first shard placement |
| `NodeCapacity`, `ClusterCapacity`, `ShardSpec`, `ModelManifest` | Placement types |

---

## ONNX Runtime Integration

VulkanVM provides an ONNX Runtime integration (`vulkanvm_onnx`) with a custom execution provider that manages tensor memory through VulkanVM pools, plus ModelHub integration for distributing ONNX models.

### Building

```bash
# Requires ONNX Runtime installed (vcpkg/conan or built from source)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DVVM_BUILD_ONNX=ON
cmake --build build --config Release
```

### Python API

```python
import vulkanvm_onnx as vvm
import numpy as np

# Create provider
provider = vvm.VulkanVMExecutionProvider(
    vvm.VulkanVMExecutionProviderConfig(
        pool_size=2_GB,
        enable_host_offload=True,
        host_shadow_size=4_GB,
        device_index=0
    )
)

# Allocate tensor memory via VulkanVM
alloc = provider.allocate_tensor([1, 3, 224, 224], vvm.TensorElementType.FLOAT, "input")

# Upload from NumPy
input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
provider.upload_tensor(alloc, input_np, input_np.nbytes)

# Download to NumPy
output_np = np.zeros((1, 1000), dtype=np.float32)
provider.download_tensor(alloc, output_np, output_np.nbytes)

# ModelHub for ONNX models
provider.publish_onnx_model("my-org/resnet50", "./resnet50.onnx", "v1")
provider.fetch_onnx_model("192.168.1.50:51010", "my-org/resnet50", "./models", "v1")

# Pool stats
print(provider.get_stats())
```

### NumPy Interop

```python
# Direct from numpy array
alloc = vvm.create_tensor_from_numpy(input_np, provider, "input")

# Back to numpy
result_np = vvm.tensor_to_numpy(alloc, provider, [1, 1000], vvm.TensorElementType.FLOAT)
```

---

## Sparse / Residency Virtual Memory

Virtual memory pool with `VK_BUFFER_CREATE_SPARSE_BINDING | SPARSE_RESIDENCY`. Allows reserving a large virtual address range and committing physical pages on demand.

```cpp
#include <vulkan_vm/sparse.hpp>

using namespace vvm;

// 1 GiB virtual address space, 32 MiB pages
SparseVirtualMemoryPool sparsePool(device, physicalDevice);
sparsePool.initialize(1_GiB, 32_MiB);

// Reserve virtual range (no physical memory yet)
auto reservation = sparsePool.reserveVirtual(256_MiB, usageFlags);

// Commit physical pages for a sub-range (e.g., 64 MiB at offset 64 MiB)
sparsePool.commit(reservation, 64_MiB, 64_MiB, memoryFlags);

// Uncommit when done
sparsePool.uncommit(reservation, 64_MiB, 64_MiB);
```

**Tested:** 1 GiB virtual / 32 MiB pages passes all sparse tests.

---

## Building

### Windows (PowerShell)

```powershell
# Using the Ninja + VS DevCmd method (avoids Windows SDK version issue)
.\scripts\build.ps1 -Tests -BuildType Release
# Or manually:
cmd /c "call \"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat\" -arch=x64 && cmake -G Ninja -B build_ninja -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_TESTS=ON && cmake --build build_ninja --config Release"
```

### Linux

```bash
./scripts/build.sh --tests
# Or manually:
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure
```

### Requirements

- CMake 3.20+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 19.30+)
- Vulkan SDK 1.3+
- **Windows SDK 10.0.26100+** (for NDKPI headers: `ws2spi.h`, `ndspi.h`)
- Optional: Volk (dynamic Vulkan loading), OpenSSL (TLS), gRPC/Protobuf (control plane), ibverbs/rdma_cm (RDMA)
- **Linux RDMA:** Kernel with `CONFIG_RDMA_RXE=y` (built-in) + `rdma-core` userspace for SoftRoCE fallback. Without it, network module uses host-staged TCP.
- **Windows NDKPI:** Vendor ND provider (e.g., NVIDIA `mlx5nd2` via WinOF-2) OR WDK 10.0.28000+ for kernel provider development

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `VVM_BUILD_SHARED` | ON | Build as shared library |
| `VVM_BUILD_TESTS` | ON | Build unit tests |
| `VVM_BUILD_EXAMPLES` | ON | Build examples (includes integration tests) |
| `VVM_ENABLE_VALIDATION` | ON | Enable Vulkan validation layers |
| `VVM_USE_VOLK` | ON | Use Volk for dynamic Vulkan loading |
| `VVM_BUILD_NETWORK` | ON | Build network/multi-node module |
| `VVM_BUILD_TENSOR_TRANSPORT` | ON | Build tensor transport module |
| `VVM_BUILD_PYTORCH` | OFF | Build PyTorch C++ extension |
| `VVM_BUILD_ONNX` | OFF | Build ONNX Runtime integration |

### Build Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build.ps1` | Windows build with tests/examples |
| `scripts/build.sh` | Linux build with tests/examples |
| `scripts/build_with_python.sh` | Linux build with `--pytorch`/`--onnx` flags |
| `scripts/softroce_persist.sh` | Linux: create/install SoftRoCE links |
| `scripts/softroce_persist.ps1` | Windows: WSL wrapper for SoftRoCE persistence |

### Key Test Targets

| Target | Description | Platform |
|--------|-------------|----------|
| `basic_test` | Core pool/buffer allocator | Linux/Windows |
| `buddy_test` | Buddy allocator logic | Linux/Windows |
| `tensor_collective_test` | Tensor collectives (allReduce, broadcast, allGather, reduceScatter) | Linux |
| `ndk_transport_test` | **ND fake provider loopback** (connect/register/write/read) | **Windows** |
| `network_test` | Multi-node TCP cluster (2-node loopback) | Linux/Windows |
| `tensor_network_test` | GPU↔GPU tensor send/recv over TCP | Linux |
| `placement_test` | Shard placement logic (10 pure-logic tests) | Linux/Windows |
| `multi_gpu_test` | Multi-GPU P2P + offload (requires 2+ GPUs) | Linux/Windows |

---

## Cross-Vendor Compatibility

| Source → Target | Linux Handle | Windows Handle |
|-----------------|--------------|----------------|
| NVIDIA → AMD    | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| NVIDIA → Intel  | DMA-BUF      | D3D12_HEAP → OPAQUE_WIN32 |
| AMD → NVIDIA    | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| Intel → NVIDIA  | DMA-BUF      | OPAQUE_WIN32 → D3D12_HEAP |
| AMD ↔ Intel     | DMA-BUF / OPAQUE_FD | OPAQUE_WIN32 |
| Same vendor     | OPAQUE_FD    | OPAQUE_WIN32 / D3D12_HEAP |

**Tested Hardware:**
- AMD Radeon RX 7900 XTX (discrete, `0x1002`)
- AMD Radeon Graphics (integrated, `0x1002`)
- Intel Arc Pro B70 (discrete, `0x8086`)

All three support sparse binding + residency.

> **Intel Arc on Windows (`igvk64.dll`) — known driver flakiness:**
> The Intel Vulkan driver on Windows is notoriously unstable. Real-world reports (including crashes blacklisted by the Khronos validation-layer test suite, IGCIT tracker issues, and DXVK upstream) consistently show `0xC0000005` access violations raised *inside* `igvk64.dll` at device/video-memory teardown, even for spec-correct applications.
> `multi_gpu_test` on Intel Arc hardware can exit with a non-zero status (`-1073741819`) during teardown *after* every test assertion passes — this is reproduced on the unmodified baseline and is not an application fault.
> Workarounds:
> - Use a newer/older known-good driver (community reports vary; some users fall back to `32.0.101.69xx`, others need the latest `101.84xx+` line).
> - Prefer NVIDIA/AMD discrete GPUs, or the Linux stack, for CI on Intel Arc.
> - Treat `multi_gpu_test` teardown crashes on Intel Windows as a driver noise, not a VulkanVM regression.

---

## APU-Specific Tuning (Strix Halo 395 / Unified Memory)

```cpp
// Auto-tuned:
PoolConfig cfg = PoolConfig::forDevice(physicalDevice);
// Sets: blockSize=1GB, maxBlocks=8, maxHeapFraction=0.7, unified memory type

// Manual for Strix Halo (96-128 GB unified RAM):
PoolConfig cfg;
cfg.blockSize = totalSystemRAM * 0.8;   // Reserve 80% at boot (e.g., 80 GB of 96 GB)
cfg.minAlignment = 256 * 1024;
cfg.enableHostVisible = true;            // Unified memory = host visible
cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
cfg.maxHeapFraction = 0.7f;              // Never exceed 70% of system heap

// Topology detection:
MemoryTopology topo = detectMemoryTopology(physicalDevice);
// topo == Unified → single heap with DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT
```

**Result:** Single massive block at boot. ROCm can't fragment what VulkanVM already owns.

---

## License & Credits

**MIT License** — see `LICENSE`.

### Contributors

- **XxChonkExX** — Project Owner, System Architect & Principal Engineer (architecture, specs, integration, validation)
- **NVIDIA** — Network module framework foundation; open-source support|
- **Khronos Group - Vulkan
- **Community** — Issues, testing, and feedback
- **LLMs: NVIDIA Nemotron, DeepSeek, Grok, and GLM
### Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`ctest --output-on-failure`)
5. Submit PR

---

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
- [x] Budget-capped pool (`maxHeapFraction`, `maxPoolBytes`, `VK_EXT_memory_budget`)
- [x] `PoolConfig::forDevice()` + `detectMemoryTopology()` (APU/Discrete/Hybrid)
- [x] `MemoryUsage` + `AllocDesc` intent-based allocation API
- [x] `VK_EXT_debug_utils` object names on buffers/memory
- [x] `ModelHub` — Hugging Face–style weight distribution (content-addressed, cache+resume)
- [x] External handle ownership fix (import consumes, `duplicateForImport` for multi-GPU)
- [x] Sparse/residency support for virtual memory
- [x] Host-staged GPU↔GPU fallback when driver refuses cross-GPU import
- [x] Direct GPU↔GPU copy (P2P) without host staging (NVIDIA/AMD/Intel vendor paths)
- [x] RDMA/verbs GPU-direct transport (NVIDIA/AMD/Intel vendor paths)
- [x] Linux SoftRoCE (`rxe`) end-to-end verification on WSL2
- [x] P0 hardening: UniqueAllocation RAII, Offload thread-safety, Logging format
- [x] Shard Placement API (capacity-first bin-packing, executor with rollback)
- [x] PyTorch C++ extension (`vulkanvm_torch`)
- [x] ONNX Runtime integration (`vulkanvm_onnx`)
- [x] Tensor Transport module (`vulkanvm_tensor`) — allocation, copy, collectives, network
- [x] Tensor Transport: layout conversion shaders (NHWC↔NCHW)
- [x] Tensor Transport: async pipeline + async collectives (`copyTensorAsync`, `allReduceAsync`, ...)
- [x] GPU-Direct RDMA wiring (`ibv_reg_dmabuf_mr` on Linux, NVIDIA peermem)
- [x] **Windows Network Direct (NDKPI) transport** — `NdkRdmaTransport` with IND2 SPI, fake provider test harness
- [x] **Tensor collectives** — Ring all-reduce, broadcast, all-gather, reduce-scatter with CPU fallback (FP32/FP16/BF16/FP8/INT4/INT8/INT32/BOOL)
- [ ] Windows WDDM2.6+ hardware scheduling hints
- [ ] Android/Vulkan support
- [ ] Kernel NDKPI provider skeleton (Windows)
- [ ] Tensor Transport: NCCL-style production collectives
