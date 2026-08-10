# VulkanVM — High-Performance GPU Networking & RDMA + Unified Memory Pool (Chonk Buffer)

**VulkanVM** is a **cross-vendor GPU networking and RDMA library** that enables zero-copy GPU↔GPU transfers over TCP, RDMA, and GPU-direct NIC DMA across AMD, NVIDIA, and Intel GPUs. Built on a hardened, zero-fragmentation Vulkan memory pool (the "Chonk Buffer"), it provides Hugging Face–style model distribution, capacity-first shard placement, and a unified tensor transport for AI inference clusters. One header, one library, zero system changes — link it into your application.

---

## Table of Contents

1. [Quick Start — GPU Networking](#quick-start--gpu-networking)
2. [GPU Networking & RDMA (Primary Feature)](#gpu-networking--rdma-primary-feature)
   - [Multi-Node Cluster over TCP/RDMA](#multi-node-cluster-over-tcprdma)
   - [Unified Tensor Transport](#unified-tensor-transport-vvmtensortransport)
   - [GPU-Direct RDMA Vendor Paths](#gpu-direct-rdma-vendor-paths)
   - [Windows NDKPI / Linux Verbs](#windows-ndkpi--linux-verbs)
   - [SoftRoCE (Linux Software RDMA)](#softroce-linux-software-rdma)
3. [ModelHub & Shard Placement](#modelhub--shard-placement)
   - [Model Registry & Distribution](#modelhub--model-weight-distribution)
   - [Capacity-First Shard Placement](#shard-placement-api)
4. [Core Memory Pool (Chonk Buffer)](#core-memory-pool-chonk-buffer)
   - [Buddy Allocator — Zero Fragmentation](#buddy-allocator--zero-fragmentation)
   - [Auto-Tuning (APU / Discrete / High-VRAM)](#auto-tuning-apu--discrete--high-vram)
   - [Memory Usage Intents](#memory-usage-intents-no-raw-flags)
5. [Cross-GPU Memory Sharing](#cross-gpu-memory-sharing)
6. [Host Offload / Demand Paging](#host-offload--demand-paging)
7. [Sparse / Residency Virtual Memory](#sparse--residency-virtual-memory)
8. [Android / Vulkan (AHardwareBuffer)](#android--vulkan-ahardwarebuffer-external-memory)
9. [PyTorch & ONNX Integration](#pytorch--onnx-integration)
10. [Building](#building)
11. [Hardware Compatibility](#hardware-compatibility)
12. [License & Credits](#license--credits)

---

## Quick Start — GPU Networking

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/tensor_transport.hpp>
using namespace vvm;
using namespace vvm::tensor;

// 1. Select GPU(s)
VkInstance instance = createInstance();
auto devices = enumerateDevices(instance);
auto best = selectBestDevice(devices, /*preferDiscrete=*/true, /*minVRAM_MiB=*/1024);
VkPhysicalDevice physicalDevice = best->device;

auto queues = findQueueFamilies(physicalDevice);
DeviceConfig devConfig{
    .physicalDevice = physicalDevice,
    .device = createDevice(physicalDevice, queues),
    .graphicsQueueFamily = queues.graphics.value_or(0),
    .computeQueueFamily  = queues.compute.value_or(0),
    .transferQueueFamily = queues.transfer.value_or(0),
};
// ... vkGetDeviceQueue for each queue ...

// 2. Create pool (auto-tuned for your hardware)
PoolConfig poolConfig = PoolConfig::forDevice(physicalDevice);
poolConfig.maxHeapFraction = 0.75f;

// 3. Configure tensor transport with networking
TransportConfig config;
config.preference = TransportConfig::Preference::Auto;  // P2P → RDMA → HostStaged → Network
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;
config.networkPort = 51000;
config.enableTLS = true;          // optional TLS for multi-node
config.tlsCertPath = "cert.pem";
config.tlsKeyPath = "key.pem";
config.tlsCaPath = "ca.pem";

// 4. Create transport and initialize
auto transport = vvm::tensor::createTensorTransport(config, {devConfig}, poolConfig);
if (!transport->initialize()) { /* handle error */ }

// 5. Allocate a tensor on GPU 0
TensorMetadata meta;
meta.dtype = DataType::Float16;
meta.layout = MemoryLayout::ChannelsLast;
meta.shape = TensorShape::makeChannelsLast({1, 32, 32, 128});
meta.name = "conv_weight";

auto tensor = transport->allocateTensor(meta, 0);

// 6. Multi-node: send/recv across machines (auto-picks P2P→RDMA→Network)
transport->sendTensor(tensor, "192.168.1.10:51001#0", 
    [](bool ok, const std::string& err) { /* done */ });

transport->recvTensor(receiver, "192.168.1.11:51002#0",
    [](bool ok, const std::string& err) { /* done */ });

// 7. Collectives (ring all-reduce across GPUs/nodes)
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
```

---

## GPU Networking & RDMA (Primary Feature)

VulkanVM's networking stack is a **production-grade multi-node GPU fabric** that moves tensor VRAM across machines with zero-copy where hardware allows, automatic fallback where it doesn't.

### Multi-Node Cluster over TCP/RDMA

| Transport | Mechanism | Staging | Platform |
|-----------|-----------|---------|----------|
| **TCP (host-staged)** | 4 MB slices direct socket↔staging | ✅ 4 MB | Windows / Linux |
| **TCP Striped** | Parallel sockets for >64 MB transfers | ✅ 4 MB | Windows / Linux |
| **Windowed Pipeline** | Double-buffered host staging (opt-in) | ✅ adaptive | Windows / Linux |
| **RDMA (Verbs)** | `ibv_reg_dmabuf_mr` / GPU-direct | ❌ No | Linux |
| **NDKPI (Windows)** | `IND2Provider` + `CreateMemoryRegion` | ✅ Host | Windows |
| **GPU-Direct RDMA** | NIC DMA to GPU VRAM | ❌ No | Linux (NVIDIA/AMD/Intel) |

**Key features:**
- **Spark-inspired wire protocol**: 32-byte versioned header + bulk stream in 4 MB slices
- **Protocol hardening**: 1 GiB body / 16 GiB stream caps, connection drop on violation
- **TLS 1.2+**: OpenSSL-backed, SNI + ALPN (`vvm/1.0`), gated by `VVM_NETWORK_HAS_TLS`
- **Automatic path selection**: `TransportConfig::Preference::Auto` → P2P → RDMA → HostStaged → Network
- **Byte-exact verified**: All paths verified on loopback with checksums

```cpp
using namespace vvm::network;

// Node A (bootstrap seed)
NetworkConfig netA; netA.listenAddress = "0.0.0.0:51001";
auto nodeA = MultiNodePoolManager::create({devCfg}, poolCfg, netA);
nodeA->start();
nodeA->registerWithCluster();

// Node B (joins A's cluster)
NetworkConfig netB; netB.listenAddress = "0.0.0.0:51002";
netB.seedNodes = { "127.0.0.1:51001" };
auto nodeB = MultiNodePoolManager::create({devCfg}, poolCfg, netB);
nodeB->start();
nodeB->registerWithCluster();

// Remote allocation + push migration (B → A)
auto src = nodeB->getLocalPool().allocate(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
std::memset(src->hostPtr, 0xAB, size);
auto dst = nodeB->allocateRemote(nodeA->getLocalNodeId(), size, usage,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeB->migrateToRemote(*src, *dst, /*useRdma=*/true);  // auto RDMA if available

// Pull migration (A pulls from B)
auto remoteDesc = nodeB->exportForRemote(*src, false, true);
auto localDst   = nodeA->getLocalPool().allocate(size, usage,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
nodeA->migrateFromRemote(*remoteDesc, *localDst, /*useRdma=*/true);
```

### Unified Tensor Transport (`vvm::tensor::Transport`)

Moves **tensors** — not raw bytes — with shape, dtype, and layout awareness. Auto-selects the best transport path.

| Feature | Description |
|---------|-------------|
| **Tensor Metadata** | Shape, dtype (FP32/FP16/BF16/INT8/INT4/FP8), strides, layout (NHWC/NCHW/Blocked) |
| **Auto Transport Selection** | P2P → RDMA → Host-Staged → Network |
| **Layout Conversion** | NHWC↔NCHW, blocked/tiling for tensor cores (compute shaders) |
| **Collectives** | Ring all-reduce, broadcast, all-gather, reduce-scatter |
| **Async Pipeline** | Worker thread, callback-based `AsyncOperation` queue, overlap compute + transfer |
| **Multi-Node** | `sendTensor`/`recvTensor` with `host:port#nodeIndex` addressing |

```cpp
#include <vulkan_vm/tensor_transport.hpp>
using namespace vvm::tensor;

TransportConfig config;
config.preference = TransportConfig::Preference::Auto;
config.enableAsyncPipeline = true;
config.maxInFlightTransfers = 4;
config.networkPort = 51000;
config.enableTLS = true;
config.tlsCertPath = "cert.pem";
config.tlsKeyPath = "key.pem";
config.tlsCaPath = "ca.pem";

std::vector<DeviceConfig> devices = { amdConfig, intelConfig, nvidiaConfig };
PoolConfig poolConfig = PoolConfig::forDevice(devices[0].physicalDevice);

auto transport = vvm::tensor::createTensorTransport(config, devices, poolConfig);
if (!transport->initialize()) { /* handle error */ }

// Allocate + distribute
auto tensor = transport->allocateTensor(meta, 0);
auto tensors = transport->allocateDistributed(meta, {0, 1, 2});

// Copy with layout conversion (compute shaders, no host round-trip)
transport->copyWithLayoutConversion(src, dst, MemoryLayout::Blocked);

// Collectives
transport->allReduce({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2});
transport->broadcast(rootTensor, {0, 1, 2}, 0);
transport->allGather({t0, t1, t2}, output, {0, 1, 2});
transport->reduceScatter({t0, t1, t2}, output, ReduceOp::Sum, {0, 1, 2});

// Async
transport->allReduceAsync({t0, t1, t2}, ReduceOp::Sum, {0, 1, 2},
    [](bool ok, const std::string& err) { /* done */ });
transport->flushAsync();  // wait for all queued ops

// Multi-node send/recv
transport->sendTensor(tensor, "192.168.1.10:51001#0", callback);
transport->recvTensor(receiver, "192.168.1.11:51002#0", callback);
```

**Transport Priority (Auto):**
| Priority | Path | Mechanism | Staging? |
|----------|------|-----------|----------|
| 1 | **P2P** | `VK_EXTERNAL_MEMORY` + `vkCmdCopyBuffer` | ❌ |
| 2 | **RDMA** | GPU-Direct: `ibv_reg_dmabuf_mr` / NDKPI | ❌ |
| 3 | **AHardwareBuffer** | Android: `VK_ANDROID_external_memory_android_hardware_buffer` | ❌ |
| 4 | **Host-Staged** | 4 MiB chunks via host memory | ✅ |
| 5 | **Network** | TCP/RDMA multi-node | ✅ |

### GPU-Direct RDMA Vendor Paths

For NIC-attached GPU memory DMA (bypassing host CPU entirely), VulkanVM provides **three vendor-specific registration paths**. The `RdmaTransport` (verbs backend) dispatches based on `vendorID`.

| GPU Vendor | Linux Path | Windows Path |
|------------|------------|--------------|
| **NVIDIA (0x10DE)** | `VK_NV_external_memory_rdma` → `vkGetMemoryRemoteAddressNV` → `ibv_reg_mr` on PCI BAR (requires `nvidia-peermem`) — **true GPU-direct BAR peer-mem** | NDKPI (`IND2Provider`) + `CreateMemoryRegion` from `D3D12_HEAP`/`OPAQUE_WIN32` — **true GPU-direct when provider supports it** |
| **AMD (0x1002)** | `DMA_BUF` → `ibv_reg_dmabuf_mr` (or mmap fallback) — **best-effort; not always BAR peer-mem class** | `OPAQUE_WIN32` → `MapViewOfFile` → NDKPI `CreateMemoryRegion` — **host/ND registration unless provider does GPU MR** |
| **Intel (0x8086)** | `DMA_BUF` → `ibv_reg_dmabuf_mr` (Xe KMD) — **best-effort; not always BAR peer-mem class** | **Level Zero** `zeMemGetAllocProperties` + `ze_external_memory_export_win32_handle_t` → NDKPI — **host/ND registration unless provider does GPU MR** |

**Note on "GPU-Direct" claims:** NVIDIA + `VK_NV_external_memory_rdma` with `nvidia-peermem` achieves true BAR peer-mem GPU↔NIC DMA. AMD and Intel paths use DMA-BUF / Level Zero export + NDKPI which are **practical host-staged or best-effort GPU registration** — they do not always achieve zero-copy NIC↔GPU BAR DMA. The transport logs which path engaged and whether zero-copy was achieved.

```cpp
// In rdma_transport.cpp (VerbsRdmaTransport)
std::optional<RdmaMemoryRegion> registerGpuMemory(
    VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkBuffer buffer) override;

// Dispatches to:
registerGpuMemoryForRdmaVendor(device, physDev, memory, offset, size, ... , vendorId);

// Returns RdmaMemoryRegion with lkey/rkey/rdmaAddr for direct NIC DMA
struct RdmaMemoryRegion {
    void* addr = nullptr;
    uint64_t length = 0;
    uint32_t lkey = 0, rkey = 0;
    uint64_t rdmaAddr = 0;  // GPU-direct remote address
    bool ownsMemory = false;
    VkDeviceMemory vkMemory = VK_NULL_HANDLE;
    VkBuffer vkBuffer = VK_NULL_HANDLE;
};
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

### Windows NDKPI / Linux Verbs

| Stack | API | Kernel Bypass | GPU-Direct |
|-------|-----|---------------|------------|
| **Linux Verbs** | `ibverbs` + `rdma_cm` | ✅ Yes | ✅ NVIDIA/AMD/Intel |
| **Windows NDKPI** | `IND2Provider/Adapter/Connector` | ✅ Yes | ❌ Host-staged only (no vendor ND provider for Intel/Arc yet) |

**NDKPI flow (Windows):**
```cpp
// 1. Enumerate ND providers via WSAEnumProtocols (PFL_NETWORKDIR_PROVIDER)
// 2. WSAStartup + WSAEnumProtocols → find IND2Provider
// 3. DllGetClassObject → IID_IND2Provider → IND2Adapter
// 4. IND2Adapter::CreateCQ / CreatePD / CreateConnector / CreateListener
// 5. IND2Connector::Connect / IND2Listener::Listen
// 6. Memory registration: IND2Adapter::CreateMemoryRegion + IND2MemoryRegion::Register
// 7. RDMA ops: IND2QueuePair::Write / Read / Send / Recv
// 8. Completion: IND2CompletionQueue::GetResults
```

**Verified:** Loopback test with fake provider (`ndfake_provider.dll`) exercises connect/accept, MR register, RDMA write+read, clean teardown — 3/3 runs clean.

### SoftRoCE (Linux Software RDMA)

For Linux environments without physical RNIC hardware (WSL2, VMs, CI, dev machines), VulkanVM supports **SoftRoCE** via the kernel's `rxe` module. This enables the verbs/RDMA transport over standard Ethernet (`eth0`) or loopback (`lo`).

**Requirements:**
- Kernel with `CONFIG_RDMA_RXE=y` (built-in, not module)
- `rdma-core` userspace (`ibverbs`, `rdma_cm`)

**Setup (WSL2 / Ubuntu 24.04):**
```bash
# 1. Build kernel with RDMA_RXE (or use distro kernel that includes it)
# 2. Create SoftRoCE links on available netdevs:
sudo rdma link add rxe0 type rxe netdev eth0   # primary interface
sudo rdma link add rxe1 type rxe netdev lo     # loopback for localhost tests
rdma link show  # verify ACTIVE state
ibv_devices     # should list rxe0, rxe1
```

**Verification (tensor_network_test):**
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

**Status:** ✅ End-to-end verified on WSL2 (custom kernel 6.18.40, `rxe0` + `rxe1`). On native Linux with physical RNIC, the same verbs path uses hardware RDMA. Without SoftRoCE or hardware RNIC, the network module falls back to **host-staged TCP** (always available).

### UCX (Unified Communication X) Transport

For production multi-transport clusters, VulkanVM integrates **UCX** (Unified Communication X) as a high-level transport backend. UCX automatically selects the best available transport at runtime: InfiniBand verbs, RoCE, TCP, shared memory, and GPU memory (CUDA/ROCm/Level Zero).

**Key features:**
- **Auto transport selection**: IB verbs → RoCE → TCP → shared memory → CUDA IPC
- **GPU memory registration**: `ucp_mem_map` for zero-copy RDMA from GPU VRAM
- **Tag matching + RMA**: `ucp_tag_send`/`recv` for control, `ucp_put`/`get` for RDMA
- **Rendezvous protocol**: Automatic switch to RDMA for large messages
- **CUDA/ROCm/Ze IPC**: Intra-node GPU copies without host staging

**Enable UCX:**
```cpp
TransportConfig config;
config.enableUCX = true;
config.ucxTLS = "rc,ud,sm,tcp";      // transport layers
config.ucxNetDevices = "mlx5_0:1";   // NIC selection
config.ucxEnableGPUMem = true;       // GPU memory registration
config.ucxEnableRndv = true;         // rendezvous for large msgs
config.ucxRndvThreshold = 8192;      // bytes
config.ucxEnableCudaIpc = true;      // intra-node GPU IPC
```

**UCX TLS (Transport Layer Selection):**
| TLS | Description |
|-----|-------------|
| `rc` | Reliable Connected (InfiniBand/RoCE) |
| `ud` | Unreliable Datagram |
| `sm` | Shared Memory (intra-node) |
| `tcp` | TCP/IP fallback |
| `cuda_ipc` | CUDA IPC (intra-node GPU) |
| `cuda_copy` | CUDA copy (staged) |

**Important:** UCX "TLS" means *Transport Layer Selection* (which fabrics UCX may use: `rc`, `ud`, `sm`, `tcp`, `cuda_ipc`, etc.). It is **not** related to OpenSSL TLS (`TransportConfig::enableTLS`, `tlsCertPath`, etc.), which secures the TCP control plane. UCX TLS is configured via `config.ucxTLS` or `UCX_TLS` env; OpenSSL TLS is configured via `config.enableTLS`.

### UCX Address Exchange

UCX endpoints are created from **opaque worker addresses** (`ucp_worker_get_address`), not TCP `host:port` strings. VulkanVM exchanges these blobs over the existing TCP control plane (`CtrlMsg::UcxAddrExchange`, length-prefixed via `vvm::network::wire::putBytes` / `getBytes`), then calls `ucp_ep_create`. IB/RoCE/TCP selection is controlled by `TransportConfig::ucxTLS` (or `UCX_TLS` env) and happens **after** the endpoint is created.

Bootstrap flow (handled by `UcxTransport::exchangeAndConnect`):

```cpp
// Active side (client that initiated TCP connect):
auto ep = ucx.exchangeAndConnect(peerKey, nodeId, true,
    // sendFn: send our UCX worker address over TCP control
    [&](const std::vector<uint8_t>& addr) {
        return session.sendControl(CtrlMsg::UcxAddrExchange, 
                                   vvm::network::wire::putBytes(body, addr));
    },
    // recvFn: receive peer's UCX worker address
    [&](std::vector<uint8_t>& outAddr) {
        return session.recvControl(CtrlMsg::UcxAddrExchange, body) &&
               vvm::network::wire::getBytes(body.data(), body.data() + body.size(), outAddr);
    });
```

Both peers must use compatible TLS sets (shared intersection that can actually connect). If you set `UCX_TLS=rc` only and there is no usable RC device, EP creation or first send fails — always keep `tcp` or `sm` in dev presets.

**Attribution:** UCX (https://github.com/openucx/ucx) — BSD-3-Clause, Pavel Shamis et al.

### GDRCopy-Style Persistent Host Pinning (NDKPI / Windows)

For high-frequency RDMA on Windows NDKPI, VulkanVM implements **persistent host memory pinning** inspired by NVIDIA's GDRCopy (`gdr_pin_buffer`/`gdr_unpin_buffer`). This avoids repeated `IND2MemoryRegion::Register`/`Deregister` overhead on hot paths.

**API:**
```cpp
// Pin host memory for repeated RDMA use (ref-counted)
bool pinPersistentHostMemory(void* ptr, size_t size);

// Release persistent pin (decrements ref count; actual deregister at 0)
void releasePersistentHostMemory(void* ptr);
```

**How it works:**
- First `pinPersistentHostMemory` → creates `IND2MemoryRegion` + `Register`
- Subsequent pins → increments ref count, reuses existing registration
- `releasePersistentHostMemory` → decrements ref count; actual `Deregister` only at 0
- Thread-safe with `std::unique_lock`

**Use case:** Pre-pin staging buffers during initialization, reuse for thousands of RDMA operations without registration overhead.

**Attribution:** Pattern inspired by GDRCopy (https://github.com/NVIDIA/gdrcopy) — MIT, NVIDIA.

---

## ModelHub & Shard Placement

### ModelHub — Model Weight Distribution

Content-addressed, Hugging Face–style model registry with TCP distribution.

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

**Cache layout:**
```
~/.cache/vvm/models/my-org/llama-3b-q4/v1/
  config.json
  weights.safetensors
  tokenizer/tokenizer.json
  .vvm_complete          # marker = cache entry valid
```

Content-addressed chunks (SHA-256, 4 MiB slices), resume support, TLS optional.

### Shard Placement API (Capacity-First Bin Packing)

Distributes model shards across a GPU cluster respecting VRAM, host-offload, activation reserves, and constraints.

```cpp
#include <vulkan_vm/placement.hpp>
using namespace vvm::placement;

// 1. Model manifest (from ModelHub)
ModelManifest model;
model.modelId = "chonk/llama-3b-q4";
model.version = "v1";
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

## Core Memory Pool (Chonk Buffer)

### Buddy Allocator — Zero Fragmentation

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
- O(1) allocation/deallocation via `unordered_set` free lists per order
- Optional internal mutex (`threadSafe_` + `withLock()`)
- Deterministic placement — lowest free offset handed out first
- Power-of-two alignment guaranteed

### Auto-Tuning (APU / Discrete / High-VRAM)

`PoolConfig::forDevice()` reads the device memory at runtime and picks the block profile:

- **APUs / unified memory (Strix Halo)** → 1-2 GB blocks, host-visible, capped heap fraction
- **Discrete cards under 24 GB** → 512 MB blocks (16 blocks, 0.75 heap fraction)
- **High-VRAM cards ≥24 GB (RTX 4090, RTX 6000 Ada)** → auto-scales to 2 GB blocks, 64 blocks, 0.8-0.85 heap fraction

Explicit overrides: `PoolConfig::forHighVRAM(physicalDevice)` (2 GB blocks, up to 0.85 heap) and `PoolConfig::forAPU(totalSystemRAM)` (RAM-budget-aware).

### Memory Usage Intents (No Raw Flags)

```cpp
enum class MemoryUsage {
    GpuOnly,        // DEVICE_LOCAL — fastest GPU access
    CpuToGpu,       // HOST_VISIBLE | HOST_COHERENT — upload staging
    GpuToCpu,       // HOST_VISIBLE | HOST_COHERENT — readback
    CpuCopy,        // HOST_VISIBLE — CPU-only copies
    Auto            // Let VulkanVM decide
};
```

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

// On target GPU (e.g., NVIDIA RTX 4090)
auto peerHandle = duplicateForImport(exportInfo.handle);  // per-peer dup
auto peerAlloc = peerPool.importMemory(std::move(peerHandle), usage);
// Both GPUs now reference the SAME physical memory

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

## Android / Vulkan (AHardwareBuffer External Memory)

Zero-copy external memory via `VK_ANDROID_external_memory_android_hardware_buffer`. Enables zero-copy sharing between Vulkan and Android's graphics pipeline (Surface, MediaCodec, Camera, etc.).

```bash
# Linux
./scripts/build_android.sh arm64-v8a android-34 Release

# Windows (PowerShell)
.\scripts\build_android.bat arm64-v8a android-34 Release
```

```cpp
#include <vulkan_vm/vulkan_vm.hpp>
#include <android/hardware_buffer.h>

// 1. Create AHardwareBuffer
AHardwareBuffer_Desc desc{};
desc.width = 1920;
desc.height = 1080;
desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | 
             AHARDWAREBUFFER_USAGE_CPU_READ_NEVER |
             AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;

AHardwareBuffer* hardwareBuffer;
AHardwareBuffer_allocate(&desc, &hardwareBuffer);

// 2. Import into VulkanVM pool
ExternalMemoryInfo extInfo;
extInfo.type = ExternalHandleType::AndroidHardwareBuffer;
extInfo.handle = ExternalHandle(hardwareBuffer);  // RAII wrapper
extInfo.size = bufferSize;
extInfo.dedicatedAllocation = true;

auto allocation = pool.importMemory(std::move(extInfo), 
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

// 3. Use in shaders via device address
auto deviceAddress = allocation->deviceAddress;
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
- **Level Zero SDK** (`C:/Program Files/LevelZeroSDK`) for Intel GPU-direct on Windows

---

## Hardware Compatibility

| GPU | Works? | Notes |
|-----|--------|-------|
| RTX 4090 / 7900 XTX (24 GB) | ✅ Excellent | 2 GB blocks (auto-detected ≥24 GB VRAM) |
| RTX 4070 / 7800 XT (12-16 GB) | ✅ Excellent | 256 MB blocks |
| Arc A770 / A750 / B70 (16/8 GB) | ✅ Good | 256 MB blocks, Level Zero GPU-direct |
| Laptop dGPU (8 GB) | ✅ Good | 128 MB blocks |
| **Strix Halo APU (96-128 GB unified)** | ✅ **BEST** | 2 GB blocks (auto-detected), sweet spot |
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

// 8. Multi-node send/recv
transport->sendTensor(tensor, "192.168.1.10:51001#0", callback);
transport->recvTensor(receiver, "192.168.1.11:51002#0", callback);

// 9. GPU-Direct RDMA
auto region = rdmaTransport->registerGpuMemory(memory, offset, size, buffer);
// region.lkey, region.rkey, region.rdmaAddr for NIC DMA
```

---

## Third-Party Licenses & Credits

### Open Source Libraries

| Component | License | Use in VulkanVM |
|-----------|---------|-----------------|
| **UCX (Unified Communication X)** | BSD-3-Clause | High-performance transport backend (InfiniBand, RoCE, TCP, shared memory, GPU) |
| **GDRCopy** | MIT | Persistent host pinning pattern for NDKPI (inspired by `gdr_pin_buffer`/`gdr_unpin_buffer`) |
| **Volk** | MIT | Vulkan function pointer loading (optional) |
| **glslang / SPIRV-Tools** | BSD-3-Clause | Shader compilation to SPIR-V |
| **Vulkan Memory Allocator (VMA)** | MIT | Reference for allocation patterns (not directly used) |
| **Vulkan-Hpp** | Apache-2.0 | C++ bindings for Vulkan API |
| **Windows NDKPI Headers** | Microsoft EULA | Vendored ND SPI headers (`third_party/ndk/`) |
| **SoftRoCE (rxe)** | GPL-2.0 | Linux software RDMA for CI/testing (system dependency) |

### Hardware Vendors & SDKs

| Company | Technology / SDK | Role in VulkanVM |
|---------|------------------|------------------|
| **NVIDIA** | CUDA, CUDA Driver, `VK_NV_external_memory_rdma`, `nvidia-peermem`, GPUDirect RDMA | GPU compute, GPU-direct RDMA registration, peer memory |
| **AMD** | ROCm / HIP, `VK_EXT_external_memory_dma_buf`, Radeon Pro / Instinct GPUs | GPU compute, DMA-BUF export, GPU-direct via HIP |
| **Intel** | Level Zero, Xe KMD, `VK_EXT_external_memory_dma_buf`, Arc / Data Center GPUs | GPU compute, DMA-BUF export, Windows GPU-direct via Level Zero |
| **Mellanox / NVIDIA Networking** | ConnectX / BlueField NICs, MLNX_OFED, `ibverbs`, `rdma_cm`, UCX | High-performance RDMA/RoCE networking |
| **Khronos Group** | Vulkan API, SPIR-V, extensions | Core graphics/compute API and specifications |
| **Microsoft** | Windows SDK, NDKPI (Network Direct), WDDM | Windows networking, RDMA via NDKPI, display |
| **Linux Kernel** | `ibverbs`, `rdma_cm`, `dma-buf`, `rxe` (SoftRoCE) | Linux RDMA stack, DMA-BUF, software RDMA |

### Attribution & Thanks

- **UCX team**: Pavel Shamis, Yossi Itigin, et al. — https://github.com/openucx/ucx
- **NVIDIA GDRCopy**: NVIDIA Corporation — https://github.com/NVIDIA/gdrcopy
- **Khronos Group**: Vulkan API specification and headers
- **Intel Level Zero team**: https://github.com/oneapi-src/level-zero
- **AMD ROCm team**: https://github.com/RadeonOpenCompute/ROCm
- **Mellanox / NVIDIA Networking**: OFED, UCX integration, hardware enablement
- **Vulkan-Hpp maintainers**: https://github.com/KhronosGroup/Vulkan-Hpp
- **Volk maintainers**: https://github.com/zeux/volk
- **glslang/SPIRV-Tools**: https://github.com/KhronosGroup/glslang

### Development Infrastructure

- **GitHub Actions**: CI/CD pipelines
- **CMake / Ninja**: Build system
- **MSVC / Clang / GCC**: Compilers
- **Visual Studio / VS Code**: IDEs
- **Vulkan SDK**: LunarG Vulkan SDK (validation layers, tools)

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