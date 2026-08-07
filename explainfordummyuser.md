# VulkanVM — Explained Like You're 5 (But Detailed Enough For Grown-Ups)

## The Big Picture

Imagine you have **three different toy boxes** (your three GPUs):
- A **red box** (AMD Radeon RX 7900 XTX) — huge, fast, but picky about how toys are arranged
- A **blue box** (AMD Radeon integrated) — smaller, shares space with your computer's brain
- A **green box** (Intel Arc Pro B70) — medium-sized, works differently than the others

**VulkanVM is a magical organizer** that lets all three boxes share toys perfectly, without ever losing anything or making a mess.

---

## The Three Problems It Solves

### Problem 1: The "Messy Room" Problem (AMD APU Fragmentation)

**What happens:** Imagine your red box has 100 compartments. You put a big toy in compartment 1-10, a medium toy in 20-25, a small toy in 30. Now you want to put in a GIANT toy that needs 20 compartments in a row. But there's no 20 empty compartments *in a row* anymore! The space is there, but it's all chopped up.

**This is called "fragmentation."** On AMD's special computer brain (Strix Halo APU), this happens ALL THE TIME and makes programs crash.

**VulkanVM's fix:** At the very start, VulkanVM says "I'm taking the WHOLE ROOM." It grabs 80% of all available space in one giant chunk. Then it uses a smart dividing system (like a perfectly organized closet) to hand out pieces. Because it never gives space back to the operating system, the room NEVER gets messy. The giant toy always fits.

---

### Problem 2: The "Different Languages" Problem (Multi-Vendor Sharing)

**What happens:** Your red box (AMD) speaks "AMD-ish," your blue box (also AMD but different) speaks "AMD-ish too," and your green box (Intel) speaks "Intel-ish." They can't directly hand toys to each other because they don't understand each other's labeling system.

**VulkanVM's fix:** VulkanVM is a **universal translator**. It knows the secret handshake for ALL three boxes:
- For AMD→AMD: Uses a special pass called "OPAQUE_FD" (Linux) or "OPAQUE_WIN32" (Windows)
- For AMD→Intel: Uses "DMA-BUF" (Linux) or "OPAQUE_WIN32" (Windows)
- For NVIDIA→anyone: Uses "D3D12_HEAP" (Windows) or "DMA-BUF" (Linux)

When you want to share a toy, VulkanVM:
1. Makes a **special dedicated copy** of that toy (so it has its very own label)
2. Translates the label into the other box's language
3. Hands it over — now BOTH boxes can see and use the SAME toy at the same time!

---

### Problem 3: The "Toy Box Full" Problem (VRAM Overflow)

**What happens:** You're playing with GIANT toys (AI models, huge textures). Your red box only holds 24 toys. You try to put in the 25th toy... **CRASH!** Program dies.

**VulkanVM's fix:** VulkanVM builds a **secret warehouse** in your computer's regular memory (system RAM). When the red box gets full:
1. VulkanVM quietly moves the least-used toy to the warehouse
2. Your program keeps running — it doesn't even know the toy moved!
3. When you need that toy again, VulkanVM brings it back from the warehouse

This happens automatically in the background. You just say "move this to the warehouse" and "bring it back" — VulkanVM handles the moving trucks (DMA engines) for you.

---

## How It Works (The Magic Inside)

### The Buddy System (How Space Gets Divided)

Imagine you have a 512-foot long hallway. You need to give people rooms of different sizes.

**VulkanVM's method (Buddy Allocator):**
- Need 64 feet? Cut the hallway in half (256), half again (128), half again (64) ✓
- Need 32 feet? Cut one 64 in half ✓
- When someone leaves, their room merges back with its "buddy" (the room next to it that was cut from the same parent)

**Why this is magic:** Rooms always merge back perfectly. No weird gaps. No fragmentation. Ever.

---

### The Three Boxes (Your Hardware)

| Your GPU | Vendor ID | What VulkanVM Calls It | Special Powers |
|----------|-----------|------------------------|----------------|
| AMD Radeon RX 7900 XTX | `0x1002` | "Red Box - Discrete" | 24 GB fast memory, owns its own warehouse |
| AMD Radeon Graphics (integrated) | `0x1002` | "Blue Box - Unified" | Shares system RAM, no separate warehouse needed |
| Intel Arc Pro B70 | `0x8086` | "Green Box - Discrete" | 16-24 GB, speaks Intel-ish |

---

## What You Can Actually Do With It

### 1. "Give Me Space For My AI Brain" (Basic Allocation)

```cpp
// You: "I need 64 MB for my neural network weights"
auto brainSpace = pool->allocateTensor(64 * 1024 * 1024);
// brainSpace->deviceAddress = the exact address your shader uses
// brainSpace->buffer = the handle for copy commands
```

### 2. "I Need A Loading Dock" (Staging Buffers)

```cpp
// You: "I need a place to load data from disk before sending to GPU"
auto loadingDock = pool->allocate({
    .size = 128_MiB,
    .memoryUsage = MemoryUsage::CpuToGpu,  // "I'll write, GPU reads"
    .mapped = true,                         // "Give me a pointer I can memcpy into"
    .name = "texture_upload"
});
// loadingDock->hostPtr is a regular C++ pointer — memcpy your PNG/JPG data here!
```

### 3. "Let My Friend Use This Too" (Cross-GPU Share)

```cpp
// On the RED box (master):
auto sharedToy = redPool.allocateDedicatedExportable(256_MiB, usage);
auto passport = redPool.exportMemory(*sharedToy, ExternalHandleType::OpaqueWin32);

// On the GREEN box (friend):
auto greenPassport = duplicateForImport(*passport);  // Make their own copy of the passport
auto greenToy = greenPool.importMemory(std::move(greenPassport), usage);
// Now BOTH boxes see the EXACT SAME DATA!
```

### 4. "My Toy Box Is Full — Use The Warehouse" (Offload)

```cpp
// Setup the warehouse (4 GB in system RAM)
OffloadConfig warehouse;
warehouse.hostShadowSize = 4_GiB;
pool->initializeOffload(warehouse);

// "Move this to the warehouse, I don't need it right now"
auto ticket = pool->offloadToHost(bigAllocation);
// ... do other work ...
pool->waitMigration(ticket);  // Wait for moving truck to finish

// "Bring it back, I need it now!"
pool->reloadToDevice(bigAllocation);
```

### 5. "Distribute This Model To All My Friends" (ModelHub)

**On your server (the library):**
```cpp
ModelHub library("/data/models");
library.start("0.0.0.0", 51010);
library.publish("my-org/llama-3b", "./model-files", "v1.0");
// Publishes: config.json, weights.safetensors, tokenizer/tokenizer.json
library.stop();
```

**On any computer (the reader):**
```cpp
// Downloads to ~/.cache/vvm/models/my-org/llama-3b/v1.0/
ModelHub::fetch("192.168.1.100:51010", "my-org/llama-3b", "./my-models", "v1.0");

// Then load into your GPU pool:
auto modelSpace = pool->allocate({ .size = modelSize, .memoryUsage = MemoryUsage::GpuOnly });
// Copy weights from ./my-models/weights.safetensors into modelSpace
```

---

## The Tensor Express (Unified Tensor Transport)

Now imagine you don't just want to move **one toy** — you want to move **entire organized collections** of toys with specific shapes, colors, and arrangements. That's what the **Tensor Transport** does for AI tensors.

### What Makes Tensors Special?

Regular memory copies don't care about **shape** (is it a 2D image? 3D volume? batch of 64 images?) or **layout** (is it NCHW or NHWC? tiled for tensor cores?).

The Tensor Transport knows:
- **Shape**: [Batch, Channels, Height, Width] or [Batch, Height, Width, Channels]
- **Data Type**: FP32, FP16, BF16, INT8, INT4, FP8
- **Layout**: NCHW (PyTorch default), NHWC (TensorFlow/TensorRT), Blocked (tiled for tensor cores)
- **Strides**: How to jump between elements in memory

### The Magic: "Just Move It"

```cpp
// "Move this 64MB FP16 tensor from GPU 0 to GPU 1, convert NHWC→NCHW on the way"
transport->copyWithLayoutConversion(srcTensor, dstTensor, MemoryLayout::Blocked);

// "Add up these gradients across 4 GPUs and give everyone the result"
transport->allReduce({gpu0_grad, gpu1_grad, gpu2_grad, gpu3_grad}, ReduceOp::Sum);

// "Send this 200MB model to the other computer, convert FP16→FP8 on the wire"
transport->sendTensor(hugeModel, "other-computer", [](bool ok) { /* done */ });
```

### The Transport Chooses the Best Road

| Road | When It's Used | Speed | CPU Involved? |
|------|---------------|-------|---------------|
| **P2P Express** | Same computer, different GPUs | 🚀 Fastest | No |
| **RDMA Super-Highway** | Different computers, RDMA NICs | 🚀🚀 Fastest | No |
| **Host-Staged Truck** | When roads are blocked | 🐢 Slow | Yes |
| **Network Ferry** | Different buildings | 🐢🐢 Slow | Yes |

The transport **automatically picks the fastest available road**. You just say "move this tensor" and it figures out the rest.

### The Three Musketeers of Collective Ops

| Operation | What It Does | Use Case |
|-----------|-------------|----------|
| **All-Reduce** | Sum/Avg/Min/Max across all GPUs, everyone gets result | Gradient sync in distributed training |
| **Broadcast** | One GPU shouts, everyone listens | Broadcast model weights |
| **All-Gather** | Everyone contributes a piece, everyone gets the whole puzzle | Gather distributed embeddings |
| **Reduce-Scatter** | Reduce + split pieces | Sharded optimizer states |

### Under the Hood (Simplified)

```cpp
// You just say:
transport->allReduce({gpu0_grad, gpu1_grad, gpu2_grad}, ReduceOp::Sum);

// Transport does ring all-reduce:
// Step 1: Each GPU sends 1/3 of its data to next GPU
// Step 2: Each GPU adds received chunk to its own
// Step 3: Repeat until everyone has sum of all 3
// Step 4: Each GPU broadcasts its 1/3 to others
// Result: All 3 GPUs have the FULL sum
```

### Cross-Computer: The Network Ferry

When GPUs are on different computers, the transport uses the **Network Ferry** (TCP or RDMA):

```
GPU A ──[P2P/RDMA]──→ NIC A ──[4MB chunks over TCP/RDMA]──→ NIC B ──[P2P/RDMA]──→ GPU B
```

**Key insight**: The transport picks the fastest available path automatically. If both computers have RDMA NICs, it uses the Super-Highway. If not, it falls back to the Network Ferry (TCP with 4MB chunks).

**How send/recv actually works today**: `sendTensor` exports your GPU memory and shouts the tensor's name to the other computer (`MsgTensorAnnounce`). `recvTensor` listens for that name, then **pulls** the VRAM over TCP in 4MB chunks straight into the destination GPU. Both tensors must have the same name — it's the "passport" that matches the sender and receiver.

```
 // Computer B (the sender)
 transport->sendTensor(sharedTensor, "192.168.1.10:51001#0", done);

 // Computer A (the receiver) — make a tensor with the SAME name, then receive:
 auto receiver = nodeA->allocateTensor(meta, 0);   // meta.name == sharedTensor name
 transport->recvTensor(*receiver, "192.168.1.11:51002#0", done);
```

### What's Still Being Built

| Feature | Status | What's Missing |
|---------|--------|----------------|
| P2P (same computer) | ✅ Done | — |
| Host-staged fallback | ✅ Done | — |
| Ring all-reduce | ✅ Done | — |
| Network send/recv (GPU⇄GPU over TCP) | ✅ Done | Verified by `tensor_network_test` |
| GPU-Direct RDMA (Linux) | 🔧 In progress | Wire up `ibv_reg_dmabuf_mr` |
| GPU-Direct RDMA (Windows) | 🔧 Planned | NDKPI implementation |

---

## The Super-Highway (GPU-Direct RDMA)

For **really fast sharing** (bypassing the CPU entirely), VulkanVM speaks the NIC's native language:

| Your GPU | The Secret Path |
|----------|-----------------|
| **NVIDIA** | `VK_NV_external_memory_rdma` → GPU gives NIC a direct map of its memory address |
| **AMD** | Export as Windows handle → `MapViewOfFile` → NIC sees the memory directly |
| **Intel** | Same as AMD — Windows handle → `MapViewOfFile` → NIC DMA |

This means: **GPU A writes, NIC reads directly from GPU A's memory, sends over wire, NIC writes directly into GPU B's memory.** CPU never touches the data. Used by supercomputers and AI clusters.

---

### Linux SoftRoCE (Software RoCE / `rxe`) — The "No Hardware Needed" Mode

**What if you don't have a fancy $10,000 NIC?** VulkanVM has a backup plan: **SoftRoCE** — it pretends your regular Ethernet cable (or even loopback `lo`) is an RDMA network card.

**It works on:** WSL2, VMs, CI runners, any Linux box with a kernel built with `CONFIG_RDMA_RXE=y`.

**Setup (WSL2 / Ubuntu 24.04):**
```bash
# 1. Build kernel with RDMA_RXE (or use distro kernel that includes it)
# 2. Create SoftRoCE links:
sudo rdma link add rxe0 type rxe netdev eth0   # primary interface
sudo rdma link add rxe1 type rxe netdev lo     # loopback for localhost tests
rdma link show  # verify ACTIVE state
ibv_devices     # should list rxe0, rxe1
```

**Result:** The `tensor_network_test` example passes with **real RDMA verbs over software**:
```
VerbsRdmaTransport initialized on device 'rxe0', RDMA listener port 51012
exportForRemote: RDMA host shadow registered for alloc 1
migrateFromRemote: pulled 16777216 bytes from 127.0.0.1:51012#0
  sendTensor: PASS
  recvTensor: PASS
  VRAM content verify on A: PASS
```

**Status:** ✅ End-to-end verified on WSL2 (custom kernel 6.18.40, `rxe0` + `rxe1`).
On native Linux with physical RNIC, the same code uses hardware RDMA.
Without SoftRoCE or hardware RNIC, it falls back to **host-staged TCP** (always works).

---

## The Shard Planner (New!) — "Who Sits Where?"

When you have a **huge model** (like Llama-70B) and a **cluster of GPUs**, you need to decide: which shard goes on which GPU? The **Shard Placement API** solves this automatically.

### What It Does

Imagine you have:
- Node A: 8 GB VRAM, 8 GB host offload
- Node B: 8 GB VRAM, 8 GB host offload
- Model: 3 shards × 2 GB each (layers 0-3, 4-7, 8-11)

The planner figures out the optimal seating chart:

```cpp
// 1. Describe the model
ModelManifest model;
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
cluster.reservedActivationBytes = 512_MiB;  // slack for KV cache

// 3. Policy
PlacementPolicy policy;
policy.allowHostOffload = true;    // spill to RAM if VRAM full
policy.preferContiguousLayers = true; // keep adjacent layers together
policy.packMode = PackDense;       // fill node-a first

// 4. Get the plan
PlacementPlan plan = ShardPlacer::plan(model, cluster, policy);
// plan.assignments = [ {blk.0-3 → node-a, DeviceLocal}, {blk.4-7 → node-a, DeviceLocal}, {blk.8-11 → node-b, DeviceLocal} ]
```

### Smart Features

| Feature | What It Means |
|---------|---------------|
| **Capacity-first** | Only places shards where they actually fit |
| **Host offload spill** | If VRAM full, automatically uses host RAM |
| **Contiguous layers** | Tries to keep layer 0-3 and 4-7 on same node |
| **Activation reserve** | Reserves per-node slack for KV cache |
| **Constraints** | `mustBeDeviceLocal` = "no offload allowed for this shard" |
| **Validation** | Catches duplicate shard IDs, empty clusters, oversized shards |
| **Best effort** | `bestEffort=true` places what fits, returns partial plan |

### Executing the Plan

```cpp
// On each node, run the executor
PlacementExecutor executor(nodeManager, modelHub);

ExecuteOptions opt;
opt.fetchIfMissing = true;   // pull from ModelHub if not cached
opt.verifyChecksum = true;   // SHA-256 verify after load

ExecuteResult result = executor.executeLocal(model, plan, opt);
// Idempotent: re-running skips already-loaded shards with same hash
// Transactional: if any shard fails, rolls back all on that node
```

### Test It

```bash
./build/tests/placement_test
# 10 pure-logic tests: simple fit, host offload, constraints, empty cluster,
# contiguous layers, ShardTooLarge, duplicate/empty shardId, activation reserve, bestEffort
```

---

## Sparse Memory (The "Infinite Room" Trick)

Imagine you want to reserve a **1 GIANT ROOM** (1 TB virtual address space) but only furnish **small corners** of it as needed.

```cpp
SparseVirtualMemoryPool magicRoom(device);
magicRoom.initialize(1_TiB, 32_MiB);  // 1 TB virtual, 32 MB pages

auto deed = magicRoom.reserveVirtual(256_GiB, usage);  // Reserve address range only
// ... later, when you actually need 64 GB at offset 64 GB ...
magicRoom.commit(deed, 64_GiB, 64_GiB, memoryFlags);
// Physical memory allocated ONLY for that 64 GB!
```

**Why this matters:** Huge virtual textures, massive embedding tables, anything where you want a giant address space but only pay for what you use.

---

## Does It Work On MY Computer?

### Windows
✅ **Yes!** Uses Ninja + Visual Studio DevCmd (avoids Windows SDK version hell). Batch scripts included: `build_only.bat`, `run_tests.bat`, `run_network_test.bat`, `run_multi_gpu_test.bat`.

### Linux
✅ **Yes!** Standard CMake + GCC/Clang. `./scripts/build.sh --tests`

### macOS (MoltenVK)
✅ **Yes!** Vulkan via MoltenVK translation layer.

### Your Hardware

| GPU Type | Works? | Notes |
|----------|--------|-------|
| RTX 4090 / 7900 XTX (24 GB) | ✅ Excellent | 512 MB blocks, 8-16 blocks |
| RTX 4070 / 7800 XT (12-16 GB) | ✅ Excellent | 256 MB blocks, 8 blocks |
| Arc A770 / A750 (16/8 GB) | ✅ Good | 256 MB blocks, 4-8 blocks |
| Laptop dGPU (4060M, 8 GB) | ✅ Good | 128 MB blocks, 4 blocks |
| **Strix Halo APU (96-128 GB unified)** | ✅ **BEST** | 1-2 GB blocks, 16 blocks — this is where it SHINES |
| Integrated (780M/890M, 2-8 GB shared) | ⚠️ Works but tight | 64 MB blocks, 4 blocks |
| Ancient GPU (Vulkan 1.0 only) | ❌ No | Needs Vulkan 1.3 + extensions |

---

## Installation = Zero Disruption

**This is a LIBRARY. Not a driver. Not a service. Not a kernel module.**

| What It Does | What It Doesn't Do |
|--------------|-------------------|
| Links into YOUR app like any `.dll`/`.so` | ❌ Touches kernel/drivers |
| Uses standard Vulkan API (`vkAllocateMemory`, `vkCreateBuffer`) | ❌ Requires admin/root |
| Creates its own memory blocks | ❌ Touches other processes |
| Optionally maps host memory | ❌ Changes GPU clocks/voltage |
| **Install = `cmake --install` → headers + lib in your prefix** | ❌ Installs background daemons |

**Your app just links it. That's it.**

---

## The Name: "Automaton" (Αὐτόματον)

> **Ancient Greek:** "Self-acting"
> 
> *Iliad 18.375:* "Twenty tripods he set against the wall... golden wheels... they moved of their own accord."
> 
> **Why it fits:**
> - **Vulkan** = Roman god of forge (Hephaestus = Greek)
> - **Automaton** = Self-moving, self-managing memory
> - **Three legs** = AMD, NVIDIA, Intel unified
> - **No human needed** — it just works across all gods (vendors)

---

## TL;DR For Your Boss

> **Vulkan Automaton** — A self-managing Vulkan memory pool that eliminates fragmentation on AMD APUs, enables zero-copy memory sharing across NVIDIA/AMD/Intel GPUs, provides demand-paged host offload, includes a Hugging Face–style model registry for distributing weights over TCP, and now adds **capacity-first shard placement** for multi-node model distribution. One header, one library, zero system changes.

---

## Recent "We Fixed That" List

- ✅ **Thread safety hardened** — budget checks and move assignment now properly mutex-protected
- ✅ **Unsafe APIs deprecated** — `madvise`/`mprotect` on GPU memory marked `[[deprecated]]` (they corrupt driver mappings)
- ✅ **Network transport hardened** — connection idle timeout (5 min default) prevents stuck connections
- ✅ **Buddy allocator robust** — OOM-safe, self-validating, graceful degradation on split failure
- ✅ **Cross-GPU fallback** — when driver refuses direct import, auto-falls back to 4 MB chunked host-staged copy
- ✅ **Vendor-specific RDMA** — NVIDIA/AMD/Intel GPU-direct registration paths implemented
- ✅ **Sparse virtual memory** — 1 TB virtual / 32 MB pages working and tested
- ✅ **P0 Audit fixes** — UniqueAllocation RAII (private ctor, `make()` factory), OffloadManager mutex-guarded, all logging converted to `fmt` style `{}`
- ✅ **SoftRoCE end-to-end** — Linux `rxe` module verified on WSL2 (kernel 6.18.40, `rxe0`/`rxe1`)
- ✅ **Shard Placement API** — capacity-first bin-packing with executor + rollback, 10 unit tests
- ✅ **ABI fix** — `VVM_NETWORK_HAS_VERBS` propagated as PUBLIC CMake definition to prevent `std::optional` layout mismatch
- ✅ **PyTorch C++ extension** — `vulkanvm_torch` with full Python bindings for pool, offload, external memory, ModelHub, shard placement
- ✅ **ONNX Runtime integration** — `vulkanvm_onnx` with `VulkanVMExecutionProvider`, NumPy interop, ModelHub for ONNX models
- ✅ **SoftRoCE persistence** — `scripts/softroce_persist.sh` (Linux) + `scripts/softroce_persist.ps1` (Windows) for auto-creation on boot

---

## PyTorch & ONNX — Use VulkanVM from Python

### PyTorch (`vulkanvm_torch`)

```python
import vulkanvm_torch as vvm

# Create pool
pool = vvm.UnifiedMemoryPool.create(dev_config, pool_config)

# Allocate tensor
alloc = pool.allocate_tensor(64 * 1024 * 1024, "weights")

# Offload/reload
pool.initialize_offload(4_GB, transfer_queue, transfer_queue_family)
op = pool.offload_to_host(alloc)
pool.wait_migration(op)
pool.reload_to_device(alloc)

# Cross-GPU share
exported = pool.export_memory(alloc, vvm.ExternalHandleType.OpaqueWin32)
peer_alloc = peer_pool.import_memory(exported, usage)

# ModelHub
hub = vvm.ModelHub("/data/models")
hub.publish("my-org/llama-3b", "./files", "v1")
vvm.ModelHub.fetch("server:51010", "my-org/llama-3b", "./models", "v1")

# Shard Placement
plan = vvm.ShardPlacer.plan(model, cluster, policy)
```

### ONNX Runtime (`vulkanvm_onnx`)

```python
import vulkanvm_onnx as vvm
import numpy as np

provider = vvm.VulkanVMExecutionProvider(
    vvm.VulkanVMExecutionProviderConfig(pool_size=2_GB, host_shadow_size=4_GB)
)

alloc = provider.allocate_tensor([1, 3, 224, 224], vvm.TensorElementType.FLOAT, "input")
provider.upload_tensor(alloc, input_np, input_np.nbytes)
provider.download_tensor(alloc, output_np, output_np.nbytes)

# Distribute ONNX models via ModelHub
provider.publish_onnx_model("my-org/resnet50", "./resnet50.onnx", "v1")
provider.fetch_onnx_model("server:51010", "my-org/resnet50", "./models", "v1")

# NumPy interop
alloc = vvm.create_tensor_from_numpy(input_np, provider, "input")
result_np = vvm.tensor_to_numpy(alloc, provider, [1, 1000], vvm.TensorElementType.FLOAT)
```

### Build

```bash
# PyTorch
cmake -B build -DVVM_BUILD_PYTORCH=ON -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")

# ONNX
cmake -B build -DVVM_BUILD_ONNX=ON
```

---

## SoftRoCE Persistence — Keep RDMA Links Across Reboots

WSL2 shuts down the VM between sessions, so SoftRoCE links (`rxe0`, `rxe1`) disappear. Use the helper scripts:

**Linux:**
```bash
sudo ./scripts/softroce_persist.sh create   # Create links now
sudo ./scripts/softroce_persist.sh install  # Install as systemd service (auto on boot)
./scripts/softroce_persist.sh status        # Show status
```

**Windows (PowerShell):**
```powershell
.\scripts\softroce_persist.ps1 -Create
.\scripts\softroce_persist.ps1 -Install
.\scripts\softroce_persist.ps1 -Status
```

Creates `rxe0` on `eth0` and `rxe1` on `lo`, waits for `ACTIVE`, optionally installs systemd service.

---

## Need Help?

- **README.md** — Full API reference, building, cross-vendor matrix, shard placement API
- **examples/** — Working demos: `network_test`, `model_registry_test`, `tensor_compute`, `offload_test`, `multi_gpu_test`, `tensor_network_test`
- **tests/** — Unit tests for buddy allocator, external handles, sparse, basic pool, **placement_test**
- **GitHub Issues** — Bug reports, feature requests

*Built with ❤️ by the Automaton team — making GPUs play nice since 2024.*
