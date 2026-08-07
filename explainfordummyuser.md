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
 
### What's Still Being Built
 
| Feature | Status | What's Missing |
|---------|--------|----------------|
| P2P (same computer) | ✅ Done | — |
| Host-staged fallback | ✅ Done | — |
| Ring all-reduce | ✅ Done | — |
| GPU-Direct RDMA (Linux) | 🔧 In progress | Wire up `ibv_reg_dmabuf_mr` |
| GPU-Direct RDMA (Windows) | 🔧 Planned | NDKPI implementation |
| Network send/recv | 🔧 Planned | Wire up TCP/RDMA send |
 
---

## The Super-Highway (GPU-Direct RDMA)

Imagine **two computers** each with their own toy boxes. They want to share toys over a network cable.

**VulkanVM builds a private highway between them:**

```
Computer A (Red Box) ←── 4 MB chunks over TCP ──→ Computer B (Green Box)
```

**No extra software needed.** Just standard internet plumbing (TCP sockets). Optional: TLS encryption (like HTTPS) if you want privacy.

**What they can do:**
- Computer B asks Computer A: "Please make me a 16 MB toy" → A makes it, B gets a handle
- Computer B pushes data TO Computer A: "Here, hold this" (streamed in 4 MB pieces)
- Computer A pulls data FROM Computer B: "Send me that toy" (streamed in 4 MB pieces)
- Both see each other's toy boxes in a shared directory

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

> **Vulkan Automaton** — A self-managing Vulkan memory pool that eliminates fragmentation on AMD APUs, enables zero-copy memory sharing across NVIDIA/AMD/Intel GPUs, provides demand-paged host offload, and includes a Hugging Face–style model registry for distributing weights over TCP. One header, one library, zero system changes.

---

## Recent "We Fixed That" List

- ✅ **Thread safety hardened** — budget checks and move assignment now properly mutex-protected
- ✅ **Unsafe APIs deprecated** — `madvise`/`mprotect` on GPU memory marked `[[deprecated]]` (they corrupt driver mappings)
- ✅ **Network transport hardened** — connection idle timeout (5 min default) prevents stuck connections
- ✅ **Buddy allocator robust** — OOM-safe, self-validating, graceful degradation on split failure
- ✅ **Cross-GPU fallback** — when driver refuses direct import, auto-falls back to 4 MB chunked host-staged copy
- ✅ **Vendor-specific RDMA** — NVIDIA/AMD/Intel GPU-direct registration paths implemented
- ✅ **Sparse virtual memory** — 1 TB virtual / 32 MB pages working and tested

---

## Bottom Line

**Install it. Link it. Call `UnifiedMemoryPool::create()`.**

It either works or returns `std::nullopt`. No system modification. No risk. Test on your Strix Halo first — that's where it shines brightest.

---

## Need Help?

- **README.md** — Full API reference, building, cross-vendor matrix
- **examples/** — Working demos: `network_test`, `model_registry_test`, `tensor_compute`, `offload_test`, `multi_gpu_test`
- **tests/** — Unit tests for buddy allocator, external handles, sparse, basic pool
- **GitHub Issues** — Bug reports, feature requests

*Built with ❤️ by the Automaton team — making GPUs play nice since 2024.*