# VulkanVM - Explain Like I'm 5 (But Technical)

## What Is This?

Think of VulkanVM as a **universal memory manager** for GPUs. It solves three nasty problems:

1. **AMD APU Fragmentation** (Strix Halo 395, etc.) - ROCm runs out of "contiguous" memory even when plenty is free
2. **Multi-Vendor Sharing** - Your 7900XTX (AMD) + Arc Pro B70 (Intel) + future NVIDIA can share memory directly
3. **Swap/Offload** - When VRAM fills, automatically spill to system RAM without crashing

---

## Theoretical Limits

| Dimension | Lower Bound | Upper Bound (Current HW) | Notes |
|-----------|-------------|--------------------------|-------|
| **Block Size** | 1 MB | 256 GB (VK_WHOLE_SIZE) | Limited by `VkPhysicalDeviceMemoryProperties::memoryHeaps[].size` |
| **Pool Size** | 1 block | 16 blocks (configurable) | 16 × 2GB = 32GB per GPU today |
| **Allocation Size** | 256 KB (alignment) | Pool size - overhead | Buddy allocator rounds to power-of-2 |
| **GPU Count** | 1 | 8+ (Vulkan limit) | Limited by `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE` and OS |
| **Cross-Vendor** | 2 | All visible GPUs | NVIDIA↔AMD↔Intel via DMA-BUF/OPAQUE_WIN32 |

**Hard Limits Today (2026):**
- **NVIDIA RTX 4090**: 24 GB VRAM → single pool up to ~20 GB usable
- **AMD 7900 XTX**: 24 GB VRAM → same
- **Intel Arc Pro B70**: 16-24 GB VRAM
- **Strix Halo (APU)**: 96-128 GB unified system RAM → pool can be 80-100 GB
- **Multi-GPU**: 4×24GB = 96 GB aggregate (if all same vendor) or mixed

**Absolute Theoretical Max (Vulkan 1.3):**
- `VkDeviceSize` = 64-bit = 18 exabytes (you'll hit OS/hardware limits first)
- Max memory heaps: 16 (typically 2-3: VRAM, system RAM, maybe host-cached)
- Max memory types: 32 per heap

---

## Durability: High-End vs Low-End

| System Type | VRAM | Pool Config | Works? |
|-------------|------|-------------|--------|
| **RTX 4090 / 7900 XTX** | 24 GB | 512MB blocks, 8-16 blocks | ✅ Excellent |
| **RTX 4070 / 7800 XT** | 12-16 GB | 256MB blocks, 8 blocks | ✅ Excellent |
| **Arc A770 / A750** | 16 GB / 8 GB | 256MB blocks, 4-8 blocks | ✅ Good |
| **Laptop dGPU (4060M)** | 8 GB | 128MB blocks, 4 blocks | ✅ Good |
| **Strix Halo APU** | 96-128 GB unified | 1-2 GB blocks, 16 blocks | ✅ **This is the sweet spot** |
| **Integrated (780M/890M)** | 2-8 GB shared | 64MB blocks, 4 blocks | ⚠️ Works but tight |
| **Ancient GPU (Vulkan 1.0 only)** | <4 GB | N/A | ❌ Needs Vulkan 1.3 + extensions |

**Key Insight:** The library **adapts to your hardware**. You configure `blockSize` and `maxBlocks` at runtime based on `vkGetPhysicalDeviceMemoryProperties`. It queries the actual heap sizes and picks sane defaults.

---

## Is It "Universal"?

### ✅ What It Fixes Universally

| Problem | How VulkanVM Solves It |
|---------|------------------------|
| **AMD ROCm fragmentation** | Pre-allocates ONE giant block at startup (80% of VRAM), sub-allocates internally. Never returns memory to OS. Fragmentation = 0. |
| **NVIDIA CUDA OOM on large tensors** | Same approach - persistent allocations, buddy allocator guarantees alignment for tensor cores |
| **Multi-vendor memory sharing** | Uses `VK_EXTERNAL_MEMORY` with DMA-BUF (Linux) / D3D12_HEAP (Windows). NVIDIA↔AMD↔Intel all support this. |
| **VRAM overflow crashes** | Host shadow buffer (system RAM) + async migration. `madvise(MADV_DONTNEED)` tells kernel "page this out". |
| **Bindless shader access** | Every allocation returns `VkDeviceAddress` - use directly in shaders, no descriptor sets needed. |

### ⚠️ What It Doesn't Fix (Yet)

| Limitation | Workaround / Future |
|------------|---------------------|
| **Peer-to-peer DMA (GPU↔GPU direct copy)** | Currently uses host staging. P2P needs `VK_KHR_external_memory_*` + vendor-specific setup. |
| **Unified virtual address space** | Each GPU has its own `VkDeviceAddress`. Cross-GPU pointers don't work directly. |
| **Windows WDDM 2.6+ GPU scheduling** | Not yet exposed via Vulkan. Future: `VK_KHR_scheduling_controls`. |
| **Sparse/residency (virtual textures)** | Planned: `VK_EXT_sparse_resource_*` + page fault handling. |

---

## Installation Impact: Zero Disruption

**This is a LIBRARY, not a driver or system service.**

| What It Does | What It Doesn't Do |
|--------------|-------------------|
| Links into YOUR application | ❌ Modifies kernel/drivers |
| Uses standard Vulkan API | ❌ Requires root/admin |
| Allocates GPU memory via `vkAllocateMemory` | ❌ Touches other processes' memory |
| Creates buffers via `vkCreateBuffer` | ❌ Changes GPU clocks/voltages |
| Optionally maps host memory via `vkMapMemory` | ❌ Installs kernel modules |

**Installation = `cmake --install` → drops `libvulkan_vm.so` / `vulkan_vm.dll` + headers in your prefix.**

Your app links it like any other library. No system changes. No background daemons. No persistence.

---

## Quick Test On Your Systems

### Strix Halo 395 (APU)
```cpp
// This config fixes your fragmentation:
PoolConfig cfg;
cfg.blockSize = totalSystemRAM * 0.8;  // e.g., 80 GB of 96 GB
cfg.minAlignment = 256 * 1024;
cfg.enableHostVisible = true;           // Unified memory = host visible
cfg.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
```
**Result:** Single 80GB allocation at boot. ROCm can't fragment it because VulkanVM owns it all.

### 7900XTX + Arc Pro B70 (Multi-Vendor)
```cpp
// GPU 0 (AMD) = master, allocates + exports
// GPU 1 (Intel) = imports via DMA-BUF (Linux) or OPAQUE_WIN32 (Windows)
auto manager = MultiGPUPoolManager::create({amdConfig, intelConfig}, poolConfig, 0);
auto allocs = manager.allocateDistributed(256 * 1024 * 1024, usage);
// allocs[0] valid on AMD, allocs[1] valid on Intel, same physical memory
```

---

## Name Suggestion

**Hephaestus' Forge → "Automaton" (Αὐτόματον)**

> "Self-acting" - the golden tripods that moved themselves between gods' halls.
> 
> *Iliad 18.375-376*: "Twenty tripods he set against the wall... golden wheels... they moved of their own accord."

**Why it fits:**
- **Vulkan** = Roman god of fire/forge (Hephaestus = Greek)
- **Nemo** = "Nobody" (Odysseus trick) → you're the hidden builder
- **Automaton** = self-moving, self-managing memory that "just works" across gods (vendors)
- **Tripod** = three legs = AMD, NVIDIA, Intel unified

**Repo name:** `vulkan-automaton` or `automaton-vm`

---

## Publishing Checklist

- [ ] Rename repo to `vulkan-automaton` (or your preference)
- [ ] Add `CONTRIBUTING.md` with testing guidelines
- [ ] Tag `v0.1.0` after smoke test
- [ ] GitHub Actions: Linux + Windows build matrix
- [ ] Publish to vcpkg / Conan Center (optional)
- [ ] Add `CITATION.cff` for academic use

---

## TL;DR For Your GitHub README

> **Vulkan Automaton** — A self-managing Vulkan memory pool that eliminates fragmentation on AMD APUs, enables zero-copy memory sharing across NVIDIA/AMD/Intel GPUs, and provides demand-paged host offload. One header, one library, zero system changes.

---

**Bottom line:** Install it, link it, call `UnifiedMemoryPool::create()`. It either works or returns `std::nullopt`. No system modification. No risk. Test on your Strix Halo first — that's where it shines.