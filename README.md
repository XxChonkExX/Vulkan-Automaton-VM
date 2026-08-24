# VulkanVM — Cross-Vendor GPU Memory & Transport Infrastructure

**VulkanVM** unifies GPU memory management and data movement behind Vulkan
external-memory primitives, then lets the frameworks you already use consume
that memory directly. One pool, one allocator family, every vendor —
AMD, Intel, (NVIDIA path designed), Tenstorrent-ICD, Android.

**Version**: 0.3.0-dev

---

## Architecture

```
VulkanVM
│
├── Core  (VulkanVM::Core)              # include <vulkan_vm/vulkan_vm.hpp>
│   ├── Chonk Buffer — unified GPU memory pool (buddy allocator, exact-fit)
│   ├── External memory — DMA-BUF / Win32 / OpaqueFd / AHardwareBuffer
│   ├── Multi-GPU allocation & cross-vendor sharing
│   ├── Host staging & DMA offload/reload
│   ├── Sparse virtual memory
│   └── Shard placement planning (pure logic)
│
├── Transport  (VulkanVM::Transport)    # include <vulkan_vm/transport.hpp>
│   ├── TCP · RDMA (verbs/SoftRoCE) · UCX · Windows ND · Android NDK
│   ├── Cluster client/server, multi-node pools
│   └── Cluster-aware placement execution
│
├── Compute  (VulkanVM::Compute)        # include <vulkan_vm/tensor_transport.hpp>
│   ├── Tensor operations & collectives
│   └── Layout conversion (SPIR-V compute)
│
└── Integrations
    ├── PyTorch — Chonk pluggable allocator + parameter binding + autograd ops
    ├── ONNX Runtime — execution provider
    └── Android — AHardwareBuffer platform backend (Core), NDK transport
```

**Dependency direction**: Integrations → {Compute, Core}; Compute → Core
(collectives add Transport); Transport → Core. **Core depends on nothing but
Vulkan** — the Chonk Buffer builds and ships standalone.

---

## Why: the relay race

A GPU workload is a relay race, and on a normal system your data keeps
getting pulled off the track. Weights live in VRAM, activations in host RAM,
optimizer states somewhere else — every handoff between them is a penalty
lap: copy to staging, copy from staging, wait, repeat. Multi-GPU makes it
worse: each vendor hands you a different baton, and the batons don't fit each
other's hands.

VulkanVM puts the entire race on one track. The **Chonk Buffer** is one
contiguous, fragmentation-free memory arena that every runner — training
framework, inference engine, second GPU, the network — reads from and writes
to directly. External memory means the baton never changes hands: the same
physical memory *is* the Vulkan buffer, *is* the HIP allocation, *is* the
PyTorch tensor, *is* the Android graphic buffer. When a second machine joins
the relay, the transport layer hands the baton across the network — same
track, next lane.

That is the whole idea: **allocate once, alias everywhere, move nothing.**

---

## Quick start

### Use the Chonk Buffer standalone (core only)

```bash
cmake -S . -B build -DVVM_BUILD_NETWORK=OFF -DVVM_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build -R "buddy_test|chonk_slab_test"
```

```cpp
#include <vulkan_vm/vulkan_vm.hpp>

vvm::DeviceConfig dev = /* enumerate + pick */;
auto pool = vvm::UnifiedMemoryPool::create(dev, vvm::PoolConfig::forDevice(dev.physicalDevice));
auto a = pool->allocateTensor(512_MB);   // fragmentation-free, device-address capable
```

### PyTorch training on the Chonk Buffer

```python
import vulkanvm_torch as vvm
vvm.init()                                  # instance/device/pool
torch.cuda.memory.change_current_allocator(vvm.allocator())  # or pluggable ABI
```

See [train_qwen_chonk.py](train_qwen_chonk.py) — Qwen 27B/40B long-context
LoRA training with **everything** (weights, optimizer, KV cache, activations)
in the pool on AMD Strix Halo (128 GB unified).

### Inference with the Chonk Buffer inside llama.cpp

Verified at parity with stock llama.cpp across dual vendors
([docs/inference_benchmarks.md](docs/inference_benchmarks.md)):
Qwen3.6-40B Q4_K_M, 21.5 t/s decode pooled across RX 7900 XTX + Arc Pro B70.

---

## Hardware support (honest tiers)

| Backend | Tier |
|---|---|
| AMD RDNA3 Vulkan (RX 7900 XTX) | **1 — Verified** |
| AMD HIP interop / DMA-BUF (Strix Halo) | **1 — Verified** |
| Intel Arc Pro B70 (Battlemage Vulkan) | **1 — Verified** |
| Android Adreno AHardwareBuffer (S24+) | **1 — Verified** |
| Intel Level Zero GPU-direct | 2 — Compile-tested |
| UCX · Windows ND · NDK transport | 2 — Compile-tested |
| NVIDIA (CUDA/Vulkan) · Tenstorrent | 3 — Designed |
| Mali · PowerVR · Xclipse | Untested |

Full evidence-linked matrix: [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md)

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/LIFETIME_CONTRACT.md](docs/LIFETIME_CONTRACT.md) | **Normative** ownership/lifetime rules (pool, allocations, external handles, PyTorch contract) |
| [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md) | Tiered hardware/platform support matrix |
| [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) | Network trust model, defenses, hardening roadmap |
| [docs/inference_benchmarks.md](docs/inference_benchmarks.md) | llama.cpp Chonk integration: parity benchmarks + debugging ladder |
| [docs/cross_machine_gpu_sharing.md](docs/cross_machine_gpu_sharing.md) | Cross-machine GPU sharing setup (TCP/RDMA) |
| [docs/X2_AMD_RDMA_SETUP.md](docs/X2_AMD_RDMA_SETUP.md) | AMD RDMA test rig setup |
| [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) | Research history: 131K/196K long-context training stabilization |
| [explainfordummyuser.md](explainfordummyuser.md) | The relay-race tour of everything above, for humans |
| [docs/AUDIT_NOTES_2026-08-15.md](docs/AUDIT_NOTES_2026-08-15.md) | External audit response + sync notes |

---

## Testing

```bash
# CPU-only regression suites (no GPU needed):
ctest --test-dir build -R "buddy_test|chonk_slab_test|placement_test"

# Slab allocator fuzz (audit Priority 3): boundaries, double-free,
# overlap tracking, invariants after every op:
./build/tests/chonk_slab_test 100000        # 1M for nightly runs

# Autograd numerical validation (needs torch + the extension):
python -m pytest python/vulkanvm_torch/test_autograd_numerics.py -v
```

CI runs the compile matrix (Linux GCC/Clang, Windows MSVC, Android NDK) plus
the CPU-only suites on every push —
[.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## Status & honesty

This is a serious experimental systems project — not a polished universal GPU
framework. The honest list:

- NVIDIA: designed, not validated (no hardware — help wanted)
- Android: verified on Adreno only; Mali/PowerVR/Xclipse untested
- Custom autograd: Vulkan compute bridge not fully integrated; ops execute
  through ATen; numerical suite in progress
- ONNX provider: CPU fallback path
- Network layer: wire parser hardened; system-level hardening roadmap in
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md)

What we claim, we test. What we haven't tested, we say.

## License & credits

See [LICENSE](LICENSE) and [README section on third-party licenses](#third-party-licenses--credits).
Built homebrew with love — local power. — Mike/ChonkE

---

## Building

```bash
# Core-only (Chonk Buffer standalone):
cmake -S . -B build -DVVM_BUILD_NETWORK=OFF -DVVM_BUILD_TESTS=ON
cmake --build build

# Full stack (adds transport):
cmake -S . -B build -DVVM_BUILD_NETWORK=ON -DVVM_BUILD_TESTS=ON
cmake --build build

# Windows: -G "Visual Studio 17 2022" -A x64 (or Ninja + vcvars)
# Android: -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a
```

Requirements: CMake >= 3.19, C++20, Vulkan SDK. Network layer adds
OpenSSL (optional TLS), libibverbs (optional RDMA), UCX (optional).

## Why "Chonk Buffer"?

The core memory pool is affectionately nicknamed the **Chonk Buffer**
("chunk" - a contiguous block of memory). One big chonk of VRAM, zero
fragmentation, every framework welcome. The name stuck; the architecture
earned it.

## Licenses

See [LICENSE](LICENSE). Third-party credits: Vulkan SDK (Khronos),
pybind11, OpenSSL, libibverbs/rdma-core, UCX - all used under their own
licenses; this project is an independent implementation.

## Changelog

Version history and the (extensive) experimental changelog live in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) and the git history. v0.3.0-dev:
layer separation (Core/Transport/Compute/Integrations), Chonk allocator
exact-fit + Chonk Chunks, llama.cpp integration at parity, Android
AHardwareBuffer verified, audit-driven correctness pass.
