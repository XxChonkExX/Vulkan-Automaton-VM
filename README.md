# VulkanVM — Cross-Vendor GPU Memory & Transport Infrastructure

**VulkanVM** unifies GPU memory management and data movement behind Vulkan
external-memory primitives, then lets the frameworks you already use consume
that memory directly. One pool, one allocator family, every vendor —
AMD, Intel, (NVIDIA path designed), Tenstorrent-ICD, Android.

**Version**: 0.3.0-dev

> **New here?** Start with [explainfordummyuser.md](explainfordummyuser.md) —
> *"the relay race tour"*: what problem this solves and how it all fits
> together, written for humans first. Then come back for the details below.

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
| [docs/LINUX_TEST_RESULTS_2026-08-25.md](docs/LINUX_TEST_RESULTS_2026-08-25.md) | Native Linux campaign: cross-vendor DMA-BUF P2P verified, verbs/RDMA fixes, full evidence |

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
[.github/workflows/ci.yml](.github/workflows/ci.yml). Hardware-run evidence and
per-suite tiers live in [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md);
the latest native-Linux campaign is documented in
[docs/LINUX_TEST_RESULTS_2026-08-25.md](docs/LINUX_TEST_RESULTS_2026-08-25.md).

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

## Licenses & credits

VulkanVM stands on a lot of shoulders. This project is an independent
implementation that borrows ideas, specifications, drivers, tools and
infrastructure from many people and organizations — all used under their own
licenses; nothing here is affiliated with or endorsed by them.

### Foundations we build on

| Project / Organization | What we use it for |
|---|---|
| **Khronos Group** — Vulkan | The entire foundation: external memory, sparse residency, device addresses, SPIR-V compute. Vulkan headers + loader power every line of this repo. |
| **Mesa3D** — RADV (AMD), ANV (Intel), lavapipe | Our primary Linux test drivers. Every Tier-1 verification row in [docs/HARDWARE_SUPPORT.md](docs/HARDWARE_SUPPORT.md) exists because Mesa's open-source Vulkan drivers let us exercise cross-vendor paths on commodity hardware. |
| **Linux kernel** — DMA-BUF, RDMA/rxe, drm | Cross-device memory sharing (DMA-BUF), SoftRoCE RDMA transport (`rdma_rxe`), and the platform substrate. Kernel-side bugs we hit are documented upstream-first in our test docs. |
| **rdma-core** (libibverbs, librdmacm) | The verbs/RDMA transport layer — connection management, memory regions, `ibv_reg_dmabuf_mr`. |
| **glslang / spirv-tools** (Khronos) | Layout-conversion compute shaders compiled at build time. |

### Integrations & optional dependencies

| Project | Role |
|---|---|
| **PyTorch** | Pluggable allocator integration + autograd ops ([train_qwen_chonk.py](train_qwen_chonk.py)) |
| **ONNX Runtime** | Execution-provider integration |
| **llama.cpp** | Inference-consumption verification ([docs/inference_benchmarks.md](docs/inference_benchmarks.md)) |
| **pybind11** | Python bindings glue |
| **OpenSSL** | Optional TLS for the network layer |
| **UCX** (optional) | Alternative transport backend |
| **Android NDK / AHardwareBuffer** | Android platform path (verified on Adreno) |

### Hardware that made the honest tiers possible

- **AMD** — Radeon RX 7900 XTX (RDNA3) and Strix Halo unified-memory APUs:
  primary verification platforms for Vulkan, HIP/DMA-BUF interop.
- **Intel** — Battlemage G31: second vendor for cross-vendor P2P evidence;
  Battlemage's open driver work makes multi-vendor rigs affordable.
- **Google/Qualcomm** (Android S24+, Adreno): AHardwareBuffer verification.

These companies' driver teams (especially Mesa's RADV/ANV contributors and
AMD's ROCm/PAL folks) do the unglamorous work that lets a homebrew project
like this one exist at all. Bugs we find in their layers get documented with
repros so they can fix them — see the Phase sections of
[docs/LINUX_TEST_RESULTS_2026-08-25.md](docs/LINUX_TEST_RESULTS_2026-08-25.md)
for current examples.

See [LICENSE](LICENSE) for this project's license.

## Why "Chonk Buffer"?

The core memory pool is affectionately nicknamed the **Chonk Buffer**
("chunk" - a contiguous block of memory). One big chonk of VRAM, zero
fragmentation, every framework welcome. The name stuck; the architecture
earned it. The keeper of this library is also known as ChonkE.

## Changelog

Version history and the (extensive) experimental changelog live in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) and the git history. v0.3.0-dev:
layer separation (Core/Transport/Compute/Integrations), Chonk allocator
exact-fit + Chonk Chunks, llama.cpp integration at parity, Android
AHardwareBuffer verified, audit-driven correctness pass. Native-Linux
campaign (2026-08-25): cross-vendor P2P + verbs/RDMA verification, RDMA
teardown hang root-caused and fixed, ANV 26.0.3 import crash characterized,
validation-layer VUID cleanup — full evidence in
[docs/LINUX_TEST_RESULTS_2026-08-25.md](docs/LINUX_TEST_RESULTS_2026-08-25.md).
