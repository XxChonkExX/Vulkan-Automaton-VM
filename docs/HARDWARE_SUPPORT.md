# Hardware & Platform Support Matrix

Status tiers (be honest — this table is a promise):

- **Tier 1 — Verified**: run end-to-end on real hardware, results published in
  this repo (benchmarks, test logs, or training runs).
- **Tier 2 — Compile-tested**: builds against the relevant SDK/driver; not
  exercised on the target hardware.
- **Tier 3 — Designed / experimental**: the code path exists; never run on
  the intended hardware.
- **Untested**: no data.

## GPU backends

| Backend | Tier | Notes |
|---|---|---|
| AMD Radeon (RDNA3, Vulkan) | **1 — Verified** | RX 7900 XTX: llama.cpp Chonk integration at parity (docs/inference_benchmarks.md), pool benchmarks |
| AMD HIP interop (DMA-BUF -> HIP) | **1 — Verified** | AMD Strix Halo (gfx1151): full 131K/196K long-context LoRA training runs (OPTIMIZATION_LOG.md) |
| AMD Vulkan (RDNA4/RDNA5) | 3 — Designed | Same code path as RDNA3; no RDNA4/5 hardware exercised |
| Intel Arc Pro (Battlemage, Vulkan) | **1 — Verified** | Arc Pro B70: llama.cpp Chonk integration verified (docs/inference_benchmarks.md). Driver note: stock Pro driver 8861 was 24x broken for compute; consumer WHQL 8974+ required |
| Intel Level Zero GPU-direct | 2 — Compile-tested | Builds against Level Zero SDK; no cross-vendor copy exercised |
| NVIDIA (CUDA path) | 3 — Designed | No hardware; community help wanted (see README notice) |
| NVIDIA (Vulkan path) | 3 — Designed | ggml-vulkan covers NVIDIA generally; VulkanVM pool integration untested on NVIDIA |
| Tenstorrent | 3 — Designed | Vulkan ICD built and submitted to TT; loosely supported |

## Android

| GPU | Tier | Notes |
|---|---|---|
| Qualcomm Adreno | **1 — Verified** | Galaxy S24+ (SM-S926U, API 36, Adreno): AHardwareBuffer import end-to-end, device address valid for shaders |
| ARM Mali | Untested | High-risk area: driver behavior differs significantly from Adreno |
| Imagination PowerVR | Untested | |
| Samsung Xclipse | Untested | |

## Transports

| Transport | Tier | Notes |
|---|---|---|
| TCP (host-staged) | **1 — Verified** | Windows (Arc B70) <-> Linux (Strix Halo) cross-machine tensor transport (docs/cross_machine_gpu_sharing.md) |
| RDMA/verbs (SoftRoCE) | **1 — Verified** | WSL2 custom kernel 6.18.40.1-wsl-rxe+: loopback 1.9 GiB/s, cross-machine cluster join (docs/cross_machine_gpu_sharing.md) |
| RDMA/verbs (hardware RoCE/iWARP) | 3 — Designed | Requires RDMA NICs; not exercised |
| UCX | 2 — Compile-tested | Builds with UCX SDK; no multi-node UCX run |
| Windows Network Direct (ND) | 2 — Compile-tested | |
| NDK transport (Android) | 2 — Compile-tested | |

## External memory

| Path | Tier | Notes |
|---|---|---|
| Vulkan -> DMA-BUF -> HIP (Linux) | **1 — Verified** | Strix Halo training pipeline |
| Vulkan -> OpaqueFd -> HIP import (Linux) | **1 — Verified** | Chonk allocator (PyTorch pluggable allocator) |
| Vulkan -> Win32 -> HIP import (Windows) | 3 — Designed | Windows uses different handle semantics; untested |
| Vulkan <-> AHardwareBuffer (Android) | **1 — Verified** | Galaxy S24+ (tests/chonk_slab_test covers the allocator; android_test covers the import) |
| CUDA external-memory import | 3 — Designed | |

## Test coverage tiers

| Suite | Runs on | Status |
|---|---|---|
| `buddy_test` | any CPU | **1 — Verified** (Windows, Linux) |
| `chonk_slab_test` | any CPU | **1 — Verified** (Windows; 2x100k-op fuzz + boundaries) |
| `placement_test` | any CPU | pure logic, no GPU |
| `chonk_slab_test 1000000` | any CPU | nightly-scale fuzz (slow: O(n^2) first-fit) |
| Android AHardwareBuffer test | device | **1 — Verified** (Galaxy S24+) |
| llama.cpp Chonk integration | 2x dGPU | **1 — Verified** (XTX + B70, docs/inference_benchmarks.md) |

## Adding a row

If you run VulkanVM on hardware listed above (or something new), open a PR
with: hardware name, driver version, which test/benchmark you ran, and the
result. Update the tier only with evidence.
