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
| AMD HIP interop / DMA-BUF (Strix Halo) | **1 — Verified** | AMD Strix Halo (gfx1151): full 131K/196K long-context LoRA training runs (OPTIMIZATION_LOG.md) |
| AMD Vulkan (RDNA4/RDNA5) | 3 — Designed | Same code path as RDNA3; no RDNA4/5 hardware exercised |
| Intel Arc Pro (Battlemage, Vulkan) | **1 — Verified** | Arc Pro B70: llama.cpp Chonk integration verified (docs/inference_benchmarks.md). Driver note: stock Pro driver 8861 was 24x broken for compute; consumer WHQL 8974+ required |
| Cross-vendor direct P2P (Linux, DMA-BUF) | **1 — Verified** | RADV NAVI31 export -> ANV Battlemage G31 import, zero-copy, data-verified (`multi_gpu_test`, native Ubuntu; docs/LINUX_TEST_RESULTS_2026-08-25.md) |
| Intel Level Zero GPU-direct | **1 — Verified** | Arc Pro B70 via ze_loader: device discovery (31.9 GB heap), device alloc plane (1 MiB/64 MiB/1 GiB alloc/echo/free), full UnifiedMemoryPool over the L0 backend (`l0_backend_test`, 2026-09-17). v1 note: heapBudget returns invalid (static heap size fallback) |
| NVIDIA (CUDA path) | **1 — Verified** | GTX 1080 Ti (sm_61, PCIe 3.0 x8): CudaMemoryBackend end-to-end — `cuda_backend_test` ALL PASS (alloc/echo/free, budget, full pool + reserve + dedicated), llama.cpp `GGML_CUDA_VVM_POOL` 4B gate byte-identical, 90 GB auto-plan coherent (SERVE_TEST_RESULTS.md 2026-09-17; llama `ccfac2928`, VulkanVM `4310d8f`). Card currently off-box; roles: small-model duty (57.7 t/s on 4B), prefill, dense/KV offload |
| NVIDIA (Vulkan path) | **1 — Verified** | Same 1080 Ti as Vulkan device (11 GB): Chonk pool exercised at 4B scale via ggml-vulkan (split follows VRAM 68/32); 90 GB sessions ran via the CUDA path |
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
| TCP (host-staged) | **1 — Verified** | Windows (Arc B70) <-> Linux (Strix Halo) cross-machine tensor transport; Linux two-node loopback cluster incl. 16 MiB migration (`network_test`, docs/LINUX_TEST_RESULTS_2026-08-25.md) |
| RDMA/verbs (SoftRoCE) | **1 — Verified** | WSL2 custom kernel 6.18.40.1-wsl-rxe+: loopback 1.9 GiB/s, cross-machine cluster join; native Ubuntu kernel 7.0: in-process client+responder verbs handshake + RDMA_WRITE verified over rxe (`multi_vendor_rdma_test`, same doc). NOTE: requires event-driven rdma_cm usage - synchronous resolve segfaults on librdmacm 61 |
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
| Cross-vendor shared host arena (VK_EXT_external_memory_host) | **1 — Verified** | XTX + Arc Pro B70 both accept HOST_ALLOCATION import (4096 alignment); zero-copy B70<->XTX 5.2 GiB/s, manager `copyDeviceToDeviceArena` 6.1 GiB/s, byte-verified both directions (`shared_arena_test`, 2026-09-17; Intel included in probe from 2026-09-17) |
| Cross-vendor direct import (Windows) | **Refused (by design)** | Opaque handles are driver-private: AMD->NVIDIA returns VkResult -13; AMD->Intel AVs inside vkAllocateMemory (B70, found via `p2p_xn_test`). `copyDeviceToDevice`/`allocateDistributed` now gate direct import to same-vendor on Windows and fall back to host-staged (3.5 GiB/s symmetric XTX<->B70). Linux dma-buf cross-vendor is real and unaffected |

## Test coverage tiers

| Suite | Runs on | Status |
|---|---|---|
| `buddy_test` | any CPU | **1 — Verified** (Windows, Linux) |
| `chonk_slab_test` | any CPU | **1 — Verified** (Windows; Linux incl. 100k-op fuzz, docs/LINUX_TEST_RESULTS_2026-08-25.md) |
| `placement_test` | any CPU | pure logic, no GPU |
| `basic_test` / `minimal_test` / `external_handle_test` | Vulkan device | **1 — Verified** on RADV NAVI31, ANV Battlemage G31, lavapipe (native Ubuntu, same doc) |
| `sparse_test` (bind/unbind/readback/zero-page) | Vulkan sparse device | **1 — Verified** (RADV NAVI31 + ANV Battlemage G31, after queue-family fix, same doc) |
| `multi_gpu_test` cross-vendor P2P | 2x dGPU | **1 — Verified** (RADV <-> ANV DMA-BUF zero-copy + host-staged fallback paths, native Ubuntu; Windows XTX <-> B70 refused-direct + host-staged verified) |
| `p2p_xn_test` staged XN | 2x dGPU | **1 — Verified** (Windows XTX <-> B70, 3.5 GiB/s symmetric; Intel pairs supported from 2026-09-17) |
| `shared_arena_test` host arena | 2x dGPU | **1 — Verified** (Windows XTX <-> B70 zero-copy 5.2 GiB/s, manager arena 6.1 GiB/s) |
| `l0_backend_test` Level Zero | Intel dGPU | **1 — Verified** (Arc Pro B70: discovery, alloc plane, full pool over L0) |
| `retirement_test` GPU reclamation | Vulkan device | **1 — Verified** (XTX + B70: pre-signal no-reclaim, post-signal exact reclaim, async copy proof) |
| `udp_verb_test` reliable transfer | loopback | **1 — Verified** (Windows; fixed 2026-09-17: missing WSAStartup + WRITE contiguity off-by-one, 10/10 runs) |
| `multi_vendor_rdma_test` (verbs) | rxe / RoCE | **1 — Verified** same-host loopback (native Ubuntu kernel 7.0; requires event-driven rdma_cm) |
| `network_test` two-node cluster | CPU/loopback | **1 — Verified** all functional checks incl. 16 MiB migration (native Ubuntu; teardown-join hang tracked in same doc) |
| `tensor_collective_test` | Vulkan device | collectives verified running (allReduce/allGather/reduceScatter/broadcast); shutdown hang tracked |
| `chonk_slab_test 1000000` | any CPU | nightly-scale fuzz (slow: O(n^2) first-fit) |
| Android AHardwareBuffer test | device | **1 — Verified** (Galaxy S24+) |
| llama.cpp Chonk integration | 2x dGPU | **1 — Verified** (XTX + B70, docs/inference_benchmarks.md) |

Known open issues from the 2026-08-25 Linux campaign (details and repros in
docs/LINUX_TEST_RESULTS_2026-08-25.md): teardown-join deadlock across
cluster managers; ANV crash on same-device DMA-BUF re-import across two
VkDevices (upstream-Mesa candidate); `vkUnmapMemory` teardown warning in
multi_gpu_test.

## Adding a row

If you run VulkanVM on hardware listed above (or something new), open a PR
with: hardware name, driver version, which test/benchmark you ran, and the
result. Update the tier only with evidence.
