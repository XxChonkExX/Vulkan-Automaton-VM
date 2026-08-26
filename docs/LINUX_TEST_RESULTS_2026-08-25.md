# Linux Native Test Results — 2026-08-25

First native-Linux hardware run (prior Linux data came from WSL2 / Strix Halo).
Machine: X870 Taichi Creator, Ryzen 9 7900X, 60 GB RAM, Ubuntu GNOME (Wayland),
kernel `7.0.0-30-generic`.

## Hardware under test

| Device | Driver | API | Notes |
|---|---|---|---|
| AMD Radeon RX 7900 XTX (Navi 31) | RADV (Mesa) | 1.4.335 | discrete |
| Intel Battlemage G31 | ANV (Mesa open-source) | 1.4.335 | discrete |
| AMD Raphael (7900X iGPU) | RADV | 1.4.335 | integrated |
| llvmpipe | lavapipe | 1.4.335 | CPU fallback |

Build: GCC, Release, core-only (`VVM_BUILD_NETWORK=OFF`), shared lib,
validation enabled (`VVM_ENABLE_VALIDATION=ON`).

## Results

| Suite | Default (XTX) | BMG G31 (ANV) | RADV-only | llvmpipe | Notes |
|---|---|---|---|---|---|
| basic_test | PASS | PASS | PASS | PASS | |
| minimal_test | PASS | PASS | PASS | PASS | BMG reports larger DEVICE_LOCAL heap than XTX (29.3 vs 22.9 GB visible budget), so score-based selection picks Intel by default |
| buddy_test | PASS | n/a | n/a | n/a | CPU-only |
| chonk_slab_test (+100k-op fuzz) | PASS | n/a | n/a | n/a | upgrades matrix row: Verified **Windows + Linux** |
| external_handle_test | PASS | PASS | PASS | PASS | |
| placement_test | PASS | n/a | n/a | n/a | CPU-only |
| sparse_test | **PASS** | **PASS** (after fix) | PASS | — | see bug #1 |
| multi_gpu_test (P2P) | **PASS — true cross-vendor DMA-BUF P2P** (after fix #2) | — | PASS (RADV↔RADV) | — | see below |

### Headline result: cross-vendor zero-copy P2P verified

`multi_gpu_test` on default devices exercised RX 7900 XTX (RADV) exporting a
dedicated allocation via **DMA-BUF** and the Battlemage G31 (ANV) importing it
directly:

```
[INFO] copyDeviceToDevice: using vendor P2P path: AMD source: OPAQUE_WIN32 (Win) / DMA-BUF (Linux) export
[INFO] Import: re-selected memory type index 0 on destination device
  D2D copy verified on dst: PASS
```

No host-staged fallback fired. Per `docs/HARDWARE_SUPPORT.md` this path had no
matrix row and has never been exercised; this run is the first evidence.
Suggested new row: *Cross-vendor direct P2P (Linux, DMA-BUF): Tier 1 — Verified
(RADV Navi31 -> ANV Battlemage G31, multi_gpu_test)*.

## Bugs found & fixed in this session

1. **sparse_test failed on ANV (Battlemage): silent NULL sparse queue**
   - ANV exposes family 2 as pure `TRANSFER` without `VK_QUEUE_SPARSE_BINDING_BIT`;
     sparse lives on families 0–1. The test created its logical device with only
     the transfer family, so `SparseVirtualMemoryPool::create` found a sparse
     family it couldn't fetch a queue from and returned `nullopt` silently
     (`src/core/sparse_memory.cpp`, `vkGetDeviceQueue` NULL check had no log).
   - Fixes: `tests/sparse_test.cpp` now selects a transfer-capable family that
     also reports the sparse bit; `sparse_memory.cpp` logs a descriptive error
     instead of failing silently. Sparse residency (bind/unbind/readback/
     zero-page semantics) then **passed identically on RADV and ANV**.

2. **Vendor P2P export never engaged on Linux: enum/bit confusion**
   - `getVendorP2PCaps()` stores raw Vulkan bits
     (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` = 0x200) which were
     `static_cast` into the sequential `ExternalHandleType` enum (`DmaBuf` == 4),
     landing in `exportMemory`'s `default: return std::nullopt`. Every Linux
     P2P copy silently degraded to host staging. Windows worked only because
     `OPAQUE_WIN32_BIT` (0x2) numerically collides with enum `OpaqueWin32` (2).
   - Fix: `toExternalHandleType()` mapping helper in `multi_gpu_manager.cpp`;
     both cast sites converted.

3. **DMA-BUF extension not enabled at logical-device creation**
   - `vkGetMemoryFdKHR` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`
     requires `VK_EXT_external_memory_dma_buf`; `tests/multi_gpu_test.cpp`
     enabled only KHR fd extensions. Added to the Linux extension list.

## Known issues opened (not yet fixed)

- `vkUnmapMemory: Invalid device [VUID-vkUnmapMemory-device-parameter]` loader
  error during `multi_gpu_test` teardown (exit code still 0). Staging memory is
  unmapped against the wrong logical device or after destruction.
- `deallocate: stale allocation handle (generation 0) rejected` warning in the
  same test's cleanup path.
- `selectBestDevice()` tie-breaks purely by reported heap size, which favors
  ANV's optimistic budget reporting over the XTX. Consider an explicit
  `VVM_DEVICE_INDEX` env override for reproducible benchmarking.

## Environment notes

- SoftRoCE link `rxe0` ACTIVE on enp14s0 (`rdma link show`) — ready for the
  verbs/RDMA phase. Kernel 7.0 >= 6.19 means `ibv_reg_dmabuf_mr` (true rxe
  dma-buf MR) may engage where older kernels used the mmap fallback;
  `scripts/test_amd_rdma.sh env` will report which path fires.
- `ibv_devices` userspace tool missing from PATH despite rdma-core installed
  (verify package set when doing the RDMA phase).

## Suggested HARDWARE_SUPPORT.md updates (evidence above)

- `chonk_slab_test`: Verified (Windows) -> Verified (**Windows, Linux**)
- New GPU-backend row: Cross-vendor direct P2P DMA-BUF — Tier 1 (this doc)
- New test-suite rows: `sparse_test` (RADV + ANV), `external_handle_test`
  (RADV/ANV/lvp), `basic/minimal` (RADV/ANV/lvp)

---

# Phase 3 — Full stack (network + tensor) on Linux, same day

Build: GCC Release, `VVM_BUILD_NETWORK=ON`, verbs + rdma_cm detected
(rdma-core 61.0-2ubuntu3, SoftRoCE `rxe0` ACTIVE on enp14s0). OpenSSL not
installed → TLS compiled out (optional dep).

## Compile fixes required (first-ever Linux compile of these paths)

CI never compiles these combos (examples OFF + no libibverbs on runners):

| File | Problem | Fix |
|---|---|---|
| examples/CMakeLists.txt | `tensor_client/server_test` link `vvm::tensor::*` undefined | link `vulkan_vm_tensor` |
| examples/network_test.cpp:221 | stale `NetworkConfig::enableAdaptiveWindow/streamPipelineBuffers` (API moved to StreamIO callbacks) | removed stale block |
| tests/multi_vendor_rdma_test.cpp:132 | WIN32 extension name used unconditionally; GetModuleFileNameA/MAX_PATH/_putenv_s in main | platform guards |
| examples/network_test.cpp devExts | **VK_KHR_external_memory_fd never enabled on Linux** → all fd export impossible | added fd + dma-buf exts |
| tests/tensor_collective_test.cpp, examples/tensor_{client,server}_test.cpp | dma-buf ext missing | added |

## Results after fixes

| Suite | Result |
|---|---|
| serialization_test (wire format) | PASS |
| dtype_conversion_test | PASS (66 checks) |
| offload_test (device↔host shadow round-trip, BMG) | PASS |
| tensor_compute example | PASS |
| model_registry_test (loopback TCP cluster, publish/fetch) | PASS |
| tensor_collective_test (self-hosted loopback cluster) | collectives all RAN (allReduce×3/allGather/reduceScatter/broadcast) then **hangs in teardown** |
| network_test (two-node loopback cluster) | **all functional checks pass incl. `migrateFromRemote: pulled 16 MiB`** (with ZC opt-out below); hangs in teardown |
| multi_vendor_rdma_test (verbs over rxe0) | transports init + listener OK; **segfault in system librdmacm** (see below) |

## New bugs / findings

1. **Cluster announce/bind mismatch (fixed):** `MultiNodePoolManager`
   auto-advertises the primary LAN IP while the test bound loopback →
   cross-node connects timed out. Fixed by setting
   `NetworkConfig::advertiseAddress` (host-only!) = listen host in
   network_test. Note: advertiseAddress must NOT include the port.
2. **System-level SoftRoCE crash (NOT a VulkanVM bug):** a 15-line standalone
   probe using only `rdma_create_id → rdma_resolve_addr → rdma_resolve_route`
   segfaults inside `rdma_resolve_route()` (librdmacm 61.0-2ubuntu3,
   kernel 7.0.0-30, rxe0 on enp14s0), destination loopback *or* LAN IP.
   Kernel-7.0 rxe has active 2026-era CVEs (e.g. CVE-2026-46133); recommend
   verifying with independent tooling (`rstream` from librdmacm-utils) and/or
   an older/HWE kernel before re-attempting `multi_vendor_rdma_test`.
3. **ANV same-device DMA-BUF re-import crashes the driver:** exporting a
   dedicated allocation as DMA-BUF from VkDevice A and importing into
   VkDevice B over the SAME physical device (BMG G31, two logical devices)
   segfaults inside `libvulkan_intel.so` via
   `MultiNodePoolManager::importRemote → UnifiedMemoryPool::importMemory`.
   Cross-*vendor* RADV→ANV import is fine (multi_gpu_test passes). Added
   `VVM_DISABLE_SAME_PROCESS_ZC=1` opt-out so cluster flows can be validated;
   worth narrowing further and possibly filing upstream Mesa.
4. **Teardown deadlock (characterized, root-cause narrowed):** both
   `tensor_collective_test` and `network_test` hang during shutdown with all
   functional checks already passed. Captured thread states
   (`ptrace_scope=0` + gdb attach):
   - main blocked in `TcpTransport::request()` awaiting a peer reply, while
   - the peer's `serveConnection()` threads sit in `readAll()` awaiting the
     *next* header -> classic missed-response / fire-and-forget mismatch.
   - `MultiNodePoolManager::heartbeatLoop` threads still alive at capture,
     i.e. `stop()` had not yet reached its heartbeat join.
   Mitigation landed: `stop()` now severs outbound peer connections
   (`shutdownConnectionPool`) before stopping its own transport, which is
   strictly better ordering even though the underlying request/response
   liveness bug remains. Next step: per-message-type logging around
   DEALLOCATE/zero-copy-import requests to find the unanswered exchange.
5. Earlier session findings still open: `vkUnmapMemory: Invalid device`
   teardown error in multi_gpu_test; stale-generation deallocate warning.

## Repro commands

```
cmake -S . -B build_rdma -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_NETWORK=ON
cmake --build build_rdma -j$(nproc)
ctest --test-dir build_rdma -I 1,8          # fast suites (pass)
timeout 120 ./build_rdma/examples/network_test                    # hang at exit
VVM_DISABLE_SAME_PROCESS_ZC=1 ./build_rdma/examples/network_test  # all checks pass
```

---

# Phase 3b — RDMA/verbs: from system-level segfault to PASS

Independent verification with rdma-core tools over the same `rxe0`:

| Tool | Result |
|---|---|
| `ibv_devinfo` | rxe0 PORT_ACTIVE |
| `rstream` (loopback) | **PASS** — full latency/BW matrix, ~1.6 GB/s @1M msgs |
| `ucmatose` (loopback) | **PASS** — full rdma_cm establishment |

## Root cause chain (three stacked bugs, all fixed)

1. **librdmacm synchronous mode crashes.** A minimal probe calling
   `rdma_resolve_route()` without draining CM events segfaults inside
   librdmacm (61.0-2ubuntu3) on kernel 7.0 — while the *event-driven* pattern
   passes on the identical stack (`rxe_probe2`: sync=SEGV, async=OK).
   `VerbsRdmaTransport::connect()` used the sync pattern.
   **Fix:** `connect()` now pumps its per-connection channel via
   `waitCmEvent()` through ADDR_RESOLVED → ROUTE_RESOLVED → ESTABLISHED
   (src/network/rdma_transport.cpp). Bonus: destinations without an RDMA
   device (127.0.0.1) now fail cleanly instead of crashing.
2. **PD/context mismatch.** `rdma_create_qp(id, ...)` requires the PD/CQ to
   come from `id->verbs` (the CM's ibv_context), not the transport's own
   `ibv_open_device` context — otherwise EINVAL. Applied on BOTH sides
   (client connect + server `handleConnectRequest`) via a lazily-allocated
   `effectivePd(id->verbs)` shared by QP creation AND all MR registrations
   so lkeys/rkeys stay QP-compatible.
3. **`RdmaMemoryRegion::rdmaAddr` never populated for host memory** (only
   the NVIDIA path set it). The test wrote to remote address **0** →
   IBV_WC_REM_ACCESS_ERR. Host and DMA-BUF-mmap registrations now set
   `rdmaAddr` to the registered CPU VA.

Also fixed en route: `connCreateMutex_` made recursive (connect() re-enters
via `effectivePd()`); removed an unreachable condvar wait that expected the
event loop to re-report an ESTABLISHED already consumed by `waitCmEvent()`.

## Result

```
$ VVM_RDMA_CONNECT_HOST=<lan-ip> ./build_rdma/tests/multi_vendor_rdma_test
[INFO] RDMA accepted incoming connection (qp 34)
[INFO] connect: RDMA established (qp 33)
Cross-vendor transport connection established!
Cross-vendor RDMA test PASSED
```

First verbs-path data transfer ever exercised on this repo's Linux client
side: two transports, cross-vendor GPUs attached, RDMA_WRITE verified over
SoftRoCE loopback in one process. Suggested HARDWARE_SUPPORT.md row:
*RDMA/verbs (SoftRoCE, same-host): Tier 1 — Verified (this doc)*.

Still open: the process hangs at exit after PASS (raw transports hit the
same teardown-join family as finding 4 above); default (no-env) ctest runs
now SKIP cleanly on loopback instead of crashing.

---

# Phase 4 — Teardown hang FIXED, ANV import crash characterized (same day, post-reinstall)

Machine re-provisioned mid-campaign (fresh Ubuntu GNOME, same kernel
`7.0.0-30-generic`, Mesa **26.0.3-1ubuntu1**, CMake 4.2.3 / GCC). Environment
rebuilt from packages only: `libvulkan-dev vulkan-tools glslang-tools
libibverbs-dev librdmacm-dev rdma-core rdmacm-utils ibverbs-utils libssl-dev
vulkan-validationlayers`. SoftRoCE recreated (`scripts/softroce_persist.sh`
updated for native NIC `enp14s0`, module persisted via
`/etc/modules-load.d/rdma_rxe.conf`); fresh-clone shader build failure fixed
(`CMakeLists.txt` now makes the SPIR-V output directory before glslang runs).

## Fixed: the teardown hang (findings 4 / Phase 3b "hangs at exit")

Root cause captured by live backtrace of the hung process:

```
main: ~MultiNodePoolManager -> cleanup() -> VerbsRdmaTransport::shutdown()
      -> std::thread::join()                      [blocked forever]
listenerEventLoop threads (x2 transports):
      rdma_get_cm_event() -> write() [kernel GET_EVENT]   [blocked forever]
```

`shutdown()` destroyed the event channel *before* joining, assuming
`rdma_destroy_event_channel()` wakes a thread parked in `rdma_get_cm_event()`.
On librdmacm 61 / kernel 7.0 it does not: the thread sleeps inside a kernel
`GET_EVENT` write that only an actual CM event can satisfy - and after peer
disconnect no event ever arrives.

**Fix** (`src/network/rdma_transport.cpp`): all CM event channels are created
`O_NONBLOCK`; `listenerEventLoop()`, `connectionEventLoop()` and
`waitCmEvent()` now `poll()` the channel fd with a bounded timeout and treat
`EAGAIN` as "no event yet", so every loop observes `shuttingDown_` and exits on
its own. `shutdown()` order flipped to join-then-destroy. No synthesized events,
no busy waiting (100 ms tick).

| Suite | Before | After |
|---|---|---|
| `multi_vendor_rdma_test` | PASS then **hang at exit** | PASS, **exit 0** |
| `tensor_collective_test` | collectives ran, **hang at teardown** | **exit 0**, PASS |
| `network_test` | checks passed, **hang at teardown** | **exit 0**, ALL TESTS PASSED |

Full ctest (11/11) and core-only ctest (7/7) pass with clean exits.

## Characterized: cross-vendor import segfault is an ANV regression in Mesa 26.0.3

With ZC enabled and an ANV destination, the process died in
`vkAllocateMemory` inside `libvulkan_intel.so`. Instrumented + bisected:

- Importing device: BMG G31 (ANV), source export: RX 7900 XTX (RADV),
  dedicated allocation, BDA allocate-flag chain - identical call sequence that
  succeeds RADV->RADV in `multi_gpu_test`.
- Crash reproduces with **both** `OPAQUE_FD` and `DMA_BUF` handle types.
- Not caused by `VK_EXT_memory_budget`, missing
  `VK_KHR_dedicated_allocation`, or memory-type selection (all bisected).
- KHR validation layers are otherwise clean up to the crash point.

Conclusion: ANV 26.0.3 segfaults importing another driver's external heap
where the previous stack imported fine; upstream-report material. The doc's
earlier "cross-vendor DMA-BUF P2P verified" row stands for the older Mesa;
re-verify after a Mesa update.

**Mitigation:** `examples/network_test.cpp` now creates node A on the best
device and node B on a different-vendor hardware device when available
(llvmpipe excluded - its DMA-BUF export is a stub and crashes importers), and
auto-disables same-process ZC for cross-vendor pairs
(`VVM_ALLOW_CROSSVENDOR_ZC=1` forces attempts). With ZC off, the zero-copy
check reports SKIP (host-staged migration is what's under test and passes:
"Pull verify: PASS", 16 MiB migrated).

## Validation-layer findings en route (real bugs, still open)

- `VUID-VkBufferDeviceAddressInfo-buffer-02601`: `vkGetBufferDeviceAddress`
  is called on buffers whose usage lacks
  `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (pool allocates with the
  allocate-flag but not the usage bit; tolerated by RADV, flagged by
  validation).
- `VUID-VkMemoryGetFdInfoKHR-handleType-00671`: the verbs GPUDirect path asks
  for DMA-BUF fds on memories never allocated with export capability
  (`gpu_direct_registration` / RDMA register path); fails non-fatally today.
- `vkUnmapMemory: Invalid device` at multi_gpu_test teardown now escalates to
  SIGABRT under the current loader (was an error print): still-open, plus the
  stale-generation deallocate warning.

## SoftRoCE notes

`rxe0` on `enp14s0`: `ibv_devinfo` PORT_ACTIVE; `ucmatose` PASS;
`rstream` full latency/BW matrix PASS (~5.2 Gb/s @1 MiB, 3.9 us @64 B) but
`-T v` data verification fails **deterministically at byte 217753681** (~208
MiB into the default run) with server-side resets - kernel rxe/rsockets data
corruption below our transport layer (raw-verbs path unaffected;
`multi_vendor_rdma_test` passes over the same device). Upstream-report
candidate alongside the known kernel 7.0 rxe CVEs.

## Repro / verify commands

```
cmake -S . -B build_rdma -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_NETWORK=ON
cmake --build build_rdma -j$(nproc)
ctest --test-dir build_rdma                                        # 11/11, no hangs
VVM_RDMA_CONNECT_HOST=<lan-ip> ./build_rdma/tests/multi_vendor_rdma_test
./build_rdma/examples/network_test                                 # exit 0
VVM_ALLOW_CROSSVENDOR_ZC=1 ./build_rdma/examples/network_test     # will hit the ANV bug
```

---

# Phase 5 — Validation-layer VUID cleanup (same day)

With `VK_LAYER_KHRONOS_validation` enabled across the suites, five violation
classes surfaced. Fixed in this phase:

| VUID | Cause | Fix |
|---|---|---|
| `VkBufferDeviceAddressInfo-buffer-02601` | `vkGetBufferDeviceAddress` queried on buffers created without `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (`allocateDedicatedExportable`, `importMemory`; tolerated by RADV, flagged by validation) | usage bit added when `config_.enableDeviceAddress` (unified_memory_pool.cpp) |
| `VkMemoryGetFdInfoKHR-handleType-00671` | verbs GPUDirect path requested DMA-BUF fds for pool-block memories that never had export capability | `registerMemoryForRdma` gates on `Allocation::isExternal` up front |
| `VkDeviceCreateInfo-pNext-02830` | network_test chained `Vulkan12Features` AND individual BDA/Timeline feature structs | single `Vulkan12Features` struct |
| `VkDeviceCreateInfo-queueFamilyIndex-02802/-06755` | duplicate queue-family entries when graphics/compute/transfer roles share a family; counts could exceed the family's queueCount | dedup + clamp against `vkGetPhysicalDeviceQueueFamilyProperties` |
| `vkCreateInstance-ppEnabledExtensionNames-01388` | unused surface extension requested | dropped |

**Leak fixes (object-lifetime):**

- `network_test`: push/pull source allocations (`srcB*`) were never returned
  to B's pool - only their promoted copies were. Explicit deallocates added.
- `tensor_collective_test`: `TensorAllocation` holds its `Allocation` by value
  and does NOT auto-free (per docs/LIFETIME_CONTRACT.md tensors are explicitly
  managed); the test dropped every handle without deallocating (20 leaked
  buffers). Handles are now tracked and returned to their pools before
  transport shutdown / device destroy (20 -> 2).

**Remaining known leaks (documented, not fixed):**

- `tensor_collective_test`: 2 buffers from transport-internal broadcast
  replication on remote devices - engine-side lifecycle work.
- `network_test`: 6 buffers of engine-internal temporaries (migration staging).
- `multi_gpu_test`: teardown `vkUnmapMemory: Invalid device` escalates to
  SIGABRT under the current loader/validation; stale-generation deallocate
  warning in the same path.

Validation status after fixes: create/destroy-path VUIDs clean in
`tensor_collective_test`; remaining hits are only the leak classes above.

## Repro

```
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ctest --test-dir build_rdma --output-on-failure
```

---

# Phase 6 — Teardown correctness, ZC policy, lifecycle hardening (same day)

## Fixed: `multi_gpu_test` SIGABRT at exit ("vkUnmapMemory: Invalid device")

Live backtrace: abort inside `vkUnmapMemory` from `~UnifiedMemoryPool`, called
via `~vector<GPUInstance>` in main. Instrumented dev/mem pairing proved the
unmap used the **correct** device/memory pair - meaning the VkDevice itself was
already destroyed. Root cause: test teardown order - `vkDestroyDevice()` ran
while the manager (and its pools) were still alive; pool destructors then
touched a dead device, which the current loader escalates to SIGABRT.

Fixes:
- `tests/multi_gpu_test.cpp`: `manager.reset()` before destroying devices
  (pools must always die before their VkDevice).
- `UnifiedMemoryPool::deallocate/subDeallocate`: stale-generation rejections
  and untracked dedicated handles now zero their handles instead of leaving
  dangling buffer/memory values that could be re-freed later.

Result: `multi_gpu_test` exits 0 with no loader errors for the first time.

## Fixed: same-process zero-copy policy moved into the library

The env-var guard in `network_test` is replaced by runtime policy in
`multi_node_manager.cpp`: exported handles are registered with the exporter's
PCI vendor ID, and `zcAllowedForPair(srcVendor,dstVendor)` refuses
cross-vendor same-process imports on Linux by default (Mesa 26.0.3 ANV crash,
Phase 4), parking the handle instead of consuming it. Overrides:
`VVM_ALLOW_CROSSVENDOR_ZC=1`. `VVM_DISABLE_SAME_PROCESS_ZC` is superseded.
`network_test` now consults the exposed
`vvm::network::sameProcessZcAllowed()` for its SKIP reporting.

## Fixed: migration staging + broadcast replica leaks

- All four host-staged migration staging sites now release via RAII
  (`StagingReleaser`) or freeing-deleter shared ownership
  (`StagingFreeDeleter`) - previously the response cleanups were `(void)x`
  no-ops and client stagings simply went out of scope. `network_test`
  leaked-buffer count under validation: 8 -> 4 hits (2 buffers).
- `~TensorTransportImpl` detaches cached distributed-tensor releasers and
  clears its handle cache before pools can go away (a use-after-teardown
  deadlock discovered by the new auto-free path). `tensor_collective_test`
  leaked-buffer count: 20 -> 4 hits (2 buffers).

## Added: tensor handle lifetime hooks

`TensorAllocation` gained `attachReleaser()` (opt-in auto-free bound to the
owning pool) plus leak warning / `VVM_TENSOR_LEAK_ABORT=1` abort semantics;
engine factories attach releasers automatically. Contract documented as
section 7 of docs/LIFETIME_CONTRACT.md.

## Added: deterministic device selection

`vvm::selectBestDevice` honors `VVM_DEVICE_INDEX=<n>` (index into
`enumerateDevices` order) before any scoring; invalid indices warn and fall
back to score selection.

## Upstream report drafts

- docs/upstream/mesa-anv-crossvendor-import-segfault.md
- docs/upstream/kernel-rxe-rstream-corruption.md

## Status after Phase 6

| Suite | Exit | Validation leaks |
|---|---|---|
| multi_gpu_test | 0 | (teardown clean) |
| network_test | 0 | 2 residual buffers |
| tensor_collective_test | 0 | 2 residual buffers |
| full ctest | 11/11 | - |
