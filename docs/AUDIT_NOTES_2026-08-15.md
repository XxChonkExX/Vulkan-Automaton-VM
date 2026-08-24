# Audit Notes — 2026-08-15

> **Please pull the latest `main` before continuing.** This file was added so that on sync you have context on the recent changes, the external audit recommendations, and what to compare against field testing. New work landed after your last commits; if you are mid-training with local edits, rebase carefully (see "Known overlap areas" below).

---

## Source

External audit of the `main` branch (as of 2026-08-15) covering the Chonk Buffer training stabilisation work, allocator hardening, and Android AHardwareBuffer support. Full text was reviewed separately; this file is the actionable summary + what changed locally afterward.

---

## 1. Auditor's recommendations (for triage)

### High priority
1. **`src/allocator/buddy_allocator.cpp` — `splitTo` cleanup** — the current pop-then-push-both-halves flow does extra insert/erase work and leaves the target block on the free list until the final `popFree(targetOrder)`. Recommended: pop once, free only the right buddy in the loop, keep the left half off the free list; return it at target order directly. Update the `allocate` call site so `order == target` still uses `popFree`. **This is an active training-critical file — coordinate before touching (see overlap).**
3. **ChonkAdamW — decoupled weight decay** — currently coupled (`grad.add(p, alpha=weight_decay)`). Standard decoupled: `p.mul_(1 - lr * weight_decay)` before moment updates. Bias-correction/step counting should be per-parameter rather than global.
4. **Hard-coded paths in `train_qwen_chonk.py`** (`/home/chonke/...`) — switch to env vars / argparse with repo-relative defaults (`MODEL_PATH`, `DATA_PATH`, `CHONK_*`).

### Medium
5. **Min-block rounding (2^31 cliff)** — document/enforce consistently: Python allocator should round up to configured min block (and preferably next power-of-two) so freed blocks stay reusable across position-dependent growth. Expose a C++-side equivalent in `PoolConfig`/`BuddyAllocator`.
6. **maxHeapFraction = 0 in training mode** — intentional for unified-memory APUs, but dangerous on discrete GPUs. Consider a soft budget + explicit "training mode" flag that logs warnings instead of silently disabling the check.

### Low / hygiene
7. Windows SDK version hard-pinned in CMake — make flexible.
8. Document exact build steps for Python bindings (`_pool_test_module` / `vulkanvm_pool_test.so`) or fold into main CMake when `VVM_BUILD_PYTORCH=ON`.
9. `find_package` fallbacks for optional deps.
10. Expand buddy unit tests with concurrent threads when `threadSafe=true`; report internal fragmentation alongside `getFragmentation()`.

---

## 2. What landed locally after the audit (already on `main`)

- **`56b1857` Android NDK compatibility fixes (full send / verified)** — `std::jthread`→`std::thread` (NDK libc++ lacks jthread/stop_token), format-security pragmas, `<android/hardware_buffer.h>` include, Android-first platform detection in `ExternalHandle`/`duplicateForImport`, Android export/import paths in `unified_memory_pool.cpp`, `supportsAndroidHardwareBuffer` in `ExternalMemoryCaps`, dedup of the duplicate `src/cross_gpu/external_memory.hpp`.
- **`11e20f7` include-path fix** — after dedup, `src/cross_gpu/external_memory.cpp` and `multi_gpu_manager.cpp` used relative `"external_memory.hpp"` that no longer resolved; pointed at canonical `vulkan_vm/cross_gpu/external_memory.hpp`. Android + network libs rebuilt clean.
- **`084f1e8` README notice update** — Android AHardwareBuffer is now tested & passing (Galaxy S24+, API 36, Adreno).

### Field testing to compare against
- **Android**: `test_ahardwarebuffer` passes on S24+ (alloc 256x256 RGBA8 = 262144 B, device addr `0x4060100000`, valid for shader use). Only Adreno covered so far — **Mali/PowerVR/Xclipse still wanted**. No read-back / compute / cross-process sharing tests yet (device address validity is confirmed, that's the current scope).
- Rebuilt libs (`libvulkan_vm.so`, `libvulkan_vm_network.so`) were re-pushed to `/data/local/tmp` and re-verified after the include fix — the pass is current, not stale.

---

## 3. Known overlap areas (sync carefully)

| File | Who owns / risk | Note |
|------|----------------|------|
| `src/core/unified_memory_pool.cpp` | Shared — **highest conflict risk** | Android branches added to `exportMemory`/`importMemory`; allocator/budget work touches this file constantly. If you restructure `importMemory`, keep the `VkImportAndroidHardwareBufferInfoANDROID` branch. |
| `src/allocator/buddy_allocator.cpp` | X2 (training) | Auditor item #1 lives here; don't rewrite without coordination — your warm-blocks / 2^31-cliff work is the current priority. |
| `README.md` notice line | Shared | Edited by both sides on 2026-08-15; expect repeat edits while training runs are being documented. |
| `external_memory.hpp` (canonical) | Local | Deduped to `include/vulkan_vm/cross_gpu/external_memory.hpp`; `src/cross_gpu/` now holds `.cpp` only. |

---

## 4. Suggested next steps after pull

1. Rebase your local training branch onto `main`; confirm `buddy_test.cpp` still passes.
2. Run the recommended smoke: `CHONK_SMOKE=1 CHONK_MIN_BLOCK_GB=4 CHONK_ALLOCATOR=1 train_qwen_chonk.py`.
3. Compare auditor items 3/4/6 against your live runs (decoupled AdamW, path cleanup, budget flag) — these are in your territory.
4. Item 2 (pool destructor mutex) is RESOLVED - no action needed.
5. Android: if you have another device (Mali/PowerVR), run the test and report the GID/device — that closes the biggest open gap.

2. **`UnifiedMemoryPool` destructor mutex** - RESOLVED (2026-08-23): the destructor takes `mutex_`; Grok's 0.3 audit cited this stale note.
