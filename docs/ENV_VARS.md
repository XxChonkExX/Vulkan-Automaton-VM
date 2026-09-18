# Environment Variables

Single reference for every `VVM_*` knob. All are optional; unset means the
documented default. Names are stable API — do not rename without an
UPGRADING note.

## Logging & diagnostics

| Variable | Default | Effect |
|---|---|---|
| `VVM_LOG_LEVEL` | `info` | `trace`/`debug`/`info`/`warn`/`error`. Filtered calls cost nothing (checked before formatting). Use `warn` for clean inference server logs. |
| `VVM_WARN_LIVE_POOL` | off | Set (any value) to warn when a live pool is destroyed (move/ownership debugging). Off by default: normal at process exit. |

## Pool & transfer behavior

| Variable | Default | Effect |
|---|---|---|
| `VVM_DEVICE_INDEX` | auto | Pin device selection to one Vulkan device index. |
| `VVM_STAGED_CHUNK_MB` | `16` | Host-staged copy chunk size in MiB (1..1024). Larger = fewer sync round-trips. |
| `VVM_STAGED_PIPELINE` | `1` | `0` = sequential chunk loop (legacy); default overlaps the dst DMA leg with the next chunk's src leg (double-buffered slots). |
| `VVM_P2P_POLICY` | `auto` | `host` = always host-staged, never attempt direct import. `force-direct` = attempt direct import even cross-vendor (refused drivers may crash; bring-up only). |
| `VVM_ALLOW_CROSSVENDOR_ZC` | off | Force same-process zero-copy import cross-vendor (same crash caveat as `force-direct`). |
| `VVM_SKIP_CMDPOOL` | off | Skip transfer command-pool creation (init-path bisect). |
| `VVM_SKIP_INITBLOCK` | off | Skip initial block bootstrap (init-path bisect). |

## Storage backend

| Variable | Default | Effect |
|---|---|---|
| `VVM_STORAGE_BACKEND` | `auto` | `ioring` / `overlapped` (Windows) — force the NVMe read path instead of probing. |

## Testing & debug

| Variable | Default | Effect |
|---|---|---|
| `VVM_TENSOR_LEAK_ABORT` | off | Abort (instead of warn) when a `TensorHandle` dies with a live buffer. ON by default under CTest for `tensor_collective_test`. |
| `VVM_BUDDY_FUZZ_ITERS` | `5000` | Iterations per thread in the concurrent buddy fuzz (up to 10M for nightlies). |
| `VVM_ND_PROVIDER_DLL` | — | Path to the NetworkDirect provider DLL (Windows ND transport tests). |
| `VVM_RDMA_BACKEND` | auto | Force the RDMA backend selection. |

## llama.cpp side (`chonk-buffer` branch)

GPU-pool knobs live there and use the `GGML_VVM_*` prefix: `GGML_VK_VVM_POOL`,
`GGML_HIP_VVM_POOL`, `GGML_CUDA_VVM_POOL`, `GGML_VVM_HEAP_FRACTION`,
`GGML_VVM_BLOCK_SIZE` (pow2, 256 KiB..8 GiB), `GGML_VVM_BASE_ALIGN`,
`GGML_VVM_CHUNK_MB`, `GGML_VVM_PURE_LOCAL=0`, `GGML_VVM_NO_DEDICATED`,
`GGML_VVM_PASSTHROUGH_ALLOC`, `--vvm-auto-kv-mib`, `--vvm-split`. See the
branch README knobs table — it is the canonical reference for those.
