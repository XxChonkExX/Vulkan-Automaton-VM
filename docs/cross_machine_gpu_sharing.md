# Cross-Machine GPU Networking

VulkanVM supports sharing GPU memory across machines over TCP. This enables distributed AI inference where tensors can be transferred between GPUs on different computers.

## Automated AMD RDMA Test

For testing the **AMD vendor-specific GPU-direct RDMA path** (DMA-BUF export → `ibv_reg_dmabuf_mr` verbs registration) over the network, use `scripts/test_amd_rdma.sh`:

```bash
# On BOTH machines (once):
sudo ./scripts/test_amd_rdma.sh setup     # deps + Soft-RoCE (rxe) link
sudo ./scripts/test_amd_rdma.sh build     # configure + build tests
sudo ./scripts/test_amd_rdma.sh env       # verify RDMA devices + GPU

# On Evo-X2 (server):
./scripts/test_amd_rdma.sh server --port 51000 --announce-count 6

# On Ubuntu client:
./scripts/test_amd_rdma.sh client --server 192.168.0.117 --port 51000 --local-port 51005
```

The script verifies the log for `VerbsRdmaTransport initialized`, `AMD GPUDirect via ibv_reg_dmabuf_mr` (or the mmap fallback), and `migrateFromRemote: pulled`.

`--announce-count N` makes the server exit after announcing N tensors (bounded run for scripts/CI). Without it the server runs until Ctrl-C and the client exits after its tests.

**Kernel note (important)**: `rdma_rxe` dma-buf MR support (`ibv_reg_dmabuf_mr` on rxe) is **not in any released kernel** — the patchset has been under upstream review (RFC v4 Feb 2026, latest posting Mar 2026) and an independent implementation proved rxe is fundamentally a CPU `memcpy` driver that needs CPU-accessible `struct page`s. So on stock kernels the AMD path uses the `DMA-BUF mmap fallback` — which works well on Strix Halo (unified memory, amdgpu exports dma-bufs with CPU-accessible pages via HMM/GTT). On a **discrete** GPU (e.g. 7900 XTX), amdgpu VRAM dma-bufs are not CPU-accessible either — expect slow/fallback behavior there; the realistic GPU-direct target is the X2 APU.

**WSL2 note**: The data-plane export step (`allocateDedicatedExportable` → `vkAllocateMemory`) fails with `VK_ERROR_OUT_OF_DEVICE_MEMORY` on WSL2's virtual GPU (`wslgd`) even at 1 MiB — the wslgd driver rejects exportable dedicated allocations. Cluster join, announce, and recv flows work in WSL. The verbs/RDMA data path can still be exercised in WSL2 by building a custom kernel with `rdma_rxe` (see below) — host-memory RDMA works, GPU dma-buf export does not.

## WSL2 development environment (optional)

Stock WSL2 kernels lack `rdma_rxe`. `scripts/wsl2_rxe_kernel.sh` builds a custom kernel (based on `microsoft/WSL2-Linux-Kernel`) with Soft-RoCE enabled:

```bash
./scripts/wsl2_rxe_kernel.sh prepare    # kernel build deps + rdma-core + perftest
./scripts/wsl2_rxe_kernel.sh build      # clone + enable CONFIG_RDMA_RXE + build
./scripts/wsl2_rxe_kernel.sh install    # copy bzImage + set kernel= in .wslconfig
# from Windows PowerShell:
wsl --shutdown
# reopen the distro, then:
./scripts/wsl2_rxe_kernel.sh verify     # modprobe rdma_rxe + rxe0 over eth0 + GID check
./scripts/wsl2_rxe_kernel.sh loopback   # veth pair + ib_write_bw self-test
```

What this enables in WSL2:
- Full verbs/rdma_cm data path over **host memory** — test the transport against the X2 without a native Ubuntu install
- `ib_write_bw`-style validation (`perftest`) before running the VulkanVM tests

What it does NOT enable in WSL2:
- GPU dma-buf export (wslgd limitation above) — the AMD GPU-direct path still needs native Linux
- `rdma_rxe` dma-buf MRs (not in any kernel, see above)

## Architecture

```
┌─────────────────┐         TCP          ┌─────────────────┐
│   Windows       │ ◄──────────────────► │   Linux (Evo-X2) │
│   Intel Arc B70 │   Tensor Transport    │   AMD Strix Halo │
│   (or 7900 XTX) │                      │   (Radeon 890M)  │
└─────────────────┘                      └─────────────────┘
     192.168.0.213                           192.168.0.117
```

## Quick Start

### Build (both machines)

```bash
git clone https://github.com/XxChonkExX/Vulkan-Automaton-VM
cd Vulkan-Automaton-VM
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVVM_BUILD_NETWORK=ON
cmake --build build --target tensor_server_test tensor_client_test
```

Both test binaries build on Linux (server is Linux-only; the client builds on Linux **and** Windows).

### Run Server (Evo-X2 / Linux)

```bash
./build/examples/tensor_server_test --port 51000 --verbose
```

### Run Client (Windows / Linux)

```powershell
# Windows
.\build\examples\tensor_client_test.exe --server 192.168.0.117 --port 51000 --local-port 51005
```

```bash
# Linux
./build/examples/tensor_client_test --server 192.168.0.117 --port 51000 --local-port 51005
```

## Command Line Options

### Server (`tensor_server_test`)

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 51000 | TCP port to listen on |
| `--size-mb` | 16 | Size of test tensor in MB |
| `--rdma-nic` | auto | RDMA device to use (e.g. `rxe0`) |
| `--announce-count` | 0 (unlimited) | Exit after announcing N tensors (bounded runs) |
| `--verbose` | off | Enable detailed operation logging |
| `--help` | | Show help |

### Client (`tensor_client_test`)

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | 192.168.0.117 | Server IP address |
| `--port` | 51000 | Server port |
| `--local-port` | 51005 | Local listen port |
| `--rdma-nic` | auto | RDMA device to use (e.g. `rxe0`) |
| `--help` | | Show help |

## Requirements

- **Network**: Both machines on same LAN (or routable)
- **Vulkan**: GPU with `VK_KHR_external_memory` support
- **Linux**: `VK_KHR_external_memory_fd` for DMA-BUF/OPAQUE_FD export
- **Windows**: `VK_KHR_external_memory_win32` for OPAQUE_WIN32 export
- **Firewall**: Allow inbound TCP on the chosen port (and port + 1 for the RDMA listener)
- **RDMA path**: `rdma-core` + `libibverbs-dev` / `librdmacm-dev`; Soft-RoCE (`rdma_rxe`) when no InfiniBand NIC — see `scripts/test_amd_rdma.sh setup`

## Transport Path Selection

VulkanVM automatically selects the best transport path:

| Priority | Path | Requirements |
|----------|------|--------------|
| 1 | P2P | Same machine, driver support |
| 2 | RDMA | Verbs transport (`librdmacm`) + GPU registration |
| 3 | Host-Staged | Always available (TCP) |
| 4 | UCX | UCX build + GPU memory registration |

The GPU-direct RDMA registration is **vendor-specific**:
- **AMD (0x1002)**: DMA-BUF export + `ibv_reg_dmabuf_mr` (mmap fallback on kernels < 6.19); ROCm/HIP optional for compute-side import
- **NVIDIA (0x10DE)**: `vkGetMemoryRemoteAddressNV` + `nvidia-peermem`
- **Intel (0x8086)**: Level Zero / DMA-BUF

For cross-machine without RDMA, **Host-Staged TCP** is used (4 MB chunks).

## Verified Configurations

| Client | Server | Status |
|--------|--------|--------|
| Windows (Intel Arc B70) | Linux/AMD Strix Halo | ✅ Working (host-staged TCP) |
| Ubuntu (AMD 7900 XTX) | Linux/AMD Strix Halo | 🔜 Pending (GPU-direct RDMA via SoftRoCE) |

## Troubleshooting

### Client registers as 127.0.0.1
**Symptom**: Server logs show client at `127.0.0.1:port`  
**Fix**: Ensure Windows LAN IP detection is working (see `getPrimaryLanIp()`)

### Cluster view shows only 1 node
**Symptom**: Server sees only itself  
**Fix**: Check firewall rules, verify TCP connection with `Test-NetConnection`

### Tensor announce fails
**Symptom**: `announceRemoteTensor: announcement failed`  
**Fix**: Check Vulkan external memory extensions are enabled

### Export handle type mismatch
**Symptom**: `exportMemory returned failed`  
**Fix**: AMD/Intel use DMA-BUF, NVIDIA uses OPAQUE_FD on Linux

## See Also

- `scripts/test_amd_rdma.sh` - Automated AMD GPU-direct RDMA network test
- `scripts/wsl2_rxe_kernel.sh` - Build a WSL2 kernel with Soft-RoCE (`rdma_rxe`)
- `scripts/softroce_persist.sh` - Soft-RoCE link persistence (systemd)
