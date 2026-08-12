# Cross-Machine GPU Networking

VulkanVM supports sharing GPU memory across machines over TCP. This enables distributed AI inference where tensors can be transferred between GPUs on different computers.

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

### Run Server (Evo-X2 / Linux)

```bash
./build/examples/tensor_server_test --port 51000 --verbose
```

### Run Client (Windows / WSL)

```powershell
.\build\examples\tensor_client_test.exe --server 192.168.0.117 --port 51000 --local-port 51005
```

## Command Line Options

### Server (`tensor_server_test`)

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 51000 | TCP port to listen on |
| `--size-mb` | 16 | Size of test tensor in MB |
| `--verbose` | off | Enable detailed operation logging |
| `--help` | | Show help |

### Client (`tensor_client_test`)

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | 192.168.0.117 | Server IP address |
| `--port` | 51000 | Server port |
| `--local-port` | 51005 | Local listen port |
| `--help` | | Show help |

## Requirements

- **Network**: Both machines on same LAN (or routable)
- **Vulkan**: GPU with `VK_KHR_external_memory` support
- **Linux**: `VK_KHR_external_memory_fd` for DMA-BUF/OPAQUE_FD export
- **Windows**: `VK_KHR_external_memory_win32` for OPAQUE_WIN32 export
- **Firewall**: Allow inbound TCP on the chosen port

## Transport Path Selection

VulkanVM automatically selects the best transport path:

| Priority | Path | Requirements |
|----------|------|--------------|
| 1 | P2P | Same machine, driver support |
| 2 | RDMA | InfiniBand/RoCE hardware + `nvidia-peermem` |
| 3 | Host-Staged | Always available (TCP) |
| 4 | UCX | UCX build + GPU memory registration |

For cross-machine without RDMA, **Host-Staged TCP** is used (4 MB chunks).

## Verified Configurations

| Client | Server | Status |
|--------|--------|--------|
| Windows (Intel Arc B70) | Linux/AMD Strix Halo | ✅ Working |
| Windows (AMD 7900 XTX) | Linux/AMD Strix Halo | 🔜 Pending |
| WSL/Ubuntu | Linux/AMD Strix Halo | 🔜 Pending (SoftRoCE) |

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

- `docs/network.md` - Protocol specification
- `docs/rdma.md` - RDMA setup guide
