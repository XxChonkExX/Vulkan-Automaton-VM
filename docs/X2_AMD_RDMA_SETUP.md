# Evo-X2 Setup — AMD GPU-Direct RDMA Test (Server Side)

Target: **Evo-X2** (192.168.0.117, AMD Strix Halo 395 / Radeon 890M, Ubuntu).
This is the **server** side of the cross-machine AMD RDMA test. The client is
the Ubuntu box (192.168.0.213) running `tensor_client_test`.

## Step 1 — Update repo on X2

```bash
cd <repo>                      # wherever Vulkan-Automaton-VM is checked out
git pull origin main           # must be >= latest (--rdma-nic + announce-count + Linux client)
git log --oneline -3
```

> If the repo isn't cloned yet:
> `git clone https://github.com/XxChonkExX/Vulkan-Automaton-VM.git && cd Vulkan-Automaton-VM`

## Step 2 — Run setup (deps + Soft-RoCE)

```bash
sudo ./scripts/test_amd_rdma.sh setup
```

This installs (idempotent, safe to re-run):
- build toolchain: `build-essential cmake ninja-build pkg-config git`
- Vulkan: `libvulkan-dev vulkan-tools mesa-vulkan-drivers`
- RDMA: `rdma-core libibverbs-dev librdmacm-dev`
- loads `rdma_rxe` and creates `rxe0` over the default-route interface

Verify: `sudo ./scripts/test_amd_rdma.sh env`

Expected markers:
- `ibv_devices` lists `rxe0`
- `rdma link show` shows `rxe0 ... state ACTIVE`
- `vulkaninfo` lists the Radeon 890M / gfx1151 device

## Step 3 — Build

```bash
sudo ./scripts/test_amd_rdma.sh build
```

Must print `Verbs transport enabled - RDMA path will be compiled in`.
Binaries: `build_rdma/examples/tensor_server_test` (+ `tensor_client_test`).

## Step 4 — Run the server

```bash
# Bounded run: announces 6 times (~12s window for the client) then exits.
./scripts/test_amd_rdma.sh server --announce-count 6

# Or interactive (Ctrl+C to stop):
./scripts/test_amd_rdma.sh server
```

Then on the client (Ubuntu box at .213):
```bash
./scripts/test_amd_rdma.sh client --server 192.168.0.117 --port 51000
```

## What "PASS" looks like (server log)

```
VerbsRdmaTransport initialized on device 'rxe0', RDMA listener port 51001
AMD GPUDirect via ibv_reg_dmabuf_mr: lkey=..., rkey=...     <- kernel >= 6.19
AMD GPUDirect via DMA-BUF mmap fallback: lkey=..., rkey=... <- kernel < 6.19
migrateFromRemote: pulled 16777216 bytes from ...
```

## Best practices

- **Run the server as your normal user, not root** — only `setup`/`build` need
  `sudo`. The server binds 51000/51001; if you get "address in use", kill the
  stale process: `pkill tensor_server_test`.
- **rdma_rxe link is NOT persistent across reboot.** After a reboot re-run
  `sudo ./scripts/test_amd_rdma.sh setup` (it detects the missing link and
  recreates it). For a permanent link, install the systemd unit:
  `sudo ./scripts/softroce_persist.sh install`.
- **Firewall**: allow inbound TCP 51000 (control) and 51001 (RDMA listener):
  `sudo ufw allow 51000/tcp && sudo ufw allow 51001/tcp` (or equivalent).
- **MTU**: rxe RoCEv2 uses UDP; keep the LAN MTU at 1500 (default). Don't mix
  with jumbo frames on the same L2 until verified.
- **Kernel < 6.19**: expect the `mmap fallback` line instead of
  `ibv_reg_dmabuf_mr` — that is expected and fine on the APU (unified memory).
- **Don't run a second server while testing** — the client connects by IP:port
  and will grab whichever is listening.
- **GPU memory**: the 890M shares system RAM; big `--size-mb` values compete
  with the desktop. Stick to 16–64 MB for tests.
- **Logs**: scripted runs tee to `/tmp/vvm_server_rdma.log` / `/tmp/vvm_client_rdma.log`.
  Grab them before rerunning; each run overwrites.

## ⚠️ Clean up reminders (remove later)

Things left in place that should be cleaned up once the test campaign is done:

- `rdma_rxe` kernel module loaded + `rxe0` link (if not needed anymore):
  `sudo rdma link del rxe0 && sudo modprobe -r rdma_rxe`
  (or `sudo ./scripts/softroce_persist.sh uninstall` if the service was installed)
- Apt packages installed by `setup` (keep while iterating; remove when done)
- `build_rdma/` build tree (safe to delete; rebuild with `build`)
- `/tmp/vvm_server_rdma.log`, `/tmp/vvm_client_rdma.log`
- This doc's "pending" status rows in `docs/cross_machine_gpu_sharing.md`
  ("Verified Configurations") — update to ✅ once the AMD RDMA path is confirmed.
