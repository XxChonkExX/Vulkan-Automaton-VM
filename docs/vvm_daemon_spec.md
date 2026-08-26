# vvm-daemon — Interface Specification v1

One long-lived process per device owning everything that should outlive apps:
the Vulkan device + Chonk pool, the cluster control plane, and the fabric data
plane (UDP mini-verbs today, verbs/ND tomorrow). Apps talk to it over a local
socket with tiny text messages; **heavy data never crosses IPC** — it moves
through the fabric directly between pools.

## Transport

| | v1 | v1.1 |
|---|---|---|
| Socket | TCP `127.0.0.1:<ctrl>` (adb-friendly, Termux/Kotlin/curl-able) | + abstract unix `\0vvm.daemon` |
| Framing | newline-delimited `KEY=VALUE`; blank line ends multi-line replies | same |
| Auth | none — loopback only | token handshake |

## Commands (v1)

| Command | Reply | Notes |
|---|---|---|
| `PING` | `OK pong=1` | liveness |
| `STATUS` | `OK device=… backend=… regions=N conns=M` | |
| `REGION new size=N` | `OK id=R rkey=K bytes=N` | host region (heap-backed); rkey is fabric-sharable |
| `REGION list` | `BEGIN`…`id=… rkey=… bytes=…`…`END` | |
| `PUT id=R off=O hex=<..>` | `OK bytes=N` | local write (≤2 KiB/call); seed patterns here |
| `GET id=R off=O len=N` | `OK hex=…` | local read-back / verification |
| `CONN add ip=A dataport=P` | `OK id=C` | registers a fabric peer (data port, not control) |
| `PUSH conn=C rkey=K src=R soff=S doff=D len=N` | `OK bytes=N` | RDMA_WRITE: R→peer's K |
| `PULL conn=C rkey=K dst=R doff=D soff=S len=N` | `OK bytes=N` | RDMA_READ: peer's K→R |
| `QUIT` | connection closes | |

Error shape: `ERR code=<n> msg=<text>`.

## CLI

```
vvm_daemon [--ctrl PORT] [--data-port PORT]
```
`--data-port` feeds `NetworkConfig.listenAddress` for the fabric transport
(UVP binds ctrl+1 by convention when unset). Control plane is TCP; the fabric
is whatever `RdmaTransport::create()` selects (`VVM_RDMA_BACKEND` honored).

## Roadmap (v2+, in order)

1. `INIT gpu=1` — full UnifiedMemoryPool on-device (pattern proven by
   basic_test on Adreno 750); `ALLOC/FREE` against it; `EXPORT` publishes a
   pool allocation's rkey.
2. `CONN` upgrades to cluster join (`MultiNodePoolManager`) once two-daemon
   heartbeat flows are validated.
3. `TENSOR put/get/allreduce` — tensor transport over the fabric.
4. `MODEL load/pull/infer` — registry publish/fetch + sharded inference
   (phone hosts layers A, tablet layers B, tokens pipeline over the fabric).

## Hardening backlog

- Loopback TCP is unauthenticated by design (device-local); before any
  non-loopback exposure: token auth + DTLS on the fabric (THREAT_MODEL §net).
- Samsung process-death mitigation: foreground service wrapper (app-side).
