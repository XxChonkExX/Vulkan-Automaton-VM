# UVP — the UDP mini-verbs wire protocol (v1)

*"Roads for the GPU-poor": a dependency-free datagram protocol that gives any
device with a UDP stack — Android phones, junk tablets, Raspberry Pis, Macs,
old PCs — RDMA-shaped semantics inside VulkanVM's `RdmaTransport` interface.
No kernel modules, no root, no vendor drivers.*

## Design goals

1. **Runs anywhere** — plain POSIX/BSD sockets; works from `adb shell`, Termux,
   locked bootloaders, unprivileged accounts.
2. **Drop-in** — implements the exact `RdmTransport` virtual surface, so the
   cluster manager, tensor collectives, and tests consume it unchanged.
3. **Safe by construction** — transfers are host-memory copies; a lost packet
   or dead peer can never corrupt GPU state or trigger the DMA-use-after-free
   class of bugs (see Phase 7 timeout work — not applicable here because the
   "NIC" is us).
4. **Correctness first, throughput second** — v1 uses Go-Back-N over large
   datagrams; congestion control and selective-repeat are roadmap items.

## Frame layout (little-endian)

```
offset  size  field
0       2     magic     = 0x5556 'UV'
2       1     version   = 1
3       1     type      = DATA | ACK | REGISTER | UNREGISTER | READREQ | PING
4       4     xid       operation id (unique per transfer, sender-chosen)
8       4     seq       datagram index within the operation
12      4     totalPkts number of DATA packets in the operation
16      2     payloadLen bytes of payload following this header
18      2     flags     reserved
20      8     regionId  the "rkey": identifies a registered region
28      8     offset    byte offset within the region
---
36      ..    payload
```

Datagram payload budget: **60000 bytes total**, i.e. ~59,964 payload bytes per
DATA packet. A 16 MiB transfer ≈ 280 packets.

## Region model

| Call | Behavior |
|---|---|
| `registerHostMemory(ptr, size)` | Registers the caller's buffer under a fresh `regionId`. Peer writes/reads hit this memory directly through the owning instance. |
| `registerGpuMemory(...)` | Returns `nullopt` in v1 (with a warning). Device contents cannot be reached from a CLI context; the cluster manager falls back to host-staged automatically. Future: hook the offload engine to sync a shadow region. |
| `rkey` handed to peers | Is simply the `regionId`. It travels the existing TCP control plane (`RemoteAllocationDesc.rkey`) unchanged. |

Staging rule: DATA packets for a region the receiver hasn't seen are dropped
with a warning — senders must have received the regionId through the control
plane, which implies the owner's `registerHostMemory` already ran.

## Transfer algorithms

**WRITE (push)** — `rdmaWrite(conn, localRegion, remoteAddr, rkey=regionId, size)`:
1. Sender streams DATA packets `[0..totalPkts)` reading from
   `localRegion.addr + i*kChunk`, `offset` added to the region-relative
   destination.
2. Go-Back-N: window of 256 packets in flight; receiver ACKs
   `{xid, highestContiguousSeq}` on every in-order arrival.
3. On ACK timeout (50 ms tick) the window is resent from the first missing
   packet. Deadline (`timeoutNs`) aborts with failure; the receiver may hold a
   partially written region — same visibility rules as a torn verbs WRITE
   against an unregistered retry.
4. Done when the final packet's ACK arrives.

**READ (pull)** — `rdmaRead(conn, localRegion, remoteAddr, rkey, size)`:
1. Sender emits `READREQ{xid, regionId, offset, totalPkts}`.
2. The OWNER streams DATA from its registered buffer toward the requester;
   requester ACKs identically.
3. Bytes land at `localRegion.addr`.

## Addressing

`connect(host, port)` records the peer endpoint; operations carry it via the
connection handle. One UDP socket per transport instance, bound to
`(listenAddress port + kRdmaPortOffset)` — same convention as the verbs
transport, so cluster wiring is identical.

## Deliberate non-goals (v1)

- Encryption/authentication — run on trusted LANs only until TLS/DTLS lands
  (tracked alongside docs/THREAT_MODEL.md).
- Congestion avoidance beyond fixed-window GBN.
- Zero-copy (recvmmsg/AF_XDP) — measured optimization once correct.
- GPU-direct — impossible from userspace on Android; host-staged only.

## Roadmap

- v1.1: selective-repeat (SACK bitmap) for lossy links.
- v1.2: DTLS option keyed off `NetworkConfig` TLS settings.
- v2: shared-memory fast path when peers share a host (loopback).
