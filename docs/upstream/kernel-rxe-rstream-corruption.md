# rxe (Soft-RoCE): rsockets data corruption at a deterministic offset under load

**Target:** linux-rdma / kernel rxe — file against
linux-rdma@vger.kernel.org (and/or github.com/linux-rdma/rdma-core issue for
the userspace side), CC linux-rdma maintainers.
**Status:** draft — re-verify on current stable first (kernel moves fast; the
session doc notes active 2026-era rxe CVEs on this tree).

## Summary

On kernel 7.0.0-30-generic with Soft-RoCE (`rdma_rxe`) bound to a physical
NIC, the rdma-core `rstream` ping-pong tool completes its latency/bandwidth
matrix but **fails data verification deterministically at byte 2,177,536,80**
(~208 MiB into the default 1 GiB transfer) when run with `-T v`. Server side
reports `rrecv: Invalid argument` storms and `Connection reset by peer`.

The failure offset is identical across runs. Latency numbers are healthy up to
that point (3.9 µs @64B → 5.2 Gb/s @1 MiB), i.e., the connection is fast and
stable right until the corruption boundary.

## Environment

- Ubuntu 26.04 (Resolute), kernel 7.0.0-30-generic
- rdma-core 61.0-2ubuntu3 (libibverbs/librdmacm userspace)
- `rdma link add rxe0 type rxe netdev enp14s0` (Intel I226-V onboard, X870E)
- Loopback-style traffic to the NIC's own LAN IP (no external peer)

## Repro

```bash
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev <NIC>
LANIP=$(ip -4 -o addr show dev <NIC> | awk '{print $4}' | cut -d/ -f1)

# terminal 1
rstream -b $LANIP -p 7473
# terminal 2
rstream -s $LANIP -p 7473 -T v
```

Observed (client):

```
name      bytes   xfers   iters   total       time     Gb/sec    usec/xfer
64_lat    64      1       100k    12m         0.78s      0.13       3.91
4k_lat    4k      1       10k     78m         0.19s      3.52       9.30
64k_lat   64k     1       1k      125m        0.21s      5.04     103.93
1m_lat    1m      1       100     200m        0.32s      5.22    1607.71
data verification failed byte 217753680
data verification failed byte 217753680
...
```

Server log tail: `rsend: Connection reset by peer`, repeated
`rrecv: Invalid argument`.

## Additional observations

- `ucmatose` (raw rdma_cm + verbs ping-pong) passes fully on the same link.
- A raw-verbs RDMA_WRITE transport built on `ibv_reg_mr` over the same rxe0
  passes an end-to-end cross-vendor GPU data-integrity test repeatedly
  (repo test `multi_vendor_rdma_test`, VVM_RDMA_CONNECT_HOST=<LAN IP>).
- The corruption therefore appears specific to the **rsockets layer over rxe**
  at larger transfer volumes; raw verbs is unaffected in our testing.
- Port choice matters only in that stale listeners from aborted runs wedge
  later connects (use a fresh `-p`); unrelated to the corruption itself.

## Notes for triage

- Deterministic byte offset suggests an accounting/offset bug (chunk boundary
  or wrap) rather than random DMA damage; ~208 MiB ≈ 208,873,472 — close to
  but not exactly 200 MiB; possibly a 217.7 MB (decimal-MB) boundary.
- Happy to run instrumented kernels/patches; box is a standing test node.
- Related context: our session notes flag kernel-7.0 rxe CVE activity
  (e.g., CVE-2026-46133 referenced in our docs) — this may share root cause.
