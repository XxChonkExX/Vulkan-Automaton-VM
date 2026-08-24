# VulkanVM Network Threat Model

> Status: **0.3 draft**. The wire-format parser is hardened (magic, version,
> message-type validation, size limits with absolute hard caps). This document
> covers the *system-level* attack surface around that parser, what is
> defended today, and what remains open.

---

## 1. Trust model

| Deployment | Trust level | Assumptions |
|---|---|---|
| Single machine (loopback, IPC) | Trusted | No network exposure |
| Private cluster (LAN, owned machines) | Semi-trusted | Peers identified by config; TLS optional; operators are the same party |
| Mixed / semi-public pool | **Untrusted** | Any peer may be malicious; TLS required; every resource limit must hold under attack |

The library must be *safe* (no memory corruption, no UB) in all three. It is
*secure* (resource and data protection) only to the extent listed below —
deploying in the untrusted model today requires the mitigations in §4 and
acceptance of the open gaps in §5.

## 2. Assets

1. GPU memory contents in flight (model weights, tensors, KV cache)
2. Pool/host memory stability (crash = DoS)
3. Cluster control plane (membership, placement decisions)
4. Host resources: sockets, fds, staging buffers, threads

## 3. Attack surface & current defenses

### 3.1 Wire protocol (defended)

| Defense | Status |
|---|---|
| Magic + protocol version check | ✅ |
| Message-type validation (reject unknown) | ✅ |
| Flag validation | ✅ |
| Configurable size limits **with absolute hard caps** | ✅ |
| `streamLen > kMaxStreamSize` cannot be disabled by config | ✅ |
| 1 GiB body cap, 16 GiB stream cap | ✅ |
| Partial-read/write correctness (`writeAll`/`readAll` loop until complete) | ✅ |

### 3.2 Known open areas (this section = the work list)

| Attack | Vector | Status |
|---|---|---|
| Connection flood | thousands of parallel TCP connects | ⚠️ accept-loop has no per-peer connection cap or rate limit |
| Slow-loris | valid headers sent byte-by-byte | ⚠️ socket read has no idle/header timeout by default |
| TLS handshake flood | repeated `SSL_accept` without completion | ⚠️ same: no handshake timeout budget |
| Legal-size disconnect | request a legal 1 GiB body, disconnect mid-stream | ⚠️ staging buffer is sized for the full body up-front → attacker controls host allocation |
| Staging exhaustion | many concurrent 1 GiB transfers | ⚠️ concurrent staging allocations are unbounded |
| Heartbeat flood | heartbeat spam | ⚠️ no per-peer heartbeat rate limit |
| Node impersonation | claim a peer ID | ✅ TLS cert identity check when TLS enabled; ❌ plaintext mode has no auth |
| Unauthorized cluster member | valid cert, not in cluster | ❌ no membership authorization layer (cert validity ≠ cluster membership) |
| Cross-peer memory confusion | malicious placement/migration requests | ⚠️ placement requests validated for shape but not for authorization |

## 4. Required configuration for untrusted deployments (today)

1. **TLS mandatory** (`VVM_NETWORK_HAS_TLS` build + cert config on both ends).
2. Firewall/namespace the cluster: only cluster peers can reach the port
   (the strongest control available until membership auth lands).
3. Run each peer under a memory cap (cgroup/job object) so staging
   exhaustion cannot take down the host.
4. Prefer host-staged TCP with small `maxBodySize` config over RDMA for
   untrusted peers (RDMA registration exposes pinned host memory).

## 5. Hardening roadmap (0.3+)

Ordered by severity:

1. **Socket timeouts** — header/idle timeout (default 30 s) and handshake
   timeout on both TLS and plaintext accept paths. (Small fix, closes the
   two cheapest DoS vectors.)
2. **Per-peer connection cap + accept rate limit** — e.g. 8 concurrent
   connections per source IP, token-bucket on new connections.
3. **Chunked staging** — never allocate body-size staging up front; stream
   into fixed 4 MiB slices (matches the existing streaming design) so a
   declared 1 GiB body cannot force a 1 GiB host allocation before a single
   byte is validated as arriving.
4. **Cluster membership authorization** — after TLS identity, check the
   cert fingerprint/CN against the configured cluster roster; reject with a
   distinct error. Config format: `cluster.members = <fingerprint list>`.
5. **Heartbeat rate limiting** — per-peer token bucket; violations drop the
   connection.
6. **Fuzz the wire parser** — libFuzzer target over
   `deserializeHeader/parseMessage` with the hard caps asserted as
   invariants (complements the existing unit tests).

## 6. Non-goals

- Encrypted-at-rest storage of models (out of scope; the filesystem owns it).
- Protection against a malicious *local* embedder (the library trusts its
  own process — same as Vulkan/CUDA).
- Anonymity: cluster peers always know each other's addresses.

## 7. Reporting

Security-relevant bugs: open a GitHub issue marked `security` or contact the
maintainer directly. Please include the deployment trust level from §1.
