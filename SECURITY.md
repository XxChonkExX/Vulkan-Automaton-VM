# Security Policy

## Supported versions

| Version | Support |
|---|---|
| 0.3.x | Security fixes for the Core (Chonk Buffer) and wire-protocol parser |
| <= 0.2.x | Experimental snapshots — **not supported**, upgrade |

## Reporting a vulnerability

Please use **GitHub Security Advisories** ("Report a vulnerability" on the
Security tab) — private by default. Include:

- Affected component (core pool / network transport / PyTorch integration /
  Android)
- Deployment trust level from [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) §1
  (single machine / private cluster / untrusted)
- A reproducer or PoC if possible

You will get an acknowledgment within a few days and a fix timeline for
confirmed issues. Credit in the release notes unless you prefer otherwise.

## Honest security posture (read before deploying)

- **The network transport is NOT production-hardened against untrusted
  peers yet.** The wire parser has hard limits (magic/version/type checks,
  absolute size caps), but system-level defenses — connection caps, socket
  timeouts, chunked staging, cluster membership authorization — are tracked
  in [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) §5. Until those ship,
  deploy only on trusted networks or behind access control (firewall /
  namespace / VPN).
- **TLS is optional** and certificate paths are user-supplied. Plaintext
  mode has no peer authentication.
- The core memory pool trusts its own process (same trust model as
  Vulkan/CUDA). It is not a sandbox boundary.

## Scope

In scope: memory safety of the pool/allocator/transport parsers, protocol
handling, external-memory handle lifecycle, TLS usage.
Out of scope: GPU driver vulnerabilities (report to AMD/Intel/NVIDIA),
the training scripts' local file handling, denial-of-service of your own
single-user local deployment.
