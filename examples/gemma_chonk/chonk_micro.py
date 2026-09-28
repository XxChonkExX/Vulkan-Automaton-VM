#!/usr/bin/env python3
"""chonk_micro.py -- surgical micro-repro for the vanishing flat buffer.

Tests, in one process:
  1. write/read coherence of a pool alloc right after alloc (copy_ H2D,
     kernel read, D2H read, hostPtr mapping read)
  2. the SAME checks after GPU allocation churn (does the pool evict or
     reuse live allocations under growth pressure?)
  3. the exact production path: .view(bf16).narrow(...).view(shape) ->
     nn.Parameter -> F.embedding
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python", "vulkanvm_torch"))
sys.path.insert(0, os.path.join(REPO, "_build"))

import ctypes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chonk import ChonkPool, install_chonk_allocator

MARK = 7.0


def host_view(host_ptr, nbytes):
    arr = np.ctypeslib.as_array(
        ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_ubyte)), shape=(nbytes,))
    u16 = np.frombuffer(arr, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def check(tag, buf_bf16, host_np, n=1024):
    kmax = float(buf_bf16[:n].float().max().item())       # kernel read
    dmax = float(buf_bf16[:n].cpu().float().max().item())  # D2H read
    hmax = None
    if host_np is not None:
        hmax = float(host_np[:n].max())
    ok = kmax == MARK
    print(f"{tag}: kernel={kmax} d2h={dmax} host={hmax} "
          f"{'OK' if ok else '*** KERNEL SEES STALE/ZERO ***'}", flush=True)
    return ok


def main():
    install_chonk_allocator()
    pool = ChonkPool()

    # small alloc with host mapping retained
    base_s, host_s = pool.alloc_base(8 << 20, "micro_small")
    hs_np = host_view(host_s, 8 << 20) if host_s else None
    t_s = base_s.view(torch.bfloat16)
    t_s[:1024].copy_(torch.full((1024,), MARK, dtype=torch.bfloat16))
    check("small/fresh      ", t_s, hs_np)

    # flat-buffer-sized alloc (the one that vanishes in production)
    base_f, host_f = pool.alloc_base(2_010_000_000, "micro_flat")
    hf_np = host_view(host_f, 2_010_000_000) if host_f else None
    t_f = base_f.view(torch.bfloat16)
    t_f[:1024].copy_(torch.full((1024,), MARK, dtype=torch.bfloat16))
    check("flat/fresh       ", t_f, hf_np)

    # exact production view path: view->narrow->reshape->Parameter->embedding
    ids = torch.arange(0, 16, dtype=torch.long, device="cuda")
    w = base_f.view(torch.bfloat16).narrow(
        0, 0, 16 * 3840).view((16, 3840)).clone()
    w.fill_(MARK)
    p = nn.Parameter(w)
    out = F.embedding(ids, p)
    print(f"embedding kernel read: max={float(out.max().item())} "
          f"{'OK' if float(out.max().item()) == MARK else '*** ZERO ***'}",
          flush=True)
    del w, p, out, ids
    torch.cuda.empty_cache()

    # churn: force pool growth/pressure with torch allocations (routed
    # through the chonk pluggable allocator), then re-check both buffers
    for i in range(12):
        junk = torch.empty(200_000_000, dtype=torch.uint8, device="cuda")
        junk[-1024:].view(torch.bfloat16).fill_(float(i))
        del junk
        if i % 4 == 3:
            torch.cuda.empty_cache()
    ok_s = check("small/after churn", t_s, hs_np)
    ok_f = check("flat/after churn ", t_f, hf_np)

    # kernel-write direction test: fill via kernel, read via D2H + host
    t_f[:1024].fill_(MARK - 1.0)
    d = float(t_f[:1024].cpu().float().max().item())
    h = float(hf_np[:1024].max()) if hf_np is not None else None
    print(f"kernel-write: d2h={d} host={h} "
          f"{'OK' if d == MARK - 1.0 else '*** KERNEL WRITES INVISIBLE ***'}",
          flush=True)

    print("stats:", pool.stats(), flush=True)
    return 0 if (ok_s and ok_f) else 1


if __name__ == "__main__":
    sys.exit(main())
