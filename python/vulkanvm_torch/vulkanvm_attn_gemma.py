# vulkanvm_attn_gemma.py
# Gemma-4 training-time tiled causal attention for the Chonk Buffer.
#
# Sibling of vulkanvm_attn_granite.py, same proven skeleton:
#   * pure torch (HIP BLAS) -- NO Triton, NO aotriton flash, NO vendor SDPA.
#     Vendor fused paths through Chonk dma-buf memory wedge the gfx1151 ring
#     (this module's Granite header documents the same failure); this kernel
#     never touches them.
#   * Flash-style online softmax over K-tiles: exact full-row softmax,
#     workspace O(B*kvH*g*QCHUNK*KTILE) regardless of sequence length.
#   * no_grad forward + hand-written exact-recompute backward (the Granite
#     ratchet fixes apply verbatim: detached stashed out, no graph in forward).
#
# Deltas vs the Granite module:
#   * No KV cache / cur_len: training forward sees the full packed block, so
#     the Q axis is ALSO tiled (QCHUNK, default 4096) -- scores never exceed
#     one (QCHUNK x KTILE) fp32 cell per visited tile.
#   * Per-layer sliding window (Gemma4: 40x sliding-1024 + 8x full): each
#     q-chunk visits only k-tiles inside its window -- sliding layers touch
#     ~2 tiles per chunk instead of all of them.
#   * Softcap replication (Gemma attention logit softcapping).
#   * Analytic causal+window mask per cell from position offsets -- no 4D
#     attention-mask tensor (which would be 2GB bf16 at 32k). If HF hands us
#     a non-None attention_mask we slice-add it on top; when None (packed
#     batches, no padding) the analytic mask carries causality alone.
#
# Interface: patches transformers.models.gemma4.modeling_gemma4
# .eager_attention_forward, same signature/contract (returns (out, None)).
#
# Geometry (Gemma4-12B): 16 Q heads / 8 KV heads (g=2), head_dim 256.
# Tiled cell fp32 at defaults (QC=4096, KT=8192): 8*2*4096*8192*4B = 1.07 GB.

import os
import torch
import torch.nn.functional as F

_QCHUNK = int(os.environ.get("CHONK_Q_CHUNK", "4096"))
_KTILE = int(os.environ.get("CHONK_ATTN_TILE", "8192"))


class GemmaAttnTiled(torch.autograd.Function):
    """Tiled exact causal attention (training, full block, no cache).

    forward inputs:
        q       : [B, H, T, D]   (full block, requires grad)
        k, v    : [B, kvH, T, D] (full block, requires grad)
        scaling : float
        softcap : float (0.0 = none)
        window  : int (0 = full attention, else sliding window width)
    Backward recomputes p_tile exactly from stashed (m, l, out) and
    accumulates dq per q-chunk, dk/dv over all visited k-tiles.
    """

    @staticmethod
    def forward(ctx, q, k, v, scaling, softcap, window):
        B, H, T, D = q.shape
        kvH = k.shape[1]
        g = H // kvH
        QC, KT = _QCHUNK, _KTILE
        dev = q.device
        DEBUG = os.environ.get("CHONK_ATTN_DEBUG", "0") == "1"
        ctx.scaling = float(scaling)
        ctx.softcap = float(softcap)
        ctx.window = int(window)
        ctx.n_groups = int(g)
        ctx.qchunk, ctx.ktile = QC, KT
        ctx.save_for_backward(q, k, v)

        with torch.no_grad():
            qg_full = q.view(B, kvH, g, T, D)
            m = torch.full((B, kvH, g, T), float("-inf"),
                           device=dev, dtype=torch.float32)
            l = torch.zeros(B, kvH, g, T, device=dev, dtype=torch.float32)
            acc = torch.zeros(B, kvH, g, T, D, device=dev,
                              dtype=torch.float32)
            out = torch.empty(B, H, T, D, device=dev, dtype=q.dtype)

            n_qchunks = (T + QC - 1) // QC
            for cq in range(n_qchunks):
                q0, q1 = cq * QC, min((cq + 1) * QC, T)
                qc = q1 - q0
                qg = qg_full[:, :, :, q0:q1, :]          # view
                # k-tile range visible to this q-chunk
                if ctx.window > 0:
                    k_lo = max(0, (q0 - ctx.window) // KT * KT)
                else:
                    k_lo = 0
                k_hi = q1  # causal: no tile starts at/after q1
                mq = m[:, :, :, q0:q1]
                lq = l[:, :, :, q0:q1]
                accq = acc[:, :, :, q0:q1, :]
                for k0 in range(k_lo, k_hi, KT):
                    k1 = min(k0 + KT, k_hi)
                    kt = k[:, :, k0:k1]                 # view
                    vt = v[:, :, k0:k1]
                    s = torch.matmul(
                        qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                    if ctx.softcap > 0:
                        s = torch.tanh(
                            s.float() / ctx.softcap) * ctx.softcap
                    s_f = s.float()
                    if DEBUG and not torch.isfinite(s_f).all():
                        print(f"[attn-debug] S_NANINF at "
                              f"cq[{q0}:{q1}] kt[{k0}:{k1}] win={ctx.window} "
                              f"qg_fin={bool(torch.isfinite(qg).all())} "
                              f"kt_fin={bool(torch.isfinite(kt).all())} "
                              f"smax={float(s_f.abs().max())} "
                              f"qmax={float(qg.abs().max())} "
                              f"kmax={float(kt.abs().max())} "
                              f"qstride={tuple(qg.stride())} "
                              f"kshape={tuple(kt.shape)} scaling={ctx.scaling}",
                              flush=True)
                    # analytic causal + window mask for this cell
                    qp = torch.arange(q0, q1, device=dev).view(-1, 1)
                    kp = torch.arange(k0, k1, device=dev).view(1, -1)
                    ok = kp <= qp
                    if ctx.window > 0:
                        ok = ok & ((qp - kp) < ctx.window)
                    s_f = s_f.masked_fill(~ok, float("-inf"))
                    m_new = torch.maximum(mq, s_f.amax(dim=-1))
                    # NaN guard: a row fully masked in this tile AND empty so
                    # far has mq == m_new == -inf, and exp(-inf - -inf) is NaN
                    # (fires on sliding layers once rows outrun the first tile
                    # by more than the window). An empty row must keep prior
                    # state (corr=1) and contribute zero mass (p_t=0).
                    valid = torch.isfinite(m_new)
                    corr = torch.exp(torch.where(
                        valid, mq - m_new, torch.zeros_like(mq)))
                    p_t = torch.exp(s_f - m_new.unsqueeze(-1))
                    p_t = torch.where(valid.unsqueeze(-1), p_t,
                                      torch.zeros_like(p_t))
                    if DEBUG:
                        tag = (f"cq[{q0}:{q1}] kt[{k0}:{k1}] win={ctx.window}")
                        for nm, t in (("m_new", m_new), ("corr", corr),
                                      ("p_t", p_t)):
                            if not torch.isfinite(t).all():
                                print(f"[attn-debug] non-finite {nm} at "
                                      f"{tag} nbad={int((~torch.isfinite(t)).sum())}",
                                      flush=True)
                    lq = lq * corr + p_t.sum(dim=-1)
                    accq = accq * corr.unsqueeze(-1) + \
                        torch.matmul(p_t.to(vt.dtype),
                                     vt.unsqueeze(2)).float()
                    mq = m_new
                out[:, :, q0:q1, :] = (
                    accq / lq.unsqueeze(-1)).to(q.dtype).reshape(B, H, qc, D)
                m[:, :, :, q0:q1] = mq
                l[:, :, :, q0:q1] = lq
                acc[:, :, :, q0:q1, :] = accq

        # Ratchet-proof stash (see Granite module header): out DETACHED.
        ctx.m_final = m
        ctx.l_final = l
        ctx.out = out.detach()
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        m = ctx.m_final
        l = ctx.l_final
        out = ctx.out
        B, H, T, D = q.shape
        kvH = k.shape[1]
        g = ctx.n_groups
        QC, KT = ctx.qchunk, ctx.ktile
        dev, qdt = q.device, q.dtype

        qg_full = q.view(B, kvH, g, T, D)
        dyg = dout.view(B, kvH, g, T, D)
        delta = (dyg.float() *
                 out.view(B, kvH, g, T, D).float()).sum(dim=-1)
        inv_l = (1.0 / l).unsqueeze(-1)

        dq = torch.zeros(B, kvH, g, T, D, device=dev, dtype=torch.float32)
        dk = torch.zeros(B, kvH, T, D, device=dev, dtype=torch.float32)
        dv = torch.zeros(B, kvH, T, D, device=dev, dtype=torch.float32)

        n_qchunks = (T + QC - 1) // QC
        for cq in range(n_qchunks):
            q0, q1 = cq * QC, min((cq + 1) * QC, T)
            qg = qg_full[:, :, :, q0:q1, :]
            dyg_c = dyg[:, :, :, q0:q1, :]
            delta_c = delta[:, :, :, q0:q1]
            mq = m[:, :, :, q0:q1]
            inv_c = inv_l[:, :, :, q0:q1, :]
            dq_c = torch.zeros(B, kvH, g, q1 - q0, D, device=dev,
                               dtype=torch.float32)
            if ctx.window > 0:
                k_lo = max(0, (q0 - ctx.window) // KT * KT)
            else:
                k_lo = 0
            for k0 in range(k_lo, q1, KT):
                k1 = min(k0 + KT, q1)
                kt = k[:, :, k0:k1]
                vt = v[:, :, k0:k1]
                s = torch.matmul(
                    qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                if ctx.softcap > 0:
                    s = torch.tanh(s.float() / ctx.softcap) * ctx.softcap
                s_f = s.float()
                qp = torch.arange(q0, q1, device=dev).view(-1, 1)
                kp = torch.arange(k0, k1, device=dev).view(1, -1)
                ok = kp <= qp
                if ctx.window > 0:
                    ok = ok & ((qp - kp) < ctx.window)
                # masked tiles recompute the same -inf cells; they
                # contribute exp(-inf - m) = 0 to p_t, so plain masking
                # of s_f is exact (matches forward cell math bit-wise).
                s_f = s_f.masked_fill(~ok, float("-inf"))
                p_t = torch.exp(s_f - mq.unsqueeze(-1)) * inv_c
                dp_t = torch.matmul(
                    dyg_c, vt.unsqueeze(2).transpose(-2, -1)).float()
                ds_t = p_t * (dp_t - delta_c.unsqueeze(-1))
                ds_bf = ds_t.to(qdt)
                dq_c += torch.matmul(
                    ds_bf, kt.unsqueeze(2)).float() * ctx.scaling
                dk[:, :, k0:k1, :] += torch.matmul(
                    ds_bf.transpose(-2, -1), qg).sum(dim=2).float() * \
                    ctx.scaling
                dv[:, :, k0:k1, :] += torch.matmul(
                    p_t.to(qdt).transpose(-2, -1), dyg_c).sum(dim=2).float()
            dq[:, :, :, q0:q1, :] = dq_c

        return (
            dq.to(qdt).reshape(B, H, T, D),
            dk.to(qdt),
            dv.to(v.dtype),
            None, None, None,
        )


def patch_gemma_attention_tiled(model):
    """Swap Gemma4's eager attention for the Chonk tiled path.

    Same contract as the Granite patch: called where HF would call
    eager_attention_forward (module-first signature). Reads per-layer
    geometry from the module + enclosing config, applies the optional
    HF-built attention_mask slice on top of the analytic cell mask.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    txt = base.model if hasattr(base, "model") else base
    layers = list(txt.layers)
    id2idx = {id(l.self_attn): i for i, l in enumerate(layers)}
    cfg = txt.config if hasattr(txt, "config") else base.config
    layer_types = list(getattr(cfg, "layer_types", ["full_attention"] * 48))
    sw = int(getattr(cfg, "sliding_window", 0) or 0)

    try:
        import transformers.models.gemma4.modeling_gemma4 as gm
    except ImportError:
        import modeling_gemma4 as gm
    orig = gm.eager_attention_forward

    # Dispatch lever: modeling_gemma4 resolves the interface FRESH every
    # forward via ALL_ATTENTION_FUNCTIONS.get(impl, <module-global default>).
    # The default is looked up in the module namespace at CALL time, so
    # replacing the global takes effect -- but ONLY when impl == "eager"
    # (the model ships "sdpa"). NOTE: per-instance rebinding
    # (layer.self_attn.attention_interface = ...) is DEAD CODE here -- the
    # forward never reads self.attention_interface, unlike older HF models.
    base_cfg = getattr(txt, "config", getattr(base, "config", None))
    if base_cfg is not None:
        base_cfg._attn_implementation = "eager"
    assert getattr(base_cfg, "_attn_implementation", None) == "eager", \
        "could not force eager dispatch"

    def patched(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        if dropout and dropout > 0:
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout=dropout, **kwargs)
        idx = id2idx.get(id(module), -1)
        lt = layer_types[idx] if 0 <= idx < len(layer_types) \
            else "full_attention"
        window = sw if lt == "sliding_attention" else 0
        g = int(getattr(module, "num_key_value_groups",
                        query.shape[1] // key.shape[1]))
        softcap = float(kwargs.get("softcap", 0.0) or 0.0)
        out = GemmaAttnTiled.apply(query, key, value, float(scaling),
                                   softcap, window)
        return out, None

    gm.eager_attention_forward = patched
    print(f"[+] Gemma4 eager attention -> Chonk tiled "
          f"(QC={_QCHUNK} KT={_KTILE}, dispatch=eager, "
          f"layers={len(layers)})",
          flush=True)
    return patched
