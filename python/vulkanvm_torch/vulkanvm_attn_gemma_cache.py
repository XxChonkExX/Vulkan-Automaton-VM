# vulkanvm_attn_gemma_cache.py
# Gemma cache-aware tiled attention for chunked Chonk training.
#
# Merge of two proven designs:
#   * vulkanvm_attn_gemma.GemmaAttnTiled (validated: forward match, backward
#     exact cos 0.9998+, masking bit-perfect, fully-masked-row NaN guard)
#   * GraniteAttnRecompute's cache interface (cur_len, layer.get_cached_kv,
#     current-chunk k_cur/v_cur with grad flow only on the chunk)
#
# The driver streams packed blocks in small chunks (1-2k) through a
# persistent pool KV cache. Per forward the model sees ONE chunk:
#   * q/k/v current: [B, H/hvH, qc, D] (qc ~1024-2048, requires grad)
#   * cached prefix: read from the layer in KT tiles (no grad, bf16 views
#     or INT4 shared-scratch dequant -- handled inside get_cached_kv)
#   * workspace per cell: [B,kvH,g,qc,KT] fp32 (~0.5-1GB, fixed forever)
# Nothing materializes at full-T: no TxT masks (chunk masks are <=260MB),
# no full-graph retention (chunk graphs only), no vendor kernels.
#
# Sliding windows ride along free: each q-chunk visits only k-tiles inside
# its window (same k_lo logic as the training kernel, offset by cur_len).

import os
import torch

_QC = int(os.environ.get("CHONK_CHUNK", "1024"))
_KTILE = int(os.environ.get("CHONK_ATTN_TILE", "8192"))


class GemmaAttnCache(torch.autograd.Function):
    """Tiled exact causal attention over (cached prefix + current chunk).

    forward inputs:
        q       : [B, H, qc, D]    (current chunk, requires grad)
        k_cur   : [B, kvH, qc, D]  (current chunk, grad flows)
        v_cur   : [B, kvH, qc, D]
        layer   : cache layer (get_cached_kv -> pool views / INT4 scratch)
        cur_len : int (cached prefix length; every cached column < cur_len
                  is visible to every query, so cached tiles need no mask)
        batch / scaling / n_groups / window / causal : metadata
    Backward recomputes p_tile exactly from stashed (m, l, out.detach())
    and computes dk_cur/dv_cur only on the current chunk (cached prefix
    has no grad path -- cross-chunk learning flows through shared params).
    """

    @staticmethod
    def forward(ctx, q, k_cur, v_cur, layer, cur_len, batch, scaling,
                n_groups, window, causal, softcap):
        KT = _KTILE
        ctx.layer = layer
        ctx.cur_len = int(cur_len)
        ctx.batch = int(batch)
        ctx.scaling = float(scaling)
        ctx.n_groups = int(n_groups)
        ctx.window = int(window)
        ctx.causal = bool(causal)
        ctx.softcap = float(softcap)
        ctx.ktile = KT
        ctx.save_for_backward(q, k_cur, v_cur)

        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]
        g = ctx.n_groups
        kc = ctx.cur_len
        dev, qdt = q.device, q.dtype

        with torch.no_grad():
            qg = q.view(B, kvH, g, qlen, D)
            m = torch.full((B, kvH, g, qlen), float("-inf"),
                           device=dev, dtype=torch.float32)
            l = torch.zeros(B, kvH, g, qlen, device=dev, dtype=torch.float32)
            acc = torch.zeros(B, kvH, g, qlen, D, device=dev,
                              dtype=torch.float32)

            def tile_step(kt, vt, k_base, masked_rows=None):
                """Fold one k-tile [k_base, k_base+w) into (m, l, acc)."""
                nonlocal m, l, acc
                s = torch.matmul(
                    qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                if ctx.softcap > 0:
                    s = torch.tanh(s.float() / ctx.softcap) * ctx.softcap
                s_f = s.float()
                # analytic causal + window mask for this cell, in GLOBAL
                # positions (queries live at [kc, kc+qlen)).
                qp = (torch.arange(qlen, device=dev) + kc).view(-1, 1)
                kp = (torch.arange(kt.shape[-2], device=dev)
                      + k_base).view(1, -1)
                ok = kp <= qp
                if ctx.window > 0:
                    ok = ok & ((qp - kp) < ctx.window)
                s_f = s_f.masked_fill(~ok, float("-inf"))
                m_new = torch.maximum(m, s_f.amax(dim=-1))
                # NaN guard (see vulkanvm_attn_gemma header): a row fully
                # masked in this tile AND empty so far must keep prior
                # state (corr=1) and contribute zero mass (p_t=0).
                valid = torch.isfinite(m_new)
                corr = torch.exp(torch.where(
                    valid, m - m_new, torch.zeros_like(m)))
                p_t = torch.exp(s_f - m_new.unsqueeze(-1))
                p_t = torch.where(valid.unsqueeze(-1), p_t,
                                  torch.zeros_like(p_t))
                l = l * corr + p_t.sum(dim=-1)
                acc = acc * corr.unsqueeze(-1) + \
                    torch.matmul(p_t.to(vt.dtype),
                                 vt.unsqueeze(2)).float()
                m = m_new

            if kc > 0:
                ck, cv = layer.get_cached_kv(kc)
                ck = ck[:batch, :, :kc]
                cv = cv[:batch, :, :kc]
                # sliding: skip tiles fully outside every query's window
                if ctx.window > 0:
                    k_lo = max(0, (kc - ctx.window) // KT * KT)
                else:
                    k_lo = 0
                for k0 in range(k_lo, kc, KT):
                    k1 = min(k0 + KT, kc)
                    tile_step(ck[:, :, k0:k1], cv[:, :, k0:k1], k0)

            # Current-chunk tile (the only grad-bearing K/V).
            tile_step(k_cur, v_cur, kc)
            out = (acc / l.unsqueeze(-1)).to(qdt).reshape(B, H, qlen, D)

        ctx.m_final = m
        ctx.l_final = l
        ctx.out = out.detach()
        return out

    @staticmethod
    def backward(ctx, dout):
        KT = ctx.ktile
        q, k_cur, v_cur = ctx.saved_tensors
        layer = ctx.layer
        m = ctx.m_final
        l = ctx.l_final
        out = ctx.out
        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]
        g = ctx.n_groups
        kc = ctx.cur_len
        dev, qdt = q.device, q.dtype

        qg = q.view(B, kvH, g, qlen, D)
        dyg = dout.view(B, kvH, g, qlen, D)
        delta = (dyg.float() *
                 out.view(B, kvH, g, qlen, D).float()).sum(dim=-1)
        inv_l = (1.0 / l).unsqueeze(-1)
        dq_acc = torch.zeros(B, kvH, g, qlen, D, device=dev,
                             dtype=torch.float32)

        def back_tile(kt, vt, k_base, acc_dk=None, acc_dv=None):
            """Recompute exact tile probs; accumulate dq (+dk/dv if asked)."""
            nonlocal dq_acc
            s = torch.matmul(
                qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
            if ctx.softcap > 0:
                s = torch.tanh(s.float() / ctx.softcap) * ctx.softcap
            s_f = s.float()
            qp = (torch.arange(qlen, device=dev) + kc).view(-1, 1)
            kp = (torch.arange(kt.shape[-2], device=dev)
                  + k_base).view(1, -1)
            ok = kp <= qp
            if ctx.window > 0:
                ok = ok & ((qp - kp) < ctx.window)
            s_f = s_f.masked_fill(~ok, float("-inf"))
            # rows empty in this tile contribute zero (guard mirrors fwd):
            # per-row running max WITHOUT keepdim (keepdim misaligns ranks:
            # [B,kvH,g,1] broadcasts against qlen and dies on g).
            valid = torch.isfinite(
                torch.maximum(m, s_f.amax(dim=-1)))
            p_t = torch.exp(s_f - m.unsqueeze(-1)) * inv_l
            p_t = torch.where(valid.unsqueeze(-1), p_t,
                              torch.zeros_like(p_t))
            dp_t = torch.matmul(dyg, vt.unsqueeze(2).transpose(-2, -1)).float()
            ds_t = p_t * (dp_t - delta.unsqueeze(-1))
            ds_bf = ds_t.to(qdt)
            dq_acc += torch.matmul(
                ds_bf, kt.unsqueeze(2)).float() * ctx.scaling
            if acc_dk is not None:
                acc_dk += torch.matmul(
                    ds_bf.transpose(-2, -1), qg).sum(dim=2).float() * \
                    ctx.scaling
                acc_dv += torch.matmul(
                    p_t.to(qdt).transpose(-2, -1), dyg).sum(dim=2).float()

        if kc > 0:
            ck, cv = layer.get_cached_kv(kc)
            ck = ck[:ctx.batch, :, :kc]
            cv = cv[:ctx.batch, :, :kc]
            if ctx.window > 0:
                k_lo = max(0, (kc - ctx.window) // KT * KT)
            else:
                k_lo = 0
            for k0 in range(k_lo, kc, KT):
                k1 = min(k0 + KT, kc)
                back_tile(ck[:, :, k0:k1], cv[:, :, k0:k1], k0)

        dk_cur = torch.zeros_like(k_cur, dtype=torch.float32)
        dv_cur = torch.zeros_like(v_cur, dtype=torch.float32)
        back_tile(k_cur, v_cur, kc, dk_cur, dv_cur)

        return (
            dq_acc.to(qdt).reshape(B, H, qlen, D),
            dk_cur.to(qdt),
            dv_cur.to(v_cur.dtype),
            None, None, None, None, None, None, None, None,
        )


def patch_gemma_attention_cache(model, kv_cache):
    """Swap Gemma4's eager attention for the Chonk cached tiled path.

    Same contract as patch_granite_attention_recompute: hands the Function
    the cache layer + metadata; current-chunk k/v come from the layer's
    stashed states so the big transient cat is never referenced.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    txt = base.model if hasattr(base, "model") else base
    layers = list(txt.layers)
    id2idx = {id(l.self_attn): i for i, l in enumerate(layers)}
    cfg = getattr(txt, "config", getattr(base, "config", None))
    layer_types = list(getattr(cfg, "layer_types", ["full_attention"] * 48))
    sw = int(getattr(cfg, "sliding_window", 0) or 0)
    if cfg is not None:
        cfg._attn_implementation = "eager"

    # Dispatch lever (see vulkanvm_attn_gemma header): the interface resolves
    # per-forward via get(impl, <module-global default>). Patch EVERY Gemma4
    # namespace present: modeling_gemma4 (text-only builds) and
    # modeling_gemma4_unified (unified builds -- the Chonk driver loads the
    # unified checkpoint's text backbone, whose attention dispatches here).
    # Contracts are identical (module-first signature, softcap kwarg).
    def patched(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        if dropout and dropout > 0:
            return _origs[0](module, query, key, value, attention_mask,
                             scaling, dropout=dropout, **kwargs)
        idx = id2idx.get(id(module), -1)
        lt = layer_types[idx] if 0 <= idx < len(layer_types) \
            else "full_attention"
        window = sw if lt == "sliding_attention" else 0
        layer = kv_cache.layers[module.layer_idx]
        cur_len = key.shape[-2] - query.shape[-2]
        k_cur = layer._last_k
        v_cur = layer._last_v
        softcap = float(kwargs.get("softcap", 0.0) or 0.0)
        out = GemmaAttnCache.apply(
            query, k_cur, v_cur, layer, cur_len, query.shape[0],
            float(scaling), int(module.num_key_value_groups), window, True,
            softcap)
        out = out.transpose(1, 2).contiguous()
        return out, None

    patched_mods = []
    _origs = []
    for _modname in (
        "transformers.models.gemma4.modeling_gemma4",
        "transformers.models.gemma4_unified.modeling_gemma4_unified",
    ):
        try:
            _gm = __import__(_modname, fromlist=["eager_attention_forward"])
            _origs.append(_gm.eager_attention_forward)
            _gm.eager_attention_forward = patched
            patched_mods.append(_modname.rsplit(".", 1)[0])
        except ImportError:
            pass
    if not patched_mods:
        raise RuntimeError("no Gemma4 attention namespace found to patch")

    print("[+] Gemma4 eager attention -> Chonk cached tiled "
          f"(QC={_QC} KT={_KTILE})", flush=True)
    return patched
