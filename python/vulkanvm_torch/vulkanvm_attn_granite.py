# vulkanvm_attn_granite.py
# Granite-specific memory-efficient causal-GQA attention for Chonk Buffer training.
#
# WHY A SECOND FILE (not editing vulkanvm_attn / vulkanvm_autograd.hpp):
#   The shared C++ recompute attention (vulkanvm_autograd.hpp :: VulkanAttentionFn)
#   computes attention INCORRECTLY for Granite (loss 12.06 vs the correct 4.625
#   from plain eager). Rather than debug the C++ GQA/cache grouping, we provide a
#   clean, correct, pure-torch implementation that:
#     * runs on Chonk/Vulkan pool memory (torch draws from the Chonk allocator),
#       i.e. "our ROCm path" -- NOT AMD's native fused SDPA/flash which reset the
#       display driver through dma-buf Chonk memory (see OPTIMIZATION_LOG.md),
#     * keeps activations FLAT via attention checkpointing: only the current
#       chunk's q/k/v are saved in the autograd graph; the cached KV span is
#       referenced from the Chonk pool (no copy, no position-proportional growth),
#     * is correct for any GQA config (Granite: 32 Q / 8 KV, head_dim 128).
#
# Interface: patches transformers.models.granite.modeling_granite.eager_attention_forward
# so it is called exactly where HF would call it, with the same signature/contract.

import os
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Causal mask cache (reconstructed, never saved in the graph -> flat memory)
# ---------------------------------------------------------------------------
_CAUSAL_MASKS = {}


def _causal_mask(qlen, klen, device, dtype):
    key = (qlen, klen, device, dtype)
    m = _CAUSAL_MASKS.get(key)
    if m is None:
        # Allowed (0) when key j <= query i + cached_len, i.e. j - i <= klen - qlen.
        # Forbidden (-inf) when j - i >= klen - qlen + 1  ->  triu diagonal = klen-qlen+1.
        diag = klen - qlen + 1
        m = torch.triu(
            torch.full((1, 1, qlen, klen), float("-inf"), device=device, dtype=dtype),
            diagonal=diag,
        )
        _CAUSAL_MASKS[key] = m
    return m


def _chunk_causal_mask(qlen, device, dtype):
    """Within-current-chunk causal mask: query i (local) may attend current-chunk
    key j (local) iff j <= i. Cached-tiles never need a mask (every cached
    position c < kc <= kc + i is visible to every query i)."""
    key = (qlen, device, dtype, "chunk")
    m = _CAUSAL_MASKS.get(key)
    if m is None:
        m = torch.triu(
            torch.full((1, 1, qlen, qlen), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )
        _CAUSAL_MASKS[key] = m
    return m


# Tiled-attention key-block size (tokens). Workspace is O(B*kvH*g*qlen*TILE)
# regardless of sequence position — with TILE=8192 and qlen=512 that is a FIXED
# ~268 MB bf16 per tile tensor instead of the O(klen) [32,512,131072] family
# (17 GB at full context) that ratcheted the pool past the wall.
_ATTN_TILE = int(os.environ.get("CHONK_ATTN_TILE", "8192"))




class GraniteAttnRecompute(torch.autograd.Function):
    """Tiled (flash-style) causal GQA attention with activation recompute.

    Memory is O(B*kvH*g*qlen*TILE) at EVERY sequence position — the key axis is
    streamed in fixed-size tiles with an online running max m / running sum l
    (corrections telescope, so the result is the exact full-row softmax). At
    qlen=512, TILE=8192: ~268 MB bf16 per tile tensor, flat for all 256 chunks
    of a 131K sequence — versus the O(klen) scores/p/dp/ds family (~17 GB at
    full context) that ratcheted the pool past the wall.

    Fixed-size tiles are also the Chonk allocator's ideal case: identical
    request sizes every chunk -> slab best-fit reuses the same blocks forever.

    forward inputs:
        q       : [B, H, qlen, D]     (current chunk, requires grad)
        k_cur   : [B, kvH, qlen, D]   (current chunk, grad flows)
        v_cur   : [B, kvH, qlen, D]
        layer   : cache layer (get_cached_kv -> pool views / INT4 scratch)
        cur_len : int (cached prefix length; cached positions are visible to
                  every query, so cached tiles are NEVER masked)
        batch / scaling / n_groups / causal : metadata
    Backward recomputes per-tile probs exactly from the saved final m/l
    (p_tile = exp(s - m_final) / l_final), accumulates dq across tiles, and
    computes dk_cur/dv_cur only on the current-chunk tile (cached prefix has
    no grad path).
    """

    @staticmethod
    def forward(ctx, q, k_cur, v_cur, layer, cur_len, batch, scaling, n_groups, causal):
        T = _ATTN_TILE
        ctx.layer = layer
        ctx.cur_len = int(cur_len)
        ctx.batch = int(batch)
        ctx.scaling = float(scaling)
        ctx.n_groups = int(n_groups)
        ctx.causal = bool(causal)
        ctx.save_for_backward(q, k_cur, v_cur)

        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]
        g = ctx.n_groups
        kc = ctx.cur_len

        qg = q.view(B, kvH, g, qlen, D)
        dev, qdt = q.device, q.dtype

        # Online-softmax state (tiny: [B,kvH,g,qlen] each).
        m = torch.full((B, kvH, g, qlen), float("-inf"), device=dev, dtype=torch.float32)
        l = torch.zeros(B, kvH, g, qlen, device=dev, dtype=torch.float32)
        acc = torch.zeros(B, kvH, g, qlen, D, device=dev, dtype=torch.float32)

        if kc > 0:
            ck, cv = layer.get_cached_kv(kc)
            ck = ck[:batch, :, :kc]
            cv = cv[:batch, :, :kc]
            for j0 in range(0, kc, T):
                j1 = min(j0 + T, kc)
                kt = ck[:, :, j0:j1]                       # [B,kvH,w,D] view
                vt = cv[:, :, j0:j1]
                s = torch.matmul(qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                s_f = s.float()
                m_new = torch.maximum(m, s_f.amax(dim=-1))
                corr = torch.exp(m - m_new)                # first tile: exp(-inf)=0
                p_t = torch.exp(s_f - m_new.unsqueeze(-1))
                l = l * corr + p_t.sum(dim=-1)
                acc = acc * corr.unsqueeze(-1) + \
                    torch.matmul(p_t.to(vt.dtype), vt.unsqueeze(2)).float()
                m = m_new

        # Current-chunk tile (grad flows through k_cur/v_cur). No mask needed
        # on cached tiles: every cached column c < kc <= kc + i is visible.
        s = torch.matmul(qg, k_cur.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
        if causal:
            s = s + _chunk_causal_mask(qlen, dev, qdt)
        s_f = s.float()
        m_new = torch.maximum(m, s_f.amax(dim=-1))
        corr = torch.exp(m - m_new)
        p_t = torch.exp(s_f - m_new.unsqueeze(-1))
        l = l * corr + p_t.sum(dim=-1)
        acc = acc * corr.unsqueeze(-1) + \
            torch.matmul(p_t.to(v_cur.dtype), v_cur.unsqueeze(2)).float()
        m = m_new

        out = (acc / l.unsqueeze(-1)).to(qdt).reshape(B, H, qlen, D)
        # Stash the tiny softmax state + output for the exact backward recompute.
        ctx.m_final = m
        ctx.l_final = l
        ctx.out = out
        return out

    @staticmethod
    def backward(ctx, dout):
        T = _ATTN_TILE
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
        # delta = rowsum(dout * out) — the softmax-backward dot product.
        delta = (dyg.float() * out.view(B, kvH, g, qlen, D).float()).sum(dim=-1)

        inv_l = (1.0 / l).unsqueeze(-1)
        dq_acc = torch.zeros(B, kvH, g, qlen, D, device=dev, dtype=torch.float32)

        if kc > 0:
            ck, cv = layer.get_cached_kv(kc)
            ck = ck[:ctx.batch, :, :kc]
            cv = cv[:ctx.batch, :, :kc]
            for j0 in range(0, kc, T):
                j1 = min(j0 + T, kc)
                kt = ck[:, :, j0:j1]
                vt = cv[:, :, j0:j1]
                s = torch.matmul(qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                p_t = torch.exp(s.float() - m.unsqueeze(-1)) * inv_l   # exact softmax tile
                dp_t = torch.matmul(dyg, vt.unsqueeze(2).transpose(-2, -1)).float()
                ds_t = p_t * (dp_t - delta.unsqueeze(-1))
                dq_acc += torch.matmul(ds_t.to(qdt), kt.unsqueeze(2)).float() * ctx.scaling

        # Current-chunk tile: dq contribution + dk_cur / dv_cur (only here).
        s = torch.matmul(qg, k_cur.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
        if ctx.causal:
            s = s + _chunk_causal_mask(qlen, dev, qdt)
        p_t = torch.exp(s.float() - m.unsqueeze(-1)) * inv_l
        dp_t = torch.matmul(dyg, v_cur.unsqueeze(2).transpose(-2, -1)).float()
        ds_t = p_t * (dp_t - delta.unsqueeze(-1))
        ds_bf = ds_t.to(qdt)
        dq_acc += torch.matmul(ds_bf, k_cur.unsqueeze(2)).float() * ctx.scaling
        dk_cur = torch.matmul(ds_bf.transpose(-2, -1), qg).sum(dim=2) * ctx.scaling
        p_bf = p_t.to(qdt)
        dv_cur = torch.matmul(p_bf.transpose(-2, -1), dyg).sum(dim=2)

        return (
            dq_acc.to(qdt).reshape(B, H, qlen, D),
            dk_cur.to(qdt),
            dv_cur.to(v_cur.dtype),
            None, None, None, None, None, None,
        )


def patch_granite_attention_recompute(model, kv_cache):
    """Swap Granite's eager attention for the Chonk tiled-recompute path.

    The patch hands the Function a reference to the cache LAYER plus metadata,
    never views of the big transient [cached|current] cats. The Function reads
    the cached span via layer.get_cached_kv() (bf16: zero-copy pool views;
    INT4: shared-scratch dequant, recomputed in backward) and takes the current
    chunk from the layer's stashed states — so nothing position-proportional
    is retained, and the tiled workspace is FIXED size at every position.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    first = next(l for l in base.model.layers if hasattr(l, "self_attn"))
    attn_mod = type(first.self_attn).__module__
    mod = __import__(attn_mod, fromlist=["eager_attention_forward"])
    orig = mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if dropout > 0:
            return orig(module, query, key, value, attention_mask, scaling, dropout, **kwargs)

        layer = kv_cache.layers[module.layer_idx]
        cur_len = key.shape[-2] - query.shape[-2]

        # Current-chunk states: stashed by the cache layer's update() — the
        # ORIGINAL grad-bearing tensors (pre-cat), tiny (~1 MB), so the big
        # transient [cached|current] cat is never referenced by the graph.
        k_cur = layer._last_k
        v_cur = layer._last_v

        out = GraniteAttnRecompute.apply(
            query,
            k_cur,
            v_cur,
            layer,
            cur_len,
            query.shape[0],
            float(scaling),
            int(module.num_key_value_groups),
            True,
        )
        out = out.transpose(1, 2).contiguous()  # [B, qlen, H, D] per HF contract
        return out, None

    mod.eager_attention_forward = patched
    print("[+] Granite eager attention -> Chonk tiled recompute (fixed workspace, zero retention)")
