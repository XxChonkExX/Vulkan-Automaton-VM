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


class GraniteAttnRecompute(torch.autograd.Function):
    """Causal GQA attention with activation recompute (checkpointing).

    forward inputs:
        q       : [B, H, qlen, D]          (current chunk, requires grad)
        k_cur   : [B, kvH, qlen, D]        (current chunk = key[...,-qlen:], grad flows)
        v_cur   : [B, kvH, qlen, D]
        scaling : float                    (config.attention_multiplier)
        n_groups: int                      (num_key_value_groups = H // kvH)
        cached_k: [B, kvH, cur_len, D]     (Chonk pool ref, DETACHED, not in graph)
        cached_v: [B, kvH, cur_len, D]
        causal  : bool
    Only q/k_cur/v_cur are saved_for_backward; cached_* are plain attrs (no
    version tracking, no copy). Backward recomputes softmax probs from q/k/v.
    """

    @staticmethod
    def forward(ctx, q, k_cur, v_cur, layer, cur_len, batch, scaling, n_groups, causal):
        """Split attention: the cached prefix and the current chunk are NEVER
        concatenated. Scores/out are computed as cached-part + current-part
        matmuls (grad flows only through the current part), so no full-prefix
        K/V copy is ever allocated (the cat was the 8.6GB/32-chunk ratchet)."""
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

        if kc > 0:
            cached_k, cached_v = layer.get_cached_kv(kc)
            cached_k = cached_k[:batch, :, :kc]
            cached_v = cached_v[:batch, :, :kc]
            klen = kc + qlen
            # One scores tensor (needed for softmax), filled in two parts —
            # NO k/v cat. cached part: no grad; current part: grad via k_cur.
            scores = torch.empty(B, kvH, g, qlen, klen, dtype=q.dtype, device=q.device)
            scores[..., :kc] = torch.matmul(qg, cached_k.unsqueeze(2).transpose(-2, -1))
            scores[..., kc:] = torch.matmul(qg, k_cur.unsqueeze(2).transpose(-2, -1))
            scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
            if causal:
                scores = scores + _causal_mask(qlen, klen, q.device, q.dtype)
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            # out = p_cached @ v_cached + p_cur @ v_cur  (split; no v cat)
            pg = p.view(B, kvH, g, qlen, klen)
            out = torch.matmul(pg[..., :kc], cached_v.unsqueeze(2)) + \
                  torch.matmul(pg[..., kc:], v_cur.unsqueeze(2))
            return out.reshape(B, H, qlen, D)
        else:
            klen = qlen
            scores = torch.matmul(qg, k_cur.unsqueeze(2).transpose(-2, -1))
            scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
            if causal:
                scores = scores + _causal_mask(qlen, klen, q.device, q.dtype)
            p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            pg = p.view(B, kvH, g, qlen, klen)
            out = torch.matmul(pg, v_cur.unsqueeze(2)).reshape(B, H, qlen, D)
            return out

    @staticmethod
    def backward(ctx, dout):
        q, k_cur, v_cur = ctx.saved_tensors
        layer = ctx.layer
        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]
        g = ctx.n_groups
        kc = ctx.cur_len
        klen = kc + qlen

        mt = torch.bfloat16 if q.dtype in (torch.bfloat16, torch.float16) else torch.float32
        qf = q.to(mt)
        qg = qf.view(B, kvH, g, qlen, D)
        dout_mt = dout.to(mt)
        dyg = dout_mt.view(B, kvH, g, qlen, D)

        if kc > 0:
            cached_k, cached_v = layer.get_cached_kv(kc)
            cached_k = cached_k[:ctx.batch, :, :kc].to(mt)
            cached_v = cached_v[:ctx.batch, :, :kc].to(mt)
            # Recompute p in two parts into one tensor (no k/v cat).
            scores = torch.empty(B, kvH, g, qlen, klen, dtype=mt, device=q.device)
            scores[..., :kc] = torch.matmul(qg, cached_k.unsqueeze(2).transpose(-2, -1))
            scores[..., kc:] = torch.matmul(qg, k_cur.unsqueeze(2).transpose(-2, -1))
            scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
            if ctx.causal:
                scores = scores + _causal_mask(qlen, klen, q.device, mt)
            p = torch.softmax(scores.float(), dim=-1).to(mt)
            pg = p.view(B, kvH, g, qlen, klen)

            # dv_cur: only the current-chunk columns of p contribute (cached
            # prefix has no grad path — skip the wasted full-length compute).
            pq2 = pg[..., kc:]                              # [B,kvH,g,qlen,qlen]
            dv_cur = torch.matmul(pq2.transpose(-2, -1), dyg).sum(dim=2)
            # dp = dout @ v^T split into one [qlen, klen] tensor (no cat)
            dp = torch.empty(B, kvH, g, qlen, klen, dtype=mt, device=q.device)
            dp[..., :kc] = torch.matmul(dyg, cached_v.unsqueeze(2).transpose(-2, -1))
            dp[..., kc:] = torch.matmul(dyg, v_cur.unsqueeze(2).to(mt).transpose(-2, -1))
            dp = dp.reshape(B, H, qlen, klen)
            ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
            dsg = ds.view(B, kvH, g, qlen, klen)
            # dq = ds @ k split
            dq = (torch.matmul(dsg[..., :kc], cached_k.unsqueeze(2)) +
                  torch.matmul(dsg[..., kc:], k_cur.unsqueeze(2).to(mt))).reshape(B, H, qlen, D) * ctx.scaling
            # dk_cur = ds[:, :, kc:]^T @ q  (only current cols)
            dk_cur = torch.matmul(dsg[..., kc:].transpose(-2, -1), qg).sum(dim=2) * ctx.scaling
        else:
            scores = torch.matmul(qg, k_cur.unsqueeze(2).to(mt).transpose(-2, -1))
            scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
            if ctx.causal:
                scores = scores + _causal_mask(qlen, klen, q.device, mt)
            p = torch.softmax(scores.float(), dim=-1).to(mt)
            pg = p.view(B, kvH, g, qlen, klen)
            dv_cur = torch.matmul(pg.transpose(-2, -1), dyg).sum(dim=2)
            dp = torch.matmul(dyg, v_cur.unsqueeze(2).to(mt).transpose(-2, -1)).reshape(B, H, qlen, klen)
            ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
            dsg = ds.view(B, kvH, g, qlen, klen)
            dq = torch.matmul(dsg, k_cur.unsqueeze(2).to(mt)).reshape(B, H, qlen, D) * ctx.scaling
            dk_cur = torch.matmul(dsg.transpose(-2, -1), qg).sum(dim=2) * ctx.scaling

        return (
            dq.to(q.dtype),
            dk_cur.to(q.dtype),
            dv_cur.to(v_cur.dtype),
            None, None, None, None, None, None,
        )


def patch_granite_attention_recompute(model, kv_cache):
    """Swap Granite's eager attention for the Chonk recompute path.

    The patch hands the Function a reference to the cache LAYER plus metadata,
    never views of the big transient [cached|current] cats. The Function reads
    the cached span via layer.get_cached_kv() (bf16: zero-copy pool views;
    INT4: transient dequant, recomputed in backward) and takes the current
    chunk from the layer's stashed states — so nothing position-proportional
    is retained between forward and backward (zero pool climb).
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
    print("[+] Granite eager attention -> Chonk recompute (layer-direct, zero retention)")
