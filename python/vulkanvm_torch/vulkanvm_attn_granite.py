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
        ctx.layer = layer        # plain attr: cache layer ref (dequant on demand)
        ctx.cur_len = int(cur_len)
        ctx.batch = int(batch)
        ctx.scaling = float(scaling)
        ctx.n_groups = int(n_groups)
        ctx.causal = bool(causal)
        ctx.save_for_backward(q, k_cur, v_cur)
        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]

        # Reconstruct full [cached | current] keys/values. The cached span comes
        # from the cache layer (bf16: zero-copy pool views; INT4: transient
        # dequant) — transient, NOT saved, so nothing position-proportional is
        # retained between forward and backward.
        if ctx.cur_len > 0:
            cached_k, cached_v = layer.get_cached_kv(ctx.cur_len)
            cached_k = cached_k[:ctx.batch, :, :ctx.cur_len]
            cached_v = cached_v[:ctx.batch, :, :ctx.cur_len]
            k = torch.cat([cached_k, k_cur], dim=-2)
            v = torch.cat([cached_v, v_cur], dim=-2)
        else:
            k, v = k_cur, v_cur
        klen = k.shape[-2]

        # Grouped GQA: match HF's repeat_kv order (kv_head first, then group).
        # Standard GQA: head h = kv_head * g + group_idx  (repeat_interleave).
        # So split q into [B, kvH, g, qlen, D].
        g = ctx.n_groups
        qg = q.view(B, kvH, g, qlen, D)
        scores = torch.matmul(qg, k.unsqueeze(2).transpose(-2, -1))
        scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
        if causal:
            scores = scores + _causal_mask(qlen, klen, q.device, q.dtype)
        # softmax in fp32 (matches HF), output back in q.dtype.
        p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        pg = p.view(B, kvH, g, qlen, klen)
        out = torch.matmul(pg, v.unsqueeze(2)).reshape(B, H, qlen, D)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k_cur, v_cur = ctx.saved_tensors
        layer = ctx.layer
        qlen = q.shape[-2]
        B, H, _, D = q.shape
        kvH = k_cur.shape[1]
        g = ctx.n_groups

        if ctx.cur_len > 0:
            cached_k, cached_v = layer.get_cached_kv(ctx.cur_len)
            cached_k = cached_k[:ctx.batch, :, :ctx.cur_len]
            cached_v = cached_v[:ctx.batch, :, :ctx.cur_len]
            k = torch.cat([cached_k, k_cur], dim=-2)
            v = torch.cat([cached_v, v_cur], dim=-2)
        else:
            k, v = k_cur, v_cur
        klen = k.shape[-2]

        # Recompute probs in bf16 (checkpointing: p was not saved), NOT fp32.
        # The position-proportional tensors (p/ds/dp, [B,H,qlen,klen]) at full
        # 131K context are 32 x qlen x 131072; fp32 makes each ~8.6GB and the
        # sum ~34GB -> HSA memory fault. bf16 halves it (matches the original
        # C++ vulkanvm_autograd.hpp mt=bf16 path) so the backward fits.
        mt = torch.bfloat16 if q.dtype in (torch.bfloat16, torch.float16) else torch.float32
        qf = q.to(mt)
        kf = k.to(mt)
        vf = v.to(mt)
        qg = qf.view(B, kvH, g, qlen, D)
        scores = torch.matmul(qg, kf.unsqueeze(2).transpose(-2, -1))
        scores = scores.reshape(B, H, qlen, klen) * ctx.scaling
        if ctx.causal:
            scores = scores + _causal_mask(qlen, klen, q.device, mt)
        p = torch.softmax(scores.float(), dim=-1).to(mt)  # [B,H,qlen,klen] bf16
        pg = p.view(B, kvH, g, qlen, klen)

        dout_mt = dout.to(mt)
        dyg = dout_mt.view(B, kvH, g, qlen, D)
        vg = vf.unsqueeze(2)  # [B,kvH,1,klen,D]

        dv = torch.matmul(pg.transpose(-2, -1), dyg).sum(dim=2)        # [B,kvH,klen,D]
        dp = torch.matmul(dyg, vg.transpose(-2, -1)).reshape(B, H, qlen, klen)
        ds = p * (dp - (dp * p).sum(dim=-1, keepdim=True))             # [B,H,qlen,klen]
        dsg = ds.view(B, kvH, g, qlen, klen)
        dq = torch.matmul(dsg, kf.unsqueeze(2)).reshape(B, H, qlen, D) * ctx.scaling
        dk = torch.matmul(dsg.transpose(-2, -1), qg).sum(dim=2) * ctx.scaling  # [B,kvH,klen,D]

        # Slice only the current-chunk portion (cached prefix has no grad path).
        # Use the SAVED k_cur's own shape as the source of truth for the
        # return length -- guarantees the returned grad matches what
        # save_for_backward stored, regardless of any view/reshape ambiguity.
        def take_last_to(x, target_len):
            n = x.shape[-2]
            if n == target_len:
                return x
            if n > target_len:
                return x[:, :, -target_len:, :]
            pad = torch.zeros(*x.shape[:-2], target_len - n, x.shape[-1],
                              device=x.device, dtype=x.dtype)
            return torch.cat([pad, x], dim=-2)
        cur_len = k_cur.shape[-2]
        dk_cur = take_last_to(dk, cur_len)
        dv_cur = take_last_to(dv, cur_len)

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
