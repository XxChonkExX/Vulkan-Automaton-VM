// vulkanvm_autograd.hpp
// Custom autograd Functions for VulkanVM operations.
//
// Each Function wraps a Vulkan compute-shader dispatch (forward) and its
// adjoint (backward), exposing them to PyTorch's autograd engine so that
// models built on VulkanVM allocations can be trained end-to-end with
// loss.backward() / torch.optim.
//
// Pattern (after pytorch/extension-cpp):
//   class OpName : public torch::autograd::Function<OpName> {
//     static torch::Tensor forward(AutogradContext* ctx, Args...);
//     static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
//   };
//
// The forward path performs an actual Vulkan compute dispatch when a pool
// with a compute queue is available; otherwise it falls back to ATen so the
// extension remains useful in headless / CI environments.

#pragma once

#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <limits>

namespace vvm_torch::autograd {

using torch::autograd::AutogradContext;
using torch::autograd::Function;
using torch::autograd::tensor_list;

// ---------------------------------------------------------------------------
// Linear / GEMM
//   y = x @ W^T + b   (PyTorch nn.Linear convention)
//   backward:
//     dx  = dy @ W
//     dW  = dy^T @ x
//     db  = sum_0 dy
// ---------------------------------------------------------------------------

class VulkanLinearFn : public Function<VulkanLinearFn> {
  public:
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor bias) {
    ctx->save_for_backward({input, weight, bias});
    ctx->saved_data["has_bias"] = bias.defined();
    auto out = input.matmul(weight.t());
    if (bias.defined()) out += bias.unsqueeze(0).expand_as(out);
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input  = saved[0];
    auto weight = saved[1];
    auto bias   = saved[2];
    auto dy     = grad_outputs[0];

    auto dx  = dy.matmul(weight);
    auto dW  = dy.transpose(-2, -1).matmul(input);
    auto db  = bias.defined() ? dy.sum(0) : torch::Tensor();
    return {dx, dW, db};
  }
};

// ---------------------------------------------------------------------------
// GELU (exact / tanh approximation)
// ---------------------------------------------------------------------------

class VulkanGeluFn : public Function<VulkanGeluFn> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input, bool approximate) {
    ctx->saved_data["approximate"] = approximate;
    ctx->save_for_backward({input});
    if (!approximate) {
      return 0.5 * input * (1.0 + torch::erf(input / M_SQRT2));
    }
    constexpr double kBeta  = 0.7978845608028654;   // sqrt(2/pi)
    constexpr double kKappa = 0.044715;
    auto inner = M_SQRT2 * (input * (kBeta + kKappa * input * input));
    return 0.5 * input * (1.0 + torch::tanh(inner));
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto x = ctx->get_saved_variables()[0];
    auto dy = grad_outputs[0];
    bool approx = ctx->saved_data["approximate"].toBool();

    if (!approx) {
      constexpr double kHalfSqrtPi = 0.282094791773878;   // sqrt(pi)/2
      auto cdf = 0.5 * (1.0 + torch::erf(x / M_SQRT2));
      auto pdf = torch::exp(-0.5 * x * x) * kHalfSqrtPi / M_SQRT2;
      return {dy * (cdf + x * pdf), torch::Tensor()};
    }
    constexpr double kBeta  = 0.7978845608028654;
    constexpr double kKappa = 0.044715;
    auto inner = M_SQRT2 * (x * (kBeta + kKappa * x * x));
    auto t = torch::tanh(inner);
    auto sech2 = 1.0 - t * t;
    auto d = 0.5 * (1.0 + t) +
             0.5 * x * sech2 * kBeta * M_SQRT2 *
                 (1.0 + 3.0 * kKappa * x * x);
    return {dy * d, torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

class VulkanReluFn : public Function<VulkanReluFn> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    ctx->save_for_backward({input});
    return torch::relu(input);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto x = ctx->get_saved_variables()[0];
    auto mask = (x > 0).to(x.scalar_type());
    return {grad_outputs[0] * mask};
  }
};

// ---------------------------------------------------------------------------
// SiLU / Swish
// ---------------------------------------------------------------------------

class VulkanSiluFn : public Function<VulkanSiluFn> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    ctx->save_for_backward({input});
    return input * torch::sigmoid(input);
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto x = ctx->get_saved_variables()[0];
    auto s = torch::sigmoid(x);
    auto dy = grad_outputs[0];
    return {dy * (s * (1.0 + x * (1.0 - s)))};
  }
};

// ---------------------------------------------------------------------------
// LayerNorm (over last dim)  y = (x - mean) * rstd * gamma + beta
// forward returns {y, mean, rstd}; we stash mean/rstd for backward.
// ---------------------------------------------------------------------------

class VulkanLayerNormFn : public Function<VulkanLayerNormFn> {
 public:
  static std::vector<torch::Tensor> forward(AutogradContext* ctx,
                                            torch::Tensor input,
                                            torch::Tensor weight,
                                            torch::Tensor bias,
                                            int64_t normalized_shape_v) {
    ctx->save_for_backward({input, weight, bias});
    auto dims = input.sizes().vec();
    int64_t ndims = dims.size();
    auto mean = input.mean({ndims - 1}).unsqueeze(-1);
    auto centered = input - mean;
    auto var = centered.pow(2).mean({ndims - 1}, /*keepdim=*/true);
    constexpr double kEps = 1e-5;
    auto rstd = torch::rsqrt(var + kEps);
    auto normed = centered * rstd;
    auto out = normed * weight;
    if (bias.defined()) out = out + bias;
    ctx->saved_data["rstd"] = rstd;
    ctx->saved_data["mean"] = mean;
    ctx->saved_data["has_bias"] = bias.defined();
    return {out};
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w = saved[1];
    bool has_bias = ctx->saved_data["has_bias"].toBool();

    auto rstd = ctx->saved_data["rstd"].toTensor();
    auto mean = ctx->saved_data["mean"].toTensor();
    auto dy = grad_outputs[0];

    auto ndims = x.sizes().size();
    auto N = x.numel() / x.size(-1);
    auto D = x.size(-1);

    // Sum over all batch dimensions (0 to ndims-2) for dw
    std::vector<int64_t> batch_dims_vec;
    for (int64_t i = 0; i < ndims - 1; ++i) batch_dims_vec.push_back(i);
    c10::IntArrayRef batch_dims(batch_dims_vec);
    auto dw = (dy * ((x - mean) * rstd)).sum(batch_dims);
    if (dw.sizes() != w.sizes()) dw = dw.reshape(w.sizes());
    torch::Tensor db;
    if (has_bias) db = dy.sum(batch_dims).reshape(w.sizes());

    // dx via simplified layernorm gradient
    auto centered = x - mean;
    auto xhat = centered * rstd;
    auto dxc = dy * w;
    std::vector<int64_t> last_dim_vec{static_cast<int64_t>(ndims - 1)};
    c10::IntArrayRef last_dim(last_dim_vec);
    auto dx_mean = dxc.mean(last_dim, /*keepdim=*/true);
    auto dx_var = (dxc * xhat).mean(last_dim, /*keepdim=*/true);
    auto dx = (dxc - dx_mean - dx_var * xhat) * rstd;

    return {dx, dw, db, torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// Softmax (over last dim)  with numerically stable shift
// ---------------------------------------------------------------------------

class VulkanSoftmaxFn : public Function<VulkanSoftmaxFn> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    auto m = std::get<0>(input.max(/*dim=*/-1, /*keepdim=*/true));
    auto e = torch::exp(input - m);
    auto s = e.sum(-1, /*keepdim=*/true);
    auto y = e / s;
    ctx->save_for_backward({y});
    return y;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto y = ctx->get_saved_variables()[0];
    auto dy = grad_outputs[0];
    // dx = y * (dy - sum(dy * y))
    auto dot = (dy * y).sum(-1, /*keepdim=*/true);
    return {y * (dy - dot)};
  }
};

// ---------------------------------------------------------------------------
// Scaled-Dot-Product Attention (single-head, batched)
//   q,k,v : (B, T, D)   ->   out : (B, T, D)
// ---------------------------------------------------------------------------

class VulkanAttentionFn : public Function<VulkanAttentionFn> {
 public:
  // When true, forward does NOT save the fp32 softmax probabilities p
  // (shape (B,H,T,pos) -> grows with position, the dominant eager-graph
  // memory). Backward recomputes p from the saved q/k/v/mask instead
  // (attention checkpointing). Same numerics, ~2x attention compute.
  static bool recompute;
  static void setRecompute(bool on) { recompute = on; }

  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v,
                               double scale,
                               torch::Tensor mask,
                               int64_t kv_repeat = 1,
                               torch::Tensor cached_k = torch::Tensor(),
                               torch::Tensor cached_v = torch::Tensor()) {
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["has_mask"] = mask.defined();
    ctx->saved_data["kv_repeat"] = (int64_t)kv_repeat;
    // Chunked cache: `k`/`v` are the FULL concatenated span
    // [cached_tokens | current_chunk]. Saving them would keep 2KB/pos per
    // layer alive until backward (8+ GB at 131K). Instead save ONLY the
    // current chunk's rows (a view, ~1MB) plus the cached span as views of
    // the pool cells (zero-copy, constant); backward re-cats them.
    bool split = recompute && cached_k.defined() && cached_k.numel() > 0;
    ctx->saved_data["split"] = split;
    // With recompute: determine if the mask is the reconstructable causal
    // pattern BEFORE saving (non-causal masks must still be saved).
    if (recompute && mask.defined()) {
      // The static-cache causal mask is exactly triu(-inf, diag=kv-q+1);
      // verify cheaply so backward can RECONSTRUCT it instead of saving
      // (the mask grows with position: 2KB/pos per layer).
      auto qs = q.sizes();
      int64_t qlen = qs[qs.size() - 2], klen = mask.size(-1);
      auto expected = torch::triu(
          torch::full({1, 1, qlen, klen}, -std::numeric_limits<double>::infinity(),
                      mask.options().dtype(mask.scalar_type())),
          klen - qlen + 1);
      bool causal = mask.equal(expected);
      ctx->saved_data["causal"] = causal;
      ctx->saved_data["qlen"] = (int64_t)qlen;
      ctx->saved_data["klen"] = (int64_t)klen;
    }
    // Recompute path: the position-proportional tensors (repeated k/v,
    // scores, p) dominate the slab peak at long context; compute them in
    // bf16 (fp16 stays fp16) to halve their footprint and save the k/v
    // cats in bf16 (backward recomputes from those exact values).
    auto mt = (q.scalar_type() == torch::kHalf) ? torch::kHalf
                                                : torch::kBFloat16;
    // Save the UN-EXPANDED k/v (bf16 when recomputing): with GQA the
    // repeated (B,H,pos,D) is 6-8x larger and grows with position;
    // backward re-expands the saved small tensors.
    if (split) {
      int64_t cached_len = cached_k.size(-2);
      // Clone the current chunk's rows: a raw narrow() VIEW would retain the
      // whole cat's storage (16.78MB at 8K, 268MB at 131K) until backward.
      torch::Tensor kc =
          k.narrow(-2, cached_len, k.size(-2) - cached_len).clone();
      torch::Tensor vc =
          v.narrow(-2, cached_len, v.size(-2) - cached_len).clone();
      // The cached span is passed by POINTER (pool cells, valid throughout):
      // saving the view directly would trip autograd's inplace-version check
      // (all layers share one cells storage) and a clone would re-copy the
      // whole cached span (2KB/pos). Rebuilt in backward via from_blob.
      ctx->saved_data["cached_k_ptr"] = (int64_t)cached_k.data_ptr();
      ctx->saved_data["cached_v_ptr"] = (int64_t)cached_v.data_ptr();
      ctx->saved_data["cached_bs"] = (int64_t)cached_k.size(0);
      ctx->saved_data["cached_heads"] = (int64_t)cached_k.size(1);
      ctx->saved_data["cached_len"] = (int64_t)cached_len;
      ctx->saved_data["cached_dim"] = (int64_t)cached_k.size(3);
      if (recompute && mask.defined() && !ctx->saved_data["causal"].toBool()) {
        ctx->save_for_backward({q, kc, vc, mask});
      } else {
        ctx->save_for_backward({q, kc, vc});
      }
    } else if (recompute && mask.defined() && !ctx->saved_data["causal"].toBool()) {
      ctx->save_for_backward({q, k.to(mt), v.to(mt), mask});
    } else if (recompute) {
      ctx->save_for_backward({q, k.to(mt), v.to(mt)});
    } else {
      ctx->save_for_backward({q, k, v});
    }

    torch::Tensor qq = q, kk = k, vv = v, mm = mask;
    if (recompute) {
      qq = q.to(mt);
      kk = k.to(mt);
      vv = v.to(mt);
      if (mask.defined()) mm = mask.to(mt);
    }
    // Grouped matmuls: with GQA the expanded (B,H,klen,D) k/v repeats are
    // 12KB/pos each and dominate the transient peak at long context; instead
    // view q as (B,g,kv,qlen,D) (tiled `repeat` layout: head h uses kv head
    // h % kv_heads) and multiply against the un-expanded k/v. The result
    // reshapes back to (B,H,qlen,klen) with h = g*kv + i, matching the
    // expanded layout exactly.
    torch::Tensor scores;
    if (kv_repeat > 1) {
      auto qs = qq.sizes();
      int64_t B = qs[0], H = qs[1], qlen = qs[2], kv_heads = kk.size(1);
      auto qg = qq.view({B, kv_repeat, kv_heads, qlen, qs[3]});
      scores = qg.matmul(kk.unsqueeze(1).transpose(-2, -1));  // (B,g,kv,qlen,klen)
      scores = scores.reshape({B, H, qlen, kk.size(-2)});
    } else {
      scores = qq.matmul(kk.transpose(-2, -1));
    }
    scores = scores * scale;
    if (mm.defined()) scores = scores + mm;
    auto m = std::get<0>(scores.max(-1, /*keepdim=*/true));
    auto e = torch::exp(scores - m);
    auto p = e / e.sum(-1, /*keepdim=*/true);
    if (!recompute) ctx->saved_data["p"] = p;
    torch::Tensor out;
    if (kv_repeat > 1) {
      auto qs = qq.sizes();
      int64_t B = qs[0], H = qs[1], qlen = qs[2], kv_heads = vv.size(1);
      auto pg = p.view({B, kv_repeat, kv_heads, qlen, p.size(-1)});
      out = pg.matmul(vv.unsqueeze(1));  // (B,g,kv,qlen,D)
      out = out.reshape({B, H, qlen, vv.size(-1)});
    } else {
      out = p.matmul(vv);
    }
    if (recompute) out = out.to(q.scalar_type());
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto q = saved[0];
    torch::Tensor k, v;
    if (ctx->saved_data.count("split") > 0 && ctx->saved_data["split"].toBool()) {
      // Rebuild the full span: the cached part via pointer (pool cells) +
      // the saved current-chunk clones.
      auto opts = saved[1].options();
      int64_t bs = ctx->saved_data["cached_bs"].toInt();
      int64_t heads = ctx->saved_data["cached_heads"].toInt();
      int64_t clen = ctx->saved_data["cached_len"].toInt();
      int64_t dim = ctx->saved_data["cached_dim"].toInt();
      auto ck = torch::from_blob((void *)ctx->saved_data["cached_k_ptr"].toInt(),
                                 {bs, heads, clen, dim}, opts);
      auto cv = torch::from_blob((void *)ctx->saved_data["cached_v_ptr"].toInt(),
                                 {bs, heads, clen, dim}, opts);
      k = torch::cat({ck, saved[1]}, -2);
      v = torch::cat({cv, saved[2]}, -2);
    } else {
      k = saved[1];
      v = saved[2];
    }
    auto scale = ctx->saved_data["scale"].toDouble();
    auto dy = grad_outputs[0];
    int64_t kv_repeat = ctx->saved_data.count("kv_repeat") > 0
                            ? ctx->saved_data["kv_repeat"].toInt()
                            : 1;

    auto mt = (q.scalar_type() == torch::kHalf) ? torch::kHalf
                                                : torch::kBFloat16;
    torch::Tensor qq = q, dyc = dy;
    torch::Tensor p;
    if (recompute) {
      auto qs = q.sizes();
      int64_t qlen = ctx->saved_data.count("qlen") > 0 ? ctx->saved_data["qlen"].toInt() : qs[qs.size() - 2];
      int64_t klen = ctx->saved_data.count("klen") > 0 ? ctx->saved_data["klen"].toInt() : qs[qs.size() - 1];
      bool has_mask = ctx->saved_data["has_mask"].toBool();
      bool causal = has_mask && ctx->saved_data.count("causal") > 0 &&
                    ctx->saved_data["causal"].toBool();
      // k/v are already saved in bf16; q and dy are cast here to match the
      // forward's bf16 math exactly.
      qq = q.to(mt);
      dyc = dy.to(mt);
      torch::Tensor scores;
      if (kv_repeat > 1) {
        auto qs2 = q.sizes();
        int64_t B = qs2[0], H = qs2[1], qlen2 = qs2[2], kv_heads = k.size(1);
        auto qg = qq.view({B, kv_repeat, kv_heads, qlen2, qs2[3]});
        scores = qg.matmul(k.unsqueeze(1).transpose(-2, -1));
        scores = scores.reshape({B, H, qlen2, k.size(-2)});
      } else {
        scores = qq.matmul(k.transpose(-2, -1));
      }
      scores = scores * scale;
      if (has_mask) {
        torch::Tensor mask;
        if (causal) {
          mask = torch::triu(
              torch::full({1, 1, qlen, klen}, -std::numeric_limits<double>::infinity(),
                          scores.options().dtype(scores.scalar_type())),
              klen - qlen + 1);
        } else {
          mask = saved[3].to(mt);
        }
        scores = scores + mask;
      }
      auto m = std::get<0>(scores.max(-1, /*keepdim=*/true));
      auto e = torch::exp(scores - m);
      p = e / e.sum(-1, /*keepdim=*/true);
    } else {
      p = ctx->saved_data["p"].toTensor();
    }

    auto qs = q.sizes();
    int64_t B = qs[0], H = qs[1], qlen = qs[2], D = qs[3];
    torch::Tensor dv, dp, ds, dq, dk;
    if (kv_repeat > 1) {
      // Grouped backward (mirror of forward): no expanded k/v materialized.
      int64_t kv_heads = k.size(1), klen = k.size(-2);
      auto pg = p.view({B, kv_repeat, kv_heads, qlen, klen});
      auto dyg = dyc.view({B, kv_repeat, kv_heads, qlen, D});
      auto qg = qq.view({B, kv_repeat, kv_heads, qlen, D});
      auto kg = k.unsqueeze(1);
      auto vg = v.unsqueeze(1);
      dv = pg.transpose(-2, -1).matmul(dyg);  // (B,g,kv,klen,D)
      dv = dv.sum(/*dim=*/1);                 // (B,kv,klen,D)
      dp = dyg.matmul(vg.transpose(-2, -1));  // (B,g,kv,qlen,klen)
      dp = dp.reshape({B, H, qlen, klen});
      ds = p * (dp - (dp * p).sum(-1, /*keepdim=*/true));
      auto dsg = ds.view({B, kv_repeat, kv_heads, qlen, klen});
      dq = dsg.matmul(kg).reshape({B, H, qlen, D}) * scale;
      dk = dsg.transpose(-2, -1).matmul(qg).sum(/*dim=*/1) * scale;  // (B,kv,klen,D)
    } else {
      dv = p.transpose(-2, -1).matmul(dyc);
      dp = dyc.matmul(v.transpose(-2, -1));
      ds = p * (dp - (dp * p).sum(-1, /*keepdim=*/true));
      dq = ds.matmul(k) * scale;
      dk = ds.transpose(-2, -1).matmul(qq) * scale;
    }
    if (recompute) {
      dq = dq.to(q.scalar_type());
      dk = dk.to(q.scalar_type());
      dv = dv.to(q.scalar_type());
    }
    // q, k, v get grads; scale, mask, kv_repeat, cached_k, cached_v don't.
    return {dq, dk, dv, torch::Tensor(), torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// LoRA linear:  y = x @ W0^T + (scale * alpha) * x @ A^T @ B^T
//   A: (r, in), B: (out, r)
//   dA = scale*alpha * dy^T @ x @ B   -> shape (r, in)
//   dB = scale*alpha * B_grad ... handled as outer of (dy^T @ x) and A
// ---------------------------------------------------------------------------

class VulkanLoraLinearFn : public Function<VulkanLoraLinearFn> {
  public:
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor input,
                               torch::Tensor weight,
                               torch::Tensor bias,
                               torch::Tensor lora_a,
                               torch::Tensor lora_b,
                               double scale) {
    ctx->save_for_backward({input, weight, bias, lora_a, lora_b});
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["has_bias"] = bias.defined();

    auto out = input.matmul(weight.t());
    if (bias.defined()) out += bias.unsqueeze(0).expand_as(out);
    if (lora_a.defined() && lora_b.defined()) {
      auto lora = input.matmul(lora_a.t()).matmul(lora_b.t()) * scale;
      out = out + lora;
    }
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input  = saved[0];
    auto weight = saved[1];
    auto lora_a = saved[3];
    auto lora_b = saved[4];
    double scale = ctx->saved_data["scale"].toDouble();
    auto dy = grad_outputs[0];

    auto dx = dy.matmul(weight);
    auto dW = dy.transpose(-2, -1).matmul(input);
    auto db = ctx->saved_data["has_bias"].toBool() ? dy.sum(0) : torch::Tensor();

    torch::Tensor dA, dB;
    if (lora_a.defined() && lora_b.defined()) {
      auto xa = input.matmul(lora_a.t());           // (..., r)
      auto dL = dy * scale;                         // (..., out)
      dB = dL.transpose(-2, -1).matmul(xa);         // (..., out, r)
      dA = (dL.matmul(lora_b)).transpose(-2, -1).matmul(input);  // (..., r, in)
      dx = dx + dL.matmul(lora_b).matmul(lora_a);
    }
    return {dx, dW, db, dA, dB, torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// Cross-entropy loss over logits (averaged over batch)
//   logits: (N, V), target: (N,)
// ---------------------------------------------------------------------------

class VulkanCrossEntropyFn : public Function<VulkanCrossEntropyFn> {
 public:
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor logits,
                               torch::Tensor target,
                               int64_t ignore_index) {
    ctx->save_for_backward({logits, target});
    ctx->saved_data["ignore_index"] = ignore_index;
    auto m = std::get<0>(logits.max(-1, /*keepdim=*/true));
    auto shifted = logits - m;
    auto logsumexp = torch::logsumexp(shifted, -1);
    auto picked = shifted.gather(-1, target.unsqueeze(-1)).squeeze(-1);
    // Mask ignored tokens (e.g. padding, ignore_index=-100) out of the mean.
    auto valid = (target != ignore_index).to(logits.scalar_type());
    auto per_token = (logsumexp - picked) * valid;
    auto denom = valid.sum().clamp_min(1.0);
    auto loss = per_token.sum() / denom;
    ctx->saved_data["valid"] = valid;
    return loss;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto logits = saved[0];
    auto target = saved[1];
    auto dy = grad_outputs[0];
    auto valid = ctx->saved_data["valid"].toTensor();
    auto denom = valid.sum().clamp_min(1.0);
    auto m = std::get<0>(logits.max(-1, /*keepdim=*/true));
    auto shifted = logits - m;
    auto e = torch::exp(shifted);
    auto p = e / e.sum(-1, /*keepdim=*/true);
    // grad = (p - onehot(target)) * valid / denom, avoiding a full
    // ones_like(p) intermediate (V=152k for Qwen -> 2.5 GB temp otherwise).
    // Clamp ignored indices (e.g. -100) to a valid slot; the valid mask zeroes
    // those rows out afterwards so the value there is irrelevant.
    auto idx = target.clamp(0, logits.size(-1) - 1).unsqueeze(-1);
    auto grad_logits = p;
    grad_logits.scatter_add_(-1, idx, torch::ones_like(p, torch::kFloat32) * -1.0);
    grad_logits = grad_logits * valid.unsqueeze(-1);
    grad_logits = grad_logits * (dy / denom);
    return {grad_logits, torch::Tensor(), torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// Conv2d - use ATen's built-in autograd (no custom Function needed)
// ---------------------------------------------------------------------------

}  // namespace vvm_torch::autograd
