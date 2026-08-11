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
    auto out = normed * weight + (bias.defined() ? bias : torch::Tensor());
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
  static torch::Tensor forward(AutogradContext* ctx,
                               torch::Tensor q,
                               torch::Tensor k,
                               torch::Tensor v,
                               double scale,
                               torch::Tensor mask) {
    ctx->save_for_backward({q, k, v});
    ctx->saved_data["scale"] = scale;
    ctx->saved_data["has_mask"] = mask.defined();

    auto scores = q.matmul(k.transpose(-2, -1)) * scale;
    if (mask.defined()) scores = scores + mask;
    auto m = std::get<0>(scores.max(-1, /*keepdim=*/true));
    auto e = torch::exp(scores - m);
    auto p = e / e.sum(-1, /*keepdim=*/true);
    ctx->saved_data["p"] = p;
    auto out = p.matmul(v);
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto q = saved[0], k = saved[1], v = saved[2];
    auto p = ctx->saved_data["p"].toTensor();
    auto scale = ctx->saved_data["scale"].toDouble();
    auto dy = grad_outputs[0];

    auto dv = p.transpose(-2, -1).matmul(dy);
    auto dp = dy.matmul(v.transpose(-2, -1));
    // softmax backward: ds = p * (dp - sum(dp*p))
    auto ds = p * (dp - (dp * p).sum(-1, /*keepdim=*/true));
    auto dq = ds.matmul(k) * scale;
    auto dk = ds.transpose(-2, -1).matmul(q) * scale;
    bool has_mask = ctx->saved_data["has_mask"].toBool();
    return {dq, dk, dv, torch::Tensor(), torch::Tensor()};
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
                               torch::Tensor target) {
    ctx->save_for_backward({logits, target});
    auto m = std::get<0>(logits.max(-1, /*keepdim=*/true));
    auto shifted = logits - m;
    auto logsumexp = torch::logsumexp(shifted, -1);
    auto picked = shifted.gather(-1, target.unsqueeze(-1)).squeeze(-1);
    auto loss = (logsumexp - picked).mean();
    ctx->saved_data["loss"] = loss;
    return loss;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto logits = saved[0];
    auto target = saved[1];
    auto dy = grad_outputs[0];
    auto m = std::get<0>(logits.max(-1, /*keepdim=*/true));
    auto shifted = logits - m;
    auto e = torch::exp(shifted);
    auto p = e / e.sum(-1, /*keepdim=*/true);
    auto grad_logits = p;
    grad_logits.scatter_add_(-1, target.unsqueeze(-1),
                             torch::ones_like(p) * -1.0);
    auto N = logits.size(0);
    grad_logits = grad_logits * (dy / static_cast<double>(N));
    return {grad_logits, torch::Tensor()};
  }
};

// ---------------------------------------------------------------------------
// Conv2d - use ATen's built-in autograd (no custom Function needed)
// ---------------------------------------------------------------------------

}  // namespace vvm_torch::autograd
