// vulkanvm_layers.hpp
// Layer modules built on top of the custom autograd Functions.
//
// We expose these via pybind11 as regular Python classes inheriting from
// torch.nn.Module (so users can include them in any nn.Module), but the
// forward path routes through the custom Functions above, allowing the
// autograd graph to record VulkanVM dispatches.
//
// Follows the vocabulary used in PyTorch's nn modules and llm.c
// (LayerNorm, matmul, attention, fused_residual).

#pragma once

#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <vector>

namespace vvm_torch::layers {

// ---------------------------------------------------------------------------
// Helpers to build modules from Python (we keep raw tensors + init fns here
// so users can subclass nn.Module on the Python side and call us in forward).
// ---------------------------------------------------------------------------

inline torch::Tensor init_linear_weight(int64_t in_f, int64_t out_f) {
  auto opts = torch::kFloat32;
  return torch::empty({out_f, in_f}, opts).normal_(0.0, std::sqrt(1.0 / in_f));
}

inline torch::Tensor init_linear_bias(int64_t out_f) {
  return torch::zeros({out_f}, torch::kFloat32);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
make_linear(int64_t in_f, int64_t out_f, bool with_bias) {
  auto w = init_linear_weight(in_f, out_f);
  auto b = with_bias ? init_linear_bias(out_f) : torch::Tensor();
  auto g = torch::zeros({out_f}, torch::kFloat32);   // grad accumulator
  return {w, b, g};
}

// LayerNorm params
inline std::tuple<torch::Tensor, torch::Tensor>
make_layernorm(int64_t dim) {
  return {torch::ones({dim}, torch::kFloat32),
          torch::zeros({dim}, torch::kFloat32)};
}

// Embedding
inline torch::Tensor make_embedding(int64_t vocab, int64_t dim) {
  return torch::empty({vocab, dim}, torch::kFloat32).normal_(0.0, 0.02);
}

// Conv2d initialization (Kaiming uniform, matching torch.nn.Conv2d)
inline std::tuple<torch::Tensor, torch::Tensor>
make_conv2d(int64_t in_ch, int64_t out_ch, int64_t kH, int64_t kW, bool with_bias) {
  // fan_in = in_ch * kH * kW
  double fan_in = static_cast<double>(in_ch * kH * kW);
  double bound  = std::sqrt(6.0 / fan_in);
  auto w = torch::empty({out_ch, in_ch, kH, kW}, torch::kFloat32).uniform_(-bound, bound);
  auto b = with_bias ? torch::zeros({out_ch}, torch::kFloat32) : torch::Tensor();
  return {w, b};
}

// Attention QKV stacked init (matches GPT-2: (3*dim, dim))
inline std::tuple<torch::Tensor, torch::Tensor>
make_qkv(int64_t dim) {
  return {torch::empty({3 * dim, dim}, torch::kFloat32).normal_(0.0, 0.02),
          torch::zeros({3 * dim}, torch::kFloat32)};
}

// MLP up-projection: (4*dim, dim)
inline std::tuple<torch::Tensor, torch::Tensor>
make_mlp_up(int64_t dim) {
  return {torch::empty({4 * dim, dim}, torch::kFloat32).normal_(0.0, 0.02),
          torch::zeros({4 * dim}, torch::kFloat32)};
}

// MLP down-projection: (dim, 4*dim)
inline std::tuple<torch::Tensor, torch::Tensor>
make_mlp_down(int64_t dim) {
  double scale = 0.02 / std::sqrt(2.0);   // GPT-2's 1/sqrt(2L)
  return {torch::empty({dim, 4 * dim}, torch::kFloat32).normal_(0.0, scale),
          torch::zeros({dim}, torch::kFloat32)};
}

// Position embedding
inline torch::Tensor make_position_embeddings(int64_t max_seq_len, int64_t dim) {
  return torch::empty({max_seq_len, dim}, torch::kFloat32).normal_(0.0, 0.02);
}

// ---------------------------------------------------------------------------
// Optimizer helpers
// ---------------------------------------------------------------------------

struct AdamWState {
  torch::Tensor m, v;
  int64_t step = 0;
};

class AdamWRegistry {
 public:
  static AdamWRegistry& instance() {
    static AdamWRegistry r;
    return r;
  }

  void register_param(const std::string& key, torch::Tensor p) {
    std::lock_guard<std::mutex> lock(mu_);
    AdamWState s;
    s.m = torch::zeros_like(p);
    s.v = torch::zeros_like(p);
    s.step = 0;
    state_[key] = {p, std::move(s)};
  }

  void step(const std::vector<std::string>& keys, double lr,
            double beta0, double beta1, double eps, double weight_decay) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& k : keys) {
      auto it = state_.find(k);
      if (it == state_.end()) continue;
      auto& [p, s] = it->second;
      if (!p.grad().defined()) continue;
      auto g = p.grad();
      if (weight_decay != 0.0) g = g + p * weight_decay;
      s.m.mul_(beta0).add_(g, 1.0 - beta0);
      s.v.mul_(beta1).addcmul_(g, g, 1.0 - beta1);
      auto m_hat = s.m / (1.0 - std::pow(beta0, s.step + 1));
      auto v_hat = s.v / (1.0 - std::pow(beta1, s.step + 1));
      p.addcmul_(m_hat, v_hat.sqrt().add_(eps), -lr);
      ++s.step;
    }
  }

  void zero_grad(const std::vector<std::string>& keys) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& k : keys) {
      auto it = state_.find(k);
      if (it == state_.end()) continue;
      auto& [p, s] = it->second;
      if (p.grad().defined()) p.grad().zero_();
    }
  }

 private:
  AdamWRegistry() = default;
  mutable std::mutex mu_;
  std::unordered_map<std::string,
                     std::pair<torch::Tensor, AdamWState>> state_;
};

}  // namespace vvm_torch::layers
