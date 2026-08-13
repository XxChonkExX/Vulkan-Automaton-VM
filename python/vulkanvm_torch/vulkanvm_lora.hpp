// vulkanvm_lora.hpp
// LoRA (Low-Rank Adaptation) adapter registry & ops.
//
// Follows the design of microsoft/LoRA (loralib): pairs of low-rank
// matrices A (r, in) and B (out, r) applied as
//   delta_W = scale * alpha * B @ A
// where typically scale = alpha / r and alpha is a user-tunable constant.
//
// Adapters are stored on the C++ side in a thread-safe registry keyed by
// string name (mirrors loralib's string-keyed approach), so that Python
// can create, query, mutate and delete them efficiently without crossing
// the language boundary per call.

#pragma once

#include <torch/extension.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vvm_torch::lora {

// ---------------------------------------------------------------------------
// LoRAAdapter  (holds A, B, optimizer state, optional base weight snapshot)
// ---------------------------------------------------------------------------

struct LoRAAdapter {
  std::string name;
  int64_t in_features = 0;
  int64_t out_features = 0;
  int64_t rank = 0;
  double alpha = 1.0;
  double scale = 1.0;          // typically alpha / rank
  bool merged = false;

  torch::Tensor a;              // shape (rank, in_features)   -> A
  torch::Tensor b;              // shape (out_features, rank)  -> B
  torch::Tensor base_weight;   // optional snapshot for merge

  // AdamW-style optimizer state
  torch::Tensor a_m, a_v;      // 1st/2nd moments for A
  torch::Tensor b_m, b_v;      // 1st/2nd moments for B
  int64_t step = 0;

  bool initialized = false;
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

class LoRARegistry {
 public:
  static LoRARegistry& instance() {
    static LoRARegistry r;
    return r;
  }

  std::shared_ptr<LoRAAdapter> create(const std::string& name,
                                      int64_t in_features,
                                      int64_t out_features,
                                      int64_t rank,
                                      double alpha) {
    std::lock_guard<std::mutex> lock(mu_);
    if (table_.count(name)) {
      throw std::runtime_error("LoRA adapter '" + name + "' already exists");
    }
    auto adapter = std::make_shared<LoRAAdapter>();
    adapter->name = name;
    adapter->in_features = in_features;
    adapter->out_features = out_features;
    adapter->rank = rank;
    adapter->alpha = alpha;
    adapter->scale = rank > 0 ? alpha / static_cast<double>(rank) : 0.0;
    init_weights(*adapter);
    table_[name] = adapter;
    return adapter;
  }

  std::shared_ptr<LoRAAdapter> get(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = table_.find(name);
    if (it == table_.end()) return nullptr;
    return it->second;
  }

  std::vector<std::string> list() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> keys;
    keys.reserve(table_.size());
    for (const auto& [k, _] : table_) keys.push_back(k);
    return keys;
  }

  bool remove(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    return table_.erase(name) > 0;
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mu_);
    table_.clear();
  }

 private:
  LoRARegistry() = default;

  void init_weights(LoRAAdapter& a) {
    // A ~ N(0, 1/r)  ; B = 0  (standard LoRA init)
    // Place adapters on the same device as the rest of training: CUDA/HIP if
    // available, CPU otherwise. (The old code assigned kFloat32 in both
    // branches, so adapters were silently stuck on CPU and matmul'ing against
    // GPU inputs would throw.)
    torch::TensorOptions opts = torch::kFloat32;
    if (torch::cuda::is_available()) opts = opts.device(torch::kCUDA);
    a.a = torch::randn({a.rank, a.in_features}, opts) /
          std::sqrt(static_cast<double>(a.rank > 0 ? a.rank : 1));
    a.b = torch::zeros({a.out_features, a.rank}, opts);
    a.a_m = torch::zeros_like(a.a);
    a.a_v = torch::zeros_like(a.a);
    a.b_m = torch::zeros_like(a.b);
    a.b_v = torch::zeros_like(a.b);
    a.step = 0;
    a.merged = false;
    a.initialized = true;
    // A and B are the trainable parameters; the caller may still .to() them
    // elsewhere, but requires_grad must be set for .grad() to be populated.
    a.a.requires_grad_(true);
    a.b.requires_grad_(true);
  }

  mutable std::mutex mu_;
  std::unordered_map<std::string, std::shared_ptr<LoRAAdapter>> table_;
};

// ---------------------------------------------------------------------------
// Merge delta_W = scale * B @ A onto the base weight (in-place);
// unmerge reverses the operation. Set adapter.scale = 0 to disable.
// ---------------------------------------------------------------------------

inline void merge_into_base(LoRAAdapter& a, torch::Tensor base_weight) {
  TORCH_CHECK(base_weight.sizes() ==
                  torch::IntArrayRef({a.out_features, a.in_features}),
              "base weight shape mismatch for LoRA merge");
  torch::NoGradGuard no_grad;
  auto delta = a.b.mm(a.a) * a.scale;
  base_weight.add_(delta);
  a.base_weight = base_weight.clone();
  a.merged = true;
}

inline void unmerge_from_base(LoRAAdapter& a) {
  if (!a.merged || !a.base_weight.defined()) return;
  torch::NoGradGuard no_grad;
  auto delta = a.b.mm(a.a) * a.scale;
  a.base_weight.sub_(delta);    // restore
  a.merged = false;
}

// ---------------------------------------------------------------------------
// Apply one AdamW step to A, B using their stored grads (.grad tensors)
// Uses TRUE decoupled AdamW: weight decay is applied to the parameter
// directly, not folded into the gradient (L2-style). All updates happen on
// .data() under NoGradGuard so in-place ops on the leaf are legal.
// ---------------------------------------------------------------------------

inline void adamw_step(LoRAAdapter& a,
                       double lr,
                       double betas0, double betas1,
                       double eps, double weight_decay) {
  torch::NoGradGuard no_grad;
  auto step_update = [&](torch::Tensor p, torch::Tensor m, torch::Tensor v) {
    if (!p.requires_grad() || !p.grad().defined()) return;
    auto g = p.grad();
    m.mul_(betas0).add_(g, 1.0 - betas0);
    v.mul_(betas1).addcmul_(g, g, 1.0 - betas1);
    auto m_hat = m / (1.0 - std::pow(betas0, a.step + 1));
    auto v_hat = v / (1.0 - std::pow(betas1, a.step + 1));
    p.data().addcmul_(m_hat, v_hat.sqrt().add_(eps), -lr);
    if (weight_decay != 0.0) p.data().mul_(1.0 - lr * weight_decay);
  };
  step_update(a.a, a.a_m, a.a_v);
  step_update(a.b, a.b_m, a.b_v);
  ++a.step;
}

inline void zero_grad(LoRAAdapter& a) {
  torch::NoGradGuard no_grad;
  if (a.a.requires_grad() && a.a.grad().defined()) a.a.grad().zero_();
  if (a.b.requires_grad() && a.b.grad().defined()) a.b.grad().zero_();
}

inline std::unordered_map<std::string, torch::Tensor>
export_state(const LoRAAdapter& a) {
  return {
      {"a", a.a.clone()},
      {"b", a.b.clone()},
      {"scale", torch::tensor({a.scale})},
      {"alpha", torch::tensor({a.alpha})},
      {"rank", torch::tensor({a.rank}, torch::kInt64)},
  };
}

inline void import_state(LoRAAdapter& a,
                        const std::unordered_map<std::string, torch::Tensor>& st) {
  if (st.count("a")) a.a = st.at("a").clone();
  if (st.count("b")) a.b = st.at("b").clone();
  if (st.count("scale")) a.scale = st.at("scale").item<double>();
  if (st.count("alpha")) a.alpha = st.at("alpha").item<double>();
}

}  // namespace vvm_torch::lora
