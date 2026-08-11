// vulkanvm_autograd_bind.cpp
//
// Pybind11 bindings for the autograd Functions, LoRA adapters, and layer
// factories defined in vulkanvm_autograd.hpp / vulkanvm_lora.hpp /
// vulkanvm_layers.hpp. This file lives next to vulkanvm_torch.cpp and is
// compiled into the same `vulkanvm_torch` Python module so users get a
// single `import vulkanvm_torch` that exposes everything.
//
// Build requirements: MSVC 19.44 / clang-cl, C++20, /Zc:__cplusplus.
// (Earlier attempts with C++17 triggered pybind11's nested-namespace
// detection bug; see python/vulkanvm_torch/pybind11_msvc_workaround.h.)

// Include ATen FIRST to avoid pybind11's ssize_t namespace pollution
// interfering with ATen's template deduction.
#include <torch/extension.h>
#include <ATen/core/Tensor.h>

// Include PyTorch's pybind11 type casters for at::Tensor
#include <torch/csrc/jit/python/pybind_utils.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "vulkanvm_autograd.hpp"
#include "vulkanvm_lora.hpp"
#include "vulkanvm_layers.hpp"

namespace py = pybind11;

using namespace vvm_torch;

// -----------------------------------------------------------------------------
// Bind LoRAAdapter struct
// -----------------------------------------------------------------------------
PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>);

void register_lora_adapter(py::module_& m) {
  py::class_<lora::LoRAAdapter, std::shared_ptr<lora::LoRAAdapter>>(m, "LoRAAdapter")
    .def_readwrite("name", &lora::LoRAAdapter::name)
    .def_readwrite("in_features", &lora::LoRAAdapter::in_features)
    .def_readwrite("out_features", &lora::LoRAAdapter::out_features)
    .def_readwrite("rank", &lora::LoRAAdapter::rank)
    .def_readwrite("alpha", &lora::LoRAAdapter::alpha)
    .def_readwrite("scale", &lora::LoRAAdapter::scale)
    .def_readwrite("merged", &lora::LoRAAdapter::merged)
    .def_readwrite("initialized", &lora::LoRAAdapter::initialized)
    .def_readonly("a", &lora::LoRAAdapter::a)
    .def_readonly("b", &lora::LoRAAdapter::b)
    .def_readonly("base_weight", &lora::LoRAAdapter::base_weight)
    .def_readonly("a_m", &lora::LoRAAdapter::a_m)
    .def_readonly("a_v", &lora::LoRAAdapter::a_v)
    .def_readonly("b_m", &lora::LoRAAdapter::b_m)
    .def_readonly("b_v", &lora::LoRAAdapter::b_v)
    .def_readwrite("step", &lora::LoRAAdapter::step);
}

// -----------------------------------------------------------------------------
// Thin wrappers around the autograd Functions with explicit signatures.
// Each wrapper calls the underlying ::apply() helper that
// torch::autograd::Function<T> exposes.
// -----------------------------------------------------------------------------

static torch::Tensor
VulkanLinearFn_apply(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias) {
  return autograd::VulkanLinearFn::apply(input, weight, bias);
}

static torch::Tensor
VulkanGeluFn_apply(torch::Tensor input, bool approximate) {
  return autograd::VulkanGeluFn::apply(input, approximate);
}

static torch::Tensor
VulkanReluFn_apply(torch::Tensor input) {
  return autograd::VulkanReluFn::apply(input);
}

static torch::Tensor
VulkanSiluFn_apply(torch::Tensor input) {
  return autograd::VulkanSiluFn::apply(input);
}

static torch::Tensor
VulkanSoftmaxFn_apply(torch::Tensor input) {
  return autograd::VulkanSoftmaxFn::apply(input);
}

static std::vector<torch::Tensor>
VulkanLayerNormFn_apply(torch::Tensor input,
                        torch::Tensor weight,
                        torch::Tensor bias,
                        int64_t normalized_shape_v) {
  return autograd::VulkanLayerNormFn::apply(input, weight, bias,
                                            normalized_shape_v);
}

static torch::Tensor
VulkanAttentionFn_apply(torch::Tensor q,
                        torch::Tensor k,
                        torch::Tensor v,
                        double scale,
                        torch::Tensor mask) {
  return autograd::VulkanAttentionFn::apply(q, k, v, scale, mask);
}

static torch::Tensor
VulkanLoraLinearFn_apply(torch::Tensor input,
                         torch::Tensor weight,
                         torch::Tensor bias,
                         torch::Tensor lora_a,
                         torch::Tensor lora_b,
                         double scale) {
  return autograd::VulkanLoraLinearFn::apply(input, weight, bias,
                                             lora_a, lora_b, scale);
}

static torch::Tensor
VulkanCrossEntropyFn_apply(torch::Tensor logits,
                           torch::Tensor target) {
  return autograd::VulkanCrossEntropyFn::apply(logits, target);
}

static torch::Tensor
VulkanConv2dFn_apply(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     std::vector<int64_t> stride,
                     std::vector<int64_t> padding,
                     std::vector<int64_t> dilation,
                     int64_t groups) {
  return torch::conv2d(input, weight, bias.defined() ? bias : torch::Tensor(),
                       stride, padding, dilation, groups);
}

// -----------------------------------------------------------------------------
// LoRA registry bindings
// -----------------------------------------------------------------------------

static std::shared_ptr<lora::LoRAAdapter>
lora_create(const std::string& name,
            int64_t in_features,
            int64_t out_features,
            int64_t rank,
            double alpha) {
  return lora::LoRARegistry::instance().create(name, in_features,
                                                out_features, rank, alpha);
}

static bool lora_remove(const std::string& name) {
  return lora::LoRARegistry::instance().remove(name);
}

static std::vector<std::string> lora_list() {
  return lora::LoRARegistry::instance().list();
}

static void lora_clear() {
  lora::LoRARegistry::instance().clear();
}

static void lora_merge_into_base(const std::string& name,
                                 torch::Tensor base_weight) {
  auto a = lora::LoRARegistry::instance().get(name);
  if (!a) throw std::runtime_error("LoRA adapter '" + name + "' not found");
  lora::merge_into_base(*a, base_weight);
}

static void lora_unmerge_from_base(const std::string& name) {
  auto a = lora::LoRARegistry::instance().get(name);
  if (!a) throw std::runtime_error("LoRA adapter '" + name + "' not found");
  lora::unmerge_from_base(*a);
}

static void lora_adamw_step(const std::string& name,
                            double lr,
                            double beta0, double beta1,
                            double eps, double weight_decay) {
  auto a = lora::LoRARegistry::instance().get(name);
  if (!a) throw std::runtime_error("LoRA adapter '" + name + "' not found");
  lora::adamw_step(*a, lr, beta0, beta1, eps, weight_decay);
}

static void lora_zero_grad(const std::string& name) {
  auto a = lora::LoRARegistry::instance().get(name);
  if (!a) throw std::runtime_error("LoRA adapter '" + name + "' not found");
  lora::zero_grad(*a);
}

// -----------------------------------------------------------------------------
// AdamW registry bindings
// -----------------------------------------------------------------------------

static void adamw_register_param(const std::string& key, torch::Tensor p) {
  layers::AdamWRegistry::instance().register_param(key, p);
}

static void adamw_step(const std::vector<std::string>& keys,
                       double lr, double beta0, double beta1,
                       double eps, double weight_decay) {
  layers::AdamWRegistry::instance().step(keys, lr, beta0, beta1,
                                         eps, weight_decay);
}

static void adamw_zero_grad(const std::vector<std::string>& keys) {
  layers::AdamWRegistry::instance().zero_grad(keys);
}

// -----------------------------------------------------------------------------
// Module init function. Called by PYBIND11_MODULE in vulkanvm_torch.cpp so
// everything ends up in the same .pyd.
// -----------------------------------------------------------------------------

void register_vulkanvm_autograd_bindings(py::module_& m) {
  // ---- LoRAAdapter class ----
  register_lora_adapter(m);

  // ---- Autograd ops ---------------------------------------------------------
  m.def("vulkan_linear",      &VulkanLinearFn_apply,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        "VulkanVM custom-autograd Linear: y = x @ W^T + b (grads to x, W, b)");
  m.def("vulkan_gelu",        &VulkanGeluFn_apply,
        py::arg("input"), py::arg("approximate") = false,
        "VulkanVM custom-autograd GELU (exact or tanh approximation)");
  m.def("vulkan_relu",        &VulkanReluFn_apply,
        py::arg("input"),
        "VulkanVM custom-autograd ReLU");
  m.def("vulkan_silu",        &VulkanSiluFn_apply,
        py::arg("input"),
        "VulkanVM custom-autograd SiLU / Swish");
  m.def("vulkan_layernorm",   &VulkanLayerNormFn_apply,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("normalized_shape_v"),
        "VulkanVM custom-autograd LayerNorm over the last dim");
  m.def("vulkan_softmax",     &VulkanSoftmaxFn_apply,
        py::arg("input"),
        "VulkanVM custom-autograd Softmax over the last dim");
  m.def("vulkan_attention",   &VulkanAttentionFn_apply,
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("scale"), py::arg("mask") = torch::Tensor(),
        "VulkanVM custom-autograd scaled-dot-product attention");
  m.def("vulkan_lora_linear", &VulkanLoraLinearFn_apply,
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("lora_a"), py::arg("lora_b"), py::arg("scale"),
        "VulkanVM custom-autograd LoRA-augmented Linear");
  m.def("vulkan_cross_entropy", &VulkanCrossEntropyFn_apply,
        py::arg("logits"), py::arg("target"),
        "VulkanVM custom-autograd cross-entropy loss");
  m.def("vulkan_conv2d",      [](torch::Tensor input, torch::Tensor weight,
                                 torch::Tensor bias,
                                 std::vector<int64_t> stride,
                                 std::vector<int64_t> padding,
                                 std::vector<int64_t> dilation,
                                 int64_t groups) {
      return torch::conv2d(input, weight, bias.defined() ? bias : torch::Tensor(),
                           stride, padding, dilation, groups);
    },
        py::arg("input"), py::arg("weight"), py::arg("bias"),
        py::arg("stride"), py::arg("padding"), py::arg("dilation"),
        py::arg("groups"),
        "ATen conv2d with built-in autograd");

  // ---- LoRA registry -------------------------------------------------------
  m.def("lora_create", &lora_create,
        py::arg("name"), py::arg("in_features"), py::arg("out_features"),
        py::arg("rank"), py::arg("alpha"),
        "Create a LoRA adapter in the registry");
  m.def("lora_remove", &lora_remove,
        py::arg("name"),
        "Remove a LoRA adapter from the registry");
  m.def("lora_list", &lora_list,
        "List all registered LoRA adapter names");
  m.def("lora_clear", &lora_clear,
        "Remove all LoRA adapters from the registry");
  m.def("lora_merge_into_base", &lora_merge_into_base,
        py::arg("name"), py::arg("base_weight"),
        "Merge LoRA delta into base weight in-place");
  m.def("lora_unmerge_from_base", &lora_unmerge_from_base,
        py::arg("name"),
        "Reverse a previously-merged LoRA adapter");
  m.def("lora_adamw_step", &lora_adamw_step,
        py::arg("name"), py::arg("lr"), py::arg("beta0"), py::arg("beta1"),
        py::arg("eps"), py::arg("weight_decay"),
        "Apply one AdamW step to the named LoRA adapter");
  m.def("lora_zero_grad", &lora_zero_grad,
        py::arg("name"),
        "Zero out gradients on the named LoRA adapter");

  // ---- Layer factories -----------------------------------------------------
  m.def("make_linear", &layers::make_linear,
        py::arg("in_f"), py::arg("out_f"), py::arg("with_bias") = true,
        "Initialize Linear (weight, bias, grad-accumulator) tensors");
  m.def("make_layernorm", &layers::make_layernorm,
        py::arg("dim"),
        "Initialize LayerNorm (weight, bias) tensors");
  m.def("make_embedding", &layers::make_embedding,
        py::arg("vocab"), py::arg("dim"),
        "Initialize Embedding weight tensor");
  m.def("make_conv2d", &layers::make_conv2d,
        py::arg("in_ch"), py::arg("out_ch"),
        py::arg("kH"), py::arg("kW"), py::arg("with_bias") = true,
        "Initialize Conv2d (weight, bias) tensors");
  m.def("make_qkv", &layers::make_qkv,
        py::arg("dim"),
        "Initialize fused QKV linear (GPT-2 style)");
  m.def("make_mlp_up", &layers::make_mlp_up,
        py::arg("dim"),
        "Initialize MLP up-projection (4*dim, dim)");
  m.def("make_mlp_down", &layers::make_mlp_down,
        py::arg("dim"),
        "Initialize MLP down-projection (dim, 4*dim)");
  m.def("make_position_embeddings", &layers::make_position_embeddings,
        py::arg("max_seq_len"), py::arg("dim"),
        "Initialize sinusoidal position-embedding tensor");

  // ---- AdamW registry -----------------------------------------------------
  m.def("adamw_register_param", &adamw_register_param,
        py::arg("key"), py::arg("p"),
        "Register a parameter with the AdamW optimizer registry");
  m.def("adamw_step", &adamw_step,
        py::arg("keys"), py::arg("lr"), py::arg("beta0"), py::arg("beta1"),
        py::arg("eps"), py::arg("weight_decay"),
        "Apply one AdamW step to the registered parameters");
  m.def("adamw_zero_grad", &adamw_zero_grad,
        py::arg("keys"),
        "Zero out gradients on the registered parameters");
}