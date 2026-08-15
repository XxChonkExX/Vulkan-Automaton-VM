// _attn_recompute_module.cpp — standalone pybind11 module exposing the
// custom-autograd attention with checkpointing (fp32 softmax probs are NOT
// saved for backward; recomputed instead). Kills the position-proportional
// growth of the eager attention graph.
#include <torch/extension.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>

#include "vulkanvm_autograd.hpp"

namespace py = pybind11;

namespace vvm_torch::autograd {
bool VulkanAttentionFn::recompute = false;
}

static torch::Tensor attention_apply(torch::Tensor q,
                                     torch::Tensor k,
                                     torch::Tensor v,
                                     double scale,
                                     torch::Tensor mask,
                                     int64_t kv_repeat,
                                     torch::Tensor cached_k,
                                     torch::Tensor cached_v) {
  return vvm_torch::autograd::VulkanAttentionFn::apply(
      q, k, v, scale, mask, kv_repeat, cached_k, cached_v);
}

PYBIND11_MODULE(vulkanvm_attn, m) {
  m.def("vulkan_attention", &attention_apply,
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("scale"), py::arg("mask") = torch::Tensor(),
        py::arg("kv_repeat") = 1,
        py::arg("cached_k").none(true) = torch::Tensor(),
        py::arg("cached_v").none(true) = torch::Tensor(),
        "Custom-autograd attention; with set_attention_recompute(True) the "
        "fp32 softmax probabilities are recomputed in backward instead of "
        "saved (attention checkpointing). cached_k/v: zero-copy views of the "
        "constant cached span; when given, only the current chunk's k/v rows "
        "are saved and backward re-cats them.");
  m.def("set_attention_recompute",
        [](bool on) { vvm_torch::autograd::VulkanAttentionFn::setRecompute(on); },
        py::arg("on"));
}