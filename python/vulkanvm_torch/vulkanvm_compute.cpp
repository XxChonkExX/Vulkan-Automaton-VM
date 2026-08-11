// vulkanvm_compute.cpp
//
// Implementation of the compute dispatch bridge. When a VulkanVM pool with a
// compute queue is available we dispatch a compute shader; otherwise we route
// through ATen so the extension remains useful in CI / headless mode.

#include "vulkanvm_compute.hpp"
#include <ATen/ops/empty.h>
#include <ATen/ops/gemm.h>

namespace vvm_torch::compute {

// ---------------------------------------------------------------------------
// Shader registration
// ---------------------------------------------------------------------------

bool register_shader(const std::string& name,
                     const std::vector<uint32_t>& spirv,
                     uint32_t push_constant_bytes,
                     uint32_t storage_buffer_count) {
  auto& s = pool_state();
  if (!has_compute()) return false;
  std::lock_guard<std::mutex> lock(s_mutex());

  auto& cache = shader_cache();
  if (cache.count(name)) {
    auto& e = cache[name];
    if (e.pipeline) vkDestroyPipeline(s.device, e.pipeline, nullptr);
    if (e.layout) vkDestroyPipelineLayout(s.device, e.layout, nullptr);
    if (e.setLayout) vkDestroyDescriptorSetLayout(s.device, e.setLayout, nullptr);
    if (e.module) vkDestroyShaderModule(s.device, e.module, nullptr);
  }

  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spirv.size() * sizeof(uint32_t);
  smci.pCode = spirv.data();
  VkShaderModule mod;
  if (vkCreateShaderModule(s.device, &smci, nullptr, &mod) != VK_SUCCESS)
    return false;

  // descriptor set layout (storage buffers, consecutive bindings 0..n-1)
  std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
  for (uint32_t i = 0; i < storage_buffer_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.bindingCount = static_cast<uint32_t>(bindings.size());
  dslci.pBindings = bindings.data();
  VkDescriptorSetLayout setLayout;
  if (vkCreateDescriptorSetLayout(s.device, &dslci, nullptr, &setLayout) !=
      VK_SUCCESS) {
    vkDestroyShaderModule(s.device, mod, nullptr);
    return false;
  }

  VkPushConstantRange pc{};
  pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pc.offset = 0;
  pc.size = push_constant_bytes;

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &setLayout;
  plci.pushConstantRangeCount = push_constant_bytes ? 1u : 0u;
  plci.pPushConstantRanges = push_constant_bytes ? &pc : nullptr;
  VkPipelineLayout layout;
  if (vkCreatePipelineLayout(s.device, &plci, nullptr, &layout) != VK_SUCCESS) {
    vkDestroyDescriptorSetLayout(s.device, setLayout, nullptr);
    vkDestroyShaderModule(s.device, mod, nullptr);
    return false;
  }

  VkComputePipelineCreateInfo pci{};
  pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pci.stage.module = mod;
  pci.stage.pName = "main";
  pci.layout = layout;
  VkPipeline pipeline;
  if (vkCreateComputePipelines(s.device, VK_NULL_HANDLE, 1, &pci, nullptr,
                               &pipeline) != VK_SUCCESS) {
    vkDestroyPipelineLayout(s.device, layout, nullptr);
    vkDestroyDescriptorSetLayout(s.device, setLayout, nullptr);
    vkDestroyShaderModule(s.device, mod, nullptr);
    return false;
  }

  cache[name] = {mod, pipeline, layout, setLayout};
  return true;
}

// ---------------------------------------------------------------------------
// ATen fallbacks (used when no Vulkan compute queue is present)
// ---------------------------------------------------------------------------

torch::Tensor gemm(torch::Tensor a, torch::Tensor b, torch::Tensor bias) {
  auto out = a.mm(b.t());
  if (bias.defined()) out += bias.unsqueeze(0).expand_as(out);
  return out;
}

torch::Tensor relu(torch::Tensor x) { return torch::relu(x); }

torch::Tensor gelu(torch::Tensor x, bool approximate) {
  return approximate ? torch::gelu(x) : torch::gelu(x, "none");
}

torch::Tensor silu(torch::Tensor x) { return torch::silu(x); }

torch::Tensor softmax(torch::Tensor x, int64_t dim) {
  return torch::softmax(x, dim);
}

torch::Tensor layernorm(torch::Tensor x, torch::Tensor w, torch::Tensor b) {
  return torch::layer_norm(x, w.sizes(), w, b.defined() ? b : torch::Tensor());
}

torch::Tensor scaled_dot_product_attention(torch::Tensor q, torch::Tensor k,
                                            torch::Tensor v, double scale) {
  int64_t d = q.size(-1);
  if (scale == 0.0) scale = 1.0 / std::sqrt(static_cast<double>(d));
  auto scores = q.matmul(k.transpose(-2, -1)) * scale;
  // optional causal mask: infer from shapes (QK same seq → assume causal)
  if (q.size(-2) == k.size(-2)) {
    int64_t T = q.size(-2);
    auto idx = torch::arange(T, q.options());
    auto causal = (idx.unsqueeze(0) <= idx.unsqueeze(1)).to(q.scalar_type());
    causal = causal.unsqueeze(0).unsqueeze(0);   // (1,1,T,T)
    scores = scores.masked_fill(~causal.bool(),
                                std::numeric_limits<float>::lowest());
  }
  auto p = torch::softmax(scores, -1);
  return p.matmul(v);
}

torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight,
                     torch::Tensor bias, std::vector<int64_t> stride,
                     std::vector<int64_t> padding,
                     std::vector<int64_t> dilation, int64_t groups) {
  return torch::conv2d(input, weight,
                       bias.defined() ? bias : torch::Tensor(),
                       stride, padding, dilation, groups);
}

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      double eps) {
  auto ms = x.pow(2).mean(-1, /*keepdim=*/true);
  auto rstd = torch::rsqrt(ms + eps);
  auto normed = x * rstd;
  return normed * w + (b.defined() ? b : torch::Tensor());
}

torch::Tensor swiglu(torch::Tensor x1, torch::Tensor x2) {
  return torch::silu(x1) * x2;
}

std::tuple<torch::Tensor, torch::Tensor>
rope(torch::Tensor q, torch::Tensor k, int64_t seq_len, int64_t head_dim,
     double base) {
  // Build inverse-freq table: theta_i = base^(-2i/d) for i in 0..d/2
  auto half = head_dim / 2;
  auto inv_freq =
      torch::pow(base, -torch::arange(0, half, q.options()).to(torch::kFloat64) /
                            half);
  auto pos = torch::arange(seq_len, q.options().dtype(torch::kFloat64));
  auto freqs = pos.unsqueeze(-1).matmul(inv_freq.unsqueeze(0));    // (T, d/2)
  auto sin = torch::sin(freqs).to(q.scalar_type());
  auto cos = torch::cos(freqs).to(q.scalar_type());

  auto rotate_half = [](torch::Tensor x) {
    int64_t d = x.size(-1);
    auto x1 = x.slice(-1, 0, d / 2);
    auto x2 = x.slice(-1, d / 2, d);
    return torch::cat({-x2, x1}, -1);
  };

  auto apply = [&](torch::Tensor t) {
    auto cos_b = cos.unsqueeze(0).unsqueeze(0);
    auto sin_b = sin.unsqueeze(0).unsqueeze(0);
    return t * cos_b + rotate_half(t) * sin_b;
  };
  return {apply(q), apply(k)};
}

void adamw_inplace(std::vector<torch::Tensor> params,
                   std::vector<torch::Tensor> grads,
                   std::vector<torch::Tensor> m,
                   std::vector<torch::Tensor> v,
                   double lr, double beta0, double beta1,
                   double eps, double weight_decay, int64_t step) {
  double bias_c0 = 1.0 - std::pow(beta0, step + 1);
  double bias_c1 = 1.0 - std::pow(beta1, step + 1);
  for (size_t i = 0; i < params.size(); ++i) {
    auto& p = params[i];
    auto& g = grads[i];
    auto& mi = m[i];
    auto& vi = v[i];
    if (weight_decay != 0.0) g.add_(p, weight_decay);
    mi.mul_(beta0).add_(g, 1.0 - beta0);
    vi.mul_(beta1).addcmul_(g, g, 1.0 - beta1);
    auto m_hat = mi / bias_c0;
    auto v_hat = vi / bias_c1;
    p.addcmul_(m_hat, v_hat.sqrt().add_(eps), -lr);
  }
}

}  // namespace vvm_torch::compute
