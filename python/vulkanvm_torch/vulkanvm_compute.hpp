// vulkanvm_compute.hpp
// Vulkan compute-shader dispatch bridge for PyTorch.
//
// Mirrors the dispatch pattern in src/tensor/tensor_transport.cpp:
//   - create shader module from SPIR-V / GLSL
//   - build a compute pipeline with push-constants + storage-buffer bindings
//   - record into a one-time command buffer and submit to the compute queue
//
// On the Python side these appear as vvm.compute.gemm/matmul/relu/gelu/...
// When no Vulkan pool/queue is available we fall back to ATen (so the
// extension still runs in CI / headless mode).

#pragma once

#include <vulkan_vm/vulkan_vm.hpp>

#include <torch/extension.h>
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <cstdint>

namespace vvm_torch::compute {

using vvm::UnifiedMemoryPool;

// ---------------------------------------------------------------------------
// Global pool accessor (the binding registers the pool once at startup)
// ---------------------------------------------------------------------------

struct PoolHandle {
  std::shared_ptr<UnifiedMemoryPool> pool;
  VkDevice     device         = VK_NULL_HANDLE;
  VkQueue      computeQueue   = VK_NULL_HANDLE;
  uint32_t     computeFamily  = 0;
  VkCommandPool cmdPool       = VK_NULL_HANDLE;
};

inline PoolHandle& pool_state() {
  static PoolHandle s;
  return s;
}

inline void set_pool(std::shared_ptr<UnifiedMemoryPool> p,
                     VkDevice dev, VkQueue q, uint32_t family) {
  auto& s = pool_state();
  std::lock_guard<std::mutex> lock(s_mutex());
  s.pool = std::move(p);
  s.device = dev;
  s.computeQueue = q;
  s.computeFamily = family;

  if (s.cmdPool) vkDestroyCommandPool(dev, s.cmdPool, nullptr);
  VkCommandPoolCreateInfo ci{};
  ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  ci.queueFamilyIndex = family;
  vkCreateCommandPool(dev, &ci, nullptr, &s.cmdPool);
}

inline std::mutex& s_mutex() {
  static std::mutex m;
  return m;
}

inline bool has_compute() {
  auto& s = pool_state();
  return s.pool && s.device != VK_NULL_HANDLE &&
         s.computeQueue != VK_NULL_HANDLE;
}

// ---------------------------------------------------------------------------
// Shader cache:  shader-name -> VkShaderModule
// ---------------------------------------------------------------------------

struct ShaderEntry {
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipeline     pipeline = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
};

inline std::unordered_map<std::string, ShaderEntry>& shader_cache() {
  static std::unordered_map<std::string, ShaderEntry> c;
  return c;
}

// Built-in GLSL compute shaders embedded as SPIR-V bytes would go here.
// For now we expose a registration API so the Python side can upload
// compiled SPIR-V (read from .comp files via glslang at build time).
bool register_shader(const std::string& name,
                     const std::vector<uint32_t>& spirv,
                     uint32_t push_constant_bytes,
                     uint32_t storage_buffer_count);

// ---------------------------------------------------------------------------
// High-level dispatch entry points. Each takes PyTorch tensors, finds the
// backing VulkanVM allocation (if registered via the pool), or stages
// the data to a temporary device buffer. Returns the output tensor.
//
// When has_compute()==false these route through ATen instead.
// ---------------------------------------------------------------------------

torch::Tensor gemm(torch::Tensor a, torch::Tensor b,
                   torch::Tensor bias = torch::Tensor());

torch::Tensor relu(torch::Tensor x);
torch::Tensor gelu(torch::Tensor x, bool approximate = true);
torch::Tensor silu(torch::Tensor x);
torch::Tensor softmax(torch::Tensor x, int64_t dim = -1);
torch::Tensor layernorm(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor scaled_dot_product_attention(torch::Tensor q,
                                            torch::Tensor k,
                                            torch::Tensor v,
                                            double scale = 0.0);

// Conv2d + transposed variant
torch::Tensor conv2d(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     std::vector<int64_t> stride,
                     std::vector<int64_t> padding,
                     std::vector<int64_t> dilation,
                     int64_t groups);

// RMSNorm (popular in modern LLMs): y = x / rms(x) * w  + b
torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                      double eps = 1e-6);

// SwiGLU activation (used in LLaMA): out = silu(x1 @ W1) * (x2 @ W2)
torch::Tensor swiglu(torch::Tensor x1, torch::Tensor x2);

// RoPE (Rotary Position Embedding) for q,k along head dim
std::tuple<torch::Tensor, torch::Tensor>
rope(torch::Tensor q, torch::Tensor k,
     int64_t seq_len, int64_t head_dim, double base = 10000.0);

// Fused AdamW kernel applied in-place to a vector of (param, grad, m, v)
void adamw_inplace(std::vector<torch::Tensor> params,
                   std::vector<torch::Tensor> grads,
                   std::vector<torch::Tensor> m,
                   std::vector<torch::Tensor> v,
                   double lr, double beta0, double beta1,
                   double eps, double weight_decay, int64_t step);

}  // namespace vvm_torch::compute
