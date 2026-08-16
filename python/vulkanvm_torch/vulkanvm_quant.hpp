// vulkanvm_quant.hpp
// Quantization utilities for Chonk Buffer: INT8 weight quantization
// with fused dequant+matmul autograd function. Stays fully in Chonk Buffer.

#pragma once

#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

namespace vvm {
    class UnifiedMemoryPool;
}

namespace vvm_torch::quant {

struct QuantizedWeight {
    torch::Tensor qweight;
    torch::Tensor scales;
    torch::Tensor zeros;
    
    int64_t in_features = 0;
    int64_t out_features = 0;
    int64_t group_size = 128;
    bool is_int4 = false;
};

inline int64_t quantized_weight_bytes(const QuantizedWeight& q);

QuantizedWeight quantize_weight(const torch::Tensor& weight, int64_t group_size, bool use_int4);
torch::Tensor dequantize_weight(const QuantizedWeight& q);

class QuantizedMatmulFn : public torch::autograd::Function<QuantizedMatmulFn> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
        const torch::Tensor& x, const torch::Tensor& qweight,
        const torch::Tensor& scales, const torch::Tensor& zeros,
        int64_t group_size, bool is_int4, const torch::Tensor& bias);

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};

torch::Tensor quantized_matmul(
    const torch::Tensor& x, const torch::Tensor& qweight,
    const torch::Tensor& scales, const torch::Tensor& zeros,
    int64_t group_size, bool is_int4, const torch::Tensor& bias);

class QuantLinear : public torch::nn::Module {
public:
    QuantLinear(int64_t in_features, int64_t out_features, 
                bool bias = true, int64_t group_size = 128, bool use_int4 = false);
    
    void load_quantized_weight(const torch::Tensor& qweight,
                               const torch::Tensor& scales,
                               const torch::Tensor& zeros,
                               const torch::Tensor& bias = torch::Tensor());
    
    torch::Tensor forward(torch::Tensor x);
    void set_lora(torch::Tensor lora_a, torch::Tensor lora_b, double scale);
    
    int64_t in_features;
    int64_t out_features;
    int64_t group_size;
    bool use_int4;
    bool has_bias;

private:
    torch::Tensor qweight_, scales_, zeros_, bias_;
    torch::Tensor lora_a_, lora_b_;
    double lora_scale_ = 1.0;
    bool has_lora_ = false;
};

struct QuantizedModelWeights {
    std::unordered_map<std::string, torch::Tensor> qweights;
    std::unordered_map<std::string, torch::Tensor> scales;
    std::unordered_map<std::string, torch::Tensor> zeros;
    std::unordered_map<std::string, torch::Tensor> biases;
    int64_t group_size = 128;
    bool is_int4 = false;
};

QuantizedWeight quantize_weight(const torch::Tensor& weight, int64_t group_size, bool use_int4);
QuantizedModelWeights quantize_model_weights(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::vector<std::string>& target_modules,
    int64_t group_size, bool use_int4);

struct QuantizedModelBuffer {
    torch::Tensor flat_buffer;
    std::unordered_map<std::string, std::tuple<int64_t, int64_t, int64_t, int64_t>> offsets;
    int64_t total_bytes = 0;
    int64_t group_size = 128;
    bool is_int4 = false;
};

QuantizedModelBuffer allocate_quantized_model_buffer(
    const QuantizedModelWeights& qweights,
    vvm::UnifiedMemoryPool& pool);

torch::Tensor quantized_matmul(
    const torch::Tensor& x, const torch::Tensor& qweight,
    const torch::Tensor& scales, const torch::Tensor& zeros,
    int64_t group_size, bool is_int4, const torch::Tensor& bias);

}  // namespace vvm_torch::quant