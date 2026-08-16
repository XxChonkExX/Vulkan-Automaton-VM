// vulkanvm_quant.cpp
// Quantization implementation: INT8 quantization, fused dequant+matmul autograd

#include "vulkanvm_quant.hpp"
#include <vulkan_vm/offload.hpp>
#include <torch/extension.h>
#include <algorithm>
#include <cmath>

namespace vvm_torch::quant {

QuantizedWeight quantize_weight(const torch::Tensor& weight, int64_t group_size, bool use_int4) {
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D (out_features, in_features)");
    
    const int64_t out_features = weight.size(0);
    const int64_t in_features = weight.size(1);
    TORCH_CHECK(in_features % group_size == 0, "in_features must be divisible by group_size");
    
    QuantizedWeight qw;
    qw.in_features = in_features;
    qw.out_features = out_features;
    qw.group_size = group_size;
    qw.is_int4 = false;
    
    auto qweight_options = torch::dtype(torch::kUInt8).device(weight.device());
    qw.qweight = torch::empty({weight.size(0), weight.size(1)}, qweight_options);
    qw.scales = torch::empty({weight.size(0), weight.size(1) / group_size}, torch::kFloat32, weight.options().device());
    qw.zeros = torch::empty({weight.size(0), weight.size(1) / group_size}, torch::dtype(torch::kUInt8).device(weight.device()));
    
    auto weight_fp32 = weight.to(torch::kFloat32);
    
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t g = 0; g < in_features / group_size; ++g) {
            int64_t start = g * group_size;
            int64_t end = start + group_size;
            
            float min_val = INFINITY;
            float max_val = -INFINITY;
            for (int64_t i = start; i < end; ++i) {
                float val = weight_fp32[o][i].item<float>();
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            
            float scale = (max_val - min_val) / 255.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            uint8_t zero_point = static_cast<uint8_t>(std::round(-min_val / scale));
            zero_point = std::clamp<int>(zero_point, 0, 255);
            
            qw.scales[o][g] = scale;
            qw.zeros[o][g] = zero_point;
            
            for (int64_t i = start; i < end; ++i) {
                float val = weight_fp32[o][i].item<float>();
                int q = static_cast<int>(std::round(val / scale)) + zero_point;
                q = std::clamp<int>(q, 0, 255);
                qw.qweight[o][i] = static_cast<uint8_t>(q);
            }
        }
    }
    
    return qw;
}

torch::Tensor dequantize_weight(const QuantizedWeight& q) {
    auto out = torch::empty({q.out_features, q.in_features}, torch::kBFloat16, q.qweight.options().device());
    
    for (int64_t o = 0; o < q.out_features; ++o) {
        for (int64_t g = 0; g < q.in_features / q.group_size; ++g) {
            int64_t start = g * q.group_size;
            int64_t end = start + q.group_size;
            float scale = q.scales[o][g].item<float>();
            float zero = static_cast<float>(q.zeros[o][g].item<uint8_t>());
            
            for (int64_t i = start; i < end; ++i) {
                float qval = static_cast<float>(q.qweight[o][i].item<uint8_t>());
                float val = (qval - zero) * scale;
                out[o][i] = torch::kBFloat16(val);
            }
        }
    }
    return out;
}

torch::Tensor QuantizedMatmulFn::forward(
    torch::autograd::AutogradContext* ctx,
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t group_size,
    bool is_int4,
    const torch::Tensor& bias
) {
    TORCH_CHECK(x.dim() >= 1, "Input must have at least 1 dimension");
    TORCH_CHECK(qweight.dim() == 2, "qweight must be 2D (out, in)");
    TORCH_CHECK(scales.dim() == 2 && zeros.dim() == 2, "scales/zeros must be 2D");
    
    const int64_t out_features = qweight.size(0);
    const int64_t in_features = qweight.size(1);
    
    ctx->save_for_backward({qweight, scales, zeros});
    ctx->saved_data["group_size"] = group_size;
    ctx->saved_data["is_int4"] = is_int4;
    ctx->saved_data["has_bias"] = bias.defined();
    
    auto weight = torch::empty({qweight.size(0), qweight.size(1)}, torch::kBFloat16, qweight.options().device());
    
    for (int64_t o = 0; o < qweight.size(0); ++o) {
        for (int64_t g = 0; g < qweight.size(1) / group_size; ++g) {
            int64_t start = g * group_size;
            int64_t end = start + group_size;
            float scale = scales[o][g].item<float>();
            float zero = static_cast<float>(zeros[o][g].item<uint8_t>());
            
            for (int64_t i = start; i < end; ++i) {
                float qval = static_cast<float>(qweight[o][i].item<uint8_t>());
                float val = (qval - zero) * scale;
                weight[o][i] = torch::kBFloat16(val);
            }
        }
    }
    
    auto x_flat = x.reshape({-1, x.size(-1)});
    auto out = x_flat.matmul(weight.t());
    out = out.reshape({x.size(0), x.size(1), out_features});
    
    if (bias.defined()) {
        out = out + bias;
    }
    
    return out;
}

torch::autograd::tensor_list QuantizedMatmulFn::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs
) {
    auto saved = ctx->get_saved_variables();
    auto qweight = saved[0];
    auto scales = saved[1];
    auto zeros = saved[2];
    
    int64_t group_size = ctx->saved_data["group_size"].toInt();
    bool is_int4 = ctx->saved_data["is_int4"].toBool();
    bool has_bias = ctx->saved_data["has_bias"].toBool();
    
    auto grad_output = grad_outputs[0];
    
    auto weight = torch::empty({qweight.size(0), qweight.size(1)}, torch::kBFloat16, qweight.options().device());
    
    for (int64_t o = 0; o < qweight.size(0); ++o) {
        for (int64_t g = 0; g < qweight.size(1) / group_size; ++g) {
            int64_t start = g * group_size;
            int64_t end = start + group_size;
            float scale = scales[o][g].item<float>();
            float zero = static_cast<float>(zeros[o][g].item<uint8_t>());
            
            for (int64_t i = start; i < end; ++i) {
                float qval = static_cast<float>(qweight[o][i].item<uint8_t>());
                float val = (qval - zero) * scale;
                weight[o][i] = torch::kBFloat16(val);
            }
        }
    }
    
    auto grad_output_flat = grad_output.reshape({-1, grad_output.size(-1)});
    auto dx = grad_output_flat.matmul(weight);
    dx = dx.reshape({grad_output.size(0), grad_output.size(1), qweight.size(1)});
    
    torch::Tensor dbias;
    if (has_bias) {
        dbias = grad_output_flat.sum(0);
    }
    
    std::vector<torch::Tensor> grads;
    grads.push_back(dx);
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(torch::Tensor());
    grads.push_back(has_bias ? dbias : torch::Tensor());
    return grads;
}

torch::Tensor quantized_matmul(
    const torch::Tensor& x,
    const torch::Tensor& qweight,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int64_t group_size,
    bool is_int4,
    const torch::Tensor& bias
) {
    return QuantizedMatmulFn::apply({x, qweight, scales, zeros, group_size, is_int4, bias})[0];
}

QuantLinear::QuantLinear(int64_t in_features, int64_t out_features, 
                         bool bias, int64_t group_size, bool use_int4)
    : in_features(in_features), out_features(out_features),
      group_size(group_size), use_int4(use_int4), has_bias(bias) {
    if (bias) {
        bias_ = torch::zeros({out_features}, torch::kBFloat16);
        register_parameter("bias", bias_);
    }
}

void QuantLinear::load_quantized_weight(const torch::Tensor& qweight,
                                        const torch::Tensor& scales,
                                        const torch::Tensor& zeros,
                                        const torch::Tensor& bias) {
    qweight_ = qweight;
    scales_ = scales;
    zeros_ = zeros;
    if (bias.defined()) {
        bias_ = bias.to(torch::kBFloat16);
        if (!has_bias) {
            has_bias = true;
            register_parameter("bias", bias_);
        }
    }
}

torch::Tensor QuantLinear::forward(torch::Tensor x) {
    auto out = quantized_matmul(x, qweight_, scales_, zeros_, group_size, use_int4, has_bias ? bias_ : torch::Tensor());
    
    if (has_lora_) {
        auto lora_out = x.matmul(lora_a_.t()).matmul(lora_b_.t()) * lora_scale_;
        out = out + lora_out;
    }
    return out;
}

void QuantLinear::set_lora(torch::Tensor lora_a, torch::Tensor lora_b, double scale) {
    lora_a_ = lora_a;
    lora_b_ = lora_b;
    lora_scale_ = scale;
    has_lora_ = true;
}

QuantizedModelWeights quantize_model_weights(
    const std::unordered_map<std::string, torch::Tensor>& state_dict,
    const std::vector<std::string>& target_modules,
    int64_t group_size,
    bool use_int4
) {
    QuantizedModelWeights result;
    result.group_size = group_size;
    result.is_int4 = false;
    
    for (const auto& module_name : target_modules) {
        std::string weight_key = module_name + ".weight";
        std::string bias_key = module_name + ".bias";
        
        auto it = state_dict.find(weight_key);
        if (it == state_dict.end()) continue;
        
        auto qw = quantize_weight(it->second, group_size, false);
        
        result.qweights[module_name] = qw.qweight;
        result.scales[module_name] = qw.scales;
        result.zeros[module_name] = qw.zeros;
        
        auto bias_it = state_dict.find(module_name + ".bias");
        if (bias_it != state_dict.end()) {
            result.biases[module_name] = bias_it->second.to(torch::kBFloat16);
        }
    }
    
    return result;
}

QuantizedModelBuffer allocate_quantized_model_buffer(
    const QuantizedModelWeights& qweights,
    vvm_torch::ChonkPool& pool
) {
    QuantizedModelBuffer result;
    result.group_size = qweights.group_size;
    result.is_int4 = false;
    
    int64_t total_bytes = 0;
    for (const auto& [name, qweight] : qweights.qweights) {
        int64_t out = qweight.size(0);
        int64_t in = qweight.size(1);
        int64_t gs = qweights.group_size;
        
        int64_t qw_bytes = out * qweight.size(1);
        int64_t scales_bytes = out * (in / gs) * 4;
        int64_t zeros_bytes = out * (in / gs);
        int64_t bias_bytes = 0;
        if (qweights.biases.count(name)) {
            bias_bytes = qweights.biases.at(name).numel() * 2;
        }
        total_bytes += qw_bytes + scales_bytes + zeros_bytes + bias_bytes;
    }
    
    auto buffer = pool.alloc_model_weights(total_bytes, "quantized_model_weights");
    auto flat_typed = buffer.view(torch::kUInt8);
    
    QuantizedModelBuffer result;
    result.flat_buffer = buffer;
    result.total_bytes = total_bytes;
    result.group_size = qweights.group_size;
    result.is_int4 = false;
    
    int64_t offset = 0;
    for (const auto& [name, qweight] : qweights.qweights) {
        int64_t out = qweight.size(0);
        int64_t in = qweight.size(1);
        int64_t gs = qweights.group_size;
        
        int64_t qw_bytes = out * qweight.size(1);
        int64_t scales_bytes = out * (in / gs) * 4;
        int64_t zeros_bytes = out * (in / gs);
        int64_t bias_bytes = 0;
        if (qweights.biases.count(name)) {
            bias_bytes = qweights.biases.at(name).numel() * 2;
        }
        
        result.offsets[name] = {offset, offset + qw_bytes, offset + qw_bytes + scales_bytes, offset + qw_bytes + scales_bytes + zeros_bytes};
        
        auto flat_typed = buffer.view(torch::kUInt8);
        flat_typed.narrow(0, offset, qweight.numel()).copy_(qweights.qweights.at(name).view(torch::kUInt8));
        offset += qweight.numel();
        
        flat_typed.narrow(0, offset, scales_bytes).copy_(qweights.scales.at(name).view(torch::kUInt8));
        offset += scales_bytes;
        
        flat_typed.narrow(0, offset, qweights.zeros.at(name).numel()).copy_(qweights.zeros.at(name));
        offset += zeros_bytes;
        
        if (qweights.biases.count(name)) {
            int64_t bias_bytes = qweights.biases.at(name).numel() * 2;
            flat_typed.narrow(0, offset, bias_bytes).copy_(qweights.biases.at(name).view(torch::kUInt8));
            offset += bias_bytes;
        }
    }
    
    return result;
}

}  // namespace vvm_torch::quant