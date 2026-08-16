# vulkanvm_quant_py.py
# Pure-Python INT8/INT4 per-group quantization for the Chonk Buffer pipeline.
# Weights are stored quantized in the pool (uint8 + fp32 scales + uint8 zeros),
# dequantized on-the-fly per module forward. Base weights are frozen; LoRA
# (via PEFT wrappers) supplies all trainable parameters.
#
# INT8:  qweight (out, in)    uint8;  scale/zero per group of `group_size`.
# INT4:  qweight (out, in/2)  uint8 (two 4-bit values packed per byte).
# Dequant:  w = (q - zero) * scale  per group.
#
# The autograd fn (QuantMatmulFn) saves ONLY the quantized buffers and re-dequants
# in backward, so the pool does not retain dequantized weights across chunks.

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight_int8(weight: torch.Tensor, group_size: int = 128):
    """Vectorized per-group asymmetric INT8 quantization.

    Returns (qweight, scales, zeros):
      qweight: (out_features, in_features) uint8
      scales:  (out_features, num_groups)  fp32
      zeros:   (out_features, num_groups)  uint8
    Dequant:  w = (q - zero) * scale  per group.
    """
    assert weight.dim() == 2, "weight must be (out_features, in_features)"
    out_f, in_f = weight.shape
    assert in_f % group_size == 0, "in_features must be divisible by group_size"
    num_groups = in_f // group_size
    device = weight.device

    w = weight.reshape(out_f, num_groups, group_size)
    wmin = w.min(dim=-1).values
    wmax = w.max(dim=-1).values
    scale = (wmax - wmin) / 255.0
    scale = torch.clamp(scale, min=1e-8)
    zero = (-wmin / scale).round().clamp(0, 255)

    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 255)
    qweight = q.reshape(out_f, in_f).to(torch.uint8)
    scales = scale.to(torch.float32).contiguous()
    zeros = zero.to(torch.uint8).contiguous()
    return qweight, scales, zeros


def quantize_weight_int4(weight: torch.Tensor, group_size: int = 128):
    """Vectorized per-group asymmetric INT4 quantization (packed 2/byte).

    Returns (qweight, scales, zeros):
      qweight: (out_features, in_features // 2) uint8  (nibble-packed)
      scales:  (out_features, num_groups)              fp32
      zeros:   (out_features, num_groups)              uint8 (0..15)
    """
    assert weight.dim() == 2, "weight must be (out_features, in_features)"
    out_f, in_f = weight.shape
    assert in_f % group_size == 0, "in_features must be divisible by group_size"
    assert in_f % 2 == 0
    num_groups = in_f // group_size
    device = weight.device

    w = weight.reshape(out_f, num_groups, group_size)
    wmin = w.min(dim=-1).values
    wmax = w.max(dim=-1).values
    scale = (wmax - wmin) / 15.0
    scale = torch.clamp(scale, min=1e-8)
    zero = (-wmin / scale).round().clamp(0, 15)

    q = (w / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, 15)
    qflat = q.reshape(out_f, in_f).to(torch.uint8)
    qweight = (qflat[:, 0::2] & 0x0F) | ((qflat[:, 1::2] & 0x0F) << 4)
    scales = scale.to(torch.float32).contiguous()
    zeros = zero.to(torch.uint8).contiguous()
    return qweight, scales, zeros


def dequantize_weight_int8(qweight, scales, zeros, group_size=128, dtype=torch.bfloat16):
    """Dequantize INT8 weight back to `dtype` (for validation / merge)."""
    out_f, in_f = qweight.shape
    num_groups = in_f // group_size
    w = (
        qweight.to(torch.float32).reshape(out_f, num_groups, group_size)
        - zeros.to(torch.float32).reshape(out_f, num_groups, 1)
    ) * scales.reshape(out_f, num_groups, 1)
    return w.reshape(out_f, in_f).to(dtype)


def dequantize_weight_int4(qweight, scales, zeros, group_size=128, dtype=torch.bfloat16):
    """Dequantize packed INT4 weight back to `dtype`."""
    out_f, packed = qweight.shape
    in_f = packed * 2
    num_groups = in_f // group_size
    lo = (qweight & 0x0F).to(torch.float32)
    hi = ((qweight >> 4) & 0x0F).to(torch.float32)
    q = torch.stack([lo, hi], dim=-1).reshape(out_f, in_f)
    q = q.reshape(out_f, num_groups, group_size)
    w = (
        q - zeros.to(torch.float32).reshape(out_f, num_groups, 1)
    ) * scales.reshape(out_f, num_groups, 1)
    return w.reshape(out_f, in_f).to(dtype)


def quantize_weight(weight: torch.Tensor, bits: int = 8, group_size: int = 128):
    if bits == 8:
        return quantize_weight_int8(weight, group_size)
    if bits == 4:
        return quantize_weight_int4(weight, group_size)
    raise ValueError(f"unsupported bits={bits}")


def dequantize_weight(qweight, scales, zeros, bits: int = 8, group_size=128,
                      dtype=torch.bfloat16):
    if bits == 8:
        return dequantize_weight_int8(qweight, scales, zeros, group_size, dtype)
    if bits == 4:
        return dequantize_weight_int4(qweight, scales, zeros, group_size, dtype)
    raise ValueError(f"unsupported bits={bits}")


class QuantMatmulFn(torch.autograd.Function):
    """Dequant + matmul with minimal graph retention.

    Saves ONLY the (pool-backed) quantized buffers for backward; the
    dequantized weight is materialized transiently in forward and REBUILT
    in backward. This keeps the autograd graph ~zero-cost: without it, every
    chunk's forward saves 16GB+ of dequantized bf16 weights and the pool
    carves that on top of the model-size savings.
    """

    @staticmethod
    def forward(ctx, x, qweight, scales, zeros, bits, group_size, bias):
        ctx.save_for_backward(qweight, scales, zeros, bias)
        ctx.bits = int(bits)
        ctx.group_size = int(group_size)
        w = dequantize_weight(qweight, scales, zeros, ctx.bits, ctx.group_size,
                              dtype=torch.bfloat16)
        return F.linear(x, w, bias)

    @staticmethod
    def backward(ctx, grad_output):
        qweight, scales, zeros, bias = ctx.saved_tensors
        w = dequantize_weight(qweight, scales, zeros, ctx.bits, ctx.group_size,
                              dtype=torch.bfloat16)
        grad_x = grad_output @ w
        return grad_x, None, None, None, None, None, None


class QuantLinear(nn.Module):
    """INT8/INT4 per-group quantized Linear. Weight frozen; grads flow via LoRA
    branches added by PEFT wrappers. qweight/scales/zeros live in the pool."""

    def __init__(self, in_features, out_features, qweight, scales, zeros,
                 bits=8, group_size=128, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zeros", zeros)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_buffer("bias", None)

    def dequantized_weight(self):
        return dequantize_weight(self.qweight, self.scales, self.zeros,
                                 self.bits, self.group_size,
                                 dtype=torch.bfloat16)

    def forward(self, x):
        return QuantMatmulFn.apply(
            x, self.qweight, self.scales, self.zeros,
            self.bits, self.group_size, self.bias)


def quantize_module_weight(linear: nn.Module, bits=8, group_size=128):
    """Quantize an nn.Linear's weight; returns QuantLinear (keeps bias)."""
    w = linear.weight.detach()
    qweight, scales, zeros = quantize_weight(w, bits=bits, group_size=group_size)
    bias = linear.bias.detach() if linear.bias is not None else None
    return QuantLinear(w.shape[1], w.shape[0], qweight, scales, zeros,
                       bits=bits, group_size=group_size, bias=bias)