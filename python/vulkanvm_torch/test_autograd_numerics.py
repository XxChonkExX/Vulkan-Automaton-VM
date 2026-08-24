# test_autograd_numerics.py - Numerical validation of the VulkanVM custom
# autograd ops against pure-ATen autograd references (audit item M4).
#
# For every op: forward outputs AND input/weight gradients are compared to a
# reference built only from standard torch ops + torch.autograd, reporting
# max-abs / mean-abs / relative error. Dtype matrix: fp32 (tight), bf16, fp16
# (scaled tolerances). Attention covers causal + non-causal masks, non-
# contiguous inputs, and recompute-on/off equivalence.
#
# Run on a machine with torch (X2 / Strix Halo):
#   python -m pytest python/vulkanvm_torch/test_autograd_numerics.py -v
#
# The suite skips itself cleanly when the vulkanvm_torch extension is absent.

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import vulkanvm_torch as vvm
    HAVE_EXT = hasattr(vvm, "vulkan_linear")
except ImportError:
    HAVE_EXT = False

pytestmark = pytest.mark.skipif(not HAVE_EXT, reason="vulkanvm_torch extension not built")

DEVICES = ["cpu"]  # the custom ops execute through ATen/Vulkan on the pool device
DTYPES = {
    "fp32": (torch.float32, 5e-5, 5e-4),    # (dtype, fwd_tol, grad_tol) max-abs
    "bf16": (torch.bfloat16, 2e-2, 5e-2),
    "fp16": (torch.float16, 1e-2, 3e-2),
}


def errs(got, ref):
    """(max_abs, mean_abs, rel_l2) between two tensors."""
    g = got.to(torch.float32)
    r = ref.to(torch.float32)
    diff = (g - r).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    denom = r.abs().clamp_min(1e-6)
    rel = ((diff / denom).pow(2).sum().sqrt() / math.sqrt(max(denom.numel(), 1))).item()
    return max_abs, mean_abs, rel


def check_match(got, ref, fwd_tol, label):
    max_abs, mean_abs, rel = errs(got, ref)
    assert max_abs <= fwd_tol, (
        f"{label}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel={rel:.3e}")


def make_inputs(shape, dtype, seed, non_contiguous=False):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(shape, generator=g, dtype=torch.float32).to(dtype)
    if non_contiguous:
        wide = torch.randn([s * 2 if i == 0 else s for i, s in enumerate(shape)],
                           generator=g, dtype=torch.float32).to(dtype)
        t = wide[::2]
    return t


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
# Generic reference harness: run the SAME op graph in pure ATen autograd
# ---------------------------------------------------------------------------

def aten_linear(x, w, b):
    return torch.nn.functional.linear(x, w, b)


def aten_gelu(x):
    return torch.nn.functional.gelu(x)


def aten_relu(x):
    return torch.relu(x)


def aten_silu(x):
    return torch.nn.functional.silu(x)


def aten_layernorm(x, w, b):
    return torch.nn.functional.layer_norm(x, x.shape[-1:], w, b)


def aten_softmax(x):
    return torch.softmax(x, -1)


def aten_attention(q, k, v, scale, mask):
    # Standard scaled-dot-product with additive/boolean mask, fp32 math.
    q32, k32, v32 = q.float(), k.float(), v.float()
    scores = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores + mask.to(torch.float32)
    p = torch.softmax(scores, -1)
    return torch.matmul(p, v32)


def aten_lora_linear(x, w, a, b_, scale):
    return torch.nn.functional.linear(x, w) + scale * (x @ a.T @ b_.T)


def aten_cross_entropy(logits, targets):
    return torch.nn.functional.cross_entropy(logits.float(), targets)


# ---------------------------------------------------------------------------
# Op-parametrized forward+backward comparisons
# ---------------------------------------------------------------------------

CASES = {
    "linear": (
        lambda vvm, t: vvm.vulkan_linear(t["x"], t["w"], t["b"]),
        lambda t: aten_linear(t["x"], t["w"], t["b"]),
        ["x", "w", "b"],
    ),
    "gelu": (
        lambda vvm, t: vvm.vulkan_gelu(t["x"]),
        lambda t: aten_gelu(t["x"]),
        ["x"],
    ),
    "relu": (
        lambda vvm, t: vvm.vulkan_relu(t["x"]),
        lambda t: aten_relu(t["x"]),
        ["x"],
    ),
    "silu": (
        lambda vvm, t: vvm.vulkan_silu(t["x"]),
        lambda t: aten_silu(t["x"]),
        ["x"],
    ),
    "layernorm": (
        lambda vvm, t: vvm.vulkan_layernorm(t["x"], t["w"], t["b"]),
        lambda t: aten_layernorm(t["x"], t["w"], t["b"]),
        ["x", "w", "b"],
    ),
    "softmax": (
        lambda vvm, t: vvm.vulkan_softmax(t["x"]),
        lambda t: aten_softmax(t["x"]),
        ["x"],
    ),
    "attention_causal": (
        lambda vvm, t: vvm.vulkan_attention(t["q"], t["k"], t["v"], t["scale"], t["mask"]),
        lambda t: aten_attention(t["q"], t["k"], t["v"], t["scale"], t["mask"]),
        ["q", "k", "v"],
    ),
    "attention_nomask": (
        lambda vvm, t: vvm.vulkan_attention(t["q"], t["k"], t["v"], t["scale"]),
        lambda t: aten_attention(t["q"], t["k"], t["v"], t["scale"], None),
        ["q", "k", "v"],
    ),
    "lora_linear": (
        lambda vvm, t: vvm.vulkan_lora_linear(t["x"], t["w"], t["a"], t["b"], t["scale"]),
        lambda t: aten_lora_linear(t["x"], t["w"], t["a"], t["b"], t["scale"]),
        ["x", "w", "a", "b"],
    ),
    "cross_entropy": (
        lambda vvm, t: vvm.vulkan_cross_entropy(t["logits"], t["targets"]),
        lambda t: aten_cross_entropy(t["logits"], t["targets"]),
        ["logits"],
    ),
}


def build_inputs(case, dtype, seed):
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, requires_grad=True):
        t = torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype)
        t.requires_grad_(requires_grad)
        return t

    t = {}
    if case == "linear":
        t["x"], t["w"], t["b"] = rnd(8, 32), rnd(48, 32), rnd(48)
    elif case in ("gelu", "relu", "silu", "softmax"):
        t["x"] = rnd(8, 64)
    elif case == "layernorm":
        t["x"], t["w"], t["b"] = rnd(8, 64), rnd(64), rnd(64)
    elif case in ("attention_causal", "attention_nomask"):
        t["q"], t["k"], t["v"] = rnd(2, 4, 16, 32), rnd(2, 4, 16, 32), rnd(2, 4, 16, 32)
        t["scale"] = 1.0 / math.sqrt(32)
        if case == "attention_causal":
            t["mask"] = torch.triu(
                torch.ones(16, 16, dtype=torch.bool), diagonal=1)
    elif case == "lora_linear":
        t["x"], t["w"] = rnd(8, 32), rnd(48, 32)
        t["a"], t["b"] = rnd(16, 32), rnd(48, 16)
        t["scale"] = 0.5
    elif case == "cross_entropy":
        t["logits"] = rnd(8, 32)
        t["targets"] = torch.randint(0, 32, (8,), generator=g)
    return t


@pytest.mark.parametrize("case", sorted(CASES.keys()))
@pytest.mark.parametrize("dtype_name", DTYPES)
def test_op_forward_backward(case, dtype_name):
    dtype, ftol, gtol = DTYPES[dtype_name]
    vvm_fn, aten_fn, grad_keys = CASES[case]

    t_v = build_inputs(case, dtype, seed=42)
    t_r = build_inputs(case, dtype, seed=42)
    # Reference always computes in fp32 for a stable ground truth.
    for v in t_r.values():
        if torch.is_tensor(v) and v.is_floating_point():
            v.data = v.data.float()
            v.requires_grad_(True)

    out_v = vvm_fn(vvm, t_v)
    out_r = aten_fn(t_r)
    check_match(out_v, out_r, ftol, f"{case} fwd {dtype_name}")

    # Backward with a fixed upstream gradient.
    grad = torch.ones_like(out_r.to(torch.float32))
    out_v.backward(grad.to(out_v.dtype))
    out_r.backward(grad)

    for key in grad_keys:
        if key not in t_v or t_v[key].grad is None:
            continue
        check_match(t_v[key].grad, t_r[key].grad, gtol,
                    f"{case} d({key}) {dtype_name}")


# ---------------------------------------------------------------------------
# Attention specifics (audit M4 checklist)
# ---------------------------------------------------------------------------

def test_attention_recompute_equivalence():
    """recompute=True must be numerically identical to recompute=False."""
    t = build_inputs("attention_causal", torch.float32, seed=7)
    scale = t["scale"]

    vvm.set_attention_recompute(False)
    out_save = vvm.vulkan_attention(t["q"], t["k"], t["v"], scale, t["mask"])
    out_save.backward(torch.ones_like(out_save))
    grads_save = [t["q"].grad.clone(), t["k"].grad.clone(), t["v"].grad.clone()]

    t2 = build_inputs("attention_causal", torch.float32, seed=7)
    vvm.set_attention_recompute(True)
    out_re = vvm.vulkan_attention(t2["q"], t2["k"], t2["v"], scale, t2["mask"])
    out_re.backward(torch.ones_like(out_re))

    check_match(out_re, out_save, 0.0, "recompute fwd")
    for g_new, g_old, name in zip(
            [t2["q"].grad, t2["k"].grad, t2["v"].grad], grads_save, "qkv"):
        check_match(g_new, g_old, 0.0, f"recompute d({name})")
    vvm.set_attention_recompute(False)


def test_attention_non_contiguous():
    t_c = build_inputs("attention_causal", torch.float32, seed=11)
    out_c = vvm.vulkan_attention(t_c["q"], t_c["k"], t_c["v"], t_c["scale"], t_c["mask"])

    # Non-contiguous variants (strided views).
    q_w = torch.randn(2, 4, 16, 64)[..., ::2]
    k_w = torch.randn(2, 4, 16, 64)[..., ::2]
    v_w = torch.randn(2, 4, 16, 64)[..., ::2]
    for t_ in (q_w, k_w, v_w):
        t_.requires_grad_(True)
    out_w = vvm.vulkan_attention(q_w, k_w, v_w, t_c["scale"], t_c["mask"])
    check_match(out_w, out_c, 1e-5, "attention non-contiguous fwd")


def test_attention_gradient_error_metrics():
    """Report (don't assert) the audit's metric set for fp32 attention."""
    t_v = build_inputs("attention_causal", torch.float32, seed=13)
    t_r = build_inputs("attention_causal", torch.float32, seed=13)
    out_v = vvm.vulkan_attention(t_v["q"], t_v["k"], t_v["v"], t_v["scale"], t_v["mask"])
    out_r = aten_attention(t_r["q"], t_r["k"], t_r["v"], t_r["scale"], t_r["mask"])
    grad = torch.randn_like(out_r)
    out_v.backward(grad)
    out_r.backward(grad)
    for key in ("q", "k", "v"):
        max_abs, mean_abs, rel = errs(t_v[key].grad, t_r[key].grad)
        # fp32 attention gradients: expect tight agreement.
        assert max_abs < 1e-4, f"d({key}) max_abs={max_abs:.3e}"
        print(f"[metrics] d({key}): max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel={rel:.3e}")


def test_cross_entropy_targets_boundary():
    """Target indices at 0 and n_classes-1 (boundary rows of the logit matrix)."""
    t = build_inputs("cross_entropy", torch.float32, seed=17)
    t["targets"] = torch.tensor([0, 31, 0, 31, 5, 17, 31, 0])
    t_r = {k: (v.clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v.clone())
           for k, v in t.items()}
    out_v = vvm.vulkan_cross_entropy(t["logits"], t["targets"])
    out_r = aten_cross_entropy(t_r["logits"], t_r["targets"])
    check_match(out_v, out_r, 5e-5, "cross_entropy boundary fwd")
    out_v.backward()
    out_r.backward()
    check_match(t["logits"].grad, t_r["logits"].grad, 5e-4, "cross_entropy boundary dlogits")

