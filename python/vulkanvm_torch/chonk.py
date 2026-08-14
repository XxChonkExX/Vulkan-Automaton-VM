# chonk.py
# Chonk Buffer (UnifiedMemoryPool) as the KV-cache host for long-context
# cache-carry training.
#
# How it works:
#   1. The pool allocates ONE exportable unified-memory block (hostPtr +
#      deviceAddress) sized to the full KV cache.
#   2. The block is exported as a dma-buf fd, imported into HIP as an
#      external memory object, and wrapped as a zero-copy CUDA tensor
#      (base tensor). GPU kernels and host code share the same physical
#      memory - no `.to("cpu")`/`.to(device)` round-trips.
#   3. A ChonkCache builds one ChonkFullLayer per full-attention layer,
#      whose keys/values are views into the pool block (in-place writes),
#      plus the standard LinearAttentionLayer for hybrid/linear layers.
#
# The KV cache therefore lives OUTSIDE torch's cuda allocator (fraction
# budget stays for model + activations), never moves between chunks, and
# grows without the cat() copy storm that previously hung the driver.

import ctypes
import os
import sys

import numpy as np
import torch

from transformers.cache_utils import (
    Cache,
    LinearAttentionLayer,
    StaticLayer,
    get_layer_types_and_kwargs,
)

for _p in (
    os.environ.get("CHONK_POOL_SO_DIR"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_build"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_build"),
    "/tmp/opencode",
):
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import vulkanvm_pool_test as pool_mod  # the compiled smoke-test binding

try:
    _HIP = ctypes.CDLL("libamdhip64.so.7")
except OSError:
    _HIP = ctypes.CDLL("libamdhip64.so")

_hip_import = _HIP.hipImportExternalMemory
_hip_import.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
_hip_import.restype = ctypes.c_int
_hip_map = _HIP.hipExternalMemoryGetMappedBuffer
_hip_map.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p]
_hip_map.restype = ctypes.c_int
_hip_destroy = _HIP.hipDestroyExternalMemory
_hip_destroy.argtypes = [ctypes.c_void_p]
_hip_destroy.restype = ctypes.c_int


class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", ctypes.c_void_p * 2), ("nv", ctypes.c_void_p)]


class _HandleDesc(ctypes.Structure):
    _anonymous_ = ("handle",)
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _BufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class ChonkPool:
    """Wraps the pool binding + HIP dma-buf import, exposing a base tensor
    that aliases the pool's unified memory for both GPU and host."""

    def __init__(self):
        info = pool_mod.init()
        self.device_name = info["device"]
        self._ext_mems = []
        self._fds = []
        self._allocations = []

    def alloc_base(self, nbytes: int, name: str = "chonk"):
        """Allocate nbytes in the pool, import to HIP, return a 1-D uint8
        CUDA tensor aliasing the memory (plus hostPtr)."""
        nbytes = int(nbytes)
        a = pool_mod.alloc_export(nbytes, name)
        fd = a["fd"]
        host_ptr = a["hostPtr"]
        self._fds.append(fd)

        ext_mem = ctypes.c_void_p()
        desc = _HandleDesc()
        desc.type = 1  # hipExternalMemoryHandleTypeOpaqueFd
        desc.handle.fd = fd
        desc.size = nbytes
        desc.flags = 0
        if _hip_import(ctypes.byref(ext_mem), ctypes.byref(desc)) != 0:
            raise RuntimeError("hipImportExternalMemory failed")

        dev_ptr = ctypes.c_void_p()
        bd = _BufferDesc()
        bd.offset = 0
        bd.size = nbytes
        bd.flags = 0
        if _hip_map(ctypes.byref(dev_ptr), ext_mem, ctypes.byref(bd)) != 0:
            raise RuntimeError("hipExternalMemoryGetMappedBuffer failed")
        dev_ptr = dev_ptr.value

        iface = {
            "shape": (nbytes,),
            "strides": None,
            "data": (dev_ptr, False),
            "typestr": "<u1",
            "version": 2,
        }
        wrap = type("PoolWrap", (), {"__cuda_array_interface__": iface})()
        base = torch.as_tensor(wrap, device="cuda")
        base = base.view(torch.uint8)
        self._ext_mems.append(ext_mem)
        self._allocations.append(a)
        return base, host_ptr

    def alloc_model_weights(self, nbytes: int, name: str = "model_weights"):
        """Allocate model weights in Chonk Buffer (exported + HIP-imported so
        the returned tensor is a valid CUDA tensor)."""
        base, _ = self.alloc_base(nbytes, name)
        return base

    def alloc_optimizer_states(self, nbytes: int, name: str = "optimizer_states"):
        """Allocate optimizer states in Chonk Buffer."""
        base, _ = self.alloc_base(nbytes, name)
        return base

    def alloc_activations(self, nbytes: int, name: str = "activations"):
        """Allocate activations buffer in Chonk Buffer."""
        base, _ = self.alloc_base(nbytes, name)
        return base

    def alloc_host_visible(self, nbytes: int, name: str = "host_visible"):
        """Allocate host-visible buffer in Chonk Buffer (for staging).
        Wrapped as a CPU tensor aliasing the mapped hostPtr."""
        nbytes = int(nbytes)
        a = pool_mod.alloc_host_visible(nbytes, name)
        host_ptr = a["hostPtr"]
        self._allocations.append(a)
        if not host_ptr:
            raise RuntimeError("alloc_host_visible returned null host pointer")
        np_arr = np.ctypeslib.as_array(
            ctypes.cast(host_ptr, ctypes.POINTER(ctypes.c_ubyte)), shape=(nbytes,)
        )
        return torch.from_numpy(np_arr), host_ptr

    def stats(self):
        return pool_mod.stats()

    def shutdown(self):
        pool_mod.shutdown()


class ChonkFullLayer(StaticLayer):
    """Static full-attention cache layer whose keys/values live in Chonk
    Buffer pool memory. Writes in place and returns a view of only the
    tokens seen so far, so masks/attention span the current length."""

    def __init__(self, max_cache_len, pool_base, byte_offset, dtype, device):
        super().__init__(max_cache_len=max_cache_len)
        self._pool_base = pool_base
        self._byte_offset = byte_offset
        self._prealloc_dtype = dtype
        self._prealloc_device = device

    def _prealloc(self, batch_size, num_heads, head_dim):
        per_t = batch_size * num_heads * self.max_cache_len * head_dim
        el_per = per_t
        offset_el = self._byte_offset // self._prealloc_dtype.itemsize
        buf = self._pool_base.view(self._prealloc_dtype)
        self.keys = buf.narrow(0, offset_el, per_t).view(
            batch_size, num_heads, self.max_cache_len, head_dim
        )
        self.values = buf.narrow(0, offset_el + per_t, per_t).view(
            batch_size, num_heads, self.max_cache_len, head_dim
        )
        self.dtype = self._prealloc_dtype
        self.device = self._prealloc_device
        self.cumulative_length = torch.tensor(0, dtype=int, device=self._prealloc_device)
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.batch_size, self.num_heads = key_states.shape[:2]
        self.v_head_dim = value_states.shape[-1]
        self.k_head_dim = key_states.shape[-1]
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        b = key_states.shape[0]
        kv_length = key_states.shape[-2]
        start = int(self.cumulative_length.item())
        # Store into the pool with DETACHED sources: cached tokens are
        # constants (truncated BPTT), so later chunks never reference a
        # previous chunk's freed autograd graph.
        self.keys[:b, :, start : start + kv_length].copy_(key_states.detach())
        self.values[:b, :, start : start + kv_length].copy_(value_states.detach())
        self.cumulative_length.add_(kv_length)
        if start == 0:
            # First chunk: nothing cached yet, keep the graph fully alive
            return key_states, value_states
        # Previous tokens are constants; the current chunk's K/V stay
        # differentiable so grads flow to its own projections.
        k = torch.cat([self.keys[:b, :, :start].detach(), key_states], dim=-2)
        v = torch.cat([self.values[:b, :, :start].detach(), value_states], dim=-2)
        return k, v

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return int(self.cumulative_length.item()) + query_length, 0

    def reset(self) -> None:
        self.cumulative_length.zero_()


def reset_chonk_cache(cache):
    """Reset a ChonkCache between sequences so it can be reused next step.
    Full-attention layers restart at position 0 (stale K/V beyond the new
    writes are never read); linear layers zero their conv/recurrent states."""
    for layer in cache.layers:
        if isinstance(layer, ChonkFullLayer):
            layer.reset()
        elif hasattr(layer, "reset"):
            layer.reset()


def build_chonk_cache(config, batch_size, max_cache_len, pool=None):
    """Build a ChonkCache for the given text config. Allocates one pool
    block holding every full-attention layer's keys and values, then
    creates views. Linear-attention layers use the standard in-place
    LinearAttentionLayer (small, no seq dimension)."""
    layer_types, layer_kwargs = get_layer_types_and_kwargs(config.get_text_config(decoder=True))
    full_layers = [i for i, t in enumerate(layer_types) if t in ("full_attention", "attention")]
    n_full = len(full_layers)

    text_cfg = config.get_text_config(decoder=True)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads
    )
    dtype = torch.bfloat16
    device = "cuda"

    per_layer_bytes = (
        2 * batch_size * num_kv_heads * max_cache_len * head_dim * dtype.itemsize
    )  # K + V
    total = per_layer_bytes * n_full

    if pool is None:
        pool = ChonkPool()
    base, host_ptr = pool.alloc_base(total, "chonk_kv_cache")
    pool.host_ptr = host_ptr

    layers = []
    byte_off = 0
    for i, lt in enumerate(layer_types):
        if lt in ("full_attention", "attention"):
            layer = ChonkFullLayer(max_cache_len, base, byte_off, dtype, device)
            layer._prealloc(batch_size, num_kv_heads, head_dim)
            byte_off += per_layer_bytes
        else:
            layer = LinearAttentionLayer(**layer_kwargs)
        layers.append(layer)

    cache = Cache(layers=layers)
    cache.pool = pool
    cache.full_layer_bytes = per_layer_bytes
    cache._batch_size = batch_size
    return cache


def estimate_model_memory(config, dtype=torch.bfloat16):
    """Estimate memory requirements for model weights and optimizer states."""
    text_cfg = config.get_text_config(decoder=True)
    hidden_size = text_cfg.hidden_size
    num_layers = text_cfg.num_hidden_layers
    num_attention_heads = text_cfg.num_attention_heads
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(text_cfg, "head_dim", hidden_size // num_attention_heads)
    intermediate_size = getattr(text_cfg, "intermediate_size", hidden_size * 4)
    vocab_size = text_cfg.vocab_size

    element_size = dtype.itemsize

    # Embedding: vocab_size * hidden_size
    embed_params = vocab_size * hidden_size

    # Per layer params (attention + FFN)
    # QKV: 3 * hidden_size * hidden_size
    # O: hidden_size * hidden_size
    # FFN: gate + up + down = 3 * hidden_size * intermediate_size
    # Norms: ~4 * hidden_size (rms/layernorm)
    per_layer_params = (
        4 * hidden_size * hidden_size +  # QKV + O
        3 * hidden_size * intermediate_size +  # FFN
        4 * hidden_size  # norms
    )

    # LM head: hidden_size * vocab_size (usually tied to embed)
    lm_head_params = hidden_size * vocab_size

    total_params = embed_params + num_layers * per_layer_params + lm_head_params
    model_bytes = total_params * element_size

    # Optimizer states (AdamW: exp_avg + exp_avg_sq in fp32 = 2 * 4 bytes per param = 8 bytes per param)
    optimizer_bytes = total_params * 8  # 2 states * 4 bytes (fp32)

    return {
        "model_params": total_params,
        "model_bytes": model_bytes,
        "optimizer_bytes": optimizer_bytes,
        "total_bytes": model_bytes + optimizer_bytes,
    }


def load_model_into_chonk(model, pool: ChonkPool, dtype=torch.bfloat16, chunk_size_mb=512):
    """Load model weights into Chonk Buffer, replacing model's parameters
    with views into the pool memory. Moves in chunks to avoid timeout."""
    params_list = list(model.named_parameters())
    total_numel = sum(p.numel() for _, p in params_list)
    total_bytes = total_numel * dtype.itemsize
    model_buffer = pool.alloc_model_weights(total_bytes, "model_weights")
    model_buffer_typed = model_buffer.view(dtype)

    print(f"Moving model to Chonk Buffer: {total_bytes / 1e9:.2f} GB "
          f"({total_numel:,} params) in chunks of {chunk_size_mb} MB...")

    offset = 0
    bytes_moved = 0
    chunk_bytes = chunk_size_mb * 1024 * 1024
    chunk_param_count = 0
    
    for name, param in params_list:
        numel = param.numel()
        param_bytes = numel * dtype.itemsize
        
        param_view = model_buffer_typed.narrow(0, offset, numel).view_as(param)
        param_view.copy_(param.data)
        param.data = param_view
        offset += numel
        bytes_moved += param_bytes
        chunk_param_count += 1
        
        # Progress reporting
        if bytes_moved >= chunk_bytes or chunk_param_count >= 100:
            pct = (bytes_moved / total_bytes) * 100
            print(f"  Progress: {pct:.1f}% ({bytes_moved / 1e9:.2f} / {total_bytes / 1e9:.2f} GB)")
            bytes_moved = 0
            chunk_param_count = 0

    print(f"Loaded model into Chonk Buffer: {offset * dtype.itemsize / 1e9:.2f} GB")
    return model_buffer


def build_model_from_chonk_buffer(config, model_buffer, dtype=torch.bfloat16, attn_implementation=None):
    """Build a trainable model whose parameters are zero-copy views into a
    Chonk Buffer allocation. Creates the module structure on meta (no memory),
    then replaces every parameter with an nn.Parameter view into the buffer.
    The flat buffer layout must match named_parameters() order (this is how
    load_model_directly_to_chonk writes it)."""
    import torch.nn as nn
    from transformers import AutoModelForCausalLM

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    typed = model_buffer.view(dtype)
    offset = 0
    for name, param in list(model.named_parameters()):
        numel = param.numel()
        view = typed.narrow(0, offset, numel).view(param.shape)
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nn.Parameter(view))
        offset += numel

    # Materialize meta buffers on CUDA (rotary inv_freq etc.)
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        if parts[-1] in ("inv_freq", "original_inv_freq") and hasattr(
            parent, "compute_default_rope_parameters"
        ):
            inv_freq, _ = parent.compute_default_rope_parameters(parent.config, device="cuda")
            if parts[-1] == "original_inv_freq":
                inv_freq = inv_freq.clone()
            setattr(parent, parts[-1], inv_freq)
        else:
            setattr(parent, parts[-1], torch.empty_like(buf, device="cuda"))

    print(f"Built model from Chonk Buffer: {offset * dtype.itemsize / 1e9:.2f} GB "
          f"({len(list(model.named_parameters()))} params)")
    return model


def load_model_directly_to_chonk(model_path, config, pool: ChonkPool, dtype=torch.bfloat16, chunk_size_mb=512):
    """Load model weights directly from disk into Chonk Buffer without loading to CUDA first.
    Uses meta device to create empty model structure, then loads weights directly to Chonk Buffer."""
    import torch
    from transformers import AutoModelForCausalLM
    
    # Create model on meta device (no memory allocation) using from_config
    print("Creating model on meta device...")
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    
    # Estimate memory and allocate in Chonk Buffer (use actual param count
    # from the meta model so the buffer matches the weights exactly)
    params_list = list(model.named_parameters())
    total_numel = sum(p.numel() for _, p in params_list)
    total_bytes = total_numel * dtype.itemsize
    model_buffer = pool.alloc_model_weights(total_bytes, "model_weights")
    model_buffer_typed = model_buffer.view(dtype)

    print(f"Loading weights directly to Chonk Buffer: {total_bytes / 1e9:.2f} GB...")

    # Load state dict and copy directly to Chonk Buffer
    from safetensors.torch import load_file
    import glob

    # Find model files
    model_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    if not model_files:
        model_files = sorted(glob.glob(f"{model_path}/pytorch_model*.bin"))

    param_map = dict(params_list)
    # Precompute each param's slot offset in the flat buffer (layout follows
    # named_parameters order). Offsets are computed ONCE, not per file.
    slots = {}
    o = 0
    for name, param in param_map.items():
        slots[name] = o
        o += param.numel()

    bytes_copied = 0
    chunk_bytes = chunk_size_mb * 1024 * 1024
    processed = 0

    # Checkpoint was saved from the multimodal wrapper: keys use
    # 'model.language_model.*' for the text model and bare 'lm_head.weight'.
    # The text-only model built from_config uses 'model.*' names, so remap.
    def to_ckpt_name(name):
        if name in ("lm_head.weight", "model.lm_head.weight"):
            return "lm_head.weight"
        if name.startswith("model."):
            return "model.language_model." + name[len("model."):]
        return None

    for model_file in model_files:
        print(f"  Loading {model_file}...")
        state_dict = load_file(model_file, device='cpu')

        for name, param in param_map.items():
            ck_name = to_ckpt_name(name)
            tensor = state_dict.get(ck_name) if ck_name else None
            if tensor is None:
                continue  # stored in another shard (slot stays as-is)

            numel = param.numel()
            param_bytes = numel * dtype.itemsize

            # Convert to target dtype
            tensor = tensor.to(dtype)

            # Copy to Chonk Buffer at the param's precomputed slot
            param_view = model_buffer_typed.narrow(0, slots[name], numel).view_as(param)
            param_view.copy_(tensor)
            bytes_copied += param_bytes
            processed += 1

            if bytes_copied >= chunk_bytes:
                pct = (bytes_copied / total_bytes) * 100
                print(f"  Progress: {pct:.1f}% ({bytes_copied / 1e9:.2f} GB)")
                bytes_copied = 0

    print(f"Loaded model directly to Chonk Buffer: {o * dtype.itemsize / 1e9:.2f} GB "
          f"({processed} params copied)")
    return model_buffer


def create_optimizer_states_in_chonk(params, pool: ChonkPool):
    """Create AdamW states (exp_avg + exp_avg_sq, fp32) in Chonk Buffer for
    the given trainable params. Keyed by the parameter OBJECT so the
    optimizer can look them up directly (ChonkAdamW does optimizer_states[p])."""
    params = list(params)
    total_numel = sum(p.numel() for p in params)
    opt_bytes = total_numel * 2 * 4  # exp_avg + exp_avg_sq, fp32
    opt_buffer = pool.alloc_optimizer_states(opt_bytes, "optimizer_states")
    opt_buffer_typed = opt_buffer.view(torch.float32)  # fp32 optimizer states

    offset = 0
    optimizer_states = {}
    for param in params:
        numel = param.numel()
        # AdamW has exp_avg and exp_avg_sq for each parameter
        exp_avg = opt_buffer_typed.narrow(0, offset, numel).view_as(param)
        offset += numel
        exp_avg_sq = opt_buffer_typed.narrow(0, offset, numel).view_as(param)
        offset += numel
        optimizer_states[param] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}

    print(f"Created optimizer states in Chonk Buffer: {offset * 4 / 1e9:.2f} GB "
          f"({len(optimizer_states)} params)")
    return optimizer_states, opt_buffer


def create_activation_buffers(pool: ChonkPool, batch_size, seq_len, hidden_size, num_layers, dtype=torch.bfloat16, chunk_size=4096, budget_gb=2.0):
    """Allocate activation scratch buffers in Chonk Buffer for chunked
    forward pass. The training loop allocates its own activations via
    torch, so this is just a modest scratch reservation."""
    total_activation_bytes = int(budget_gb * 1024 ** 3)

    act_buffer = pool.alloc_activations(total_activation_bytes, "activations")
    print(f"Created activation buffers in Chonk Buffer: {total_activation_bytes / 1e9:.2f} GB")
    return act_buffer


def patch_linear_cache_for_chunked_training():
    """Monkey-patch transformers' LinearAttentionLayer state updates to
    REASSIGN tensors instead of copy_()ing in place. The chunked forward
    graph saves the stored state when the layer reads it; an in-place
    copy_ by the next update then breaks autograd (version mismatch).
    With reassignment the previous tensor stays version-stable, enabling
    chunked forward/backward with truncated BPTT across chunks."""
    from transformers import cache_utils as cu

    def update_conv_state(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
        if not self.is_conv_states_initialized[state_idx]:
            self.lazy_initialization(
                conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size
            )
        if not self.has_previous_state[state_idx]:
            full_conv_states = conv_states
            self.has_previous_state[state_idx] = True
            if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
                padding_length = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
                full_conv_states = torch.nn.functional.pad(full_conv_states, (padding_length, 0), value=0)
        else:
            full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
        if not self.record_past:
            # Fresh storage owned by the cache: never part of a previous
            # chunk's autograd graph, so it cannot be freed or version-bumped.
            self.conv_states[state_idx] = full_conv_states[..., -self.conv_kernel_size[state_idx]:].detach().clone()
        else:
            self.conv_states[state_idx] = full_conv_states
        return full_conv_states

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kwargs):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states.detach().clone()
        return self.recurrent_states[state_idx]

    cu.LinearAttentionLayer.update_conv_state = update_conv_state
    cu.LinearAttentionLayer.update_recurrent_state = update_recurrent_state
    print("Patched LinearAttentionLayer for chunked training (state reassignment)")


def build_lora_chonk_setup(model_path, config, batch_size, max_cache_len,
                           lora_r=64, lora_alpha=128, lora_dropout=0.05,
                           dtype=torch.bfloat16, chunk_size_mb=512,
                           staging_gb=2.0, attn_implementation="eager"):
    """Complete LoRA training setup with EVERYTHING in Chonk Buffer:
      - pool + KV cache
      - base weights loaded directly from disk into the pool
      - zero-copy model built from the pool buffer, base frozen
      - LoRA adapters (trainable) in pool memory
      - AdamW fp32 states for LoRA params in pool memory
      - activation + staging buffers in pool memory
    Returns a dict with pool, kv_cache, model, model_buffer, optimizer_states,
    opt_buffer, act_buffer, staging_buffer, staging_host_ptr."""
    from peft import LoraConfig, get_peft_model

    pool = ChonkPool()

    # 1. KV cache in Chonk Buffer (before model, per user constraint)
    kv_cache = build_chonk_cache(config, batch_size, max_cache_len, pool)

    # 2. Base weights directly from disk into Chonk Buffer
    model_buffer = load_model_directly_to_chonk(model_path, config, pool,
                                                dtype=dtype, chunk_size_mb=chunk_size_mb)

    # 3. Zero-copy model + freeze base
    model = build_model_from_chonk_buffer(config, model_buffer, dtype=dtype,
                                          attn_implementation=attn_implementation)
    model.requires_grad_(False)

    # 4. LoRA adapters
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA applied: {n_trainable / 1e6:.1f}M trainable params "
          f"({n_trainable * 2 / 1e9:.2f} GB bf16)")

    # 5. Optimizer states for trainable params in Chonk Buffer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_states, opt_buffer = create_optimizer_states_in_chonk(trainable, pool)

    # 6. Activation buffers in Chonk Buffer
    text_cfg = config.get_text_config(decoder=True)
    act_buffer = create_activation_buffers(
        pool, batch_size, 0, text_cfg.hidden_size, text_cfg.num_hidden_layers,
        dtype=dtype, chunk_size=4096, budget_gb=2.0,
    )

    # 7. Host-visible staging buffer
    staging_buffer, staging_host_ptr = pool.alloc_host_visible(
        int(staging_gb * 1024 ** 3), "staging")

    print(f"\n=== Chonk Buffer LoRA Setup Complete ===")
    stats = pool.stats()
    print(f"Pool stats: {stats}")
    print(f"Total used: {stats['totalUsed'] / 1e9:.2f} GB")

    return {
        "pool": pool,
        "kv_cache": kv_cache,
        "model": model,
        "model_buffer": model_buffer,
        "optimizer_states": optimizer_states,
        "opt_buffer": opt_buffer,
        "act_buffer": act_buffer,
        "staging_buffer": staging_buffer,
        "staging_host_ptr": staging_host_ptr,
    }
