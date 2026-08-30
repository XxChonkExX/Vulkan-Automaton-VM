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
import math
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


def install_chonk_allocator():
    """Replace torch's HIP caching allocator with the pool-backed allocator.

    Every segment torch allocates is carved from a Chonk Buffer block
    (Vulkan dma-buf -> hipImportExternalMemory) and freed back into the
    pool, keeping one allocator family over the unified heap. Must be
    called before any CUDA tensor exists.

    Order matters: the pool must exist and the HIP context must be
    initialized BEFORE torch's allocator is swapped in, otherwise the
    first allocation re-enters HIP context init from inside the
    allocator (deadlock/hang).
    """
    try:
        pool_mod.init()
    except RuntimeError as e:
        if "already initialized" not in str(e):
            raise
    _hip_malloc = _HIP.hipMalloc
    _hip_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    _hip_malloc.restype = ctypes.c_int
    _hip_free = _HIP.hipFree
    _hip_free.argtypes = [ctypes.c_void_p]
    _hip_free.restype = ctypes.c_int
    probe = ctypes.c_void_p()
    if _hip_malloc(ctypes.byref(probe), 4096) != 0:
        raise RuntimeError("hipMalloc probe failed (HIP context init)")
    _hip_free(probe)

    from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator

    allocator = CUDAPluggableAllocator(
        pool_mod.__file__,
        "chonk_allocator_alloc",
        "chonk_allocator_free",
    )
    change_current_allocator(allocator)


class ChunkedPoolBuffer:
    """A logical contiguous byte region backed by multiple smaller exportable
    pool blocks. Present narrow(start, len) as a single contiguous tensor so
    callers (quant buffer writes, model buffer) work unchanged. Within a block
    it is a zero-copy view; across a block boundary it copies (rare for the
    per-module quant writes, which fit in one ~1GB block)."""

    def __init__(self, blocks, block_bytes, total_bytes, name):
        self.blocks = blocks          # list of uint8 CUDA tensors
        self.block_bytes = block_bytes
        self.total_bytes = total_bytes
        self.name = name
        # Each block's byte-offset base in the logical range.
        self._starts = []
        off = 0
        for b in blocks:
            self._starts.append(off)
            off += b.shape[0]

    @classmethod
    def from_single(cls, base_tensor, total_bytes, name):
        obj = cls([base_tensor], max(base_tensor.shape[0], 1), total_bytes, name)
        return obj

    def _locate(self, byte_off, length):
        """Return (tensor, in_block_off) for the block containing byte_off."""
        for i, b in enumerate(self.blocks):
            if byte_off < self._starts[i] + b.shape[0]:
                return b, byte_off - self._starts[i]
        raise IndexError(f"offset {byte_off} out of range")

    def narrow(self, start, length):
        """start/length in BYTES (the buffer is uint8). Returns a contiguous
        uint8 tensor of `length` bytes spanning the logical range."""
        if length == 0:
            return self.blocks[0].narrow(0, 0, 0)
        # Single-block case (common): zero-copy view.
        b, off = self._locate(start, length)
        if off + length <= b.shape[0]:
            return b.narrow(0, off, length)
        # Cross-block: copy into a fresh contiguous tensor.
        parts = []
        rem = length
        cur = start
        while rem > 0:
            blk, boff = self._locate(cur, rem)
            take = min(rem, blk.shape[0] - boff)
            parts.append(blk.narrow(0, boff, take))
            cur += take
            rem -= take
        out = torch.empty(length, dtype=torch.uint8, device=parts[0].device)
        o = 0
        for p in parts:
            out.narrow(0, o, p.shape[0]).copy_(p)
            o += p.shape[0]
        return out

    def view(self, dtype):
        # Not a true contiguous view across blocks; used mainly to get a typed
        # handle for narrow-based writes. Return self (caller uses narrow).
        return self

    def __len__(self):
        return self.total_bytes


class ChonkPool:
    """Wraps the pool binding + HIP dma-buf import, exposing a base tensor
    that aliases the pool's unified memory for both GPU and host."""

    def __init__(self):
        try:
            info = pool_mod.init()
        except RuntimeError as e:
            if "already initialized" not in str(e):
                raise
            # The pluggable allocator already created the pool on first
            # allocation; adopt the existing one.
            info = pool_mod.info()
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

    def alloc_chunked(self, nbytes: int, name: str = "chunked", max_block_mb: int = 1024):
        """Allocate a large region as MULTIPLE smaller exportable blocks (each
        <= max_block_mb) to avoid a single monolithic vkAllocateMemory that the
        radv driver rejects at scale (e.g. the 14.43 GB INT4 q buffer). Returns
        a ChunkedPoolBuffer that presents the region as one logical contiguous
        byte range but is backed by many small, driver-friendly blocks. This is
        the "smaller blocks / fine ladder" fix."""
        nbytes = int(nbytes)
        if nbytes <= max_block_mb * 1024 * 1024:
            # Small enough for a single block; behave like alloc_model_weights.
            base, _ = self.alloc_base(nbytes, name)
            return ChunkedPoolBuffer.from_single(base, nbytes, name)
        block_bytes = max_block_mb * 1024 * 1024
        n_blocks = (nbytes + block_bytes - 1) // block_bytes
        blocks = []
        remaining = nbytes
        for i in range(n_blocks):
            this = min(block_bytes, remaining)
            b, _ = self.alloc_base(this, f"{name}_block_{i}")
            blocks.append(b)
            remaining -= this
        return ChunkedPoolBuffer(blocks, block_bytes, nbytes, name)

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
    tokens seen so far, so masks/attention span the current length.
    Supports INT4 quantization (packed uint8) when enabled via CHONK_QUANTIZE_KV.
    In quantized mode, only INT4 data lives in the pool; bf16 tensors are
    materialized transiently during the forward pass."""

    def __init__(self, max_cache_len, pool_base, byte_offset, storage_dtype, compute_dtype, device, quantize_kv, num_kv_heads, head_dim):
        super().__init__(max_cache_len=max_cache_len)
        self._pool_base = pool_base
        self._byte_offset = byte_offset
        self._storage_dtype = storage_dtype
        self._compute_dtype = compute_dtype
        self._prealloc_device = device
        self._quantize = quantize_kv
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._kv_elements_per_layer = 2 * num_kv_heads * max_cache_len * head_dim

    def _prealloc(self, batch_size, num_heads, head_dim):
        if self._quantize:
            kv_elements = batch_size * num_heads * self.max_cache_len * head_dim
            elements_per_head = kv_elements // num_heads
            packed_head_dim = head_dim // 2
            bytes_per_head = elements_per_head // 2
            kv_bytes = int(kv_elements * 0.5)
            # Per-position per-head scales: [max_cache_len, num_heads] for K and V
            scale_per_pos_bytes = num_heads * self.max_cache_len * self._compute_dtype.itemsize
            scale_bytes = 2 * scale_per_pos_bytes
            # Layout: [k_data(kv_bytes)] [v_data(kv_bytes)] [k_scales] [v_scales]
            total_bytes = 2 * kv_bytes + scale_bytes

            buf = self._pool_base.view(torch.uint8)
            offset = self._byte_offset
            self._k_data = buf.narrow(0, offset, kv_bytes)
            self._v_data = buf.narrow(0, offset + kv_bytes, kv_bytes)
            scale_per_pos_bytes = num_heads * self.max_cache_len * self._compute_dtype.itemsize
            scale_offset = offset + 2 * kv_bytes
            self._k_scales = buf.narrow(0, scale_offset, scale_per_pos_bytes).view(
                self._compute_dtype
            ).view(self.max_cache_len, num_heads)
            self._v_scales = buf.narrow(0, scale_offset + scale_per_pos_bytes, scale_per_pos_bytes).view(
                self._compute_dtype
            ).view(self.max_cache_len, num_heads)

            self._keys = None
            self._values = None
            self._is_quantized = True
        else:
            per_t = batch_size * num_heads * self.max_cache_len * head_dim
            offset_el = self._byte_offset // self._storage_dtype.itemsize
            buf = self._pool_base.view(self._storage_dtype)
            self._keys = buf.narrow(0, offset_el, per_t).view(
                batch_size, num_heads, self.max_cache_len, head_dim
            )
            self._values = buf.narrow(0, offset_el + per_t, per_t).view(
                batch_size, num_heads, self.max_cache_len, head_dim
            )
            self._is_quantized = False

        self.dtype = self._compute_dtype
        self.device = self._prealloc_device
        # Create on CPU first to avoid ROCm lazy allocation issues
        self.cumulative_length = torch.tensor(0, dtype=torch.int32, device="cpu").to(self._prealloc_device)
        self.is_initialized = True

    @staticmethod
    def _dequantize_int4(packed, scales, start_pos, length, head_idx):
        """Dequantize INT4 (packed uint8) to bf16. Per-position per-head scales.
        If scale is 0 (empty marker), returns zeros for that position.
        Runs in no_grad - cached prefix is constant (truncated BPTT)."""
        with torch.no_grad():
            if packed.numel() == 0:
                return torch.empty(0, dtype=torch.bfloat16, device=packed.device)
            last_dim = packed.shape[-1]
            low = (packed & 0xF).to(torch.int8)
            high = ((packed >> 4) & 0xF).to(torch.int8)
            low = torch.where(low >= 8, low - 16, low)
            high = torch.where(high >= 8, high - 16, high)
            interleaved = torch.stack([low, high], dim=-1)
            dequant = interleaved.reshape(*packed.shape[:-1], last_dim * 2)
            # scales: (max_cache_len, num_heads), slice [start_pos:start_pos+length] for head_idx
            scale_slice = scales[start_pos:start_pos + length, head_idx]  # (length,)
            scale_slice = scale_slice.view(1, length, 1)  # broadcast: (1, length, 1)
            return (dequant.to(torch.bfloat16) * scale_slice)

    def _write_quantized(self, key_states, value_states, start, kv_length):
        b = key_states.shape[0]
        # key_states: (batch, num_heads, kv_length, head_dim) -- the incoming chunk
        # We write to the cache at positions [start:start+kv_length]
        k_chunk = key_states[:b]  # entire incoming chunk
        v_chunk = value_states[:b]

        for h in range(self._num_kv_heads):
            k_h = k_chunk[:, h].to(self._compute_dtype)  # (b, kv_length, head_dim)
            v_h = v_chunk[:, h].to(self._compute_dtype)

            # Quantize per-position (per-token) within this head
            abs_max = k_h.abs().amax(dim=-1)  # (b, kv_length)
            min_scale = 1e-3
            min_max = min_scale * 7.0
            too_small = abs_max < min_max
            k_scale_val = torch.where(too_small, torch.zeros_like(abs_max), abs_max / 7.0)
            k_scale_val = torch.clamp(k_scale_val, min=min_scale)
            self._k_scales[start:start+kv_length, h] = k_scale_val[0]

            abs_max_v = v_h.abs().amax(dim=-1)
            too_small_v = abs_max_v < min_max
            v_scale_val = torch.where(too_small_v, torch.zeros_like(abs_max_v), abs_max_v / 7.0)
            v_scale_val = torch.clamp(v_scale_val, min=min_scale)
            self._v_scales[start:start+kv_length, h] = v_scale_val[0]

            k_quant = (k_h / k_scale_val.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
            v_quant = (v_h / v_scale_val.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)

            packed_head_dim = self._head_dim // 2
            packed_stride = self.max_cache_len * packed_head_dim
            offset = h * packed_stride + start * packed_head_dim
            k_packed = (k_quant[..., 0::2] & 0xF) | ((k_quant[..., 1::2] & 0xF) << 4)
            v_packed = (v_quant[..., 0::2] & 0xF) | ((v_quant[..., 1::2] & 0xF) << 4)
            bytes_to_copy = kv_length * packed_head_dim
            self._k_data[offset:offset+bytes_to_copy].copy_(k_packed.view(-1))
            self._v_data[offset:offset+bytes_to_copy].copy_(v_packed.view(-1))

    def _dequantize_prefix(self, b, length):
        """Dequantize prefix [0:length] into a temporary bf16 tensor.
        Uses per-position scales stored alongside the quantized data."""
        k_out = torch.empty(b, self._num_kv_heads, length, self._head_dim,
                           dtype=self._compute_dtype, device=self._prealloc_device)
        v_out = torch.empty(b, self._num_kv_heads, length, self._head_dim,
                           dtype=self._compute_dtype, device=self._prealloc_device)
        packed_head_dim = self._head_dim // 2
        packed_stride = self.max_cache_len * packed_head_dim
        for h in range(self._num_kv_heads):
            offset = h * packed_stride
            k_packed = self._k_data[offset:offset + length * packed_head_dim].view(1, length, packed_head_dim)
            v_packed = self._v_data[offset:offset + length * packed_head_dim].view(1, length, packed_head_dim)
            k_out[:, h] = self._dequantize_int4(k_packed, self._k_scales, 0, length, h)
            v_out[:, h] = self._dequantize_int4(v_packed, self._v_scales, 0, length, h)
        return k_out, v_out

    def get_cached_kv(self, length):
        """Return dequantized keys/values for the first 'length' tokens.
        Returns (k, v) with shape (batch, num_heads, length, head_dim).
        Used by attention recompute patch."""
        if not self._quantize:
            return self.keys, self.values
        b = 1
        k, v = self._dequantize_prefix(b, length)
        return k, v

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

        if self._quantize:
            ks = key_states.to(self._compute_dtype)
            vs = value_states.to(self._compute_dtype)
            self._write_quantized(ks, vs, start, kv_length)
        else:
            ks = key_states.to(self.keys.dtype)
            vs = value_states.to(self.values.dtype)
            self.keys[:b, :, start:start+kv_length].copy_(ks.detach())
            self.values[:b, :, start:start+kv_length].copy_(vs.detach())

        self.cumulative_length.add_(kv_length)
        if start == 0:
            return ks, vs

        if self._quantize:
            k_cached, v_cached = self._dequantize_prefix(b, start)
            k = torch.cat([k_cached.detach(), ks], dim=-2)
            v = torch.cat([v_cached.detach(), vs], dim=-2)
        else:
            k = torch.cat([self.keys[:b, :, :start].detach(), ks], dim=-2)
            v = torch.cat([self.values[:b, :, :start].detach(), vs], dim=-2)
        return k, v

    @property
    def keys(self):
        """Access keys for attention patch. In quantized mode, dequantizes on the fly."""
        if self._quantize:
            b = 1
            length = int(self.cumulative_length.item())
            if length == 0:
                return torch.empty(1, self._num_kv_heads, 0, self._head_dim,
                                  dtype=self._compute_dtype, device=self._prealloc_device)
            k, _ = self._dequantize_prefix(b, length)
            return k
        return self._keys

    @property
    def values(self):
        """Access values for attention patch. In quantized mode, dequantizes on the fly."""
        if self._quantize:
            b = 1
            length = int(self.cumulative_length.item())
            if length == 0:
                return torch.empty(1, self._num_kv_heads, 0, self._head_dim,
                                  dtype=self._compute_dtype, device=self._prealloc_device)
            _, v = self._dequantize_prefix(b, length)
            return v
        return self._values

    @keys.setter
    def keys(self, val):
        self._keys = val

    @values.setter
    def values(self, val):
        self._values = val

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
    import os
    layer_types, layer_kwargs = get_layer_types_and_kwargs(config.get_text_config(decoder=True))
    full_layers = [i for i, t in enumerate(layer_types) if t in ("full_attention", "attention")]
    n_full = len(full_layers)

    text_cfg = config.get_text_config(decoder=True)
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", None)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads
    )
    device = "cuda"

    # Weights are INT4 (via build_lora_chonk_setup quantize=True); the KV cache
    # stays bf16/fp16 ("weights 4, cache 16"). Quantizing the cache (uint8 packed)
    # causes explosive forward/backward dequant tensors at long context -- the
    # exact failure the user hit. Gate behind CHONK_QUANTIZE_KV (default OFF).
    quantize_kv = os.environ.get("CHONK_QUANTIZE_KV", "0") == "1"
    if quantize_kv:
        storage_dtype = torch.uint8
        compute_dtype = torch.bfloat16
        bytes_per_element = 0.5
    else:
        storage_dtype = torch.bfloat16
        compute_dtype = torch.bfloat16
        bytes_per_element = storage_dtype.itemsize

    per_layer_kv_elements = 2 * batch_size * num_kv_heads * max_cache_len * head_dim
    per_layer_bytes = int(per_layer_kv_elements * bytes_per_element)
    if quantize_kv:
        # Per-position per-head scales (fp16): 2 * num_kv_heads * max_cache_len
        scale_per_pos_bytes = num_kv_heads * max_cache_len * compute_dtype.itemsize
        scale_bytes = 2 * scale_per_pos_bytes
    else:
        scale_bytes = 0
    per_layer_total = per_layer_bytes + scale_bytes
    total = per_layer_total * n_full

    if pool is None:
        pool = ChonkPool()

    layers = []
    for i, lt in enumerate(layer_types):
        if lt in ("full_attention", "attention"):
            # Per-layer allocation: avoid one monolithic 32GB exportable block
            # (drivers reject a single contiguous vkAllocateMemory that large;
            # per-layer ~0.5GB blocks fit the bucket ladder and the GTT heap).
            base, host_ptr = pool.alloc_base(per_layer_total, f"chonk_kv_cache_layer_{i}")
            layer = ChonkFullLayer(max_cache_len, base, 0, storage_dtype, compute_dtype, device, quantize_kv, num_kv_heads, head_dim)
            layer._prealloc(batch_size, num_kv_heads, head_dim)
        else:
            layer = LinearAttentionLayer(**layer_kwargs)
        layers.append(layer)

    cache = Cache(layers=layers)
    cache.pool = pool
    cache.full_layer_bytes = per_layer_total
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


def build_model_from_chonk_buffer(config, model_buffer, dtype=torch.bfloat16,
                                  attn_implementation=None, skip_modules=None):
    """Build a trainable model whose parameters are zero-copy views into a
    Chonk Buffer allocation. Creates the module structure on meta (no memory),
    then replaces every parameter with an nn.Parameter view into the buffer.
    The flat buffer layout must match named_parameters() order (this is how
    load_model_directly_to_chonk writes it).
    skip_modules: set of module names whose params are left as meta tensors
    (they are quantized; the PEFT LoraLayer base will be swapped later)."""
    import torch.nn as nn
    from transformers import AutoModelForCausalLM

    skip_modules = set(skip_modules or [])
    skip_prefixes = tuple(f"{m}." for m in skip_modules)

    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    typed = model_buffer.view(dtype)
    offset = 0
    for name, param in list(model.named_parameters()):
        skip = name.startswith(skip_prefixes)
        if name.startswith(skip_prefixes):
            continue  # quantized: module swapped to QuantLinear after PEFT
        numel = param.numel()
        if offset + numel > typed.numel():
            raise RuntimeError(f"Offset overflow: offset={offset}, numel={numel}, buffer_size={typed.numel()}, name={name}")
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


def swap_quantized_base_layers(model, quant_dict, group_size=128, bits=8):
    """After get_peft_model wrapped the (meta-weight) nn.Linear targets, swap
    each LoraLayer's base_layer for a QuantLinear backed by pool buffers.
    PEFT's LoraLayer.forward calls self.base_layer(x) generically, so this is
    the only change needed; LoRA branches stay trainable on top."""
    from vulkanvm_quant_py import QuantLinear

    swapped = 0
    for name, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        base = module.base_layer
        if not hasattr(base, "in_features"):
            continue
        # Match against quant_dict keys (e.g. 'model.layers.3.self_attn.q_proj').
        entry = quant_dict.get(name)
        if entry is None:
            # PEFT layer may sit under model.base_model.base.model.* prefix.
            for m_name, (qw, sw, zw, bw) in quant_dict.items():
                if name.endswith("." + m_name) or name == m_name:
                    entry = (qw, sw, zw, bw)
                    break
        if entry is None:
            continue
        qw, sw, zw, bw = entry
        module.base_layer = QuantLinear(
            base.in_features, base.out_features, qw, sw, zw,
            bits=bits, group_size=group_size, bias=bw)
        swapped += 1
    print(f"Swapped {swapped} quantized base layers into PEFT wrappers")
    return swapped


def replace_plain_quantized_layers(model, quant_dict, group_size=128, bits=8):
    """Replace plain (non-PEFT-wrapped) nn.Linear modules whose weights were
    quantized in the pool with QuantLinear. Used when quantizing ALL linear
    layers (not just LoRA targets): the LoRA targets live inside PEFT wrappers
    (handled by swap_quantized_base_layers); every other quantized Linear is a
    standalone module and gets swapped here."""
    from vulkanvm_quant_py import QuantLinear
    import torch.nn as nn

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        entry = quant_dict.get(name)
        if entry is None:
            for m_name, e in quant_dict.items():
                if name.endswith("." + m_name) or name == m_name:
                    entry = e
                    break
        if entry is None:
            continue
        qw, sw, zw, bw = entry
        ql = QuantLinear(
            module.in_features, module.out_features, qw, sw, zw,
            bits=bits, group_size=group_size, bias=bw)
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], ql)
        replaced += 1
    print(f"Replaced {replaced} plain quantized Linear layers")
    return replaced


def load_model_directly_to_chonk(model_path, config, pool: ChonkPool, dtype=torch.bfloat16,
                                 chunk_size_mb=512, quantize_modules=None, quant_group_size=128,
                                 quant_bits=8):
    """Load model weights directly from disk into Chonk Buffer without loading to CUDA first.
    Uses meta device to create empty model structure, then loads weights directly to Chonk Buffer.

    If quantize_modules is given (list of module names, e.g. 'model.layers.0.self_attn.q_proj'),
    those Linear weights are stored INT8/INT4-quantized in separate pool allocations instead of
    bf16 in the flat model buffer. Returns (model_buffer, quant_dict) where quant_dict maps
    module name -> (qweight, scales, zeros) pool-backed tensors (empty dict when disabled)."""
    import torch
    from transformers import AutoModelForCausalLM
    from vulkanvm_quant_py import quantize_weight

    # Handle "all" special keyword to quantize all linear layers
    if quantize_modules == "all":
        print("Creating meta model for quantization module detection...")
        # Use the SAME config as the actual model (language_model_only=True)
        # so module names match (no "language_model." prefix)
        config.language_model_only = True
        with torch.device('meta'):
            meta_model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=True)
        quantize_modules = {
            name for name, m in meta_model.named_modules()
            if isinstance(m, torch.nn.Linear)
        }
        quantize_modules.discard("lm_head")
        print(f"Quantizing ALL {len(quantize_modules)} linear modules (lm_head excluded)")
    quantize_modules = set(quantize_modules or [])
    quant_weight_names = {f"{m}.weight" for m in quantize_modules}
    quant_bias_names = {f"{m}.bias" for m in quantize_modules}
    quant_dict = {}

    # Create model on meta device (no memory allocation) using from_config
    # Qwen3.5/3.6/3.8 ship as multimodal wrappers (Qwen3_5ForConditionalGeneration):
    # force text-only so the meta model is 'model.layers.*' + lm_head, matching
    # the to_ckpt_name remap below. Without this, the meta model includes the
    # vision tower + MTP head and every weight lookup fails (silent zeros).
    print("Creating model on meta device...")
    config.language_model_only = True
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    # Estimate memory and allocate in Chonk Buffer (use actual param count
    # from the meta model so the buffer matches the weights exactly)
    params_list = list(model.named_parameters())
    total_numel = sum(p.numel() for name, p in params_list
                      if name not in quant_weight_names)
    total_bytes = total_numel * dtype.itemsize
    model_buffer = pool.alloc_model_weights(total_bytes, "model_weights")
    model_buffer_typed = model_buffer.view(dtype)

    print(f"Loading weights directly to Chonk Buffer: {total_bytes / 1e9:.2f} GB "
          f"(excluding {len(quantize_modules)} quantized modules)")

    # Allocate ALL quantized weights as three flat buffers (qweight/scales/
    # zeros) so the buddy allocator sees ~3 large allocations instead of
    # ~3*len(quantize_modules) tiny ones. Tiny allocations fragment the
    # sub-allocator and force fresh blocks during the position-growing steps.
    quant_bytes = {"q": 0, "s": 0, "z": 0}
    if quantize_modules:
        for pname, p in params_list:
            if pname not in quant_weight_names:
                continue
            out_f, in_f = p.shape
            ng = max(1, in_f // quant_group_size)
            quant_bytes["q"] += (out_f * in_f) if quant_bits == 8 else (out_f * in_f) // 2
            quant_bytes["s"] += out_f * ng * 4
            quant_bytes["z"] += out_f * ng
        # +8 per module to cover the 8-byte alignment padding in the loop.
        n_mod = len(quant_weight_names)
        quant_bytes["q"] += n_mod * 8
        quant_bytes["s"] += n_mod * 8
        quant_bytes["z"] += n_mod * 8
        # Allocate ALL quantized weights as chunked flat buffers (qweight/
        # scales/zeros) split into <=1GB exportable blocks so no single
        # monolithic vkAllocateMemory (14.43GB q) is needed -- the radv driver
        # rejects large single exportable blocks under load. This is the
        # "smaller blocks / fine ladder" fix.
        max_q_block_mb = int(os.environ.get("CHONK_MAX_EXPORTABLE_MB", "1024"))
        quant_buffers = {
            "q": pool.alloc_chunked(quant_bytes["q"], "quant_qweight", max_q_block_mb),
            "s": pool.alloc_chunked(quant_bytes["s"], "quant_scales", max_q_block_mb),
            "z": pool.alloc_chunked(quant_bytes["z"], "quant_zeros", max_q_block_mb),
        }
        quant_offsets = {"q": 0, "s": 0, "z": 0}
        print(f"Quant buffers: q={quant_bytes['q']/1e9:.2f}GB "
              f"s={quant_bytes['s']/1e9:.2f}GB z={quant_bytes['z']/1e9:.2f}GB")

# Load state dict and copy directly to Chonk Buffer
    from safetensors.torch import load_file
    import glob

    # Find model files
    model_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    if not model_files:
        model_files = sorted(glob.glob(f"{model_path}/pytorch_model*.bin"))

    param_map = dict(params_list)
    # Precompute each param's slot offset in the flat buffer (layout follows
    # named_parameters() order, skipping quantized params). Offsets are
    # computed ONCE, not per file.
    slots = {}
    o = 0
    for name, param in param_map.items():
        if name in quant_weight_names:
            continue
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
        return name

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

            if name in quant_bias_names:
                # Tiny bf16 bias for a quantized module, kept in its own alloc.
                module_name = name[: -len(".bias")]
                b_buf = pool.alloc_model_weights(numel * dtype.itemsize,
                                                 f"b_{module_name}")
                b_view = b_buf.view(dtype).view_as(param)
                b_view.copy_(tensor.to(dtype))
                entry = quant_dict.setdefault(module_name, (None, None, None, None))
                quant_dict[module_name] = (entry[0], entry[1], entry[2], b_view)
                processed += 1
                continue

            if name in quant_weight_names:
                # Quantize into the flat quant buffers (no bf16 copy).
                module_name = name[: -len(".weight")]
                if processed % 25 == 0:
                    print(f"  Quantizing [{processed}/~{len(param_map)}]: {module_name}...", flush=True)
                qweight, scales, zeros = quantize_weight(
                    tensor.to(torch.float32), bits=quant_bits,
                    group_size=quant_group_size)
                # Keep 8-byte element alignment inside the flat buffers so the
                # float32 scales view is always well-aligned.
                align8 = lambda x: (x + 7) & ~7
                qo = align8(quant_offsets["q"])
                so = align8(quant_offsets["s"])
                zo = align8(quant_offsets["z"])
                q_buf = quant_buffers["q"].narrow(0, qo, qweight.numel())
                s_buf = quant_buffers["s"].narrow(0, so, scales.numel() * 4)
                z_buf = quant_buffers["z"].narrow(0, zo, zeros.numel())
                q_buf.view(torch.uint8).view_as(qweight).copy_(qweight)
                s_buf.view(torch.float32).view_as(scales).copy_(scales)
                z_buf.view(torch.uint8).view_as(zeros).copy_(zeros)
                quant_offsets["q"] = qo + qweight.numel()
                quant_offsets["s"] = so + scales.numel() * 4
                quant_offsets["z"] = zo + zeros.numel()
                quant_dict[module_name] = (
                    q_buf.view(torch.uint8).view_as(qweight),
                    s_buf.view(torch.float32).view_as(scales),
                    z_buf.view(torch.uint8).view_as(zeros),
                    None,
                )
                processed += 1
                continue

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
          f"({processed} params copied, {len(quant_dict)} quantized)")
    return model_buffer, quant_dict


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
                            staging_gb=2.0, attn_implementation="eager",
                            quantize=False, quant_group_size=128,
                            quant_bits=8, act_budget_gb=2.0):
    """Complete LoRA training setup with EVERYTHING in Chonk Buffer:
      - pool + KV cache
      - base weights loaded directly from disk into the pool
      - zero-copy model built from the pool buffer, base frozen
      - LoRA adapters (trainable) in pool memory
      - AdamW fp32 states for LoRA params in pool memory
      - activation + staging buffers in pool memory
    quantize=True stores the LoRA target weights INT8/INT4 in the pool
    (quant_bits=8 or 4; the PEFT base layers are swapped for QuantLinear
    after wrapping). Returns a dict with pool, kv_cache, model, model_buffer,
    optimizer_states, opt_buffer, act_buffer, staging_buffer, staging_host_ptr."""
    from peft import LoraConfig, get_peft_model

    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    pool = ChonkPool()

    # 1. KV cache in Chonk Buffer (before model, per user constraint)
    kv_cache = build_chonk_cache(config, batch_size, max_cache_len, pool)

    # 2. Base weights directly from disk into Chonk Buffer
    quantize_modules = None
    if quantize:
        # Quantize ALL Linear layers into the pool (INT8/INT4); LoRA adapters
        # only ever target TARGET_MODULES. Skipping every quantized weight from
        # the bf16 flat buffer is the big memory win (the 27B model's non-target
        # linears stay bf16 otherwise).
        with torch.device("meta"):
            from transformers import AutoModelForCausalLM
            import torch.nn as nn
            meta_model = AutoModelForCausalLM.from_config(
                config, torch_dtype=dtype, trust_remote_code=True)
        quantize_modules = [
            name for name, m in meta_model.named_modules()
            if isinstance(m, nn.Linear)
        ]
        print(f"Quantizing {len(quantize_modules)} Linear modules to "
              f"INT{quant_bits} (group_size={quant_group_size})")

    model_buffer, quant_dict = load_model_directly_to_chonk(
        model_path, config, pool, dtype=dtype, chunk_size_mb=chunk_size_mb,
        quantize_modules=quantize_modules, quant_group_size=quant_group_size,
        quant_bits=quant_bits)

    # 3. Zero-copy model + freeze base
    model = build_model_from_chonk_buffer(
        config, model_buffer, dtype=dtype,
        attn_implementation=attn_implementation,
        skip_modules=quantize_modules)

    # 4. LoRA adapters
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA applied: {n_trainable / 1e6:.1f}M trainable params "
          f"({n_trainable * 2 / 1e9:.2f} GB bf16)")

    # 4b. When target base weights were left on meta (quantized path), PEFT
    #     creates the LoRA adapters on meta too (it dispatches adapters to
    #     base_layer.weight.device). Re-materialize them on cuda with the
    #     standard PEFT gaussian init before swapping in the QuantLinear.
    for pname, p in list(model.named_parameters()):
        if p.requires_grad and p.device.type == "meta":
            fresh = torch.empty_like(p, device="cuda")
            if "lora_A" in pname:
                torch.nn.init.kaiming_uniform_(fresh, a=math.sqrt(5))
            else:
                torch.nn.init.zeros_(fresh)
            parts = pname.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], torch.nn.Parameter(fresh))
    if any(p.device.type == "meta" and p.requires_grad
           for p in model.parameters()):
        raise RuntimeError("trainable params left on meta after LoRA rematerialize")

    # 4c. Swap quantized base layers inside the PEFT wrappers, then replace
    #     the standalone (non-target) quantized Linears.
    if quant_dict:
        swap_quantized_base_layers(model, quant_dict, group_size=quant_group_size,
                                   bits=quant_bits)
        replace_plain_quantized_layers(model, quant_dict,
                                       group_size=quant_group_size, bits=quant_bits)

    # 5. Optimizer states for trainable params in Chonk Buffer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_states, opt_buffer = create_optimizer_states_in_chonk(trainable, pool)

    # 6. Activation buffers in Chonk Buffer
    text_cfg = config.get_text_config(decoder=True)
    act_buffer = create_activation_buffers(
        pool, batch_size, 0, text_cfg.hidden_size, text_cfg.num_hidden_layers,
        dtype=dtype, chunk_size=4096, budget_gb=act_budget_gb,
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
