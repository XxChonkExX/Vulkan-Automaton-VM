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
        self._ext_mem = ctypes.c_void_p()
        self._fds = []

    def alloc_base(self, nbytes: int, name: str = "chonk"):
        """Allocate nbytes in the pool, import to HIP, return a 1-D uint8
        CUDA tensor aliasing the memory (plus hostPtr)."""
        nbytes = int(nbytes)
        a = pool_mod.alloc_export(nbytes, name)
        fd = a["fd"]
        host_ptr = a["hostPtr"]
        self._fds.append(fd)

        desc = _HandleDesc()
        desc.type = 1  # hipExternalMemoryHandleTypeOpaqueFd
        desc.handle.fd = fd
        desc.size = nbytes
        desc.flags = 0
        if _hip_import(ctypes.byref(self._ext_mem), ctypes.byref(desc)) != 0:
            raise RuntimeError("hipImportExternalMemory failed")

        dev_ptr = ctypes.c_void_p()
        bd = _BufferDesc()
        bd.offset = 0
        bd.size = nbytes
        bd.flags = 0
        if _hip_map(ctypes.byref(dev_ptr), self._ext_mem, ctypes.byref(bd)) != 0:
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
        return base, host_ptr

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
        self.keys[:b, :, start : start + kv_length].copy_(key_states)
        self.values[:b, :, start : start + kv_length].copy_(value_states)
        self.cumulative_length.add_(kv_length)
        return self.keys[:b, :, : start + kv_length], self.values[:b, :, : start + kv_length]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return int(self.cumulative_length.item()), 0

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
