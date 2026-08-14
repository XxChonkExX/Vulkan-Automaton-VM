import sys, os, ctypes, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "vulkanvm_torch"))
from chonk import install_chonk_allocator, _HandleDesc, _BufferDesc, _hip_import, _hip_map
import vulkanvm_pool_test as pool_mod

install_chonk_allocator()

a = pool_mod.alloc_export(1 << 20, "probe")
fd = a["fd"]
ext_mem = ctypes.c_void_p()
desc = _HandleDesc()
desc.type = 1
desc.handle.fd = fd
desc.size = 1 << 20
assert _hip_import(ctypes.byref(ext_mem), ctypes.byref(desc)) == 0
dev_ptr = ctypes.c_void_p()
bd = _BufferDesc()
bd.offset = 0
bd.size = 1 << 20
assert _hip_map(ctypes.byref(dev_ptr), ext_mem, ctypes.byref(bd)) == 0
dev_ptr = dev_ptr.value

iface = {"shape": (1 << 20,), "strides": None, "data": (dev_ptr, False), "typestr": "<u1", "version": 2}
wrap = type("PoolWrap", (), {"__cuda_array_interface__": iface})()
base = torch.as_tensor(wrap, device="cuda")
print("base created, ptr:", dev_ptr)
del base
print("base deleted OK")
print("survived - as_tensor deleter does NOT throw")