import sys, os, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python", "vulkanvm_torch"))
from chonk import install_chonk_allocator
import vulkanvm_pool_test as pool_mod

install_chonk_allocator()

print("torch allocator installed; first allocations:")
x = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
print("  alloc1 ok", x.shape)
y = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
print("  alloc2 ok", y.shape)
z = torch.randn(64 * 1024 * 1024, device="cuda")
print("  alloc3 ok", z.shape)
a = x + y[: x.numel()]
print("  op ok", a.shape)
print("pool stats:", pool_mod.stats())
del x, y, z, a
torch.cuda.empty_cache()
print("pool stats after empty_cache:", pool_mod.stats())
big = torch.empty(3 * 1024 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
print("  big alloc ok", big.shape)
print("pool stats with 3GB live:", pool_mod.stats())
del big
torch.cuda.empty_cache()
print("pool stats after freeing big:", pool_mod.stats())
print("ALLOCATOR TEST PASSED")