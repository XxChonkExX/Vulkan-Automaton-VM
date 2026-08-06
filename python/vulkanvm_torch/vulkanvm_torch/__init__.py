"""
VulkanVM PyTorch Integration
=============================

High-level Python API for using VulkanVM with PyTorch.

Example:
    import vulkanvm_torch as vvm
    import torch

    # Create pool from existing Vulkan device
    vvm.create_pool(device_props)
    
    # Allocate tensor
    t = vvm.allocate(64 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuOnly)
    
    # Use device address in custom kernels
    addr = vvm.get_device_address(t)
"""

import os
import sys
import torch

# Try to load the C++ extension
try:
    import vulkanvm_torch as _C
except ImportError:
    # Try to find the built extension
    build_paths = [
        os.path.join(os.path.dirname(__file__), 'build', 'lib.*'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', 'python', 'vulkanvm_torch'),
    ]
    for path in build_paths:
        for p in glob.glob(path):
            if os.path.isdir(p):
                sys.path.insert(0, p)
                try:
                    import vulkanvm_torch as _C
                    break
                except ImportError:
                    pass

from ._C import (
    create_pool,
    destroy_pool,
    is_pool_created,
    allocate,
    deallocate,
    get_device_address,
    export_memory,
    get_pool_stats,
    MemoryUsage,
    ExternalHandleType,
)

import torch
from typing import Optional, List, Tuple, Union
import ctypes

__all__ = [
    'create_pool',
    'destroy_pool', 
    'is_pool_created',
    'allocate',
    'deallocate',
    'get_device_address',
    'export_memory',
    'get_pool_stats',
    'MemoryUsage',
    'ExternalHandleType',
    'VulkanVMPool',
    'VulkanVMTensor',
]

# -----------------------------------------------------------------------------
# High-level Python wrappers
# -----------------------------------------------------------------------------

class VulkanVMPool:
    """High-level wrapper for VulkanVM pool."""
    
    def __init__(self, 
                 physical_device: int,
                 device: int,
                 graphics_queue_family: int,
                 compute_queue_family: int,
                 transfer_queue_family: int,
                 graphics_queue: int,
                 compute_queue: int,
                 transfer_queue: int,
                 block_size: int = 512 * 1024 * 1024,
                 max_heap_fraction: float = 0.75):
        
        props = torch.tensor([
            physical_device, device,
            graphics_queue_family, compute_queue_family, transfer_queue_family,
            graphics_queue, compute_queue, transfer_queue
        ], dtype=torch.int64)
        
        # Store config for reference
        self._config = {
            'block_size': block_size,
            'max_heap_fraction': max_heap_fraction,
        }
        
        create_pool(torch.tensor([
            physical_device, device,
            graphics_queue_family, compute_queue_family, transfer_queue_family,
            graphics_queue, compute_queue, transfer_queue
        ], dtype=torch.int64))
    
    @classmethod
    def from_torch_device(cls, device: torch.device, block_size: int = 512 * 1024 * 1024):
        """Create pool from a torch.device (CUDA)."""
        # This requires extracting Vulkan handles from CUDA interop
        # For now, require explicit handles
        raise NotImplementedError("Use explicit Vulkan handles for now")
    
    def allocate(self,
                 size: int,
                 usage: Optional[int] = None,
                 memory_usage: MemoryUsage = MemoryUsage.GpuOnly,
                 exportable: bool = False,
                 mapped: bool = False,
                 name: Optional[str] = None) -> 'VulkanVMTensor':
        """Allocate a tensor from the pool."""
        info = allocate(size, usage, memory_usage.value, exportable, mapped, name)
        return VulkanVMTensor(info)
    
    def deallocate(self, tensor: 'VulkanVMTensor'):
        deallocate(tensor._info)
        tensor._info = None
    
    def stats(self) -> dict:
        s = get_pool_stats()
        return {
            'total_allocated': s[0].item(),
            'total_used': s[1].item(),
            'total_free': s[2].item(),
            'largest_free_block': s[3].item(),
            'total_capacity': s[4].item(),
            'allocation_count': s[5].item(),
            'block_count': s[6].item(),
            'dedicated_count': s[7].item(),
            'fragmentation_ratio': s[8].item() / 10000.0,
        }
    
    def destroy(self):
        destroy_pool()


class VulkanVMTensor:
    """Tensor backed by VulkanVM allocation."""
    
    def __init__(self, info: torch.Tensor):
        self._info = info
    
    @property
    def buffer(self) -> int:
        return self._info[0].item()
    
    @property
    def memory(self) -> int:
        return self._info[1].item()
    
    @property
    def offset(self) -> int:
        return self._info[2].item()
    
    @property
    def size(self) -> int:
        return self._info[3].item()
    
    @property
    def device_address(self) -> int:
        return self._info[4].item()
    
    @property
    def host_ptr(self) -> int:
        return self._info[5].item()
    
    @property
    def block_index(self) -> int:
        return self._info[6].item()
    
    def is_dedicated(self) -> bool:
        return self.block_index == 0xFFFFFFFF
    
    def get_device_address(self) -> int:
        return get_device_address(self._info).item()
    
    def export(self, handle_type: int) -> dict:
        ext = export_memory(self._info, handle_type)
        return {
            'handle': ext[0].item(),
            'size': ext[1].item(),
            'memory_type_index': ext[2].item(),
            'dedicated': bool(ext[3].item()),
        }
    
    def __del__(self):
        if self._info is not None:
            try:
                deallocate(self._info)
            except:
                pass


# Convenience functions
def create_pool(device_props: torch.Tensor) -> int:
    """Create pool from device properties tensor."""
    return create_pool(device_props)

def is_pool_ready() -> bool:
    return is_pool_created()

def get_pool_statistics() -> dict:
    s = get_pool_stats()
    return {
        'total_allocated': s[0].item(),
        'total_used': s[1].item(),
        'total_free': s[2].item(),
        'largest_free_block': s[3].item(),
        'total_capacity': s[4].item(),
        'allocation_count': s[5].item(),
        'block_count': s[6].item(),
        'dedicated_count': s[7].item(),
        'fragmentation_ratio': s[8].item() / 10000.0,
    }


# Context manager for easy pool lifecycle
class vulkanvm_pool:
    """Context manager for VulkanVM pool lifecycle."""
    
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._pool = None
    
    def __enter__(self) -> VulkanVMPool:
        self._pool = VulkanVMPool(*self._args, **self._kwargs)
        return self._pool
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool:
            self._pool.destroy()
            self._pool = None


# Example usage and testing
if __name__ == '__main__':
    print("VulkanVM PyTorch Integration")
    print("Import successful!" if 'vulkanvm_torch' in sys.modules else "C++ extension not loaded")
    print("Use: from vulkanvm_torch import VulkanVMPool, MemoryUsage")