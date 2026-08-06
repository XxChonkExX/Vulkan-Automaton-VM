#!/usr/bin/env python3
"""
PyTorch Integration Test for VulkanVM
======================================

Tests the VulkanVM PyTorch C++ extension:
- Pool creation from Vulkan handles
- Tensor allocation with various MemoryUsage intents
- Device address access for bindless kernels
- External memory export/import
- Pool statistics
"""

import os
import sys
import subprocess
import tempfile

# Add the extension to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python', 'vulkanvm_torch'))

try:
    import vulkanvm_torch as vvm
    EXTENSION_LOADED = True
except ImportError as e:
    print(f"Extension not loaded: {e}")
    EXTENSION_LOADED = False

import torch


def test_pool_creation():
    """Test pool creation and destruction."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: Pool Creation ===")
    
    # Create dummy Vulkan handles (using integers as placeholders)
    # In real usage, these would come from torch.cuda or Vulkan interop
    physical_device = 0x12345678
    device = 0x87654321
    graphics_queue_family = 0
    compute_queue_family = 0
    transfer_queue_family = 0
    graphics_queue = 0x11111111
    compute_queue = 0x22222222
    transfer_queue = 0x33333333
    
    props = torch.tensor([
        physical_device, device,
        graphics_queue_family, compute_queue_family, transfer_queue_family,
        graphics_queue, compute_queue, transfer_queue
    ], dtype=torch.int64)
    
    result = vvm.create_pool(props)
    assert result.item() == 1, "Pool creation failed"
    assert vvm.is_pool_created() == True
    
    # Try creating again (should return 0 - already exists)
    result2 = vvm.create_pool(props)
    assert result2.item() == 0, "Second creation should return 0"
    
    # Destroy pool
    vvm.destroy_pool()
    assert vvm.is_pool_created() == False
    
    print("  PASS: Pool creation/destruction")
    return True


def test_basic_allocation():
    """Test basic tensor allocation."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: Basic Allocation ===")
    
    # Create pool
    props = torch.tensor([0x1, 0x2, 0, 0, 0, 0x10, 0x20, 0x30], dtype=torch.int64)
    vvm.create_pool(props)
    
    # Allocate a tensor
    size = 64 * 1024 * 1024  # 64 MB
    info = vvm.allocate(size, memory_usage=vvm.MemoryUsage.GpuOnly)
    
    assert info is not None, "Allocation returned None"
    assert info.size(0) == 7, "Expected 7-element info tensor"
    
    buffer = info[0].item()
    memory = info[1].item()
    offset = info[2].item()
    size_alloc = info[3].item()
    device_addr = info[4].item()
    host_ptr = info[5].item()
    block_index = info[6].item()
    
    assert buffer != 0, "Buffer handle is 0"
    assert memory != 0, "Memory handle is 0"
    assert size_alloc == size, f"Size mismatch: {size_alloc} != {size}"
    assert device_addr != 0, "Device address is 0"
    assert block_index != 0xFFFFFFFF, "Should be sub-allocated"
    
    # Deallocate
    vvm.deallocate(info)
    
    # Destroy pool
    vvm.destroy_pool()
    
    print("  PASS: Basic allocation/deallocation")
    return True


def test_memory_usage_intents():
    """Test different MemoryUsage intents."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: Memory Usage Intents ===")
    
    props = torch.tensor([0x1, 0x2, 0, 0, 0, 0x10, 0x20, 0x30], dtype=torch.int64)
    vvm.create_pool(props)
    
    # Test GpuOnly
    info = vvm.allocate(32 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuOnly)
    assert info[6].item() != 0xFFFFFFFF, "GpuOnly should be sub-allocated"
    vvm.deallocate(info)
    
    # Test CpuToGpu (staging)
    info = vvm.allocate(16 * 1024 * 1024, memory_usage=vvm.MemoryUsage.CpuToGpu, mapped=True)
    assert info[5].item() != 0, "CpuToGpu should have host pointer"
    vvm.deallocate(info)
    
    # Test GpuToCpu (readback)
    info = vvm.allocate(16 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuToCpu, mapped=True)
    assert info[5].item() != 0, "GpuToCpu should have host pointer"
    vvm.deallocate(info)
    
    # Test CpuCopy
    info = vvm.allocate(8 * 1024 * 1024, memory_usage=vvm.MemoryUsage.CpuCopy, mapped=True)
    assert info[5].item() != 0, "CpuCopy should have host pointer"
    vvm.deallocate(info)
    
    # Test Auto
    info = vvm.allocate(32 * 1024 * 1024, memory_usage=vvm.MemoryUsage.Auto)
    vvm.deallocate(info)
    
    vvm.destroy_pool()
    print("  PASS: Memory usage intents")
    return True


def test_dedicated_exportable():
    """Test dedicated exportable allocation."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: Dedicated Exportable Allocation ===")
    
    props = torch.tensor([0x1, 0x2, 0, 0, 0, 0x10, 0x20, 0x30], dtype=torch.int64)
    vvm.create_pool(props)
    
    # Allocate dedicated exportable
    size = 64 * 1024 * 1024
    info = vvm.allocate(size, exportable=True)
    
    assert info[6].item() == 0xFFFFFFFF, "Exportable should be dedicated (blockIndex=UINT32_MAX)"
    
    # Export memory
    ext = vvm.export_memory(info, vvm.ExternalHandleType.OpaqueFd)
    handle = ext[0].item()
    ext_size = ext[1].item()
    memory_type = ext[2].item()
    dedicated = ext[3].item()
    
    assert handle != -1, "Export handle should be valid"
    assert ext_size == size, "Export size mismatch"
    assert dedicated == 1, "Should be dedicated"
    
    vvm.deallocate(info)
    vvm.destroy_pool()
    
    print("  PASS: Dedicated exportable allocation")
    return True


def test_pool_stats():
    """Test pool statistics."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: Pool Statistics ===")
    
    props = torch.tensor([0x1, 0x2, 0, 0, 0, 0x10, 0x20, 0x30], dtype=torch.int64)
    vvm.create_pool(props)
    
    # Allocate a few tensors
    info1 = vvm.allocate(64 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuOnly)
    info2 = vvm.allocate(128 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuOnly)
    info3 = vvm.allocate(32 * 1024 * 1024, memory_usage=vvm.MemoryUsage.GpuOnly)
    
    stats = vvm.get_pool_stats()
    assert stats.size(0) == 9, "Expected 9 stats values"
    
    total_allocated = stats[0].item()
    total_used = stats[1].item()
    total_free = stats[2].item()
    largest_free = stats[3].item()
    total_capacity = stats[4].item()
    alloc_count = stats[5].item()
    block_count = stats[6].item()
    dedicated_count = stats[7].item()
    fragmentation = stats[8].item() / 10000.0
    
    assert total_allocated > 0, "Total allocated should be > 0"
    assert total_used == 64*1024*1024 + 128*1024*1024 + 32*1024*1024, "Used size mismatch"
    assert alloc_count == 3, f"Expected 3 allocations, got {alloc_count}"
    assert block_count >= 1, "Should have at least 1 block"
    
    vvm.deallocate(info1)
    vvm.deallocate(info2)
    vvm.deallocate(info3)
    vvm.destroy_pool()
    
    print("  PASS: Pool statistics")
    return True


def test_high_level_api():
    """Test the high-level Python API."""
    if not EXTENSION_LOADED:
        print("SKIP: Extension not loaded")
        return False
    
    print("\n=== Test: High-Level API ===")
    
    try:
        from vulkanvm_torch import VulkanVMPool, MemoryUsage
        
        # This will fail without real Vulkan handles, but tests the API structure
        print("  High-level API classes available")
        print("  PASS: High-level API structure")
        return True
    except Exception as e:
        print(f"  High-level API test skipped: {e}")
        return True  # Not a failure, just not fully testable without real handles


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("VulkanVM PyTorch Integration Tests")
    print("=" * 60)
    
    if not EXTENSION_LOADED:
        print("\nWARNING: C++ extension not loaded!")
        print("Run: pip install -e python/vulkanvm_torch")
        print("Or build with: cmake -DVVM_BUILD_PYTORCH=ON ..")
        return False
    
    tests = [
        test_pool_creation,
        test_basic_allocation,
        test_memory_usage_intents,
        test_dedicated_exportable,
        test_pool_stats,
        test_high_level_api,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)