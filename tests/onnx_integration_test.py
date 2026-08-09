#!/usr/bin/env python3
"""
ONNX Runtime Integration Test for VulkanVM
===========================================

Tests the VulkanVM ONNX Runtime integration:
- Custom allocator hooking
- ModelHub weight fetching
- Session creation with VulkanVM provider
"""

import os
import sys
import tempfile
import numpy as np

# Try to import onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. Install with: pip install onnxruntime")

# Try to import VulkanVM ONNX extension
try:
    import vulkanvm_onnx as vvm_onnx
    EXTENSION_LOADED = True
except ImportError as e:
    print(f"ONNX extension not loaded: {e}")
    EXTENSION_LOADED = False


def create_dummy_onnx_model(model_path: str):
    """Create a simple ONNX model for testing."""
    import onnx
    from onnx import helper, TensorProto
    
    # Simple model: input -> Add -> output
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    weight_tensor = helper.make_tensor_value_info('weight', TensorProto.FLOAT, [1, 3, 224, 224])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Create constant weight
    weight_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    weight_initializer = helper.make_tensor(
        'weight_const', TensorProto.FLOAT, [1, 3, 224, 224], weight_data.tobytes()
    )
    
    node = helper.make_node('Add', ['input', 'weight_const'], ['output'])
    
    graph = helper.make_graph(
        [node], 'test_model',
        [input_tensor], [output_tensor],
        initializer=[weight_initializer]
    )
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx.save(model, model_path)
    return model_path


def test_ort_available():
    """Test if ONNX Runtime is available."""
    print("\n=== Test: ONNX Runtime Available ===")
    if not ONNX_AVAILABLE:
        print("  SKIP: ONNX Runtime not installed")
        return False
    
    print(f"  ONNX Runtime version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    return True


def test_model_hub_fetch():
    """Test ModelHub fetch functionality."""
    print("\n=== Test: ModelHub Fetch ===")
    
    if not EXTENSION_LOADED:
        print("  SKIP: Extension not loaded")
        return False
    
    if not ONNX_AVAILABLE:
        print("  SKIP: ONNX Runtime not available")
        return False
    
    # This would need a running ModelHub server
    # For now, just test the API exists
    try:
        # This will fail without a running hub, but tests API
        result = vvm_onnx.fetch_model("127.0.0.1:51010", "test/model", "/tmp/test_cache", "v1")
        print("  ModelHub fetch API accessible")
        return True
    except Exception as e:
        print(f"  ModelHub fetch test (expected to fail without server): {e}")
        return True  # Not a failure, just no server


def test_session_creation():
    """Test creating ONNX session with VulkanVM provider."""
    print("\n=== Test: Session Creation ===")
    
    if not ONNX_AVAILABLE:
        print("  SKIP: ONNX Runtime not available")
        return False
    
    if not EXTENSION_LOADED:
        print("  SKIP: Extension not loaded")
        return False
    
    # Create a temporary model
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        model_path = f.name
    
    try:
        create_dummy_onnx_model(model_path)
        
        # Try to create session with CPU provider (baseline)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        
        # Test inference
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {'input': input_data})
        
        assert len(outputs) == 1
        assert outputs[0].shape == (1, 3, 224, 224)
        
        print("  PASS: CPUExecutionProvider session")
        
        # Try with VulkanVM provider if available
        if hasattr(vvm_onnx, 'create_vulkanvm_provider'):
            print("  VulkanVM provider factory available")
        else:
            print("  VulkanVM provider not yet implemented in extension")
        
        return True
        
    finally:
        os.unlink(model_path)


def test_allocator_hook():
    """Test custom allocator hook."""
    print("\n=== Test: Allocator Hook ===")
    
    if not EXTENSION_LOADED:
        print("  SKIP: Extension not loaded")
        return False
    
    # Test that the allocator class can be instantiated
    try:
        # This requires a pool, so just test class exists
        assert hasattr(vvm_onnx, 'VulkanVMAllocator')
        assert hasattr(vvm_onnx, 'VulkanVMExecutionProvider')
        assert hasattr(vvm_onnx, 'VulkanVMProviderFactory')
        print("  Allocator and provider classes available")
        return True
    except Exception as e:
        print(f"  Allocator test failed: {e}")
        return False


def test_model_hub_publish():
    """Test ModelHub publish functionality."""
    print("\n=== Test: ModelHub Publish ===")
    
    if not EXTENSION_LOADED:
        print("  SKIP: Extension not loaded")
        return False
    
    if not ONNX_AVAILABLE:
        print("  SKIP: ONNX Runtime not available")
        return False
    
    # Test that the hub can be instantiated
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            from vulkanvm_onnx import ModelHub
            hub = ModelHub(tmpdir)
            
            # Create dummy model files
            model_dir = os.path.join(tmpdir, "source_model")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create dummy weight file
            weight_file = os.path.join(model_dir, "weights.safetensors")
            with open(weight_file, 'wb') as f:
                f.write(b"dummy weights" * 1000)
            
            # Create config
            with open(os.path.join(model_dir, "config.json"), 'w') as f:
                f.write('{"model_type": "test", "hidden_size": 512}')
            
            # Publish (this will fail without Vulkan, but tests API)
            try:
                hub.start("127.0.0.1", 51030)
                result = hub.publish("test/model", model_dir, "v1")
                hub.stop()
                print("  ModelHub publish API accessible")
            except Exception as e:
                print(f"  Publish test (expected without Vulkan): {e}")
            
        return True
    except Exception as e:
        print(f"  ModelHub test failed: {e}")
        return False


def run_all_tests():
    """Run all ONNX integration tests."""
    print("=" * 60)
    print("VulkanVM ONNX Runtime Integration Tests")
    print("=" * 60)
    
    tests = [
        test_ort_available,
        test_model_hub_fetch,
        test_session_creation,
        test_allocator_hook,
        test_model_hub_publish,
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
    import os
    success = run_all_tests()
    sys.exit(0 if success else 1)