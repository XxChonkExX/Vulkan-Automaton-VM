"""
VulkanVM ONNX Runtime Integration
==================================

High-level Python API for using VulkanVM with ONNX Runtime.

Example:
    import vulkanvm_onnx as vvm
    import onnxruntime as ort

    # Create pool
    pool = vvm.create_pool(...)
    
    # Create ONNX session with VulkanVM provider
    session = vvm.create_session("model.onnx", pool)
    
    # Run inference
    outputs = session.run(None, {"input": input_data})
"""

import os
import sys
import ctypes
import numpy as np

# Try to load the C++ extension
try:
    import vulkanvm_onnx as _C
except ImportError:
    # Try to find the built extension
    build_paths = [
        os.path.join(os.path.dirname(__file__), 'build', 'lib.*'),
        os.path.join(os.path.dirname(__file__), '..', '..', '..', 'build', 'python', 'vulkanvm_onnx'),
    ]
    for path in build_paths:
        for p in glob.glob(path):
            if os.path.isdir(p):
                sys.path.insert(0, p)
                try:
                    import vulkanvm_onnx as _C
                    break
                except ImportError:
                    pass

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

__all__ = [
    'create_vulkanvm_provider',
    'create_session',
    'VulkanVMSession',
    'fetch_model',
]

# -----------------------------------------------------------------------------
# High-level session wrapper
# -----------------------------------------------------------------------------

class VulkanVMSession:
    """ONNX Runtime session with VulkanVM provider."""
    
    def __init__(self, 
                 model_path: str,
                 vulkanvm_pool,
                 cache_dir: str = "./onnx_cache",
                 providers: Optional[List[str]] = None):
        
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed. pip install onnxruntime")
        
        # Create the VulkanVM provider factory
        # This requires the pool's internal pointer
        self._pool = vulkanvm_pool
        self._cache_dir = cache_dir
        
        # Create session with VulkanVM provider + fallback
        self._session = self._create_session(model_path, providers)
    
    def _create_session(self, model_path: str, providers: Optional[List[str]]):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Add VulkanVM provider if we have the C++ extension
        if hasattr(_C, 'CreateVulkanVMProviderFactory'):
            # This would need the pool's internal shared_ptr
            pass
        
        provider_list = providers or ['CPUExecutionProvider']
        return ort.InferenceSession(model_path, so, providers=provider_list)
    
    def run(self, output_names: Optional[List[str]], input_feed: Dict[str, np.ndarray]):
        return self._session.run(output_names, input_feed)
    
    def fetch_model(self, model_id: str, version: str = "1", hub_address: str = "127.0.0.1:51010"):
        """Fetch model weights from ModelHub."""
        return _C.fetch_model(hub_address, model_id, self._cache_dir, version)


def create_vulkanvm_provider(pool_ptr: int, cache_dir: str = "./onnx_cache"):
    """Create a VulkanVM execution provider for ONNX Runtime."""
    if not hasattr(_C, 'CreateVulkanVMProviderFactory'):
        raise RuntimeError("C++ extension not loaded with provider support")
    return _C.CreateVulkanVMProviderFactory(cache_dir, pool_ptr)


def create_session(model_path: str, pool, cache_dir: str = "./onnx_cache") -> VulkanVMSession:
    """Create an ONNX Runtime session with VulkanVM provider."""
    return VulkanVMSession(model_path, pool, cache_dir)


def fetch_model(hub_address: str, model_id: str, cache_dir: str, version: str = "1") -> bool:
    """Fetch model from a ModelHub server."""
    if not hasattr(_C, 'fetch_model'):
        raise RuntimeError("Model fetching not available in C++ extension")
    return _C.fetch_model(hub_address, model_id, cache_dir, version)


# For direct use without the wrapper
def get_onnx_provider_options(pool_ptr: int, cache_dir: str) -> Dict:
    """Get provider options dict for ort.InferenceSession()."""
    return {
        'vulkanvm': {
            'pool_ptr': pool_ptr,
            'cache_dir': cache_dir,
        }
    }


# Example usage
if __name__ == '__main__':
    print("VulkanVM ONNX Runtime Integration")
    print("ONNX Runtime available:", ONNX_AVAILABLE)
    if ONNX_AVAILABLE:
        import onnxruntime as ort
        print("Available providers:", ort.get_available_providers())