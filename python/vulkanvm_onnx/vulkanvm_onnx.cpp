// VulkanVM ONNX Runtime Execution Provider
// Provides a custom EP that uses VulkanVM for memory allocation and execution

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/provider_factory_interface.h>
#include <onnxruntime/core/providers/execution_provider.h>
#include <onnxruntime/core/providers/execution_provider_factory.h>

#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/network/model_registry.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>

namespace vulkanvm_onnx {

using vvm::UnifiedMemoryPool;
using vvm::DeviceConfig;
using vvm::PoolConfig;
using vvm::Allocation;
using vvm::MemoryUsage;

class VulkanVMAllocator : public onnxruntime::IAllocator {
public:
    explicit VulkanVMAllocator(std::shared_ptr<UnifiedMemoryPool> pool)
        : pool_(std::move(pool)) {}

    void* Alloc(size_t size) override {
        // ONNX Runtime requests host allocations - we provide host-visible Vulkan memory
        vvm::AllocDesc desc;
        desc.size = size;
        desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        desc.memoryUsage = vvm::MemoryUsage::CpuCopy;
        desc.mapped = true;
        desc.name = "onnx_host_buffer";
        
        auto allocOpt = pool_->allocate(desc);
        if (!allocOpt) return nullptr;
        
        // Store allocation for later free
        void* ptr = allocOpt->hostPtr;
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = std::move(*allocOpt);
        return ptr;
    }

    void Free(void* p) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(p);
        if (it != allocations_.end()) {
            pool_->deallocate(std::move(it->second));
            allocations_.erase(it);
        }
    }

    const OrtMemoryInfo& Info() const override {
        static const OrtMemoryInfo info = {"VulkanVM", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault};
        return info;
    }

private:
    std::shared_ptr<UnifiedMemoryPool> pool_;
    std::mutex mutex_;
    std::unordered_map<void*, Allocation> allocations_;
};

// -----------------------------------------------------------------------------
// VulkanVM Execution Provider
// -----------------------------------------------------------------------------

class VulkanVMExecutionProvider : public onnxruntime::IExecutionProvider {
public:
    VulkanVMExecutionProvider(const std::string& cache_dir, 
                               const std::shared_ptr<UnifiedMemoryPool>& pool)
        : onnxruntime::IExecutionProvider(onnxruntime::kVulkanVMExecutionProvider),
          pool_(pool),
          cache_dir_(cache_dir) {
        
        // Register our custom allocator
        allocator_ = std::make_unique<VulkanVMAllocator>(pool_);
        
        // ModelHub for weight fetching
        hub_ = std::make_unique<vvm::network::ModelHub>(cache_dir);
    }

    ~VulkanVMExecutionProvider() override = default;

    std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>> CreatePreferredAllocators() override {
        std::vector<std::unique_ptr<onnxruntime::IExecutionProvider>> providers;
        providers.push_back(std::unique_ptr<onnxruntime::IExecutionProvider>(this));
        return providers;
    }

    onnxruntime::AllocatorPtr GetAllocator(int ort_mem_type) const override {
        if (ort_mem_type == OrtMemTypeDefault) {
            return allocator_.get();
        }
        return onnxruntime::IExecutionProvider::GetAllocator(ort_mem_type);
    }

    onnxruntime::common::Status Compile(const std::vector<onnxruntime::GraphViewer*>& graph_viewers,
                                         const std::string& model_path) override {
        // For now, delegate to default CPU EP
        // TODO: Implement Vulkan-backed kernels
        return onnxruntime::common::Status::OK();
    }

    // ModelHub integration for weight fetching
    bool FetchModel(const std::string& model_id, const std::string& version,
                    const std::string& hub_address) {
        return vvm::network::ModelHub::fetch(hub_address, model_id, cache_dir_, version);
    }

    std::shared_ptr<UnifiedMemoryPool> GetPool() const { return pool_; }
    vvm::network::ModelHub* GetHub() { return hub_.get(); }

private:
    std::shared_ptr<UnifiedMemoryPool> pool_;
    std::string cache_dir_;
    std::unique_ptr<VulkanVMAllocator> allocator_;
    std::unique_ptr<vvm::network::ModelHub> hub_;
};

// -----------------------------------------------------------------------------
// Provider Factory
// -----------------------------------------------------------------------------

class VulkanVMProviderFactory : public onnxruntime::IExecutionProviderFactory {
public:
    VulkanVMProviderFactory(const std::string& cache_dir,
                            const std::shared_ptr<UnifiedMemoryPool>& pool)
        : cache_dir_(cache_dir), pool_(pool) {}

    std::unique_ptr<onnxruntime::IExecutionProvider> CreateProvider() override {
        return std::make_unique<VulkanVMExecutionProvider>(cache_dir_, pool_);
    }

private:
    std::string cache_dir_;
    std::shared_ptr<UnifiedMemoryPool> pool_;
};

// -----------------------------------------------------------------------------
// C API for Python bindings
// -----------------------------------------------------------------------------

extern "C" {

// Create the provider factory
OrtExecutionProviderFactory* CreateVulkanVMProviderFactory(
    const char* cache_dir, void* pool_ptr) {
    
    auto* pool = static_cast<std::shared_ptr<UnifiedMemoryPool>*>(pool_ptr);
    auto factory = std::make_unique<VulkanVMProviderFactory>(
        cache_dir ? cache_dir : "./onnx_cache", *pool);
    return factory.release();
}

void DestroyProviderFactory(OrtExecutionProviderFactory* factory) {
    delete factory;
}

} // extern "C"

} // namespace vulkanvm_onnx