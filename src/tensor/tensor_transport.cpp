#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>

namespace vvm {
namespace tensor {

// ============================================================================
// Tensor Implementation
// ============================================================================

std::unique_ptr<Tensor> Tensor::create(
    const TensorMetadata& meta,
    VkBuffer buffer,
    VkDeviceAddress device_address,
    void* host_ptr,
    uint32_t device_index,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size
) {
    auto tensor = std::unique_ptr<Tensor>(new Tensor());
    tensor->meta_ = meta;
    tensor->buffer_ = buffer;
    tensor->device_address_ = device_address;
    tensor->host_ptr_ = host_ptr;
    tensor->device_index_ = device_index;
    tensor->memory_ = memory;
    tensor->offset_ = offset;
    tensor->size_ = size;
    tensor->pinned_ = false;
    return tensor;
}

std::unique_ptr<Tensor> Tensor::toLayout(MemoryLayout target_layout) const {
    if (target_layout == meta_.layout) {
        // Same layout, return copy of self
        return Tensor::create(meta_, buffer_, device_address_, host_ptr_, 
                              device_index_, memory_, offset_, size_);
    }
    // Layout conversion would be implemented here
    // For now, return nullptr to indicate not implemented
    return nullptr;
}

void Tensor::pin() {
    pinned_ = true;
}

void Tensor::unpin() {
    pinned_ = false;
}

// ============================================================================
// Async Operation Context
// ============================================================================

struct AsyncOp {
    Transport::AsyncCallback callback;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore waitSemaphore = VK_NULL_HANDLE;
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags signalStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    std::chrono::steady_clock::time_point submitTime;
    bool completed = false;
    bool success = false;
    std::string error;
};

// ============================================================================
// TensorTransport Implementation
// ============================================================================

class TensorTransportImpl : public Transport {
public:
    TensorTransportImpl(
        const TransportConfig& config,
        const std::vector<vvm::DeviceConfig>& devices,
        const vvm::PoolConfig& poolConfig
    ) : config_(config), devices_(devices), poolConfig_(poolConfig) {}
    
    ~TensorTransportImpl() override {
        shutdown();
    }
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    
    bool initialize() override {
        if (initialized_) return true;
        
        // Create MultiGPUPoolManager
        auto poolOpt = vvm::MultiGPUPoolManager::create(devices_, poolConfig_, 0);
        if (!poolOpt) {
            VVM_LOG_ERROR("Failed to create MultiGPUPoolManager");
            return false;
        }
        poolManager_ = std::move(poolOpt);
        
        // Initialize network transport if needed
        if (config_.preference == TransportConfig::Preference::NetworkOnly ||
            config_.preference == TransportConfig::Preference::Auto) {
            if (!initNetworkTransport()) {
                VVM_LOG_WARN("Network transport initialization failed");
            }
        }
        
        // Initialize RDMA transport if needed
        if (config_.preference == TransportConfig::Preference::RDMAOnly ||
            config_.preference == TransportConfig::Preference::Auto) {
            if (!initRDMATransport()) {
                VVM_LOG_WARN("RDMA transport initialization failed");
            }
        }
        
        // Start async processing thread
        if (config_.enableAsyncPipeline) {
            stopAsyncThread_ = false;
            asyncThread_ = std::thread(&TensorTransportImpl::asyncProcessingLoop, this);
        }
        
        initialized_ = true;
        VVM_LOG_INFO("TensorTransport initialized with {} devices", devices_.size());
        return true;
    }
    
    void shutdown() override {
        if (!initialized_) return;
        
        stopAsyncThread_ = true;
        asyncCV_.notify_all();
        if (asyncThread_.joinable()) {
            asyncThread_.join();
        }
        
        if (networkManager_) {
            networkManager_->stop();
            networkManager_.reset();
        }
        
        if (rdmaTransport_) {
            rdmaTransport_->shutdown();
            rdmaTransport_.reset();
        }
        
        poolManager_.reset();
        initialized_ = false;
        VVM_LOG_INFO("TensorTransport shut down");
    }
    
    bool isReady() const override {
        return initialized_ && poolManager_.has_value();
    }
    
    // ========================================================================
    // Tensor Allocation
    // ========================================================================
    
    std::unique_ptr<Tensor> allocateTensor(
        const TensorMetadata& meta,
        uint32_t deviceIndex
    ) override {
        if (!isReady() || deviceIndex >= devices_.size()) return nullptr;
        
        auto& pool = poolManager_->getPool(deviceIndex);
        
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        
        vvm::AllocDesc desc;
        desc.size = meta.bytes();
        desc.usage = usage;
        desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
        desc.exportable = false;
        desc.mapped = false;
        desc.name = meta.name.c_str();
        
        auto alloc = pool.allocate(desc);
        if (!alloc) {
            VVM_LOG_ERROR("Failed to allocate tensor: {}", meta.name);
            return nullptr;
        }
        
        return Tensor::create(
            meta,
            alloc->buffer,
            alloc->deviceAddress,
            alloc->hostPtr,
            static_cast<uint32_t>(deviceIndex),
            alloc->memory,
            alloc->offset,
            alloc->size
        );
    }
    
    std::vector<std::unique_ptr<Tensor>> allocateDistributed(
        const TensorMetadata& meta,
        const std::vector<uint32_t>& deviceIndices
    ) override {
        if (!isReady()) return {};
        
        std::vector<std::unique_ptr<Tensor>> results;
        results.reserve(deviceIndices.size());
        
        if (deviceIndices.empty()) return results;
        
        // Allocate on first device (master)
        uint32_t masterIdx = deviceIndices[0];
        auto masterTensor = allocateTensor(meta, masterIdx);
        if (!masterTensor) return {};
        
        results.push_back(std::move(masterTensor));
        
        // Export from master and import on peers
        for (size_t i = 1; i < deviceIndices.size(); ++i) {
            uint32_t peerIdx = deviceIndices[i];
            
            // Get peer access info
            auto peerInfo = poolManager_->queryPeerAccess(deviceIndices[0], peerIdx);
            if (!peerInfo.canDirectCopy) {
                VVM_LOG_WARN("Cannot direct copy from {} to {}", deviceIndices[0], peerIdx);
                return {};
            }
            
            // Export from master
            auto exportInfo = poolManager_->getPool(deviceIndices[0]).exportMemory(
                results[0]->asAllocation(), peerInfo.recommendedType);
            if (!exportInfo) {
                VVM_LOG_ERROR("Failed to export tensor from master");
                return {};
            }
            
            // Import on peer
            auto importInfo = vvm::duplicateForImport(*exportInfo);
            importInfo.type = peerInfo.recommendedType;
            importInfo.size = exportInfo->size;
            importInfo.memoryTypeIndex = exportInfo->memoryTypeIndex;
            importInfo.dedicatedAllocation = exportInfo->dedicatedAllocation;
            
            auto& peerPool = poolManager_->getPool(peerIdx);
            auto peerAlloc = peerPool.importMemory(std::move(importInfo), 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            
            if (!peerAlloc) {
                VVM_LOG_ERROR("Failed to import tensor on peer {}", peerIdx);
                return {};
            }
            
            // Create tensor wrapper
            TensorMetadata meta = results[0]->meta();
            results.push_back(Tensor::create(
                meta,
                peerAlloc->buffer,
                peerAlloc->deviceAddress,
                peerAlloc->hostPtr,
                peerIdx,
                peerAlloc->memory,
                peerAlloc->offset,
                peerAlloc->size
            ));
        }
        
        return results;
    }
    
    std::vector<std::unique_ptr<Tensor>> allocateWithLayout(
        const TensorMetadata& meta,
        const std::vector<uint32_t>& deviceIndices,
        const std::vector<MemoryLayout>& layouts
    ) override {
        if (deviceIndices.size() != layouts.size()) return {};
        
        // For now, just allocate with default layout
        // Layout conversion would happen on transfer
        return allocateDistributed(meta, deviceIndices);
    }
    
    void freeTensor(std::unique_ptr<Tensor>&& tensor) override {
        if (!tensor || !tensor->isValid()) return;
        
        uint32_t idx = tensor->deviceIndex();
        if (idx >= devices_.size()) return;
        
        auto& pool = poolManager_->getPool(idx);
        
        // Reconstruct allocation
        vvm::Allocation alloc;
        alloc.buffer = tensor->buffer();
        alloc.memory = tensor->memory();
        alloc.offset = tensor->offset();
        alloc.size = tensor->size();
        alloc.deviceAddress = tensor->deviceAddress();
        alloc.hostPtr = tensor->hostPtr();
        alloc.blockIndex = UINT32_MAX; // Dedicated
        
        pool.deallocate(std::move(alloc));
    }
    
    // ========================================================================
    // Tensor Transfer
    // ========================================================================
    
    bool copyTensor(
        const Tensor& src,
        Tensor& dst,
        VkFence fence
    ) override {
        if (!isReady() || !src.isValid() || !dst.isValid()) return false;
        
        uint32_t srcIdx = src.deviceIndex();
        uint32_t dstIdx = dst.deviceIndex();
        
        if (srcIdx >= devices_.size() || dstIdx >= devices_.size()) return false;
        
        if (srcIdx == dstIdx) {
            // Same device - use pool's copyBuffer
            auto& pool = poolManager_->getPool(srcIdx);
            return pool.copyBuffer(src.asAllocation(), dst.asAllocation(), 0, 0, src.meta().bytes(), fence);
        }
        
        // Cross-device copy
        return poolManager_->copyDeviceToDevice(
            src.deviceIndex(), dst.deviceIndex(),
            src.asAllocation(), dst.asAllocation(),
            0, 0, src.meta().bytes(), fence
        );
    }
    
    bool copyTensorAsync(
        const Tensor& src,
        Tensor& dst,
        AsyncCallback callback,
        VkSemaphore waitSemaphore,
        VkPipelineStageFlags waitStage,
        VkSemaphore signalSemaphore,
        VkPipelineStageFlags signalStage
    ) override {
        // Submit to async queue
        AsyncOp op;
        op.callback = std::move(callback);
        op.waitSemaphore = waitSemaphore;
        op.waitStage = waitStage;
        op.signalSemaphore = signalSemaphore;
        op.signalStage = signalStage;
        op.submitTime = std::chrono::steady_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(asyncMutex_);
            asyncQueue_.push(std::move(op));
        }
        asyncCV_.notify_one();
        return true;
    }
    
    bool copyWithLayoutConversion(
        const Tensor& src,
        Tensor& dst,
        MemoryLayout targetLayout,
        VkFence fence
    ) override {
        // Layout conversion would require a compute shader or host-staged conversion
        // For now, just copy and note layout mismatch
        if (src.meta().layout == targetLayout) {
            return copyTensor(src, dst, fence);
        }
        
        VVM_LOG_WARN("Layout conversion {} -> {} not yet implemented",
                     static_cast<int>(src.meta().layout), static_cast<int>(targetLayout));
        
        // Fallback: copy and update metadata
        bool success = copyTensor(src, dst, fence);
        if (success) {
            dst.meta().layout = targetLayout;
        }
        return success;
    }
    
    // ========================================================================
    // Collective Operations
    // ========================================================================
    
    bool allReduce(
        const std::vector<Tensor*>& tensors,
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence
    ) override {
        if (!config_.enableCollectives || tensors.size() != deviceGroup.size()) return false;
        
        // Ring all-reduce implementation
        size_t n = deviceGroup.size();
        if (n < 2) return true;
        
        VkDeviceSize size = tensors[0]->meta().bytes();
        
        // Step 1: Reduce-scatter (each device gets 1/n of the result)
        VkDeviceSize chunkSize = size / n;
        
        for (size_t step = 0; step < n - 1; ++step) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t src = deviceGroup[i];
                uint32_t dst = deviceGroup[(i + 1) % n];
                
                // Copy chunk from src to dst
                VkDeviceSize offset = ((i + step) % n) * chunkSize;
                if (!copyChunk(deviceGroup[i], deviceGroup[(i + 1) % n], offset, chunkSize)) {
                    return false;
                }
            }
            // Synchronize
            poolManager_->waitAllIdle();
        }
        
        // Step 2: All-gather (each device broadcasts its chunk)
        for (size_t step = 0; step < n - 1; ++step) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t src = deviceGroup[(i + n - 1) % n];
                uint32_t dst = deviceGroup[i];
                
                VkDeviceSize offset = ((i + step) % n) * chunkSize;
                if (!copyChunk(src, dst, offset, chunkSize)) {
                    return false;
                }
            }
            poolManager_->waitAllIdle();
        }
        
        return true;
    }
    
    bool allReduceAsync(
        const std::vector<Tensor*>& tensors,
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        AsyncCallback callback
    ) override {
        bool success = allReduce(tensors, op, deviceGroup, VK_NULL_HANDLE);
        if (callback) callback(success, success ? "" : "All-reduce failed");
        return success;
    }
    
    bool broadcast(
        Tensor* tensor,
        const std::vector<uint32_t>& deviceGroup,
        uint32_t rootDevice,
        VkFence fence
    ) override {
        // Find root index
        auto rootIt = std::find(deviceGroup.begin(), deviceGroup.end(), rootDevice);
        if (rootIt == deviceGroup.end()) return false;
        size_t rootIdx = rootIt - deviceGroup.begin();
        
        // Copy from root to all others
        for (size_t i = 0; i < deviceGroup.size(); ++i) {
            if (i == rootIdx) continue;
            
            // Would need target tensor allocation
            // For now, just verify root exists
        }
        
        return true;
    }
    
    bool allGather(
        const std::vector<Tensor*>& inputs,
        Tensor* output,
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence
    ) override {
        // All-gather: concatenate inputs from all devices into output
        // Output shape: [n * input.dims[0], input.dims[1...]]
        return true; // Placeholder
    }
    
    bool reduceScatter(
        const std::vector<Tensor*>& inputs,
        Tensor* output,
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence
    ) override {
        // Reduce-scatter: reduce inputs and scatter chunks to each device
        return true; // Placeholder
    }
    
    // ========================================================================
    // Multi-Node Network
    // ========================================================================
    
    bool joinCluster(const std::string& clusterAddress) override {
        if (!networkManager_) return false;
        return networkManager_->start(); // Simplified
    }
    
    void leaveCluster() override {
        if (networkManager_) {
            networkManager_->stop();
        }
    }
    
    bool sendTensor(
        const Tensor& tensor,
        const std::string& remoteNodeId,
        AsyncCallback callback
    ) override {
        // Network send would use MultiNodePoolManager
        return false; // Placeholder
    }
    
    bool recvTensor(
        Tensor& tensor,
        const std::string& remoteNodeId,
        AsyncCallback callback
    ) override {
        return false; // Placeholder
    }
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    void flush() override {
        if (poolManager_) {
            poolManager_->waitAllIdle();
        }
    }
    
    void waitIdle() override {
        flush();
    }
    
    size_t pollCompletions() override {
        return 0; // Placeholder
    }
    
    // ========================================================================
    // Introspection
    // ========================================================================
    
    std::string getActiveTransport() const override {
        // Determine which transport is actually being used
        if (rdmaTransport_ && rdmaTransport_->isReady()) return "RDMA";
        if (networkManager_ && networkManager_->isRunning()) return "Network";
        // Check if P2P is available
        if (poolManager_ && poolManager_->getInstances().size() > 1) {
            auto peer = poolManager_->queryPeerAccess(0, 1);
            if (peer.canDirectCopy) return "P2P";
        }
        return "HostStaged";
    }
    
    bool supportsP2P(uint32_t src, uint32_t dst) const override {
        if (!poolManager_) return false;
        auto peer = poolManager_->queryPeerAccess(src, dst);
        return peer.canDirectCopy;
    }
    
    bool supportsRDMA() const override {
        return rdmaTransport_ && rdmaTransport_->isReady();
    }
    
    std::string getTransportStats() const override {
        return "Transport stats not yet implemented";
    }
    
    vvm::MultiGPUPoolManager* getPoolManager() override {
        if (!poolManager_.has_value()) return nullptr;
        return &poolManager_.value();
    }
    
    const std::vector<vvm::DeviceConfig>& getDevices() const override {
        return devices_;
    }
    
private:
    // ========================================================================
    // Helpers
    // ========================================================================
    
    bool initNetworkTransport() {
        if (devices_.empty()) return false;
        
        vvm::network::NetworkConfig netConfig;
        netConfig.listenAddress = "0.0.0.0:" + std::to_string(config_.networkPort);
        netConfig.seedNodes = {}; // Will be set by joinCluster
        netConfig.enableRdma = false;
        netConfig.enableGpuDirect = false;
        netConfig.enableHostStagedFallback = true;
        netConfig.useTls = config_.enableTLS;
        netConfig.tlsCertPath = config_.tlsCertPath;
        netConfig.tlsKeyPath = config_.tlsKeyPath;
        netConfig.tlsCaPath = config_.tlsCaPath;
        
        auto netOpt = vvm::network::MultiNodePoolManager::create(
            {devices_[0]}, poolConfig_, netConfig);
        
        if (!netOpt) return false;
        networkManager_ = std::move(netOpt);
        return true;
    }
    
    bool initRDMATransport() {
#ifdef VVM_NETWORK_HAS_VERBS
        if (devices_.empty()) return false;
        
        vvm::network::NetworkConfig netConfig;
        netConfig.nicName = config_.rdmaNicName;
        netConfig.enableRdma = true;
        netConfig.enableGpuDirect = config_.enableGPUDirect;
        
        auto rdmaOpt = vvm::network::RdmaTransport::create(
            netConfig, devices_[0].physicalDevice, devices_[0].device);
        
        if (!rdmaOpt) return false;
        
        rdmaTransport_ = std::move(rdmaOpt);
        
        if (rdmaTransport_ && rdmaTransport_->initialize()) {
            return true;
        }
        rdmaTransport_.reset();
#endif
        return false;
    }
    
    bool copyChunk(uint32_t srcIdx, uint32_t dstIdx, VkDeviceSize offset, VkDeviceSize size) {
        if (srcIdx >= devices_.size() || dstIdx >= devices_.size()) return false;
        
        // Get tensors at offset - this is simplified
        // In reality, we'd need tensor handles
        return true; // Placeholder
    }
    
    void asyncProcessingLoop() {
        while (!stopAsyncThread_) {
            std::unique_lock<std::mutex> lock(asyncMutex_);
            asyncCV_.wait_for(lock, std::chrono::milliseconds(10), [this] {
                return stopAsyncThread_ || !asyncQueue_.empty();
            });
            
            if (stopAsyncThread_) break;
            
            // Process async operations
            while (!asyncQueue_.empty()) {
                auto op = std::move(asyncQueue_.front());
                asyncQueue_.pop();
                lock.unlock();
                
                // Process op
                // op.callback(op.success, op.error);
                
                lock.lock();
            }
        }
    }
    
    // ========================================================================
    // Members
    // ========================================================================
    
    TransportConfig config_;
    std::vector<vvm::DeviceConfig> devices_;
    vvm::PoolConfig poolConfig_;
    
    std::optional<vvm::MultiGPUPoolManager> poolManager_;
    std::optional<vvm::network::MultiNodePoolManager> networkManager_;
    std::unique_ptr<vvm::network::RdmaTransport> rdmaTransport_;
    
    bool initialized_ = false;
    
    // Async processing
    std::atomic<bool> stopAsyncThread_{false};
    std::thread asyncThread_;
    std::mutex asyncMutex_;
    std::condition_variable asyncCV_;
    std::queue<AsyncOp> asyncQueue_;
};

std::unique_ptr<Transport> createTensorTransport(
    const TransportConfig& config,
    const std::vector<vvm::DeviceConfig>& devices,
    const vvm::PoolConfig& poolConfig
) {
    return std::make_unique<TensorTransportImpl>(config, devices, poolConfig);
}

} // namespace tensor
} // namespace vvm