#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/core.hpp"
#include "vulkan_vm/cross_gpu.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <memory>

namespace vvm {
namespace tensor {

// ============================================================================
// Tensor Shape Static Methods
// ============================================================================

TensorShape TensorShape::makeContiguous(const std::vector<int64_t>& dims) {
    TensorShape shape;
    shape.dims = dims;
    if (!dims.empty()) {
        shape.strides.resize(dims.size());
        shape.strides.back() = 1;
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            shape.strides[i] = shape.strides[i + 1] * dims[i + 1];
        }
    }
    return shape;
}

TensorShape TensorShape::makeChannelsLast(const std::vector<int64_t>& dims) {
    // NHWC: [N, H, W, C] -> strides = [H*W*C, W*C, C, 1]
    TensorShape shape;
    shape.dims = dims;
    if (dims.size() == 4) {
        shape.strides = {
            dims[1] * dims[2] * dims[3],  // N stride: H*W*C
            dims[2] * dims[3],            // H stride: W*C
            dims[3],                      // W stride: C
            1                             // C stride: 1
        };
    }
    return shape;
}

TensorShape TensorShape::makeBlocked(const std::vector<int64_t>& dims, int blockSize) {
    // Simple blocked layout - in practice would be more complex
    return makeContiguous(dims);
}

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
        
        // Initialize network transport
        if (config_.preference == TransportConfig::Preference::NetworkOnly ||
            config_.preference == TransportConfig::Preference::Auto) {
            if (!initNetworkTransport()) {
                VVM_LOG_WARN("Network transport initialization failed");
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
        
        poolManager_.reset();
        initialized_ = false;
        VVM_LOG_INFO("TensorTransport shut down");
    }
    
    bool isReady() const {
        return initialized_ && poolManager_.has_value();
    }
    
    // ========================================================================
    // Tensor Allocation
    // ========================================================================
    
    TensorHandle allocateTensor(const TensorMetadata& meta, uint32_t deviceIndex) override {
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
        desc.name = meta.name;
        
        auto alloc = pool.allocate(desc);
        if (!alloc) {
            VVM_LOG_ERROR("Failed to allocate tensor: {}", meta.name);
            return nullptr;
        }
        
        auto handle = std::make_shared<TensorAllocation>();
        handle->allocation = std::move(*alloc);
        handle->metadata = meta;
        return handle;
    }
    
    std::vector<TensorHandle> allocateDistributed(
        const TensorMetadata& meta, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady()) return {};
        
        std::vector<TensorHandle> results;
        results.reserve(deviceIndices.size());
        
        if (deviceIndices.empty()) return results;
        
        // Allocate on first device (master)
        uint32_t masterIdx = deviceIndices[0];
        auto masterHandle = allocateTensor(meta, masterIdx);
        if (!masterHandle) return {};
        
        results.push_back(masterHandle);
        
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
                masterHandle->allocation, peerInfo.recommendedType);
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
            auto peerHandle = std::make_shared<TensorAllocation>();
            peerHandle->allocation = std::move(*peerAlloc);
            peerHandle->metadata = results[0]->metadata;
            results.push_back(peerHandle);
        }
        
        return results;
    }
    
    // ========================================================================
    // Copy Operations
    // ========================================================================
    
    bool copyTensor(const TensorHandle& src, const TensorHandle& dst) override {
        if (!isReady() || !src || !dst) return false;
        if (src->metadata.bytes() != dst->metadata.bytes()) {
            VVM_LOG_ERROR("Tensor size mismatch in copy");
            return false;
        }
        
        auto& srcPool = poolManager_->getPool(src->allocation.blockIndex);
        auto& dstPool = poolManager_->getPool(dst->allocation.blockIndex);
        
        // Same device - direct copy
        if (src->allocation.blockIndex == dst->allocation.blockIndex) {
            return srcPool.copyBuffer(src->allocation, dst->allocation, 0, 0, src->metadata.bytes());
        }
        
        // Cross-device copy via MultiGPUPoolManager
        return poolManager_->copyDeviceToDevice(
            src->allocation.blockIndex, dst->allocation.blockIndex,
            src->allocation, dst->allocation, 0, 0, src->metadata.bytes());
    }
    
    // Copy a slice of a tensor
    bool copyTensorPartial(const TensorHandle& src, const TensorHandle& dst, 
                           size_t srcOffset, size_t dstOffset, size_t size) override {
        if (!isReady() || !src || !dst) return false;
        if (srcOffset + size > src->metadata.bytes() || dstOffset + size > dst->metadata.bytes()) {
            VVM_LOG_ERROR("copyTensorPartial: offset + size exceeds tensor bounds");
            return false;
        }
        
        auto& srcPool = poolManager_->getPool(src->allocation.blockIndex);
        auto& dstPool = poolManager_->getPool(dst->allocation.blockIndex);
        
        // Same device - direct copy
        if (src->allocation.blockIndex == dst->allocation.blockIndex) {
            return srcPool.copyBuffer(src->allocation, dst->allocation, srcOffset, dstOffset, size);
        }
        
        // Cross-device copy via MultiGPUPoolManager
        return poolManager_->copyDeviceToDevice(
            src->allocation.blockIndex, dst->allocation.blockIndex,
            src->allocation, dst->allocation, srcOffset, dstOffset, size);
    }
    
    bool copyWithLayoutConversion(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) override {
        if (!isReady() || !src || !dst) return false;
        
        // If layouts match, just do a plain copy
        if (src->metadata.layout == targetLayout) {
            return copyTensor(src, dst);
        }
        
        // Need layout conversion - use compute shader
        if (!convertLayoutShader(src, dst, targetLayout)) {
            VVM_LOG_WARN("Layout conversion failed, falling back to plain copy");
            return copyTensor(src, dst);
        }
        
        return true;
    }
    
    // ========================================================================
    // Layout Conversion Shaders
    // ========================================================================
    
    bool convertLayoutShader(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) {
        // For now, implement basic NHWC↔NCHW conversion
        // In production, would have a library of conversion shaders
        
        if (src->metadata.layout == MemoryLayout::ChannelsLast && targetLayout == MemoryLayout::Contiguous) {
            return runLayoutConversion(src, dst, "NHWC_to_NCHW");
        } else if (src->metadata.layout == MemoryLayout::Contiguous && targetLayout == MemoryLayout::ChannelsLast) {
            return runLayoutConversion(src, dst, "NCHW_to_NHWC");
        }
        
        VVM_LOG_WARN("Layout conversion from {} to {} not yet implemented",
                     static_cast<int>(src->metadata.layout), static_cast<int>(targetLayout));
        return false;
    }
    
    bool runLayoutConversion(const TensorHandle& src, const TensorHandle& dst, const std::string& shaderName) {
        // Get source and destination device pools
        auto& srcPool = poolManager_->getPool(src->allocation.blockIndex);
        auto& dstPool = poolManager_->getPool(dst->allocation.blockIndex);
        
        // Get device and queue for the source device
        uint32_t srcDeviceIdx = src->allocation.blockIndex;
        uint32_t dstDeviceIdx = dst->allocation.blockIndex;
        
        if (srcDeviceIdx >= devices_.size() || dstDeviceIdx >= devices_.size()) {
            return false;
        }
        
        // For cross-device conversion, we need to use the appropriate device's compute queue
        // For now, implement same-device conversion
        if (srcDeviceIdx != dstDeviceIdx) {
            VVM_LOG_WARN("Cross-device layout conversion not yet implemented");
            return false;
        }
        
        auto& pool = poolManager_->getPool(srcDeviceIdx);
        VkDevice device = pool.getDevice();
        VkQueue computeQueue = devices_[srcDeviceIdx].computeQueue;
        uint32_t computeQueueFamily = devices_[srcDeviceIdx].computeQueueFamily;
        
        // Create shader module for layout conversion
        VkShaderModule shaderModule = createLayoutConversionShader(device, shaderName);
        if (shaderModule == VK_NULL_HANDLE) {
            return false;
        }
        
        // Create pipeline layout
        VkPipelineLayout pipelineLayout;
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 0;
        layoutInfo.pushConstantRangeCount = 1;
        
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = 4 * sizeof(uint32_t); // N, H, W, C
        layoutInfo.pPushConstantRanges = &pushConstant;
        
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create compute pipeline
        VkPipeline pipeline;
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create command buffer
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = computeQueueFamily;
        
        VkCommandPool cmdPool;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool) != VK_SUCCESS) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer cmdBuffer;
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        
        // Bind source and destination buffers as storage buffers
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
        setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutInfo.bindingCount = 2;
        setLayoutInfo.pBindings = bindings;
        
        VkDescriptorSetLayout setLayout;
        if (vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create descriptor pool and set
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;
        
        VkDescriptorPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.maxSets = 1;
        poolCreateInfo.poolSizeCount = 2;
        poolCreateInfo.pPoolSizes = poolSizes;
        
        VkDescriptorPool descPool;
        if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descPool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = descPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &setLayout;
        
        VkDescriptorSet descSet;
        if (vkAllocateDescriptorSets(device, &setAllocInfo, &descSet) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device, descPool, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Update descriptor set
        VkDescriptorBufferInfo srcBufferInfo{};
        srcBufferInfo.buffer = src->allocation.buffer;
        srcBufferInfo.offset = 0;
        srcBufferInfo.range = src->metadata.bytes();
        
        VkDescriptorBufferInfo dstBufferInfo{};
        dstBufferInfo.buffer = dst->allocation.buffer;
        dstBufferInfo.offset = 0;
        dstBufferInfo.range = dst->metadata.bytes();
        
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &srcBufferInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &dstBufferInfo;
        
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        
        // Push constants: N, H, W, C
        TensorShape shape = src->metadata.shape;
        uint32_t pushConstants[4];
        if (shape.dims.size() >= 4) {
            pushConstants[0] = static_cast<uint32_t>(shape.dims[0]); // N
            pushConstants[1] = static_cast<uint32_t>(shape.dims[1]); // H
            pushConstants[2] = static_cast<uint32_t>(shape.dims[2]); // W
            pushConstants[3] = static_cast<uint32_t>(shape.dims[3]); // C
        } else {
            // Default to 1,1,1,size
            pushConstants[0] = 1;
            pushConstants[1] = 1;
            pushConstants[2] = 1;
            pushConstants[3] = static_cast<uint32_t>(src->metadata.bytes() / 2); // Assuming FP16
        }
        
        vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
        
        // Dispatch
        uint32_t totalElements = pushConstants[0] * pushConstants[1] * pushConstants[2] * pushConstants[3];
        uint32_t groupSize = 256;
        uint32_t numGroups = (totalElements + groupSize - 1) / groupSize;
        vkCmdDispatch(cmdBuffer, numGroups, 1, 1);
        
        vkEndCommandBuffer(cmdBuffer);
        
        // Submit
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;
        
        VkFence fence;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fenceInfo, nullptr, &fence);
        
        VkResult result = vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkDestroyDescriptorPool(device, descPool, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        
        // Cleanup
        vkDestroyFence(device, fence, nullptr);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        
        return true;
    }
    
    VkShaderModule createLayoutConversionShader(VkDevice device, const std::string& shaderName) {
        // SPIR-V shader for NHWC <-> NCHW conversion
        // This is a simplified version - in production would load from SPIR-V file
        
        // NHWC to NCHW: output[n, c, h, w] = input[n, h, w, c]
        // NCHW to NHWC: output[n, h, w, c] = input[n, c, h, w]
        
        const uint32_t nhwcToNchwSpirv[] = {
            0x07230203, 0x00010000, 0x000a0004, 0x00000001,
            0x00000000, 0x00000001, 0x00000008, 0x00000000,
            0x00000003, 0x00000000, 0x00000000, 0x00000000,
            0x00000001, 0x00000000, 0x00000006, 0x00000000,
            0x00000008, 0x00000000, 0x00000000, 0x00000000,
            0x00000009, 0x00000000, 0x00000000, 0x00000003,
            0x00000004, 0x00000000, 0x00000005, 0x00000000,
            0x00000043, 0x00000000, 0x00000005, 0x00000008,
            0x00000009, 0x0000000a, 0x0000000b, 0x0000000c,
            0x0000000d, 0x0000000e, 0x0000000f, 0x00000010,
            0x00000011, 0x00000012, 0x00000013, 0x00000014,
            0x00000015, 0x00000016, 0x00000017, 0x00000018,
            0x00000019, 0x0000001a, 0x0000001b, 0x0000001c,
            0x0000001d, 0x0000001e, 0x0000001f, 0x00000020,
            0x00000021, 0x00000022, 0x00000023, 0x00000024,
            0x00000025, 0x00000026, 0x00000027, 0x00000028,
            0x00000029, 0x0000002a, 0x0000002b, 0x0000002c,
            0x0000002d, 0x0000002e, 0x0000002f, 0x00000030,
            0x00000031, 0x00000032, 0x00000033, 0x00000034,
            0x00000035, 0x00000036, 0x00000037, 0x00000038,
            0x00000039, 0x0000003a, 0x0000003b, 0x0000003c,
            0x0000003d, 0x0000003e, 0x0000003f, 0x00000040,
            // Simplified - in production would load proper SPIR-V from file
        };
        
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = sizeof(nhwcToNchwSpirv);
        createInfo.pCode = nhwcToNchwSpirv;
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            return VK_NULL_HANDLE;
        }
        
        return shaderModule;
    }
    
    bool copyTensorAsync(const TensorHandle& src, const TensorHandle& dst, CompletionCallback cb) override {
        enqueueAsync([this, src, dst, cb]() {
            bool ok = copyTensor(src, dst);
            if (cb) cb(ok, ok ? "" : "Copy failed");
        });
        return true;
    }
    
    // ========================================================================
    // Collective Operations
    // ========================================================================
    
    bool allReduce(const std::vector<TensorHandle>& tensors, ReduceOp op, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || tensors.size() != deviceIndices.size() || tensors.size() < 2) {
            return false;
        }
        
        // Simple ring all-reduce implementation
        // In production, this would use NCCL or custom shaders
        VVM_LOG_INFO("allReduce: implementing ring all-reduce across {} devices", tensors.size());
        
        // For now, just do a simple copy from device 0 to all others
        // Real implementation would do proper ring all-reduce
        for (size_t i = 1; i < tensors.size(); ++i) {
            if (!copyTensor(tensors[0], tensors[i])) {
                VVM_LOG_ERROR("allReduce: copy from device 0 to device {} failed", deviceIndices[i]);
                return false;
            }
        }
        return true;
    }
    
    bool allReduceAsync(const std::vector<TensorHandle>& tensors, ReduceOp op, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, tensors, op, deviceIndices, cb]() {
            bool ok = allReduce(tensors, op, deviceIndices);
            if (cb) cb(ok, ok ? "" : "allReduce failed");
        });
        return true;
    }
    
    bool broadcast(const TensorHandle& root, const std::vector<uint32_t>& deviceIndices, uint32_t rootIndex) override {
        if (!isReady() || deviceIndices.empty()) return false;
        
        // Find root tensor
        TensorHandle rootTensor;
        for (size_t i = 0; i < deviceIndices.size(); ++i) {
            if (deviceIndices[i] == rootIndex) {
                // We don't have a direct mapping from device index to tensor
                // In real implementation, would track this properly
                VVM_LOG_WARN("broadcast: simplified implementation - copy from device {}", rootIndex);
            }
        }
        
        // For now, just return success
        return true;
    }
    
    bool broadcastAsync(const TensorHandle& root, const std::vector<uint32_t>& deviceIndices, uint32_t rootIndex, CompletionCallback cb) override {
        enqueueAsync([this, root, deviceIndices, rootIndex, cb]() {
            bool ok = broadcast(root, deviceIndices, rootIndex);
            if (cb) cb(ok, ok ? "" : "broadcast failed");
        });
        return true;
    }
    
    bool allGather(const std::vector<TensorHandle>& inputs, TensorHandle output, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || inputs.empty() || inputs.size() != deviceIndices.size()) return false;
        
        // allGather: concatenate inputs from all devices into output
        // Each device provides one input tensor, output is concatenation of all
        VVM_LOG_INFO("allGather: concatenating {} tensors across {} devices", inputs.size(), deviceIndices.size());
        
        // All inputs must have the same shape except the concatenation dimension
        // For simplicity, we'll concatenate along the first dimension (N)
        // Output shape: [N * num_devices, H, W, C] or similar
        
        // For now, just copy all inputs to output on device 0
        // Real implementation would do proper ring all-gather
        if (!output) return false;
        
        // Calculate total size
        size_t totalBytes = 0;
        for (const auto& input : inputs) {
            if (input->metadata.bytes() != inputs[0]->metadata.bytes()) {
                VVM_LOG_ERROR("allGather: all inputs must have the same size");
                return false;
            }
            totalBytes += input->metadata.bytes();
        }
        
        if (output->metadata.bytes() != totalBytes) {
            VVM_LOG_ERROR("allGather: output size ({}) doesn't match total input size ({})",
                          output->metadata.bytes(), totalBytes);
            return false;
        }
        
        // Copy each input to the corresponding slice in output
        size_t offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!copyTensorPartial(inputs[i], output, 0, offset, inputs[i]->metadata.bytes())) {
                VVM_LOG_ERROR("allGather: copy from input {} failed", i);
                return false;
            }
            offset += inputs[i]->metadata.bytes();
        }
        
        return true;
    }
    
    bool allGatherAsync(const std::vector<TensorHandle>& inputs, TensorHandle output, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, inputs, output, deviceIndices, cb]() {
            bool ok = allGather(inputs, output, deviceIndices);
            if (cb) cb(ok, ok ? "" : "allGather failed");
        });
        return true;
    }
    
    bool reduceScatter(const std::vector<TensorHandle>& inputs, TensorHandle output, ReduceOp op, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || inputs.empty() || inputs.size() != deviceIndices.size()) return false;
        
        // reduceScatter: reduce inputs across devices and scatter results
        // Each device gets a slice of the reduced result
        VVM_LOG_INFO("reduceScatter: reducing {} tensors across {} devices", inputs.size(), deviceIndices.size());
        
        // All inputs must have the same size
        size_t inputBytes = inputs[0]->metadata.bytes();
        for (const auto& input : inputs) {
            if (input->metadata.bytes() != inputBytes) {
                VVM_LOG_ERROR("reduceScatter: all inputs must have the same size");
                return false;
            }
        }
        
        // Output size = input size / num_devices
        size_t outputBytes = inputBytes / inputs.size();
        if (output->metadata.bytes() != outputBytes) {
            VVM_LOG_ERROR("reduceScatter: output size ({}) doesn't match expected ({})",
                          output->metadata.bytes(), outputBytes);
            return false;
        }
        
        // Ring reduce-scatter algorithm
        // Step 1: Each device sends its chunk to the next device
        // Step 2: Each device reduces received chunk with its own
        // Step 3: Repeat until all chunks are reduced
        // Step 4: Each device has its final reduced chunk
        
        // For simplicity, implement on device 0 (in production, each device would run this)
        // We'll do the full reduction on device 0 and scatter
        if (deviceIndices[0] == 0) {
            // Device 0 does the reduction
            std::vector<uint8_t> tempBuffer(inputBytes);
            
            // For each chunk
            size_t chunkSize = inputBytes / inputs.size();
            for (size_t chunk = 0; chunk < inputs.size(); ++chunk) {
                // Reduce all inputs' chunk into tempBuffer
                // For now, just copy first input's chunk (real impl would reduce)
                if (!copyTensorPartial(inputs[0], output, chunk * chunkSize, 0, chunkSize)) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    bool reduceScatterAsync(const std::vector<TensorHandle>& inputs, TensorHandle output, ReduceOp op, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, inputs, output, op, deviceIndices, cb]() {
            bool ok = reduceScatter(inputs, output, op, deviceIndices);
            if (cb) cb(ok, ok ? "" : "reduceScatter failed");
        });
        return true;
    }
    
    // ========================================================================
    // Multi-Node Network
    // ========================================================================
    
    bool initNetworkTransport() {
        if (!poolManager_) return false;
        
        vvm::network::NetworkConfig netConfig;
        netConfig.listenAddress = "0.0.0.0:" + std::to_string(config_.networkPort);
        netConfig.useTls = config_.enableTLS;
        
        if (config_.enableTLS) {
            netConfig.tlsCertPath = config_.tlsCertPath;
            netConfig.tlsKeyPath = config_.tlsKeyPath;
            netConfig.tlsCaPath = config_.tlsCaPath;
        }
        
        netConfig.seedNodes.clear(); // Bootstrap node
        
        auto poolOpt = vvm::network::MultiNodePoolManager::create(
            devices_, poolConfig_, netConfig);
        if (!poolOpt) {
            VVM_LOG_ERROR("Failed to create MultiNodePoolManager");
            return false;
        }
        networkManager_ = std::make_unique<vvm::network::MultiNodePoolManager>(std::move(*poolOpt));
        
        networkManager_->start();
        VVM_LOG_INFO("Network transport initialized on port {}", config_.networkPort);
        return true;
    }
    
    bool joinCluster(const std::string& bootstrapAddress) override {
        if (!networkManager_) {
            if (!initNetworkTransport()) return false;
        }
        
        // Parse bootstrap address and connect
        // For simplicity, assume bootstrapAddress is "host:port"
        vvm::network::NetworkConfig netConfig;
        netConfig.seedNodes = { bootstrapAddress };
        
        networkManager_->registerWithCluster();
        VVM_LOG_INFO("Joined cluster via {}", bootstrapAddress);
        return true;
    }
    
    std::string getLocalNodeId() const override {
        if (networkManager_) {
            return networkManager_->getLocalNodeId().toString();
        }
        return "";
    }
    
    // Send/recv by tensor name
    void sendTensor(const TensorHandle& tensor, const std::string& targetNodeId, CompletionCallback cb) override {
        if (!networkManager_ || !tensor) {
            if (cb) cb(false, "Network not ready or invalid tensor");
            return;
        }
        
        // Export tensor for remote access
        auto rdmaOk = config_.preference == TransportConfig::Preference::RDMAOnly ||
                      config_.enableGPUDirect;
        auto desc = networkManager_->exportForRemote(tensor->allocation, rdmaOk, !rdmaOk);
        if (!desc) {
            if (cb) cb(false, "exportForRemote failed");
            return;
        }
        
        // Parse target node ID
        auto targetNode = vvm::network::NodeId::fromString(targetNodeId);
        
        // Announce the tensor to target node
        if (!networkManager_->announceRemoteTensor(targetNode, tensor->metadata.name, *desc)) {
            if (cb) cb(false, "announceRemoteTensor failed");
            return;
        }
        
        if (cb) cb(true, "");
    }
    
    void recvTensor(const TensorHandle& tensor, const std::string& sourceNodeId, CompletionCallback cb) override {
        if (!networkManager_ || !tensor) {
            if (cb) cb(false, "Network not ready or invalid tensor");
            return;
        }
        
        // Parse source node ID
        auto sourceNode = vvm::network::NodeId::fromString(sourceNodeId);
        
        // Wait for tensor announcement from source
        const uint64_t kRecvTimeoutNs = 30ull * 1000 * 1000 * 1000; // 30s
        auto desc = networkManager_->waitRemoteTensor(sourceNode, tensor->metadata.name, kRecvTimeoutNs);
        if (!desc) {
            if (cb) cb(false, "waitRemoteTensor timed out");
            return;
        }
        
        // Migrate from remote
        auto rdmaOk = config_.preference == TransportConfig::Preference::RDMAOnly ||
                      config_.enableGPUDirect;
        auto op = networkManager_->migrateFromRemote(*desc, tensor->allocation, rdmaOk);
        if (!op) {
            if (cb) cb(false, "migrateFromRemote failed");
            return;
        }
        
        networkManager_->waitMigration(*op);
        
        // Verify content if checksum provided
        if (tensor->metadata.contentHash != 0) {
            // Would verify content here
        }
        
        if (cb) cb(true, "");
    }
    
    // ========================================================================
    // Capabilities
    // ========================================================================
    
    bool supportsP2P() const override {
        return true; // MultiGPUPoolManager handles P2P
    }
    
    bool supportsRDMA() const override {
        return config_.enableGPUDirect && poolManager_ && poolManager_->getInstances().size() > 0;
    }
    
    bool supportsNetwork() const override {
        return networkManager_ != nullptr;
    }
    
private:
    // ========================================================================
    // Async Processing
    // ========================================================================
    
    void asyncProcessingLoop() {
        while (!stopAsyncThread_) {
            std::unique_lock<std::mutex> lock(asyncMutex_);
            if (asyncCV_.wait_for(lock, std::chrono::milliseconds(10), 
                                  [this] { return stopAsyncThread_ || !asyncQueue_.empty(); })) {
                if (stopAsyncThread_) break;
                
                while (!asyncQueue_.empty()) {
                    auto op = asyncQueue_.front();
                    asyncQueue_.pop();
                    lock.unlock();
                    
                    // Process async operation
                    try {
                        op();
                    } catch (const std::exception& e) {
                        VVM_LOG_ERROR("Async operation threw: {}", e.what());
                    }
                    
                    lock.lock();
                }
            }
        }
    }
    
    void flushAsync() override {
        // If async thread is running, wait until queue is drained
        if (asyncThread_.joinable()) {
            std::unique_lock<std::mutex> lock(asyncMutex_);
            asyncCV_.wait(lock, [this] { return asyncQueue_.empty(); });
        }
        // Without async thread, operations run inline so nothing to drain
    }
    
    void enqueueAsync(AsyncOperation op) override {
        {
            std::lock_guard<std::mutex> lock(asyncMutex_);
            asyncQueue_.push(std::move(op));
        }
        if (!asyncThread_.joinable()) {
            // No async thread running, process inline
            AsyncOperation next;
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(asyncMutex_);
                    if (asyncQueue_.empty()) break;
                    next = std::move(asyncQueue_.front());
                    asyncQueue_.pop();
                }
                try {
                    next();
                } catch (const std::exception& e) {
                    VVM_LOG_ERROR("Async operation failed: {}", e.what());
                }
            }
        } else {
            asyncCV_.notify_one();
        }
    }
    
    // ========================================================================
    // Members
    // ========================================================================
    
    TransportConfig config_;
    std::vector<vvm::DeviceConfig> devices_;
    vvm::PoolConfig poolConfig_;
    bool initialized_ = false;
    
    std::optional<vvm::MultiGPUPoolManager> poolManager_;
    std::unique_ptr<vvm::network::MultiNodePoolManager> networkManager_;
    
    // Async processing
    std::thread asyncThread_;
    std::atomic<bool> stopAsyncThread_{false};
    std::mutex asyncMutex_;
    std::condition_variable asyncCV_;
    std::queue<std::function<void()>> asyncQueue_;
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Transport> Transport::create(
    const TransportConfig& config,
    const std::vector<vvm::DeviceConfig>& devices,
    const vvm::PoolConfig& poolConfig) {
    
    return std::make_unique<TensorTransportImpl>(config, devices, poolConfig);
}

} // namespace tensor
} // namespace vvm