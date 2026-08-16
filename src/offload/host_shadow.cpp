#include "vulkan_vm/offload.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <unordered_map>

#ifdef VVM_PLATFORM_LINUX
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace vvm {

// ============================================================================
// HostShadowManager Implementation
// ============================================================================

HostShadowManager::HostShadowManager(VkPhysicalDevice physicalDevice, VkDevice device, const OffloadConfig& config)
    : physicalDevice_(physicalDevice), device_(device), config_(config) {
    VVM_LOG_INFO("HostShadowManager: physicalDevice={}, device={}, hostShadowSize={}",
                 physicalDevice, device, config.hostShadowSize);
    createShadowBuffer();
    VVM_LOG_INFO("HostShadowManager: shadow buffer created, size={}", shadowBuffer_.size);
    
    // Create command pool for copy operations using the transfer queue family
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = config.transferQueueFamily;  // Use the configured transfer queue family
    VkResult res = vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create HostShadowManager command pool: {}", vkResultToString(res));
    }
}

HostShadowManager::~HostShadowManager() {
    destroyShadowBuffer();
    if (cmdPool_) {
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
    }
}

bool HostShadowManager::createShadowBuffer() {
    VVM_LOG_INFO("createShadowBuffer: allocating buffer of size {}", config_.hostShadowSize);
    // A zero-sized shadow is a valid "disabled" state (APU/forAPU sets
    // hostShadowMultiplier=0). Creating and binding a 0-size buffer hangs
    // the Vulkan driver (observed on gfx1151 / RADV) and serves no purpose.
    if (config_.hostShadowSize == 0) {
        VVM_LOG_INFO("hostShadowSize=0, shadow buffer disabled");
        shadowBuffer_.size = 0;
        return true;
    }
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = config_.hostShadowSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult res = vkCreateBuffer(device_, &bufferInfo, nullptr, &shadowBuffer_.buffer);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create shadow buffer: {}", vkResultToString(res));
        return false;
    }
    VVM_LOG_INFO("Shadow buffer created: {}", shadowBuffer_.buffer);
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, shadowBuffer_.buffer, &memReq);
    VVM_LOG_INFO("Memory requirements: size={}, alignment={}, memoryTypeBits={}", 
                 memReq.size, memReq.alignment, memReq.memoryTypeBits);
    
    // Find HOST_VISIBLE | HOST_COHERENT memory
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    
    uint32_t memTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((memReq.memoryTypeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & 
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memTypeIndex = i;
            break;
        }
    }
    
    if (memTypeIndex == UINT32_MAX) {
        VVM_LOG_ERROR("No HOST_VISIBLE|HOST_COHERENT memory type found");
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    VVM_LOG_INFO("Found memory type index {}", memTypeIndex);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &shadowBuffer_.memory) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to allocate memory for shadow buffer");
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    VVM_LOG_INFO("Shadow memory allocated: {}", shadowBuffer_.memory);
    
    VkResult bindRes = vkBindBufferMemory(device_, shadowBuffer_.buffer, shadowBuffer_.memory, 0);
    if (bindRes != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to bind buffer memory: {}", vkResultToString(bindRes));
        vkFreeMemory(device_, shadowBuffer_.memory, nullptr);
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    VVM_LOG_INFO("Buffer memory bound successfully");
    
    VkResult mapRes = vkMapMemory(device_, shadowBuffer_.memory, 0, VK_WHOLE_SIZE, 0, &shadowBuffer_.mappedPtr);
    if (mapRes != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to map shadow memory: {}", vkResultToString(mapRes));
        vkUnmapMemory(device_, shadowBuffer_.memory);
        vkFreeMemory(device_, shadowBuffer_.memory, nullptr);
        shadowBuffer_.memory = VK_NULL_HANDLE;
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        shadowBuffer_.buffer = VK_NULL_HANDLE;
        return false;
    }
    VVM_LOG_INFO("Shadow memory mapped at {}", shadowBuffer_.mappedPtr);
    shadowBuffer_.size = memReq.size;
    shadowBuffer_.freeRanges.emplace_back(0, memReq.size);
    
    return true;
}

void HostShadowManager::destroyShadowBuffer() {
    if (shadowBuffer_.memory) {
        if (shadowBuffer_.mappedPtr) {
            vkUnmapMemory(device_, shadowBuffer_.memory);
        }
        vkFreeMemory(device_, shadowBuffer_.memory, nullptr);
        shadowBuffer_.memory = VK_NULL_HANDLE;
    }
    if (shadowBuffer_.buffer) {
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        shadowBuffer_.buffer = VK_NULL_HANDLE;
    }
}

VkDeviceSize HostShadowManager::alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    if ((alignment & (alignment - 1)) != 0) return value;
    VkDeviceSize remainder = value & (alignment - 1);
    if (remainder == 0) return value;
    VkDeviceSize delta = alignment - remainder;
    if (value > std::numeric_limits<VkDeviceSize>::max() - delta) return value;
    return value + delta;
}

std::optional<VkDeviceSize> HostShadowManager::allocateRegion(VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    size = alignUp(size, 4096);  // Page align for madvise/mprotect
    
    for (auto it = shadowBuffer_.freeRanges.begin(); it != shadowBuffer_.freeRanges.end(); ++it) {
        if (it->second >= size) {
            VkDeviceSize offset = it->first;
            it->first += size;
            it->second -= size;
            
            if (it->second == 0) {
                shadowBuffer_.freeRanges.erase(it);
            }
            
            shadowBuffer_.used += size;
            return offset;
        }
    }
    return std::nullopt;
}

void HostShadowManager::freeRegion(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    size = alignUp(size, 4096);

    // Guard against underflow from double-free or mismatched size
    if (shadowBuffer_.used >= size) {
        shadowBuffer_.used -= size;
    } else {
        VVM_LOG_ERROR("HostShadowManager::freeRegion: used underflow "
                      "(used={}, free_size={}, offset={})",
                      shadowBuffer_.used, size, offset);
        shadowBuffer_.used = 0;
    }

    shadowBuffer_.freeRanges.emplace_back(offset, size);

    // Merge adjacent ranges
    std::sort(shadowBuffer_.freeRanges.begin(), shadowBuffer_.freeRanges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> merged;
    for (const auto& range : shadowBuffer_.freeRanges) {
        if (!merged.empty() && merged.back().first + merged.back().second == range.first) {
            merged.back().second += range.second;
        } else {
            merged.push_back(range);
        }
    }
    shadowBuffer_.freeRanges = std::move(merged);
}

void* HostShadowManager::mapRegion(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!shadowBuffer_.mappedPtr) return nullptr;
    return static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
}

void HostShadowManager::unmapRegion(VkDeviceSize offset, VkDeviceSize size) {
    // Host coherent - no explicit flush needed
    (void)offset;
    (void)size;
}

#ifdef VVM_PLATFORM_LINUX
void HostShadowManager::adviseDontNeed(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_DONTNEED);
}

void HostShadowManager::adviseWillNeed(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_WILLNEED);
}

void HostShadowManager::adviseFree(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_FREE);
}

void HostShadowManager::protectRegion(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    mprotect(ptr, size, PROT_NONE);
}

void HostShadowManager::unprotectRegion(VkDeviceSize offset, VkDeviceSize size) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    mprotect(ptr, size, PROT_READ | PROT_WRITE);
}
#else
void HostShadowManager::adviseDontNeed(VkDeviceSize, VkDeviceSize) {}
void HostShadowManager::adviseWillNeed(VkDeviceSize, VkDeviceSize) {}
void HostShadowManager::adviseFree(VkDeviceSize, VkDeviceSize) {}
void HostShadowManager::protectRegion(VkDeviceSize, VkDeviceSize) {}
void HostShadowManager::unprotectRegion(VkDeviceSize, VkDeviceSize) {}
#endif

// ============================================================================
// MigrationEngine Implementation
// ============================================================================

MigrationEngine::MigrationEngine(VkDevice device, VkQueue transferQueue, 
                                  uint32_t queueFamily, uint32_t maxConcurrent)
    : device_(device), transferQueue_(transferQueue), queueFamily_(queueFamily),
      maxConcurrent_(maxConcurrent) {
    
    contexts_.resize(maxConcurrent_);
    
    // Create timeline semaphore
    VkSemaphoreTypeCreateInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineInfo.initialValue = 0;
    
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semInfo.pNext = &timelineInfo;
    vkCreateSemaphore(device_, &semInfo, nullptr, &timelineSemaphore_);
    
    // Command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamily_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);
    
    // Shared fence for context recycling
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device_, &fenceInfo, nullptr, &contextFence_);
    
    // Initialize contexts
    for (auto& ctx : contexts_) {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool_;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device_, &allocInfo, &ctx.cmdBuffer);
        
        VkFenceCreateInfo fInfo{};
        fInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(device_, &fInfo, nullptr, &ctx.fence);
    }
}

MigrationEngine::~MigrationEngine() {
    waitIdle();
    
    for (auto& ctx : contexts_) {
        if (ctx.cmdBuffer) vkFreeCommandBuffers(device_, cmdPool_, 1, &ctx.cmdBuffer);
        if (ctx.fence) vkDestroyFence(device_, ctx.fence, nullptr);
        if (ctx.signalSemaphore) vkDestroySemaphore(device_, ctx.signalSemaphore, nullptr);
        if (ctx.waitSemaphore) vkDestroySemaphore(device_, ctx.waitSemaphore, nullptr);
    }
    
    if (timelineSemaphore_) vkDestroySemaphore(device_, timelineSemaphore_, nullptr);
    if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
    if (contextFence_) vkDestroyFence(device_, contextFence_, nullptr);
}

std::optional<MigrationContext*> MigrationEngine::acquireContext() {
    std::lock_guard<std::mutex> lock(contextMutex_);
    for (uint32_t i = 0; i < maxConcurrent_; ++i) {
        uint32_t idx = (nextContext_ + i) % maxConcurrent_;
        auto& ctx = contexts_[idx];

        if (!ctx.inUse) {
            VkResult res = vkGetFenceStatus(device_, ctx.fence);
            if (res == VK_SUCCESS) {
                vkResetFences(device_, 1, &ctx.fence);
                vkResetCommandBuffer(ctx.cmdBuffer, 0);
                ctx.inUse = true;
                nextContext_ = (idx + 1) % maxConcurrent_;
                return &ctx;
            }
        }
    }
    return std::nullopt;
}

void MigrationEngine::releaseContext(MigrationContext* ctx) {
    if (!ctx) return;
    ctx->inUse = false;
    ctx->timelineValue = 0;
}

bool MigrationEngine::submitCopy(MigrationContext* ctx, const MigrationRequest& req) {
    VVM_LOG_INFO("submitCopy: ctx={}, toHost={}, size={}, deviceBuf={}, hostBuf={}, srcOffset={}, dstOffset={}",
                 ctx, req.toHost, req.size, req.allocation ? req.allocation->buffer : VK_NULL_HANDLE,
                 req.hostShadowBuffer, req.srcOffset, req.dstOffset);

    if (!ctx || !req.allocation || !req.allocation->buffer || !req.hostShadowBuffer) {
        VVM_LOG_ERROR("submitCopy: null parameter (ctx={}, allocation={}, hostShadowBuffer={})",
                      reinterpret_cast<const void*>(ctx),
                      req.allocation ? reinterpret_cast<const void*>(req.allocation->buffer) : nullptr,
                      reinterpret_cast<const void*>(req.hostShadowBuffer));
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult res = vkBeginCommandBuffer(ctx->cmdBuffer, &beginInfo);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("vkBeginCommandBuffer failed: {}", vkResultToString(res));
        return false;
    }

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = req.srcOffset;
    copyRegion.dstOffset = req.dstOffset;
    copyRegion.size = req.size;

    VkBuffer deviceBuf = req.allocation->buffer;
    VkBuffer hostBuf = req.hostShadowBuffer;

    if (req.toHost) {
        VVM_LOG_INFO("Copying device->host: src={}, dst={}, size={}", deviceBuf, hostBuf, req.size);
        vkCmdCopyBuffer(ctx->cmdBuffer, deviceBuf, hostBuf, 1, &copyRegion);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = hostBuf;
        barrier.offset = req.dstOffset;
        barrier.size = req.size;
        vkCmdPipelineBarrier(ctx->cmdBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    } else {
        VVM_LOG_INFO("Copying host->device: src={}, dst={}, size={}", hostBuf, deviceBuf, req.size);
        vkCmdCopyBuffer(ctx->cmdBuffer, hostBuf, deviceBuf, 1, &copyRegion);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = deviceBuf;
        barrier.offset = req.dstOffset;
        barrier.size = req.size;
        vkCmdPipelineBarrier(ctx->cmdBuffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    VkResult endRes = vkEndCommandBuffer(ctx->cmdBuffer);
    if (endRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkEndCommandBuffer failed: {}", vkResultToString(endRes));
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(contextMutex_);
        timelineValue_++;
        ctx->timelineValue = timelineValue_;
    }

    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timelineInfo.waitSemaphoreValueCount = 0;
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &ctx->timelineValue;

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx->cmdBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timelineSemaphore_;

    if (ctx->waitSemaphore) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &ctx->waitSemaphore;
        submitInfo.pWaitDstStageMask = &req.waitStage;
    }

    VkResult queueRes = vkQueueSubmit(transferQueue_, 1, &submitInfo, ctx->fence);
    if (queueRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkQueueSubmit failed: {}", vkResultToString(queueRes));
        return false;
    }

    VVM_LOG_INFO("vkQueueSubmit succeeded, fence={}", ctx->fence);
    return true;
}

std::optional<MigrationOperation> MigrationEngine::submitMigration(const MigrationRequest& req) {
    auto ctxOpt = acquireContext();
    if (!ctxOpt) return std::nullopt;

    MigrationContext* ctx = *ctxOpt;

    if (!submitCopy(ctx, req)) {
        releaseContext(ctx);
        return std::nullopt;
    }

    MigrationOperation op;
    op.id = nextOpId_++;
    op.allocation = req.allocation;
    op.toHost = req.toHost;
    op.completionFence = ctx->fence;
    op.signalSemaphore = timelineSemaphore_;
    op.waitSemaphore = ctx->waitSemaphore;
    op.owningContext = ctx;

    {
        std::lock_guard<std::mutex> lock(pendingOpsMutex_);
        pendingOps_[op.id] = op;
    }
    return op;
}

void MigrationEngine::waitMigration(const MigrationOperation& op) {
    if (op.completionFence) {
        vkWaitForFences(device_, 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
    if (op.onComplete) {
        op.onComplete();
    }
    if (op.owningContext) {
        releaseContext(static_cast<MigrationContext*>(op.owningContext));
    }
    removePendingOp(op.id);
}

bool MigrationEngine::pollMigration(const MigrationOperation& op) {
    if (!op.completionFence) {
        if (op.onComplete) op.onComplete();
        if (op.owningContext) releaseContext(static_cast<MigrationContext*>(op.owningContext));
        removePendingOp(op.id);
        return true;
    }
    if (vkGetFenceStatus(device_, op.completionFence) == VK_SUCCESS) {
        if (op.onComplete) op.onComplete();
        if (op.owningContext) releaseContext(static_cast<MigrationContext*>(op.owningContext));
        removePendingOp(op.id);
        return true;
    }
    return false;
}

void MigrationEngine::removePendingOp(MigrationId id) {
    std::lock_guard<std::mutex> lock(pendingOpsMutex_);
    pendingOps_.erase(id);
}

void MigrationEngine::flush() {
    // All submissions are immediate in this implementation
}

void MigrationEngine::waitIdle() {
    if (transferQueue_ != VK_NULL_HANDLE) {
        vkQueueWaitIdle(transferQueue_);
    }
    
    for (auto& ctx : contexts_) {
        if (ctx.inUse) {
            vkWaitForFences(device_, 1, &ctx.fence, VK_TRUE, UINT64_MAX);
            ctx.inUse = false;
            ctx.timelineValue = 0;
        }
    }
}

uint32_t MigrationEngine::getPendingCount() const {
    std::lock_guard<std::mutex> lock(pendingOpsMutex_);
    return static_cast<uint32_t>(pendingOps_.size());
}

// ============================================================================
// OffloadManager Implementation
// ============================================================================

OffloadManager::OffloadManager(UnifiedMemoryPool* pool, const OffloadConfig& config)
    : pool_(pool), config_(config),
      completionThread_([this]() { processCompletions(); }) {
    VVM_LOG_INFO("OffloadManager: pool={}, transferQueue={}, transferQueueFamily={}",
                 pool, config.transferQueue, config.transferQueueFamily);

    try {
        shadowManager_ = std::make_unique<HostShadowManager>(
            pool->getPhysicalDevice(),
            pool->getDevice(), config);
        VVM_LOG_INFO("HostShadowManager created, buffer={}, size={}",
                     shadowManager_->getBuffer(), shadowManager_->getSize());
    } catch (const std::exception& e) {
        VVM_LOG_ERROR("Exception creating HostShadowManager: {}", e.what());
        throw;
    } catch (...) {
        VVM_LOG_ERROR("Unknown exception creating HostShadowManager");
        throw;
    }

    VkQueue transferQueue = config.transferQueue;
    uint32_t transferQueueFamily = config.transferQueueFamily;

    VVM_LOG_INFO("Creating MigrationEngine with transferQueue={}, queueFamily={}",
                 transferQueue, transferQueueFamily);

    migrationEngine_ = std::make_unique<MigrationEngine>(
        pool->getDevice(), transferQueue, transferQueueFamily, 4);
    VVM_LOG_INFO("MigrationEngine created");
}

OffloadManager::~OffloadManager() {
    shutdown_.store(true, std::memory_order_release);
    threadShouldStop_.store(true, std::memory_order_release);
    completionCv_.notify_all();
    if (completionThread_.joinable()) {
        completionThread_.join();
    }
    waitAll();
}

void OffloadManager::processCompletions() {
    while (!shutdown_.load(std::memory_order_acquire) && !threadShouldStop_.load(std::memory_order_acquire)) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(completionMutex_);
            completionCv_.wait(lock, [this] {
                return !completionQueue_.empty() || shutdown_.load(std::memory_order_acquire);
            });
            if (completionQueue_.empty()) continue;
            task = std::move(completionQueue_.front());
            completionQueue_.erase(completionQueue_.begin());
        }
        if (task) task();
    }
}

std::optional<MigrationOperation> OffloadManager::offload(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    return offloadLocked(alloc);
}

std::optional<MigrationOperation> OffloadManager::offloadLocked(Allocation& alloc) {
    VVM_LOG_INFO("offload: alloc={}, size={}, offset={}", 
                  &alloc, alloc.size, alloc.offset);
    
    // Allocate region in host shadow
    auto shadowOffset = shadowManager_->allocateRegion(alloc.size);
    if (!shadowOffset) {
        VVM_LOG_ERROR("Failed to allocate shadow region for offload (size={})", alloc.size);
        return std::nullopt;
    }
    
    VVM_LOG_INFO("Allocated shadow region at offset {} for size {}", *shadowOffset, alloc.size);
    
    // NOTE: We deliberately do NOT call mprotect(PROT_NONE) on the shadow
    // region. mprotect on memory mapped via vkMapMemory is undefined
    // behavior -- the Vulkan driver owns the underlying mmap, and changing
    // page protections can SIGSEGV the driver or corrupt GPU-side data.
    // (Config.useMprotect is now the default-false sentinel for user-
    // provided mmap'd regions only; see OffloadConfig docs.)
    
    // Submit migration
    MigrationEngine::MigrationRequest req;
    req.allocation = &alloc;
    req.srcOffset = alloc.offset;
    req.dstOffset = *shadowOffset;
    req.size = alloc.size;
    req.toHost = true;
    req.hostShadowBuffer = shadowManager_->getBuffer();
    
    VVM_LOG_INFO("Submitting migration to host: srcOffset={}, dstOffset={}, size={}, hostShadowBuffer={}", 
                  alloc.offset, *shadowOffset, alloc.size, shadowManager_->getBuffer());
    
    auto op = migrationEngine_->submitMigration(req);
    if (op) {
        VVM_LOG_INFO("Migration submitted successfully, op={}", &*op);
        // Mark allocation as offloaded and track the shadow offset for reload.
        alloc.shadowOffset = *shadowOffset;
        alloc.savedHostPtr = alloc.hostPtr;   // remember the pool mapping
        alloc.hostPtr = shadowManager_->mapRegion(*shadowOffset, alloc.size);
        
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesOffloaded += alloc.size;
        stats_.activeMigrations++;
    } else {
        VVM_LOG_ERROR("MigrationEngine::submitMigration returned nullopt");
        // Roll back the shadow region allocation.  Safe here because no
        // GPU work was submitted.
        shadowManager_->freeRegion(*shadowOffset, alloc.size);
    }
    
    return op;
}

std::optional<MigrationOperation> OffloadManager::reload(Allocation& alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    return reloadLocked(alloc);
}

std::optional<MigrationOperation> OffloadManager::reloadLocked(Allocation& alloc) {
    // Use the tracked shadow offset from the matching offload.
    if (alloc.shadowOffset == static_cast<VkDeviceSize>(-1)) {
        VVM_LOG_ERROR("reload: allocation is not currently offloaded (no shadow offset)");
        return std::nullopt;
    }
    VkDeviceSize shadowOffset = alloc.shadowOffset;

    // NOTE: No unprotectRegion() call here -- see offload() comment above.

    MigrationEngine::MigrationRequest req;
    req.allocation = &alloc;
    req.srcOffset = shadowOffset;
    req.dstOffset = alloc.offset;
    req.size = alloc.size;
    req.toHost = false;
    req.hostShadowBuffer = shadowManager_->getBuffer();

    auto op = migrationEngine_->submitMigration(req);
    if (op) {
        // Restore the original pool mapping that was saved during offload.
        alloc.hostPtr = alloc.savedHostPtr;
        alloc.savedHostPtr = nullptr;
        alloc.shadowOffset = static_cast<VkDeviceSize>(-1);
        shadowManager_->unmapRegion(shadowOffset, alloc.size);

        // Free the shadow region after GPU copy completes. The migration engine's
        // waitMigration releases the context once the fence signals; we hook the
        // cleanup into the operation's completion by wrapping the shadow free.
        VkDeviceSize regionSize = alloc.size;
        MigrationOperation* opPtr = &*op;
        opPtr->onComplete = [this, shadowOffset, regionSize]() {
            shadowManager_->freeRegion(shadowOffset, regionSize);
        };

        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesReloaded += alloc.size;
        stats_.activeMigrations++;
    }

    return op;
}

bool OffloadManager::offloadSync(Allocation& alloc, uint64_t timeoutNs) {
    std::lock_guard<std::mutex> lock(mutex_);
    VVM_LOG_DEBUG("offloadSync called: alloc={}, size={}, timeout={}", 
                  &alloc, alloc.size, timeoutNs);
    
    auto op = offloadLocked(alloc);
    if (!op) {
        VVM_LOG_ERROR("offload returned nullopt");
        return false;
    }
    
    return waitSync(*op, timeoutNs);
}

bool OffloadManager::reloadSync(Allocation& alloc, uint64_t timeoutNs) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto op = reloadLocked(alloc);
    if (!op) return false;
    
    return waitSync(*op, timeoutNs);
}

bool OffloadManager::waitSync(MigrationOperation& op, uint64_t timeoutNs) {
    if (timeoutNs == UINT64_MAX) {
        migrationEngine_->waitMigration(op);
        finishSync();
        return true;
    }
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::nanoseconds(timeoutNs);
    while (std::chrono::steady_clock::now() < deadline) {
        if (migrationEngine_->pollMigration(op)) {
            finishSync();
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    VVM_LOG_WARN("sync migration did not complete within timeout; finishing in background");
    {
        std::lock_guard<std::mutex> lock(completionMutex_);
        completionQueue_.push_back([this, op]() mutable {
            migrationEngine_->waitMigration(op);
            finishSync();
        });
    }
    completionCv_.notify_one();
    return false;
}

void OffloadManager::finishSync() {
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    stats_.activeMigrations--;
    stats_.completedMigrations++;
}

void OffloadManager::waitAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    migrationEngine_->waitIdle();
    
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    stats_.activeMigrations = 0;
}

OffloadManager::Stats OffloadManager::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    Stats s = stats_;
    s.activeMigrations = migrationEngine_->getPendingCount();
    return s;
}

void OffloadManager::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> statsLock(statsMutex_);
    stats_ = {};
}

} // namespace vvm