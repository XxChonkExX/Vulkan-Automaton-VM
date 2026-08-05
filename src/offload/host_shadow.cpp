#include "vulkan_vm/offload.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstring>

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
    createShadowBuffer();
    
    // Create command pool for copy operations
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = 0;  // Would be transfer queue family
    vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);
}

HostShadowManager::~HostShadowManager() {
    destroyShadowBuffer();
    if (cmdPool_) {
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
    }
}

bool HostShadowManager::createShadowBuffer() {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = config_.hostShadowSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &shadowBuffer_.buffer) != VK_SUCCESS) {
        return false;
    }
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, shadowBuffer_.buffer, &memReq);
    
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
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &shadowBuffer_.memory) != VK_SUCCESS) {
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    
    if (vkBindBufferMemory(device_, shadowBuffer_.buffer, shadowBuffer_.memory, 0) != VK_SUCCESS) {
        vkFreeMemory(device_, shadowBuffer_.memory, nullptr);
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    
    vkMapMemory(device_, shadowBuffer_.memory, 0, VK_WHOLE_SIZE, 0, &shadowBuffer_.mappedPtr);
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
    return (value + alignment - 1) & ~(alignment - 1);
}

std::optional<VkDeviceSize> HostShadowManager::allocateRegion(VkDeviceSize size) {
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
    size = alignUp(size, 4096);
    shadowBuffer_.freeRanges.emplace_back(offset, size);
    shadowBuffer_.used -= size;
    
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
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_DONTNEED);
}

void HostShadowManager::adviseWillNeed(VkDeviceSize offset, VkDeviceSize size) {
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_WILLNEED);
}

void HostShadowManager::adviseFree(VkDeviceSize offset, VkDeviceSize size) {
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    madvise(ptr, size, MADV_FREE);
}

void HostShadowManager::protectRegion(VkDeviceSize offset, VkDeviceSize size) {
    void* ptr = static_cast<char*>(shadowBuffer_.mappedPtr) + offset;
    mprotect(ptr, size, PROT_NONE);
}

void HostShadowManager::unprotectRegion(VkDeviceSize offset, VkDeviceSize size) {
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

void HostShadowManager::defragment() {
    // TODO: Implement compaction of host shadow buffer
}

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
    ctx->inUse = false;
    ctx->timelineValue = 0;
}

void MigrationEngine::submitCopy(MigrationContext* ctx, const MigrationRequest& req) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ctx->cmdBuffer, &beginInfo);
    
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = req.srcOffset;
    copyRegion.dstOffset = req.dstOffset;
    copyRegion.size = req.size;
    
    if (req.toHost) {
        // Device -> Host: src is allocation buffer, dst is shadow buffer
        vkCmdCopyBuffer(ctx->cmdBuffer, req.allocation->buffer, 
                       req.allocation->buffer, 1, &copyRegion);  // Simplified
    } else {
        // Host -> Device: src is shadow buffer, dst is allocation buffer
        vkCmdCopyBuffer(ctx->cmdBuffer, req.allocation->buffer,
                       req.allocation->buffer, 1, &copyRegion);  // Simplified
    }
    
    vkEndCommandBuffer(ctx->cmdBuffer);
    
    // Submit with timeline semaphore
    timelineValue_++;
    ctx->timelineValue = timelineValue_;
    
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
    
    if (ctx->signalSemaphore) {
        submitInfo.signalSemaphoreCount++;
        // Would need array for multiple semaphores
    }
    
    vkQueueSubmit(transferQueue_, 1, &submitInfo, ctx->fence);
}

std::optional<MigrationOperation> MigrationEngine::submitMigration(const MigrationRequest& req) {
    auto ctxOpt = acquireContext();
    if (!ctxOpt) return std::nullopt;
    
    MigrationContext* ctx = *ctxOpt;
    submitCopy(ctx, req);
    
    MigrationOperation op;
    op.allocation = req.allocation;
    op.toHost = req.toHost;
    op.completionFence = ctx->fence;
    op.signalSemaphore = timelineSemaphore_;
    op.waitSemaphore = ctx->waitSemaphore;
    
    pendingOps_.push_back(op);
    return op;
}

void MigrationEngine::waitMigration(const MigrationOperation& op) {
    if (op.completionFence) {
        vkWaitForFences(device_, 1, &op.completionFence, VK_TRUE, UINT64_MAX);
    }
}

bool MigrationEngine::pollMigration(const MigrationOperation& op) {
    if (!op.completionFence) return true;
    return vkGetFenceStatus(device_, op.completionFence) == VK_SUCCESS;
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
        }
    }
}

uint32_t MigrationEngine::getPendingCount() const {
    return static_cast<uint32_t>(pendingOps_.size());
}

// ============================================================================
// OffloadManager Implementation
// ============================================================================

OffloadManager::OffloadManager(UnifiedMemoryPool* pool, const OffloadConfig& config)
    : pool_(pool), config_(config) {
    shadowManager_ = std::make_unique<HostShadowManager>(
        pool->getPhysicalDevice(),
        pool->getDevice(), config);
    
    VkQueue transferQueue = config.transferQueue;
    uint32_t transferQueueFamily = config.transferQueueFamily;
    
    migrationEngine_ = std::make_unique<MigrationEngine>(
        pool->getDevice(), transferQueue, transferQueueFamily, 4);
}

OffloadManager::~OffloadManager() {
    waitAll();
}

std::optional<MigrationOperation> OffloadManager::offload(Allocation& alloc) {
    // Allocate region in host shadow
    auto shadowOffset = shadowManager_->allocateRegion(alloc.size);
    if (!shadowOffset) return std::nullopt;
    
    // Protect region (optional, for page fault detection)
    if (config_.useMprotect) {
        shadowManager_->protectRegion(*shadowOffset, alloc.size);
    }
    
    // Submit migration
    MigrationEngine::MigrationRequest req;
    req.allocation = &alloc;
    req.srcOffset = alloc.offset;
    req.dstOffset = *shadowOffset;
    req.size = alloc.size;
    req.toHost = true;
    
    auto op = migrationEngine_->submitMigration(req);
    if (op) {
        // Mark allocation as offloaded
        alloc.hostPtr = shadowManager_->mapRegion(*shadowOffset, alloc.size);
        
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesOffloaded += alloc.size;
        stats_.activeMigrations++;
    }
    
    return op;
}

std::optional<MigrationOperation> OffloadManager::reload(Allocation& alloc) {
    // Find shadow offset (would need tracking)
    // Simplified: assume we track it
    VkDeviceSize shadowOffset = 0;  // Would look up
    
    // Unprotect region
    if (config_.useMprotect) {
        shadowManager_->unprotectRegion(shadowOffset, alloc.size);
    }
    
    MigrationEngine::MigrationRequest req;
    req.allocation = &alloc;
    req.srcOffset = shadowOffset;
    req.dstOffset = alloc.offset;
    req.size = alloc.size;
    req.toHost = false;
    
    auto op = migrationEngine_->submitMigration(req);
    if (op) {
        alloc.hostPtr = nullptr;
        
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesReloaded += alloc.size;
        stats_.activeMigrations++;
    }
    
    return op;
}

bool OffloadManager::offloadSync(Allocation& alloc, uint64_t timeoutNs) {
    auto op = offload(alloc);
    if (!op) return false;
    
    VkResult res = vkWaitForFences(pool_->getDevice(), 1, &op->completionFence, 
                                   VK_TRUE, timeoutNs);
    
    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.activeMigrations--;
        stats_.completedMigrations++;
        return true;
    }
    return false;
}

bool OffloadManager::reloadSync(Allocation& alloc, uint64_t timeoutNs) {
    auto op = reload(alloc);
    if (!op) return false;
    
    VkResult res = vkWaitForFences(pool_->getDevice(), 1, &op->completionFence,
                                   VK_TRUE, timeoutNs);
    
    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.activeMigrations--;
        stats_.completedMigrations++;
        return true;
    }
    return false;
}

void OffloadManager::waitAll() {
    migrationEngine_->waitIdle();
    
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.activeMigrations = 0;
}

OffloadManager::Stats OffloadManager::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    Stats s = stats_;
    s.activeMigrations = migrationEngine_->getPendingCount();
    return s;
}

void OffloadManager::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_ = {};
}

} // namespace vvm