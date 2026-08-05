#include "vulkan_vm/offload.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <algorithm>
#include <cstring>

#ifdef VVM_PLATFORM_WINDOWS
#include <windows.h>
#endif

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
    char logPath[MAX_PATH];
    DWORD tempPathLen = GetTempPathA(MAX_PATH, logPath);
    if (tempPathLen == 0 || tempPathLen >= MAX_PATH) {
        strcpy_s(logPath, MAX_PATH, "C:\\temp\\");
    }
    strcat_s(logPath, MAX_PATH, "debug_host_shadow.log");
    FILE* f = fopen(logPath, "a");
    if (f) { fprintf(f, "=== HostShadowManager constructor START ===\n"); fclose(f); }
    else {
        // Try current directory
        FILE* f2 = fopen("debug_host_shadow.log", "a");
        if (f2) { fprintf(f2, "=== HostShadowManager constructor START (fallback) ===\n"); fclose(f2); }
    }
    VVM_LOG_INFO("HostShadowManager constructor: physicalDevice=%p, device=%p, hostShadowSize=%llu", 
                 physicalDevice, device, config.hostShadowSize);
    createShadowBuffer();
    VVM_LOG_INFO("createShadowBuffer completed, buffer=%p", shadowBuffer_.buffer);
    fflush(stderr);
    
    // Create command pool for copy operations using the transfer queue family
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | 
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = config.transferQueueFamily;  // Use the configured transfer queue family
    VkResult res = vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create HostShadowManager command pool: %s", vkResultToString(res).c_str());
    }
}

HostShadowManager::~HostShadowManager() {
    destroyShadowBuffer();
    if (cmdPool_) {
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
    }
}

bool HostShadowManager::createShadowBuffer() {
    VVM_LOG_INFO("createShadowBuffer: allocating buffer of size %llu", config_.hostShadowSize);
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = config_.hostShadowSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | 
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkResult res = vkCreateBuffer(device_, &bufferInfo, nullptr, &shadowBuffer_.buffer);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to create shadow buffer: %s", vkResultToString(res).c_str());
        return false;
    }
    VVM_LOG_INFO("Shadow buffer created: %p", shadowBuffer_.buffer);
    
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, shadowBuffer_.buffer, &memReq);
    VVM_LOG_INFO("Memory requirements: size=%llu, alignment=%llu, memoryTypeBits=%u", 
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
    VVM_LOG_INFO("Found memory type index %u", memTypeIndex);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIndex;
    
    if (vkAllocateMemory(device_, &allocInfo, nullptr, &shadowBuffer_.memory) != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to allocate memory for shadow buffer");
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    VVM_LOG_INFO("Shadow memory allocated: %p", shadowBuffer_.memory);
    
    VkResult bindRes = vkBindBufferMemory(device_, shadowBuffer_.buffer, shadowBuffer_.memory, 0);
    if (bindRes != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to bind buffer memory: %s", vkResultToString(bindRes).c_str());
        vkFreeMemory(device_, shadowBuffer_.memory, nullptr);
        vkDestroyBuffer(device_, shadowBuffer_.buffer, nullptr);
        return false;
    }
    VVM_LOG_INFO("Buffer memory bound successfully");
    
    VkResult mapRes = vkMapMemory(device_, shadowBuffer_.memory, 0, VK_WHOLE_SIZE, 0, &shadowBuffer_.mappedPtr);
    if (mapRes != VK_SUCCESS) {
        VVM_LOG_ERROR("Failed to map shadow memory: %s", vkResultToString(mapRes).c_str());
        return false;
    }
    VVM_LOG_INFO("Shadow memory mapped at %p", shadowBuffer_.mappedPtr);
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
    VVM_LOG_INFO("submitCopy: ctx=%p, toHost=%d, size=%llu, deviceBuf=%p, hostBuf=%p, srcOffset=%llu, dstOffset=%llu", 
                 ctx, req.toHost, req.size, req.allocation->buffer, req.hostShadowBuffer, req.srcOffset, req.dstOffset);
    
    if (!req.allocation || !req.allocation->buffer) {
        VVM_LOG_ERROR("Device allocation buffer is NULL!");
        return;
    }
    if (!req.hostShadowBuffer) {
        VVM_LOG_ERROR("Host shadow buffer is NULL!");
        return;
    }
    
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult res = vkBeginCommandBuffer(ctx->cmdBuffer, &beginInfo);
    if (res != VK_SUCCESS) {
        VVM_LOG_ERROR("vkBeginCommandBuffer failed: %s", vkResultToString(res).c_str());
        return;
    }
    
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = req.srcOffset;
    copyRegion.dstOffset = req.dstOffset;
    copyRegion.size = req.size;
    
    VkBuffer deviceBuf = req.allocation->buffer;
    VkBuffer hostBuf = req.hostShadowBuffer;
    
    if (req.toHost) {
        // Device -> Host: src is allocation buffer, dst is shadow buffer
        VVM_LOG_INFO("Copying device->host: src=%p, dst=%p, size=%llu", deviceBuf, hostBuf, req.size);
        vkCmdCopyBuffer(ctx->cmdBuffer, deviceBuf, hostBuf, 1, &copyRegion);
    } else {
        // Host -> Device: src is shadow buffer, dst is allocation buffer
        VVM_LOG_INFO("Copying host->device: src=%p, dst=%p, size=%llu", hostBuf, deviceBuf, req.size);
        vkCmdCopyBuffer(ctx->cmdBuffer, hostBuf, deviceBuf, 1, &copyRegion);
    }
    
    VkResult endRes = vkEndCommandBuffer(ctx->cmdBuffer);
    if (endRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkEndCommandBuffer failed: %s", vkResultToString(endRes).c_str());
        return;
    }
    
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
    
    VkResult queueRes = vkQueueSubmit(transferQueue_, 1, &submitInfo, ctx->fence);
    if (queueRes != VK_SUCCESS) {
        VVM_LOG_ERROR("vkQueueSubmit failed: %s", vkResultToString(queueRes).c_str());
    } else {
        VVM_LOG_INFO("vkQueueSubmit succeeded, fence=%p", ctx->fence);
    }
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
    char logPath[MAX_PATH];
    DWORD tempPathLen = GetTempPathA(MAX_PATH, logPath);
    if (tempPathLen == 0 || tempPathLen >= MAX_PATH) {
        strcpy_s(logPath, MAX_PATH, "C:\\temp\\");
    }
    strcat_s(logPath, MAX_PATH, "debug_offload.log");
    FILE* f = fopen(logPath, "a");
    if (!f) {
        f = fopen("debug_offload.log", "a");
    }
    if (f) { fprintf(f, "=== OffloadManager constructor START ===\n"); fclose(f); }
    VVM_LOG_INFO("OffloadManager constructor: pool=%p, transferQueue=%p, transferQueueFamily=%u", 
                 pool, config.transferQueue, config.transferQueueFamily);
    fflush(stderr);
    
    try {
        shadowManager_ = std::make_unique<HostShadowManager>(
            pool->getPhysicalDevice(),
            pool->getDevice(), config);
        VVM_LOG_INFO("HostShadowManager created, buffer=%p, size=%llu", 
                     shadowManager_->getBuffer(), shadowManager_->getSize());
    } catch (const std::exception& e) {
        VVM_LOG_ERROR("Exception creating HostShadowManager: %s", e.what());
        throw;
    } catch (...) {
        VVM_LOG_ERROR("Unknown exception creating HostShadowManager");
        throw;
    }
    
    VkQueue transferQueue = config.transferQueue;
    uint32_t transferQueueFamily = config.transferQueueFamily;
    
    VVM_LOG_INFO("Creating MigrationEngine with transferQueue=%p, queueFamily=%u", 
                 transferQueue, transferQueueFamily);
    fflush(stderr);
    
    migrationEngine_ = std::make_unique<MigrationEngine>(
        pool->getDevice(), transferQueue, transferQueueFamily, 4);
    VVM_LOG_INFO("MigrationEngine created");
}

OffloadManager::~OffloadManager() {
    waitAll();
}

std::optional<MigrationOperation> OffloadManager::offload(Allocation& alloc) {
    VVM_LOG_INFO("offload: alloc=%p, size=%llu, offset=%llu", 
                  &alloc, alloc.size, alloc.offset);
    
    // Allocate region in host shadow
    auto shadowOffset = shadowManager_->allocateRegion(alloc.size);
    if (!shadowOffset) {
        VVM_LOG_ERROR("Failed to allocate shadow region for offload (size=%llu)", alloc.size);
        return std::nullopt;
    }
    
    VVM_LOG_INFO("Allocated shadow region at offset %llu for size %llu", *shadowOffset, alloc.size);
    
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
    req.hostShadowBuffer = shadowManager_->getBuffer();
    
    VVM_LOG_INFO("Submitting migration to host: srcOffset=%llu, dstOffset=%llu, size=%llu, hostShadowBuffer=%p", 
                  alloc.offset, *shadowOffset, alloc.size, shadowManager_->getBuffer());
    
    auto op = migrationEngine_->submitMigration(req);
    if (op) {
        VVM_LOG_INFO("Migration submitted successfully, op=%p", &*op);
        // Mark allocation as offloaded
        alloc.hostPtr = shadowManager_->mapRegion(*shadowOffset, alloc.size);
        
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.bytesOffloaded += alloc.size;
        stats_.activeMigrations++;
    } else {
        VVM_LOG_ERROR("MigrationEngine::submitMigration returned nullopt");
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
    req.hostShadowBuffer = shadowManager_->getBuffer();
    
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
    VVM_LOG_DEBUG("offloadSync called: alloc=%p, size=%llu, timeout=%llu", 
                  &alloc, alloc.size, timeoutNs);
    
    auto op = offload(alloc);
    if (!op) {
        VVM_LOG_ERROR("offload returned nullopt");
        return false;
    }
    
    VkResult res = vkWaitForFences(pool_->getDevice(), 1, &op->completionFence, 
                                   VK_TRUE, timeoutNs);
    
    if (res == VK_SUCCESS) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.activeMigrations--;
        stats_.completedMigrations++;
        stats_.bytesOffloaded += alloc.size;
        return true;
    }
    
    VVM_LOG_ERROR("vkWaitForFences failed: %s", vkResultToString(res).c_str());
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