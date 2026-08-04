#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <cstdio>

namespace vvm {

// ============================================================================
// Vulkan Result Conversion
// ============================================================================

Result makeResult(VkResult vkResult, const char* operation) {
    if (vkResult == VK_SUCCESS) return Result::success();
    
    ErrorCode code = ErrorCode::VulkanError;
    switch (vkResult) {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        case VK_ERROR_OUT_OF_POOL_MEMORY:
            code = ErrorCode::OutOfMemory;
            break;
        case VK_ERROR_FEATURE_NOT_PRESENT:
        case VK_ERROR_EXTENSION_NOT_PRESENT:
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            code = ErrorCode::UnsupportedFeature;
            break;
        case VK_ERROR_FRAGMENTED_POOL:
        case VK_ERROR_FRAGMENTATION:
            code = ErrorCode::AllocationFailed;
            break;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            code = ErrorCode::ExportFailed;
            break;
        default:
            break;
    }
    
    return Result::error(code, 
        std::string(operation) + " failed: " + vkResultToString(vkResult), 
        vkResult);
}

void checkVk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        auto err = makeResult(result, operation);
        VVM_LOG_ERROR("%s", err.message.c_str());
        // In production, might throw or use expected<T>
    }
}

// ============================================================================
// Validation Helpers
// ============================================================================

bool validateDevice(const DeviceConfig& config) {
    if (!config.physicalDevice || !config.device) return false;
    if (config.graphicsQueueFamily == UINT32_MAX) return false;
    if (!config.graphicsQueue) return false;
    return true;
}

bool validatePoolConfig(const PoolConfig& config) {
    if (config.blockSize < 1024 * 1024) return false;  // At least 1MB
    if (config.minAlignment < 4096) return false;      // At least page size
    if (config.maxBlocks == 0) return false;
    return true;
}

// ============================================================================
// Debug Utilities
// ============================================================================

void printMemoryTypes(const VkPhysicalDeviceMemoryProperties& props) {
    VVM_LOG_INFO("Memory Heaps (%u):", props.memoryHeapCount);
    for (uint32_t i = 0; i < props.memoryHeapCount; ++i) {
        const auto& heap = props.memoryHeaps[i];
        VVM_LOG_INFO("  Heap %u: %zu MB, flags: 0x%x", i, heap.size / (1024*1024), heap.flags);
    }
    
    VVM_LOG_INFO("Memory Types (%u):", props.memoryTypeCount);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        const auto& type = props.memoryTypes[i];
        VVM_LOG_INFO("  Type %u: heap=%u, flags=0x%x", i, type.heapIndex, type.propertyFlags);
    }
}

void printQueueFamilies(VkPhysicalDevice physicalDevice) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    
    VVM_LOG_INFO("Queue Families (%u):", count);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& fam = families[i];
        VVM_LOG_INFO("  Family %u: count=%u, flags=0x%x", i, fam.queueCount, fam.queueFlags);
    }
}

void printDeviceProperties(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    
    VVM_LOG_INFO("Device: %s", props.deviceName);
    VVM_LOG_INFO("  Type: %u, API: %u.%u.%u, Driver: %u", 
                 props.deviceType,
                 VK_API_VERSION_MAJOR(props.apiVersion),
                 VK_API_VERSION_MINOR(props.apiVersion),
                 VK_API_VERSION_PATCH(props.apiVersion),
                 props.driverVersion);
    VVM_LOG_INFO("  Vendor: 0x%04x, Device: 0x%04x", props.vendorID, props.deviceID);
    VVM_LOG_INFO("  Limits: maxMemAlloc=%zu MB, bufferAlignment=%u",
                 props.limits.maxMemoryAllocationCount / (1024*1024),
                 props.limits.minUniformBufferOffsetAlignment);
}

} // namespace vvm