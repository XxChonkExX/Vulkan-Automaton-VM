#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include <vector>
#include <string>
#include <optional>
#include <mutex>
#include <chrono>
#include <functional>
#include <cstdio>
#include <cstdarg>
#include <sstream>
#include <string_view>
#include <tuple>
#include <cstring>
#include <type_traits>

namespace vvm {

// ============================================================================
// Vulkan Utility Functions
// ============================================================================

// Queue family selection
struct QueueFamilies {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> compute;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> present;
};

QueueFamilies findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface = VK_NULL_HANDLE);

// Memory type finding
std::optional<uint32_t> findMemoryTypeIndex(const VkPhysicalDeviceMemoryProperties& memProps,
                                             VkMemoryPropertyFlags required,
                                             VkMemoryPropertyFlags preferred = 0);

void getMemoryTypeProperties(uint32_t memoryTypeIndex, 
                             VkMemoryPropertyFlags& flags,
                             const VkPhysicalDeviceMemoryProperties& memProps);

// Device selection
struct DeviceScore {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    int score = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memProps{};
    bool discrete = false;
    bool integrated = false;
    uint32_t vendorID = 0;
    uint32_t deviceID = 0;
};

std::vector<DeviceScore> enumerateDevices(VkInstance instance);
std::optional<DeviceScore> selectBestDevice(const std::vector<DeviceScore>& devices,
                                             bool preferDiscrete = true,
                                             uint32_t minHeapSizeMB = 0);

// Format helpers
VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice,
                              const std::vector<VkFormat>& candidates,
                              VkImageTiling tiling,
                              VkFormatFeatureFlags features);

// Extension checking
bool checkDeviceExtensionSupport(VkPhysicalDevice device, 
                                  const std::vector<const char*>& required);
bool checkInstanceExtensionSupport(const std::vector<const char*>& required);

// Memory type selection with budget awareness
struct MemoryTypeSelector {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    bool hasBudgetExt = false;
    
    explicit MemoryTypeSelector(VkPhysicalDevice pd = VK_NULL_HANDLE);
    void refresh();
    
    // Find best memory type considering budget, heap, and flags
    struct SelectionResult {
        uint32_t memoryTypeIndex = UINT32_MAX;
        VkDeviceSize heapBudget = 0;
        VkDeviceSize heapUsage = 0;
        float heapUtilization = 0.0f;
    };
    
    SelectionResult select(VkMemoryPropertyFlags required,
                           VkMemoryPropertyFlags preferred = 0,
                           VkDeviceSize minHeapBudget = 0) const;
    
    // Get dedicated allocation requirements
    DedicatedAllocationInfo getDedicatedAllocationInfo(
        VkBufferCreateInfo* bufferInfo,
        VkImageCreateInfo* imageInfo = nullptr) const;
};

inline VkDeviceSize getHeapBudget(const VkPhysicalDeviceMemoryBudgetPropertiesEXT& budget, uint32_t heapIndex) {
    return budget.heapBudget[heapIndex];
}

inline VkDeviceSize getHeapUsage(const VkPhysicalDeviceMemoryBudgetPropertiesEXT& budget, uint32_t heapIndex) {
    return budget.heapUsage[heapIndex];
}

// Debug helpers
void printMemoryTypes(const VkPhysicalDeviceMemoryProperties& props);
void printQueueFamilies(VkPhysicalDevice physicalDevice);
void printDeviceProperties(VkPhysicalDevice physicalDevice);

// Debug utils
std::string vkResultToString(VkResult result);
std::string vkErrorToString(VkResult result);

void setDebugName(VkDevice device, VkObjectType type, uint64_t handle, const char* name);

// ============================================================================
// Synchronization Helpers
// ============================================================================

class FencePool {
public:
    FencePool(VkDevice device, uint32_t initialSize = 8);
    ~FencePool();
    
    VkFence acquire();
    void release(VkFence fence);
    void resetAll();
    
private:
    VkDevice device_;
    std::vector<VkFence> available_;
    std::vector<VkFence> inUse_;
    std::mutex mutex_;
};

class SemaphorePool {
public:
    SemaphorePool(VkDevice device, bool timeline = false, uint32_t initialSize = 8);
    ~SemaphorePool();
    
    VkSemaphore acquire(uint64_t initialValue = 0);
    void release(VkSemaphore semaphore);
    
private:
    VkDevice device_;
    bool timeline_;
    std::vector<VkSemaphore> available_;
    std::vector<VkSemaphore> inUse_;
    std::mutex mutex_;
};

// ============================================================================
// Command Buffer Helpers
// ============================================================================

class CommandBufferPool {
public:
    CommandBufferPool(VkDevice device, uint32_t queueFamily, 
                       VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                       uint32_t initialSize = 16);
    ~CommandBufferPool();
    
    VkCommandBuffer acquire(bool begin = true);
    void release(VkCommandBuffer cmdBuffer, bool end = true);
    void resetAll();
    
private:
    VkDevice device_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBufferLevel level_;
    std::vector<VkCommandBuffer> available_;
    std::vector<VkCommandBuffer> inUse_;
    std::mutex mutex_;
};

VkCommandBuffer beginSingleTimeCommands(VkDevice device, VkCommandPool pool);
void endSingleTimeCommands(VkDevice device, VkCommandPool pool, VkCommandBuffer cmdBuffer, VkQueue queue);

// ============================================================================
// Profiling / Timing
// ============================================================================

class GpuTimer {
public:
    GpuTimer(VkDevice device, VkQueue queue, uint32_t queueFamily, uint32_t maxFrames = 16);
    ~GpuTimer();
    
    void beginFrame();
    void endFrame();
    
    struct Timestamp {
        const char* name;
        uint64_t start;
        uint64_t end;
        double ms;
    };
    
    void timestamp(const char* name);
    std::vector<Timestamp> getResults();
    void reset();
    
private:
    VkDevice device_;
    VkQueue queue_;
    VkQueryPool queryPool_ = VK_NULL_HANDLE;
    uint32_t maxFrames_;
    uint32_t currentFrame_ = 0;
    uint32_t queryCount_ = 0;
    std::vector<std::pair<const char*, uint32_t>> timestamps_;
};

// ============================================================================
// Resource Management
// ============================================================================

template<typename T>
class ResourceCache {
public:
    using Creator = std::function<std::optional<T>()>;
    using Destructor = std::function<void(T&)>;
    
    ResourceCache(Creator create, Destructor destroy)
        : create_(std::move(create)), destroy_(std::move(destroy)) {}
    
    ~ResourceCache() { clear(); }
    
    std::optional<T> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_.empty()) {
            T resource = std::move(available_.back());
            available_.pop_back();
            return resource;
        }
        return create_();
    }
    
    void release(T&& resource) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_.push_back(std::move(resource));
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& res : available_) {
            destroy_(res);
        }
        available_.clear();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return available_.size();
    }
    
private:
    Creator create_;
    Destructor destroy_;
    std::vector<T> available_;
    mutable std::mutex mutex_;
};

// ============================================================================
// Logging
// ============================================================================

enum class LogLevel { Trace, Debug, Info, Warning, Error };

namespace detail {

// Render a single argument for {} substitution.
inline std::string logArgToString(const std::string& v) { return v; }
inline std::string logArgToString(const char* v) { return v ? v : "(null)"; }
inline std::string logArgToString(char* v) { return v ? v : "(null)"; }
inline std::string logArgToString(bool v) { return v ? "true" : "false"; }
inline std::string logArgToString(char v) { return std::string(1, v); }
template <typename T>
std::string logArgToString(const T& v) {
    if constexpr (std::is_integral_v<T>) {
        return std::to_string(v);
    } else if constexpr (std::is_floating_point_v<T>) {
        std::ostringstream os;
        os << v;
        return os.str();
    } else if constexpr (std::is_pointer_v<T>) {
        std::ostringstream os;
        os << static_cast<const void*>(v);
        return os.str();
    } else {
        std::ostringstream os;
        os << v;
        return os.str();
    }
}

// Substitute {} placeholders with the rendered arguments.
// Falls back to the raw format string when argument count mismatches.
template <typename... Args>
std::string logFormat(const char* fmt, Args&&... args) {
    if (fmt == nullptr) return "";
    std::string_view fv(fmt);
    std::vector<std::string> rendered;
    rendered.reserve(sizeof...(Args));
    (rendered.push_back(logArgToString(std::forward<Args>(args))), ...);

    std::string out;
    out.reserve(fv.size());
    size_t argIndex = 0;
    for (size_t i = 0; i < fv.size(); ++i) {
        if (i + 1 < fv.size() && fv[i] == '{' && fv[i + 1] == '}') {
            if (argIndex < rendered.size()) {
                out += rendered[argIndex++];
            } else {
                out += "{}";
            }
            ++i;
        } else {
            out += fv[i];
        }
    }
    return out;
}

}  // namespace detail

class Logger {
public:
    static Logger& instance();
    
    void setLevel(LogLevel level) { level_ = level; }
    void setCallback(std::function<void(LogLevel, const std::string&)> cb) { callback_ = std::move(cb); }

    // Supports both printf-style (%d, %s, ...) and {}-style formats.
    template<typename... Args>
    void log(LogLevel lvl, const char* fmt, Args&&... args) {
        if (lvl < level_) return;

        std::string msg;
        if (fmt != nullptr && std::strstr(fmt, "{}") != nullptr) {
            msg = detail::logFormat(fmt, std::forward<Args>(args)...);
        } else {
            char buffer[1024];
            int len = snprintf(buffer, sizeof(buffer), fmt ? fmt : "", std::forward<Args>(args)...);
            if (len < 0) return;
            msg.assign(buffer, static_cast<size_t>(len));
        }

        if (callback_) {
            callback_(lvl, msg);
        } else {
            static const char* levelStr[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR"};
            fprintf(stderr, "[%s] %s\n", levelStr[static_cast<int>(lvl)], msg.c_str());
        }
    }

private:
    Logger() = default;
    LogLevel level_ = LogLevel::Info;
    std::function<void(LogLevel, const std::string&)> callback_;
};

#define VVM_LOG_TRACE(...)  vvm::Logger::instance().log(vvm::LogLevel::Trace, __VA_ARGS__)
#define VVM_LOG_DEBUG(...)  vvm::Logger::instance().log(vvm::LogLevel::Debug, __VA_ARGS__)
#define VVM_LOG_INFO(...)   vvm::Logger::instance().log(vvm::LogLevel::Info, __VA_ARGS__)
#define VVM_LOG_WARN(...)   vvm::Logger::instance().log(vvm::LogLevel::Warning, __VA_ARGS__)
#define VVM_LOG_ERROR(...)  vvm::Logger::instance().log(vvm::LogLevel::Error, __VA_ARGS__)

} // namespace vvm