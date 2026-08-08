#pragma once

#include <vulkan/vulkan.h>

#ifdef VVM_PLATFORM_LINUX
#include <unistd.h>
#endif

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

// Find memory type index for importing external memory on a DESTINATION device.
// Given the source device's memory type index and required flags, this queries
// the destination device's memory properties to find a compatible type.
// This is required because memoryTypeIndex is NOT portable across devices.
std::optional<uint32_t> findImportMemoryTypeIndex(VkPhysicalDevice dstPhysicalDevice,
                                                   uint32_t srcMemoryTypeIndex,
                                                   VkMemoryPropertyFlags requiredFlags,
                                                   VkExternalMemoryHandleTypeFlagBits handleType);

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
struct MemoryTypeSelector;
struct DedicatedAllocationInfo;  // Forward declaration

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
// RAII Wrappers for Vulkan Handles
// ============================================================================

template<typename Handle, typename Deleter>
class UniqueHandle {
public:
    UniqueHandle() = default;
    UniqueHandle(Handle h, Deleter d) : handle_(h), deleter_(std::move(d)) {}
    
    UniqueHandle(UniqueHandle&& other) noexcept
        : handle_(other.handle_), deleter_(std::move(other.deleter_)) {
        other.handle_ = VK_NULL_HANDLE;
    }
    
    UniqueHandle& operator=(UniqueHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            deleter_ = std::move(other.deleter_);
            other.handle_ = VK_NULL_HANDLE;
        }
        return *this;
    }
    
    ~UniqueHandle() { reset(); }
    
    void reset() {
        if (handle_ != VK_NULL_HANDLE && deleter_) {
            deleter_(handle_);
            handle_ = VK_NULL_HANDLE;
        }
    }
    
    Handle get() const { return handle_; }
    explicit operator bool() const { return handle_ != VK_NULL_HANDLE; }
    Handle release() { Handle h = handle_; handle_ = VK_NULL_HANDLE; return h; }
    
private:
    Handle handle_ = VK_NULL_HANDLE;
    Deleter deleter_;
};

// Convenience aliases for common Vulkan handles
using UniqueDeviceMemory = UniqueHandle<VkDeviceMemory, std::function<void(VkDeviceMemory)>>;
using UniqueBuffer = UniqueHandle<VkBuffer, std::function<void(VkBuffer)>>;
using UniqueCommandPool = UniqueHandle<VkCommandPool, std::function<void(VkCommandPool)>>;
using UniqueFence = UniqueHandle<VkFence, std::function<void(VkFence)>>;
using UniqueSemaphore = UniqueHandle<VkSemaphore, std::function<void(VkSemaphore)>>;
using UniqueQueryPool = UniqueHandle<VkQueryPool, std::function<void(VkQueryPool)>>;

// Helper factories
inline UniqueDeviceMemory makeUniqueDeviceMemory(VkDevice device, VkDeviceMemory memory) {
    return UniqueDeviceMemory(memory, [device](VkDeviceMemory m) { vkFreeMemory(device, m, nullptr); });
}

inline UniqueBuffer makeUniqueBuffer(VkDevice device, VkBuffer buffer) {
    return UniqueBuffer(buffer, [device](VkBuffer b) { vkDestroyBuffer(device, b, nullptr); });
}

inline UniqueCommandPool makeUniqueCommandPool(VkDevice device, VkCommandPool pool) {
    return UniqueCommandPool(pool, [device](VkCommandPool p) { vkDestroyCommandPool(device, p, nullptr); });
}

inline UniqueFence makeUniqueFence(VkDevice device, VkFence fence) {
    return UniqueFence(fence, [device](VkFence f) { vkDestroyFence(device, f, nullptr); });
}

inline UniqueSemaphore makeUniqueSemaphore(VkDevice device, VkSemaphore semaphore) {
    return UniqueSemaphore(semaphore, [device](VkSemaphore s) { vkDestroySemaphore(device, s, nullptr); });
}

inline UniqueQueryPool makeUniqueQueryPool(VkDevice device, VkQueryPool pool) {
    return UniqueQueryPool(pool, [device](VkQueryPool p) { vkDestroyQueryPool(device, p, nullptr); });
}

// ============================================================================
// External Handle Types
// ============================================================================

enum class ExternalHandleType {
    OpaqueFd,              // Linux: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    OpaqueWin32,           // Windows: VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
    D3D12Heap,             // Windows: VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT
    DmaBuf,                // Linux: VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
    AndroidHardwareBuffer  // Android: VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID
};

// RAII wrapper for external memory handles (FD on Linux, HANDLE on Windows, AHardwareBuffer on Android)
class ExternalHandle {
public:
    ExternalHandle() = default;
    
    #ifdef VVM_PLATFORM_LINUX
    // Allow implicit construction from int (fd)
    ExternalHandle(int fd) : fd_(fd) {}
    ~ExternalHandle() { if (fd_ >= 0) close(fd_); }
    
    ExternalHandle(ExternalHandle&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
    ExternalHandle& operator=(ExternalHandle&& other) noexcept {
        if (this != &other) { if (fd_ >= 0) close(fd_); fd_ = other.fd_; other.fd_ = -1; }
        return *this;
    }
    
    int get() const { return fd_; }
    explicit operator bool() const { return fd_ >= 0; }
    int release() { int fd = fd_; fd_ = -1; return fd; }
    
    #elif defined(VVM_PLATFORM_WINDOWS)
    // Allow implicit construction from HANDLE
    ExternalHandle(HANDLE handle) : handle_(handle) {}
    ~ExternalHandle() { if (handle_) CloseHandle(handle_); }
    
    ExternalHandle(ExternalHandle&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    ExternalHandle& operator=(ExternalHandle&& other) noexcept {
        if (this != &other) { if (handle_) CloseHandle(handle_); handle_ = other.handle_; other.handle_ = nullptr; }
        return *this;
    }
    
    HANDLE get() const { return handle_; }
    explicit operator bool() const { return handle_ != nullptr; }
    HANDLE release() { HANDLE h = handle_; handle_ = nullptr; return h; }
    
    #elif defined(VVM_PLATFORM_ANDROID)
    // Android: AHardwareBuffer handle
    ExternalHandle(AHardwareBuffer* buffer) : hardwareBuffer_(buffer) {}
    ~ExternalHandle() { if (hardwareBuffer_) AHardwareBuffer_release(hardwareBuffer_); }
    
    ExternalHandle(ExternalHandle&& other) noexcept : hardwareBuffer_(other.hardwareBuffer_) { other.hardwareBuffer_ = nullptr; }
    ExternalHandle& operator=(ExternalHandle&& other) noexcept {
        if (this != &other) { if (hardwareBuffer_) AHardwareBuffer_release(hardwareBuffer_); hardwareBuffer_ = other.hardwareBuffer_; other.hardwareBuffer_ = nullptr; }
        return *this;
    }
    
    AHardwareBuffer* get() const { return hardwareBuffer_; }
    explicit operator bool() const { return hardwareBuffer_ != nullptr; }
    AHardwareBuffer* release() { AHardwareBuffer* h = hardwareBuffer_; hardwareBuffer_ = nullptr; return h; }
#endif
    
    private:
        #ifdef VVM_PLATFORM_LINUX
        int fd_ = -1;
        #elif defined(VVM_PLATFORM_WINDOWS)
        HANDLE handle_ = nullptr;
        #elif defined(VVM_PLATFORM_ANDROID)
        AHardwareBuffer* hardwareBuffer_ = nullptr;
        #endif
    };

// ============================================================================
// External Memory Info
// ============================================================================

struct ExternalMemoryInfo {
    ExternalHandleType type = ExternalHandleType::OpaqueFd;
    ExternalHandle handle;  // RAII wrapper for FD (Linux) or HANDLE (Windows)
    VkDeviceSize size = 0;
    uint32_t memoryTypeIndex = UINT32_MAX;
    bool dedicatedAllocation = false;

    // Move-only due to ExternalHandle member
    ExternalMemoryInfo() = default;
    ExternalMemoryInfo(ExternalMemoryInfo&& other) noexcept
        : type(other.type)
        , handle(std::move(other.handle))
        , size(other.size)
        , memoryTypeIndex(other.memoryTypeIndex)
        , dedicatedAllocation(other.dedicatedAllocation) {
        other.type = ExternalHandleType::OpaqueFd;
        other.size = 0;
        other.memoryTypeIndex = UINT32_MAX;
        other.dedicatedAllocation = false;
    }

    ExternalMemoryInfo& operator=(ExternalMemoryInfo&& other) noexcept {
        if (this != &other) {
            type = other.type;
            handle = std::move(other.handle);
            size = other.size;
            memoryTypeIndex = other.memoryTypeIndex;
            dedicatedAllocation = other.dedicatedAllocation;
            other.type = ExternalHandleType::OpaqueFd;
            other.size = 0;
            other.memoryTypeIndex = UINT32_MAX;
            other.dedicatedAllocation = false;
        }
        return *this;
    }

    ExternalMemoryInfo(const ExternalMemoryInfo&) = delete;
    ExternalMemoryInfo& operator=(const ExternalMemoryInfo&) = delete;
};

// Duplicate an exported handle so multiple peers can import the SAME memory
// independently. Vulkan consumes one handle per successful import, so N peers
// need N distinct handles. dup() on Linux, DuplicateHandle() on Windows.
// The returned ExternalMemoryInfo independently owns its OWN handle; the
// source stays untouched (its handle remains valid for the original exporter).
inline ExternalMemoryInfo duplicateForImport(const ExternalMemoryInfo& src) {
    ExternalMemoryInfo copy;
    copy.type = src.type;
    copy.size = src.size;
    copy.memoryTypeIndex = src.memoryTypeIndex;
    copy.dedicatedAllocation = src.dedicatedAllocation;
#ifdef VVM_PLATFORM_LINUX
    if (src.handle) {
        int dupFd = dup(src.handle.get());
        if (dupFd >= 0) copy.handle = ExternalHandle(dupFd);
    }
#elif defined(VVM_PLATFORM_WINDOWS)
    if (src.handle) {
        HANDLE dupHandle = nullptr;
        if (DuplicateHandle(GetCurrentProcess(), src.handle.get(),
                            GetCurrentProcess(), &dupHandle, 0, FALSE,
                            DUPLICATE_SAME_ACCESS)) {
            copy.handle = ExternalHandle(dupHandle);
        }
    }
#elif defined(VVM_PLATFORM_ANDROID)
    if (src.handle) {
        // AHardwareBuffer doesn't need explicit duplication - it's reference counted
        // Just retain a new reference
        copy.handle = src.handle;
    }
#endif
    return copy;
}

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

// ============================================================================
// Serialization helpers (used by network transport)
// ============================================================================

namespace detail {

inline void putU8(std::vector<uint8_t>& b, uint8_t v) { b.push_back(v); }

inline void putU16(std::vector<uint8_t>& b, uint16_t v) {
    b.push_back(static_cast<uint8_t>(v & 0xFF));
    b.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

inline void putU32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void putU64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

inline void putStr(std::vector<uint8_t>& b, const std::string& s) {
    putU32(b, static_cast<uint32_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}

inline void putBytes(std::vector<uint8_t>& b, const std::vector<uint8_t>& data) {
    putU32(b, static_cast<uint32_t>(data.size()));
    b.insert(b.end(), data.begin(), data.end());
}

inline bool getU8(const uint8_t*& p, const uint8_t* end, uint8_t& out) {
    if (p >= end) return false;
    out = *p++;
    return true;
}

inline bool getU16(const uint8_t*& p, const uint8_t* end, uint16_t& out) {
    if (p + 2 > end) return false;
    out = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    p += 2;
    return true;
}

inline bool getU32(const uint8_t*& p, const uint8_t* end, uint32_t& out) {
    if (p + 4 > end) return false;
    out = 0;
    for (int i = 0; i < 4; ++i) out |= static_cast<uint32_t>(p[i]) << (8 * i);
    p += 4;
    return true;
}

inline bool getU64(const uint8_t*& p, const uint8_t* end, uint64_t& out) {
    if (p + 8 > end) return false;
    out = 0;
    for (int i = 0; i < 8; ++i) out |= static_cast<uint64_t>(p[i]) << (8 * i);
    p += 8;
    return true;
}

inline bool getStr(const uint8_t*& p, const uint8_t* end, std::string& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(reinterpret_cast<const char*>(p), len);
    p += len;
    return true;
}

inline bool getBytes(const uint8_t*& p, const uint8_t* end, std::vector<uint8_t>& out) {
    uint32_t len = 0;
    if (!getU32(p, end, len)) return false;
    if (p + len > end) return false;
    out.assign(p, p + len);
    p += len;
    return true;
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