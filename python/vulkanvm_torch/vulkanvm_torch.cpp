// VulkanVM PyTorch C++ Extension
// Provides a custom PyTorch allocator backed by UnifiedMemoryPool

#include <torch/extension.h>
#include <vulkan_vm/vulkan_vm.hpp>

#include <memory>
#include <mutex>
#include <unordered_map>

namespace vulkanvm_torch {

using vvm::UnifiedMemoryPool;
using vvm::DeviceConfig;
using vvm::PoolConfig;
using vvm::Allocation;
using vvm::MemoryUsage;
using vvm::ExternalHandleType;
using vvm::ExternalMemoryInfo;

static std::unique_ptr<UnifiedMemoryPool> g_pool = nullptr;
static std::mutex g_pool_mutex;

// -----------------------------------------------------------------------------
// Pool lifecycle
// -----------------------------------------------------------------------------

torch::Tensor create_pool(torch::Tensor device_props) {
    // device_props is a tensor with [physicalDeviceHandle, deviceHandle, 
    // graphicsQueueFamily, computeQueueFamily, transferQueueFamily,
    // graphicsQueue, computeQueue, transferQueue]
    // All as int64_t handles
    TORCH_CHECK(device_props.dim() == 1 && device_props.size(0) >= 8,
                "device_props must have at least 8 elements");

    auto* data = device_props.data_ptr<int64_t>();
    
    DeviceConfig devConfig;
    devConfig.physicalDevice = reinterpret_cast<VkPhysicalDevice>(data[0]);
    devConfig.device = reinterpret_cast<VkDevice>(data[1]);
    devConfig.graphicsQueueFamily = static_cast<uint32_t>(data[2]);
    devConfig.computeQueueFamily = static_cast<uint32_t>(data[3]);
    devConfig.transferQueueFamily = static_cast<uint32_t>(data[4]);
    devConfig.graphicsQueue = reinterpret_cast<VkQueue>(data[5]);
    devConfig.computeQueue = reinterpret_cast<VkQueue>(data[6]);
    devConfig.transferQueue = reinterpret_cast<VkQueue>(data[7]);

    PoolConfig poolConfig;
    poolConfig.blockSize = 512 * 1024 * 1024;
    poolConfig.minAlignment = 256 * 1024;
    poolConfig.enableHostVisible = true;
    poolConfig.enableExternal = true;
    poolConfig.enableDeviceAddress = true;
    poolConfig.maxBlocks = 16;

    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_pool) {
        return torch::tensor(0, torch::dtype(torch::kInt64));  // already created
    }

    auto poolOpt = UnifiedMemoryPool::create(devConfig, poolConfig);
    if (!poolOpt) {
        TORCH_CHECK(false, "Failed to create UnifiedMemoryPool");
    }
    g_pool = std::make_unique<UnifiedMemoryPool>(std::move(*poolOpt));
    
    return torch::tensor(1, torch::dtype(torch::kInt64));
}

torch::Tensor destroy_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.reset();
    return torch::tensor(1, torch::dtype(torch::kInt64));
}

bool is_pool_created() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    return g_pool != nullptr;
}

// -----------------------------------------------------------------------------
// Allocation wrapper
// -----------------------------------------------------------------------------

struct TorchAllocation {
    Allocation alloc;
    std::shared_ptr<UnifiedMemoryPool> pool_ref;  // keep pool alive
};

static std::unordered_map<void*, TorchAllocation> g_alloc_map;
static std::mutex g_alloc_mutex;

// Custom deleter for PyTorch tensor storage
struct VulkanStorageDeleter {
    void operator()(void* ptr) {
        std::lock_guard<std::mutex> lock(g_alloc_mutex);
        auto it = g_alloc_map.find(ptr);
        if (it != g_alloc_map.end()) {
            it->second.pool_ref->deallocate(std::move(it->second.alloc));
            g_alloc_map.erase(it);
        }
    }
};

// -----------------------------------------------------------------------------
// Allocation API
// -----------------------------------------------------------------------------

torch::Tensor allocate_tensor(
    int64_t size,
    c10::optional<int64_t> usage_ = c10::nullopt,
    c10::optional<int64_t> memory_usage_ = c10::nullopt,
    c10::optional<bool> exportable_ = c10::nullopt,
    c10::optional<bool> mapped_ = c10::nullopt,
    c10::optional<std::string> name_ = c10::nullopt) {
    
    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    TORCH_CHECK(g_pool, "Pool not created. Call create_pool() first.");

    VkBufferUsageFlags usage = usage_.has_value() 
        ? static_cast<VkBufferUsageFlags>(usage_.value())
        : (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
           VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    vvm::MemoryUsage memUsage = memory_usage_.has_value()
        ? static_cast<vvm::MemoryUsage>(memory_usage_.value())
        : vvm::MemoryUsage::GpuOnly;

    bool exportable = exportable_.value_or(false);
    bool mapped = mapped_.value_or(false);

    std::optional<Allocation> allocOpt;
    if (exportable) {
        VkMemoryPropertyFlags flags = 0;
        if (memUsage == vvm::MemoryUsage::CpuToGpu || memUsage == vvm::MemoryUsage::GpuToCpu) {
            flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        allocOpt = g_pool->allocateDedicatedExportable(size, usage, flags);
    } else {
        vvm::AllocDesc desc;
        desc.size = size;
        desc.usage = usage;
        desc.memoryUsage = memUsage;
        desc.exportable = exportable;
        desc.mapped = mapped;
        desc.name = name_.value_or("").c_str();
        allocOpt = g_pool->allocate(desc);
    }

    TORCH_CHECK(allocOpt.has_value(), "Allocation failed");

    Allocation alloc = std::move(*allocOpt);

    // Create PyTorch tensor from the allocation
    // We create a storage backed by the Vulkan memory
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        size,
        alloc.hostPtr ? std::make_unique<c10::Allocator>(/* custom */) : nullptr,
        c10::Device(c10::DeviceType::CUDA, 0)  // placeholder; we use device addresses
    );

    // Better approach: create a tensor with a custom allocator
    // For now, return a tensor that holds the allocation info
    // The actual data lives in Vulkan memory, accessible via device address

    // Store the allocation so it doesn't get destroyed
    void* key = reinterpret_cast<void*>(alloc.buffer);
    {
        std::lock_guard<std::mutex> alloc_lock(g_alloc_mutex);
        g_alloc_map.emplace(key, TorchAllocation{std::move(alloc), g_pool ? std::shared_ptr<UnifiedMemoryPool>(g_pool.get(), [](auto*){}) : nullptr});
    }

    // Return a tensor with metadata (we'll use device address for actual compute)
    auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::kCUDA, 0);  // CUDA device for compatibility

    // Create an empty tensor that we'll use as a handle
    // The real data is in Vulkan memory
    torch::Tensor handle = torch::empty({1}, options);
    
    // Store the Allocation info in the tensor's metadata via a custom class
    // For now, return allocation info as a tensor
    return torch::tensor({
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(alloc.buffer)),
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(alloc.memory)),
        static_cast<int64_t>(alloc.offset),
        static_cast<int64_t>(alloc.size),
        static_cast<int64_t>(alloc.deviceAddress),
        static_cast<int64_t>(reinterpret_cast<uintptr_t>(alloc.hostPtr)),
        static_cast<int64_t>(alloc.blockIndex)
    }, torch::dtype(torch::kInt64));
}

torch::Tensor deallocate_tensor(torch::Tensor alloc_info) {
    TORCH_CHECK(alloc_info.dim() == 1 && alloc_info.size(0) == 7,
                "alloc_info must have 7 elements");
    
    auto* data = alloc_info.data_ptr<int64_t>();
    VkBuffer buffer = reinterpret_cast<VkBuffer>(data[0]);
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(data[1]);
    VkDeviceSize offset = data[2];
    VkDeviceSize size = data[3];
    VkDeviceAddress deviceAddress = data[4];
    void* hostPtr = reinterpret_cast<void*>(data[5]);
    uint32_t blockIndex = static_cast<uint32_t>(data[6]);

    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    TORCH_CHECK(g_pool, "Pool not created");

    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = offset;
    alloc.size = size;
    alloc.deviceAddress = deviceAddress;
    alloc.hostPtr = hostPtr;
    alloc.blockIndex = blockIndex;

    g_pool->deallocate(std::move(alloc));
    return torch::tensor(1, torch::dtype(torch::kInt64));
}

// -----------------------------------------------------------------------------
// Device address access
// -----------------------------------------------------------------------------

torch::Tensor get_device_address(torch::Tensor alloc_info) {
    auto* data = alloc_info.data_ptr<int64_t>();
    VkDeviceAddress addr = data[4];
    return torch::tensor(static_cast<int64_t>(addr), torch::dtype(torch::kInt64));
}

// -----------------------------------------------------------------------------
// External memory export/import
// -----------------------------------------------------------------------------

torch::Tensor export_memory(torch::Tensor alloc_info, int64_t handle_type) {
    auto* data = alloc_info.data_ptr<int64_t>();
    VkBuffer buffer = reinterpret_cast<VkBuffer>(data[0]);
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(data[1]);
    VkDeviceSize offset = data[2];
    VkDeviceSize size = data[3];

    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    TORCH_CHECK(g_pool, "Pool not created");

    // Reconstruct allocation
    Allocation alloc;
    alloc.buffer = buffer;
    alloc.memory = memory;
    alloc.offset = offset;
    alloc.size = size;
    alloc.blockIndex = UINT32_MAX;  // dedicated

    auto extInfoOpt = g_pool->exportMemory(alloc, static_cast<ExternalHandleType>(handle_type));
    TORCH_CHECK(extInfoOpt.has_value(), "Export failed");

    auto& extInfo = *extInfoOpt;
    return torch::tensor({
        static_cast<int64_t>(extInfo.handle.get()),
        static_cast<int64_t>(extInfo.size),
        static_cast<int64_t>(extInfo.memoryTypeIndex),
        static_cast<int64_t>(extInfo.dedicatedAllocation ? 1 : 0)
    }, torch::dtype(torch::kInt64));
}

// -----------------------------------------------------------------------------
// Pool stats
// -----------------------------------------------------------------------------

torch::Tensor get_pool_stats() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    TORCH_CHECK(g_pool, "Pool not created");

    auto stats = g_pool->getStats();
    return torch::tensor({
        static_cast<int64_t>(stats.totalAllocated),
        static_cast<int64_t>(stats.totalUsed),
        static_cast<int64_t>(stats.totalFree),
        static_cast<int64_t>(stats.largestFreeBlock),
        static_cast<int64_t>(stats.totalCapacity),
        static_cast<int64_t>(stats.allocationCount),
        static_cast<int64_t>(stats.blockCount),
        static_cast<int64_t>(stats.dedicatedCount),
        static_cast<int64_t>(stats.fragmentationRatio * 10000)  // x10000 for precision
    }, torch::dtype(torch::kInt64));
}

// -----------------------------------------------------------------------------
// Module registration
// -----------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_pool", &create_pool, "Create VulkanVM pool");
    m.def("destroy_pool", &destroy_pool, "Destroy VulkanVM pool");
    m.def("is_pool_created", &is_pool_created, "Check if pool exists");
    
    m.def("allocate", &allocate_tensor, "Allocate tensor from VulkanVM pool",
          py::arg("size"), py::arg("usage") = c10::nullopt,
          py::arg("memory_usage") = c10::nullopt,
          py::arg("exportable") = c10::nullopt,
          py::arg("mapped") = c10::nullopt,
          py::arg("name") = c10::nullopt);
    
    m.def("deallocate", &deallocate_tensor, "Deallocate tensor");
    m.def("get_device_address", &get_device_address, "Get device address");
    m.def("export_memory", &export_memory, "Export memory for sharing");
    m.def("get_pool_stats", &get_pool_stats, "Get pool statistics");
    
    // Enums
    py::enum_<vvm::MemoryUsage>(m, "MemoryUsage")
        .value("GpuOnly", vvm::MemoryUsage::GpuOnly)
        .value("CpuToGpu", vvm::MemoryUsage::CpuToGpu)
        .value("GpuToCpu", vvm::MemoryUsage::GpuToCpu)
        .value("CpuCopy", vvm::MemoryUsage::CpuCopy)
        .value("Auto", vvm::MemoryUsage::Auto);
    
    py::enum_<ExternalHandleType>(m, "ExternalHandleType")
        .value("OpaqueFd", ExternalHandleType::OpaqueFd)
        .value("OpaqueWin32", ExternalHandleType::OpaqueWin32)
        .value("D3D12Heap", ExternalHandleType::D3D12Heap)
        .value("DmaBuf", ExternalHandleType::DmaBuf);
}