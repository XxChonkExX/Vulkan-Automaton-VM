// vulkanvm_torch.cpp
//
// PyTorch C++ Extension for VulkanVM — FULL AI TRAINING EDITION.
//
// This is a STANDALONE pybind11 binding that uses ONLY:
//   - pybind11 3.1.0 (from pip, MSVC 19.44 compatible)
//   - VulkanVM C++ API
//   - Python C API
//
// NO ATen, NO torch/, NO c10/ - avoids MSVC 19.44 incompatibility with
// PyTorch's bundled pybind11 2.13.6.

// MSVC 19.44 workaround: include Python.h first so Py_ssize_t is defined
// for pybind11's `using ssize_t = Py_ssize_t;` in detail/common.h
#if defined(_MSC_VER) && _MSC_VER >= 1940
#include <Python.h>
#endif

// Include pybind11 FIRST (standalone 3.1.0 with MSVC fixes)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

// VulkanVM - only include core headers we need (avoid offload.hpp which uses C++20)
#include <vulkan_vm/core.hpp>
#include <vulkan_vm/allocator.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// Forward declaration for autograd/LoRA/layer bindings defined in the
// sibling translation unit vulkanvm_autograd_bind.cpp.
void register_vulkanvm_autograd_bindings(py::module_& m);
using vvm::UnifiedMemoryPool;
using vvm::DeviceConfig;
using vvm::PoolConfig;
using vvm::Allocation;
using vvm::MemoryUsage;
using vvm::ExternalHandleType;
using vvm::ExternalMemoryInfo;

static std::unique_ptr<UnifiedMemoryPool> g_pool = nullptr;
static std::mutex g_pool_mutex;

struct TorchAllocation {
    Allocation alloc;
    std::shared_ptr<UnifiedMemoryPool> pool_ref;
};

static std::unordered_map<void*, TorchAllocation> g_alloc_map;
static std::mutex g_alloc_mutex;

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
// Pool lifecycle
// -----------------------------------------------------------------------------

py::object create_pool(py::object device_props_obj) {
    auto device_props = py::cast<py::array_t<int64_t>>(device_props_obj);
    py::buffer_info buf = device_props.request();
    if (!(buf.ndim == 1 && buf.size >= 8)) {
        throw std::runtime_error("device_props must have at least 8 elements");
    }
    int64_t* data = static_cast<int64_t*>(buf.ptr);

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
        return py::int_(0);  // already created
    }

    auto poolOpt = UnifiedMemoryPool::create(devConfig, poolConfig);
    if (!poolOpt) {
        throw std::runtime_error("Failed to create UnifiedMemoryPool");
    }
    g_pool = std::make_unique<UnifiedMemoryPool>(std::move(*poolOpt));
    return py::int_(1);
}

py::int_ destroy_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.reset();
    return py::int_(1);
}

bool is_pool_created() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    return g_pool != nullptr;
}

// -----------------------------------------------------------------------------
// Allocation API
// -----------------------------------------------------------------------------

py::array_t<int64_t> allocate_tensor(
    int64_t size,
    py::object usage_ = py::none(),
    py::object memory_usage_ = py::none(),
    py::object exportable_ = py::none(),
    py::object mapped_ = py::none(),
    py::object name_ = py::none()) {

    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    if (!g_pool) throw std::runtime_error("Pool not created. Call create_pool() first.");

    VkBufferUsageFlags usage = usage_.is_none()
        ? (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
           VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : static_cast<VkBufferUsageFlags>(usage_.cast<uint64_t>());

    vvm::MemoryUsage memUsage = memory_usage_.is_none()
        ? vvm::MemoryUsage::GpuOnly
        : static_cast<vvm::MemoryUsage>(memory_usage_.cast<int>());

    bool exportable = exportable_.is_none() ? false : exportable_.cast<bool>();
    bool mapped = mapped_.is_none() ? false : mapped_.cast<bool>();

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
        desc.name = name_.is_none() ? "" : name_.cast<std::string>().c_str();
        allocOpt = g_pool->allocate(desc);
    }

    if (!allocOpt.has_value()) {
        throw std::runtime_error("Allocation failed");
    }

    Allocation alloc_opt = std::move(*allocOpt);
    void* key = reinterpret_cast<void*>(alloc_opt.buffer);
    {
        std::lock_guard<std::mutex> alloc_lock(g_alloc_mutex);
        auto shared = std::shared_ptr<UnifiedMemoryPool>(g_pool.get(), [](auto*){});
        g_alloc_map.emplace(key, TorchAllocation{std::move(alloc_opt), shared});
    }
    auto& a = g_alloc_map.find(key)->second.alloc;

    py::array_t<int64_t> result({7});
    auto* out = static_cast<int64_t*>(result.request().ptr);
    out[0] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(a.buffer));
    out[1] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(a.memory));
    out[2] = static_cast<int64_t>(a.offset);
    out[3] = static_cast<int64_t>(a.size);
    out[4] = static_cast<int64_t>(a.deviceAddress);
    out[5] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(a.hostPtr));
    out[6] = static_cast<int64_t>(a.blockIndex);
    return result;
}

py::int_ deallocate_tensor(py::array_t<int64_t> alloc_info) {
    auto buf = alloc_info.request();
    if (!(buf.ndim == 1 && buf.size == 7)) {
        throw std::runtime_error("alloc_info must have 7 elements");
    }
    int64_t* data = static_cast<int64_t*>(buf.ptr);
    VkBuffer buffer = reinterpret_cast<VkBuffer>(data[0]);
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(data[1]);
    VkDeviceSize offset = data[2];
    VkDeviceSize size = data[3];
    VkDeviceAddress deviceAddress = data[4];
    void* hostPtr = reinterpret_cast<void*>(data[5]);
    uint32_t blockIndex = static_cast<uint32_t>(data[6]);

    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    if (!g_pool) throw std::runtime_error("Pool not created");

    std::lock_guard<std::mutex> alloc_lock(g_alloc_mutex);
    auto it = g_alloc_map.find(buffer);
    if (it != g_alloc_map.end()) {
        it->second.pool_ref->deallocate(std::move(it->second.alloc));
        g_alloc_map.erase(it);
        return py::int_(1);
    }

    Allocation my_alloc;
    my_alloc.buffer = buffer;
    my_alloc.memory = memory;
    my_alloc.offset = offset;
    my_alloc.size = size;
    my_alloc.deviceAddress = deviceAddress;
    my_alloc.hostPtr = hostPtr;
    my_alloc.blockIndex = blockIndex;
    g_pool->deallocate(std::move(my_alloc));
    return py::int_(1);
}

py::int_ get_device_address(py::array_t<int64_t> alloc_info) {
    auto buf = alloc_info.request();
    int64_t* data = static_cast<int64_t*>(buf.ptr);
    return py::int_(data[4]);
}

py::array_t<int64_t> export_memory(py::array_t<int64_t> alloc_info, int64_t handle_type) {
    auto buf = alloc_info.request();
    int64_t* data = static_cast<int64_t*>(buf.ptr);
    VkBuffer buffer = reinterpret_cast<VkBuffer>(data[0]);
    VkDeviceMemory memory = reinterpret_cast<VkDeviceMemory>(data[1]);
    VkDeviceSize offset = data[2];
    VkDeviceSize size = data[3];

    std::lock_guard<std::mutex> pool_lock(g_pool_mutex);
    if (!g_pool) throw std::runtime_error("Pool not created");

    Allocation alloc_ext;
    alloc_ext.buffer = buffer;
    alloc_ext.memory = memory;
    alloc_ext.offset = offset;
    alloc_ext.size = size;
    alloc_ext.blockIndex = UINT32_MAX;

    auto extInfoOpt = g_pool->exportMemory(alloc_ext, static_cast<ExternalHandleType>(handle_type));
    if (!extInfoOpt.has_value()) throw std::runtime_error("Export failed");

    auto& extInfo = *extInfoOpt;
    py::array_t<int64_t> result({4});
    auto* out = static_cast<int64_t*>(result.request().ptr);
    out[0] = reinterpret_cast<int64_t>(extInfo.handle.get());
    out[1] = static_cast<int64_t>(extInfo.size);
    out[2] = static_cast<int64_t>(extInfo.memoryTypeIndex);
    out[3] = static_cast<int64_t>(extInfo.dedicatedAllocation ? 1 : 0);
    return result;
}

py::array_t<int64_t> get_pool_stats() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (!g_pool) throw std::runtime_error("Pool not created");

    auto stats = g_pool->getStats();
    py::array_t<int64_t> result({9});
    auto* out = static_cast<int64_t*>(result.request().ptr);
    out[0] = static_cast<int64_t>(stats.totalAllocated);
    out[1] = static_cast<int64_t>(stats.totalUsed);
    out[2] = static_cast<int64_t>(stats.totalFree);
    out[3] = static_cast<int64_t>(stats.largestFreeBlock);
    out[4] = static_cast<int64_t>(stats.totalCapacity);
    out[5] = static_cast<int64_t>(stats.allocationCount);
    out[6] = static_cast<int64_t>(stats.blockCount);
    out[7] = static_cast<int64_t>(stats.dedicatedCount);
    out[8] = static_cast<int64_t>(stats.fragmentationRatio * 10000);
    return result;
}

// -----------------------------------------------------------------------------
// Module registration
// -----------------------------------------------------------------------------

PYBIND11_MODULE(vulkanvm_torch, m) {
    m.doc() = "VulkanVM PyTorch Extension — pool lifecycle & tensor allocation";

    m.def("create_pool", &create_pool, "Create VulkanVM pool");
    m.def("destroy_pool", &destroy_pool, "Destroy VulkanVM pool");
    m.def("is_pool_created", &is_pool_created, "Check if pool exists");
    m.def("allocate", &allocate_tensor, "Allocate tensor from VulkanVM pool",
          py::arg("size"), py::arg("usage") = py::none(),
          py::arg("memory_usage") = py::none(), py::arg("exportable") = py::none(),
          py::arg("mapped") = py::none(), py::arg("name") = py::none());
    m.def("deallocate", &deallocate_tensor, "Deallocate tensor");
    m.def("get_device_address", &get_device_address, "Get device address");
    m.def("export_memory", &export_memory, "Export memory for sharing");
    m.def("get_pool_stats", &get_pool_stats, "Get pool statistics");

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

    // Autograd Functions, LoRA registry, layer factories, AdamW optimizer.
    register_vulkanvm_autograd_bindings(m);
}