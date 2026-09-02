// pool_bindings.cpp - pybind11 glue for the Chonk Buffer PyTorch integration.
//
// Thin layer: device/pool lifecycle lives in device/pool_device.cpp, the slab
// allocator in allocator/chonk_allocator.cpp, and the HIP bridge in
// interop/hip_external_memory.cpp. This file only converts to/from py::dict
// and declares the pybind module.

#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/utils.hpp>

#include "../device/pool_device.hpp"
#include "../allocator/chonk_allocator.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace vvm;

// Init info dict: heap-allocated and never destroyed at exit (py::dict dtor
// requires the interpreter to be alive).
static py::dict* g_lastInitInfo = nullptr;

static void requirePool() {
    if (!vvm_torch::pool()) throw std::runtime_error("not initialized");
}

static py::dict alloc_to_dict(const vvm::Allocation& a) {
    py::dict out;
    out["size"] = static_cast<uint64_t>(a.size);
    out["offset"] = static_cast<uint64_t>(a.offset);
    out["block"] = static_cast<uint64_t>(a.blockIndex);
    out["deviceAddress"] = static_cast<uint64_t>(a.deviceAddress);
    out["hostPtr"] = reinterpret_cast<uint64_t>(a.hostPtr);
    out["buffer"] = reinterpret_cast<uint64_t>(a.buffer);
    out["memory"] = reinterpret_cast<uint64_t>(a.memory);
    out["hostPtrValid"] = (a.hostPtr != nullptr);
    return out;
}

// Shared allocation wrapper: flags per legacy wrapper, dict shape identical.
static py::dict alloc_wrapper(size_t size, const std::string& name,
                              bool mapped, bool exportable,
                              vvm::MemoryUsage usage,
                              bool keep, bool doExport) {
    requirePool();
    vvm::UnifiedMemoryPool* pool = vvm_torch::pool();
    vvm::AllocDesc desc;
    desc.size = size;
    desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    desc.memoryUsage = usage;
    desc.mapped = mapped;
    desc.exportable = exportable;
    desc.name = name;
    auto allocOpt = pool->allocate(desc);
    if (!allocOpt) throw std::runtime_error("allocation failed");
    vvm::Allocation a = std::move(*allocOpt);
    py::dict out = alloc_to_dict(a);
    if (doExport) {
        auto info = pool->exportMemory(a, vvm::ExternalHandleType::OpaqueFd);
        if (!info) {
            pool->deallocate(std::move(a));
            throw std::runtime_error("exportMemory failed");
        }
        out["fd"] = info->handle.release();
    }
    if (keep || doExport) {
        vvm_torch::keptAllocations().push_back(std::move(a));
    } else {
        pool->deallocate(std::move(a));
    }
    return out;
}

static py::dict initPool() {
    vvm_torch::PoolInitInfo info = vvm_torch::initPoolCore();
    py::dict out;
    out["device"] = info.deviceName;
    out["heap_mb"] = info.heapBytes >> 20;
    py::list devlist;
    for (auto& d : info.devices) {
        py::dict e;
        e["name"] = d.name;
        e["vendor"] = d.vendor;
        e["score"] = d.score;
        e["heap_mb"] = d.heapBytes >> 20;
        devlist.append(e);
    }
    out["devices"] = devlist;
    return out;
}

static py::dict init() {
    if (vvm_torch::pool()) {
        throw std::runtime_error("already initialized");
    }
    g_lastInitInfo = new py::dict(initPool());
    return *g_lastInitInfo;
}

static py::dict alloc(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, /*mapped=*/true, /*exportable=*/false,
                         vvm::MemoryUsage::GpuOnly, /*keep=*/false, /*doExport=*/false);
}

static py::dict allocKeep(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, true, false, vvm::MemoryUsage::GpuOnly, true, false);
}

static py::dict allocExport(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, true, true, vvm::MemoryUsage::GpuOnly, true, true);
}

static py::dict allocModelWeights(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, false, false, vvm::MemoryUsage::GpuOnly, true, false);
}

static py::dict allocOptimizerStates(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, false, false, vvm::MemoryUsage::GpuOnly, true, false);
}

static py::dict allocActivations(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, false, false, vvm::MemoryUsage::GpuOnly, true, false);
}

static py::dict allocHostVisible(size_t size, const std::string& name) {
    return alloc_wrapper(size, name, true, false, vvm::MemoryUsage::CpuToGpu, true, false);
}

static py::dict stats() {
    requirePool();
    auto s = vvm_torch::pool()->getStats();
    py::dict out;
    out["totalAllocated"] = static_cast<uint64_t>(s.totalAllocated);
    out["totalUsed"] = static_cast<uint64_t>(s.totalUsed);
    out["totalFree"] = static_cast<uint64_t>(s.totalFree);
    out["largestFreeBlock"] = static_cast<uint64_t>(s.largestFreeBlock);
    out["fragmentationRatio"] = s.fragmentationRatio;
    out["blockCount"] = s.blockCount;
    out["allocationCount"] = s.allocationCount;
    return out;
}

static void shutdown() {
    if (!vvm_torch::pool()) return;
    // Allocator blocks hold pool memory: release them first, then the pool
    // (shutdownPoolCore also disposes any kept allocations).
    vvm_torch::vvm_torch_chonk_allocator_reset();
    vvm_torch::shutdownPoolCore();
}

PYBIND11_MODULE(vulkanvm_pool_test, m) {
    m.doc() = "Chonk Buffer (UnifiedMemoryPool) PyTorch integration";
    m.def("init", &init, "create instance/device/pool");
    m.def("alloc", &alloc, py::arg("size"), py::arg("name") = "");
    m.def("alloc_keep", &allocKeep, py::arg("size"), py::arg("name") = "");
    m.def("alloc_export", &allocExport, py::arg("size"), py::arg("name") = "");
    m.def("alloc_model_weights", &allocModelWeights, py::arg("size"), py::arg("name") = "");
    m.def("alloc_optimizer_states", &allocOptimizerStates, py::arg("size"), py::arg("name") = "");
    m.def("alloc_activations", &allocActivations, py::arg("size"), py::arg("name") = "");
    m.def("alloc_host_visible", &allocHostVisible, py::arg("size"), py::arg("name") = "");
    m.def("stats", &stats);
    m.def("release_empty_blocks", [](size_t keepFloor) {
        vvm_torch::vvm_torch_chonk_allocator_release_empty(keepFloor);
    }, py::arg("keepFloor") = 2, "release fully-free slab blocks down to keepFloor");
    m.def("info", []() {
        if (!g_lastInitInfo) throw std::runtime_error("not initialized");
        return *g_lastInitInfo;
    });
    m.def("shutdown", &shutdown);
}
