// PyTorch C++ Extension for VulkanVM
// Exposes UnifiedMemoryPool, Tensor operations, and ModelHub to Python

#include <torch/extension.h>
#include <vulkan_vm/vulkan_vm.hpp>
#include <vulkan_vm/network/model_registry.hpp>
#include <vulkan_vm/placement.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace vvm;

// ============================================================================
// Exception handling
// ============================================================================

static void translate_vulkan_error(const std::exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

static void translate_vulkan_nullopt(const char* msg) {
    PyErr_SetString(PyExc_RuntimeError, msg);
}

// ============================================================================
// DeviceConfig binding
// ============================================================================

PYBIND11_MODULE(vulkanvm_torch, m) {
    m.doc() = "VulkanVM PyTorch Extension - Unified Vulkan Memory Pool for PyTorch";
    
    // Register exception translators
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            translate_vulkan_error(e);
        }
    });

    // -------------------------------------------------------------------------
    // Enums
    // -------------------------------------------------------------------------
    
    py::enum_<MemoryUsage>(m, "MemoryUsage")
        .value("GpuOnly", MemoryUsage::GpuOnly)
        .value("CpuToGpu", MemoryUsage::CpuToGpu)
        .value("GpuToCpu", MemoryUsage::GpuToCpu)
        .value("CpuCopy", MemoryUsage::CpuCopy)
        .value("Auto", MemoryUsage::Auto);
    
    py::enum_<MemTier>(m, "MemTier")
        .value("DeviceLocal", MemTier::DeviceLocal)
        .value("HostOffload", MemTier::HostOffload)
        .value("DiskCache", MemTier::DiskCache);
    
    py::enum_<ShardKind>(m, "ShardKind")
        .value("Weights", ShardKind::Weights)
        .value("Tokenizer", ShardKind::Tokenizer)
        .value("Config", ShardKind::Config)
        .value("Extra", ShardKind::Extra);
    
    py::enum_<ErrorCode>(m, "ErrorCode")
        .value("Ok", ErrorCode::Ok)
        .value("InvalidManifest", ErrorCode::InvalidManifest)
        .value("InvalidCluster", ErrorCode::InvalidCluster)
        .value("InsufficientCapacity", ErrorCode::InsufficientCapacity)
        .value("UnsatisfiableConstraint", ErrorCode::UnsatisfiableConstraint)
        .value("ShardTooLarge", ErrorCode::ShardTooLarge)
        .value("ActivationReserveFailed", ErrorCode::ActivationReserveFailed);
    
    // -------------------------------------------------------------------------
    // PoolConfig
    // -------------------------------------------------------------------------
    
    py::class_<PoolConfig>(m, "PoolConfig")
        .def(py::init<>())
        .def_readwrite("blockSize", &PoolConfig::blockSize)
        .def_readwrite("minAlignment", &PoolConfig::minAlignment)
        .def_readwrite("maxBlocks", &PoolConfig::maxBlocks)
        .def_readwrite("maxHeapFraction", &PoolConfig::maxHeapFraction)
        .def_readwrite("enableHostVisible", &PoolConfig::enableHostVisible)
        .def_readwrite("preferredFlags", &PoolConfig::preferredFlags)
        .def_readwrite("exportable", &PoolConfig::exportable)
        .def_static("for_device", &PoolConfig::forDevice, "physical_device"_a)
        .def_static("for_apu", &PoolConfig::forAPU, "total_system_ram"_a);
    
    // -------------------------------------------------------------------------
    // DeviceConfig
    // -------------------------------------------------------------------------
    
    py::class_<DeviceConfig>(m, "DeviceConfig")
        .def(py::init<>())
        .def_readwrite("physicalDevice", &DeviceConfig::physicalDevice)
        .def_readwrite("device", &DeviceConfig::device)
        .def_readwrite("graphicsQueueFamily", &DeviceConfig::graphicsQueueFamily)
        .def_readwrite("computeQueueFamily", &DeviceConfig::computeQueueFamily)
        .def_readwrite("transferQueueFamily", &DeviceConfig::transferQueueFamily)
        .def_readwrite("graphicsQueue", &DeviceConfig::graphicsQueue)
        .def_readwrite("computeQueue", &DeviceConfig::computeQueue)
        .def_readwrite("transferQueue", &DeviceConfig::transferQueue);
    
    // -------------------------------------------------------------------------
    // AllocDesc
    // -------------------------------------------------------------------------
    
    py::class_<AllocDesc>(m, "AllocDesc")
        .def(py::init<>())
        .def_readwrite("size", &AllocDesc::size)
        .def_readwrite("usage", &AllocDesc::usage)
        .def_readwrite("memoryUsage", &AllocDesc::memoryUsage)
        .def_readwrite("exportable", &AllocDesc::exportable)
        .def_readwrite("mapped", &AllocDesc::mapped)
        .def_readwrite("name", &AllocDesc::name);
    
    // -------------------------------------------------------------------------
    // Allocation (opaque handle returned to Python)
    // -------------------------------------------------------------------------
    
    py::class_<Allocation>(m, "Allocation")
        .def_readonly("size", &Allocation::size)
        .def_readonly("offset", &Allocation::offset)
        .def_readonly("blockIndex", &Allocation::blockIndex)
        .def_readonly("deviceAddress", &Allocation::deviceAddress)
        .def_readonly("buffer", &Allocation::buffer)
        .def_readonly("memory", &Allocation::memory)
        .def_readonly("hostPtr", &Allocation::hostPtr)
        .def_readonly("mapped", &Allocation::mapped)
        .def_readonly("exportable", &Allocation::exportable)
        .def_readonly("name", &Allocation::name);
    
    // -------------------------------------------------------------------------
    // UniqueAllocation (smart pointer)
    // -------------------------------------------------------------------------
    
    py::class_<UniqueAllocation>(m, "UniqueAllocation")
        .def_property_readonly("size", [](const UniqueAllocation& ua) { return ua->size; })
        .def_property_readonly("device_address", [](const UniqueAllocation& ua) { return ua->deviceAddress; })
        .def_property_readonly("buffer", [](const UniqueAllocation& ua) { return ua->buffer; })
        .def_property_readonly("host_ptr", [](const UniqueAllocation& ua) { return ua->hostPtr; })
        .def("get", &UniqueAllocation::get)
        .def("release", &UniqueAllocation::release)
        .def("__bool__", [](const UniqueAllocation& ua) { return static_cast<bool>(ua); });
    
    // -------------------------------------------------------------------------
    // PoolStats
    // -------------------------------------------------------------------------
    
    py::class_<PoolStats>(m, "PoolStats")
        .def_readonly("totalAllocated", &PoolStats::totalAllocated)
        .def_readonly("totalUsed", &PoolStats::totalUsed)
        .def_readonly("totalFree", &PoolStats::totalFree)
        .def_readonly("largestFreeBlock", &PoolStats::largestFreeBlock)
        .def_readonly("fragmentationRatio", &PoolStats::fragmentationRatio)
        .def_readonly("blockCount", &PoolStats::blockCount)
        .def_readonly("allocationCount", &PoolStats::allocationCount);
    
    // -------------------------------------------------------------------------
    // MemoryTopology
    // -------------------------------------------------------------------------
    
    py::class_<MemoryTopology>(m, "MemoryTopology")
        .def_readonly("type", &MemoryTopology::type)
        .def_readonly("heapCount", &MemoryTopology::heapCount)
        .def_readonly("hasUnifiedHeap", &MemoryTopology::hasUnifiedHeap)
        .def_readonly("totalDeviceLocalBytes", &MemoryTopology::totalDeviceLocalBytes)
        .def_readonly("totalHostVisibleBytes", &MemoryTopology::totalHostVisibleBytes);
    
    // -------------------------------------------------------------------------
    // UnifiedMemoryPool
    // -------------------------------------------------------------------------
    
    py::class_<UnifiedMemoryPool, std::shared_ptr<UnifiedMemoryPool>>(m, "UnifiedMemoryPool")
        .def_static("create", [](const DeviceConfig& dev, const PoolConfig& cfg) {
            auto pool = UnifiedMemoryPool::create(dev, cfg);
            if (!pool) {
                throw std::runtime_error("Failed to create UnifiedMemoryPool");
            }
            return pool;
        }, "dev_config"_a, "pool_config"_a)
        .def("allocate_tensor", [](UnifiedMemoryPool& pool, size_t size, const std::string& name = "") {
            auto alloc = pool.allocateTensor(size);
            if (!alloc) throw std::runtime_error("Allocation failed");
            return alloc;
        }, "size"_a, "name"_a = "")
        .def("allocate", [](UnifiedMemoryPool& pool, const AllocDesc& desc) {
            auto alloc = pool.allocate(desc);
            if (!alloc) throw std::runtime_error("Allocation failed");
            return alloc;
        }, "desc"_a)
        .def("allocate_dedicated_exportable", [](UnifiedMemoryPool& pool, size_t size, VkBufferUsageFlags usage) {
            auto alloc = pool.allocateDedicatedExportable(size, usage);
            if (!alloc) throw std::runtime_error("Dedicated exportable allocation failed");
            return alloc;
        }, "size"_a, "usage"_a)
        .def("deallocate", &UnifiedMemoryPool::deallocate, "alloc"_a)
        .def("copy_buffer", &UnifiedMemoryPool::copyBuffer, "src"_a, "dst"_a, "src_offset"_a, "dst_offset"_a, "size"_a)
        .def("get_stats", &UnifiedMemoryPool::getStats)
        .def("get_stats_string", [](UnifiedMemoryPool& pool) {
            auto stats = pool.getStats();
            std::ostringstream oss;
            oss << "Allocated: " << (stats.totalAllocated >> 20) << " MiB\n"
                << "Used: " << (stats.totalUsed >> 20) << " MiB\n"
                << "Free: " << (stats.totalFree >> 20) << " MiB\n"
                << "Largest Free: " << (stats.largestFreeBlock >> 20) << " MiB\n"
                << "Fragmentation: " << (stats.fragmentationRatio * 100) << "%\n"
                << "Blocks: " << stats.blockCount << "\n"
                << "Allocations: " << stats.allocationCount;
            return oss.str();
        })
        // Offload
        .def("initialize_offload", [](UnifiedMemoryPool& pool, size_t host_shadow_size, VkQueue transfer_queue, uint32_t transfer_queue_family) {
            OffloadConfig cfg;
            cfg.hostShadowSize = host_shadow_size;
            cfg.transferQueue = transfer_queue;
            cfg.transferQueueFamily = transfer_queue_family;
            cfg.persistentMapping = true;
            cfg.useCoherentMapping = true;
            pool.initializeOffload(cfg);
        }, "host_shadow_size"_a, "transfer_queue"_a, "transfer_queue_family"_a)
        .def("offload_to_host", [](UnifiedMemoryPool& pool, const UniqueAllocation& alloc) {
            return pool.offloadToHost(*alloc);
        }, "alloc"_a)
        .def("reload_to_device", [](UnifiedMemoryPool& pool, const UniqueAllocation& alloc) {
            pool.reloadToDevice(*alloc);
        }, "alloc"_a)
        .def("wait_migration", [](UnifiedMemoryPool& pool, const std::optional<MigrationOp>& op) {
            if (op) pool.waitMigration(*op);
        }, "op"_a)
        // External memory
        .def("export_memory", [](UnifiedMemoryPool& pool, const UniqueAllocation& alloc, int handle_type) {
            return pool.exportMemory(*alloc, static_cast<ExternalHandleType>(handle_type));
        }, "alloc"_a, "handle_type"_a)
        .def("import_memory", [](UnifiedMemoryPool& pool, const std::string& handle_json, VkBufferUsageFlags usage) {
            // Import from JSON handle (platform-specific)
            auto alloc = pool.importMemory(handle_json, usage);
            if (!alloc) throw std::runtime_error("Import failed");
            return alloc;
        }, "handle_json"_a, "usage"_a)
        // Sparse
        .def("create_sparse_pool", [](UnifiedMemoryPool& pool, size_t virtual_size, size_t page_size) {
            // Returns a SparseVirtualMemoryPool
            throw std::runtime_error("Use SparseVirtualMemoryPool directly");
        });
    
    // -------------------------------------------------------------------------
    // SparseVirtualMemoryPool
    // -------------------------------------------------------------------------
    
    py::class_<SparseVirtualMemoryPool>(m, "SparseVirtualMemoryPool")
        .def(py::init<VkDevice, VkPhysicalDevice>(), "device"_a, "physical_device"_a)
        .def("initialize", &SparseVirtualMemoryPool::initialize, "virtual_size"_a, "page_size"_a)
        .def("reserve_virtual", &SparseVirtualMemoryPool::reserveVirtual, "size"_a, "usage"_a)
        .def("commit", &SparseVirtualMemoryPool::commit, "reservation"_a, "offset"_a, "size"_a, "flags"_a)
        .def("uncommit", &SparseVirtualMemoryPool::uncommit, "reservation"_a, "offset"_a, "size"_a);
    
    // -------------------------------------------------------------------------
    // MigrationOp
    // -------------------------------------------------------------------------
    
    py::class_<MigrationOp>(m, "MigrationOp")
        .def_readonly("fence", &MigrationOp::fence)
        .def_readonly("srcOffset", &MigrationOp::srcOffset)
        .def_readonly("dstOffset", &MigrationOp::dstOffset)
        .def_readonly("size", &MigrationOp::size)
        .def_readonly("toHost", &MigrationOp::toHost);
    
    // -------------------------------------------------------------------------
    // MultiGPUPoolManager
    // -------------------------------------------------------------------------
    
    py::class_<MultiGPUPoolManager, std::shared_ptr<MultiGPUPoolManager>>(m, "MultiGPUPoolManager")
        .def_static("create", [](const std::vector<DeviceConfig>& devices, const PoolConfig& cfg, int master_idx) {
            auto mgr = MultiGPUPoolManager::create(devices, cfg, master_idx);
            if (!mgr) throw std::runtime_error("Failed to create MultiGPUPoolManager");
            return mgr;
        }, "devices"_a, "pool_config"_a, "master_idx"_a)
        .def("allocate_distributed", &MultiGPUPoolManager::allocateDistributed, "size"_a, "usage"_a)
        .def("get_local_pool", &MultiGPUPoolManager::getLocalPool)
        .def("query_peer_access", &MultiGPUPoolManager::queryPeerAccess, "src"_a, "dst"_a)
        .def("copy_device_to_device", &MultiGPUPoolManager::copyDeviceToDevice, "src_idx"_a, "dst_idx"_a, "src_alloc"_a, "dst_alloc"_a, "src_offset"_a, "dst_offset"_a, "size"_a)
        .def("submit_migration_barrier", &MultiGPUPoolManager::submitMigrationBarrier, "ops"_a)
        .def("wait_all_idle", &MultiGPUPoolManager::waitAllIdle);
    
    // -------------------------------------------------------------------------
    // PeerAccess
    // -------------------------------------------------------------------------
    
    py::class_<PeerAccess>(m, "PeerAccess")
        .def_readonly("canDirectCopy", &PeerAccess::canDirectCopy)
        .def_readonly("canDirectRead", &PeerAccess::canDirectRead)
        .def_readonly("canDirectWrite", &PeerAccess::canDirectWrite);
    
    // -------------------------------------------------------------------------
    // ModelHub (network)
    // -------------------------------------------------------------------------
    
    py::class_<ModelHub>(m, "ModelHub")
        .def(py::init<const std::string&>(), "cache_dir"_a)
        .def("start", &ModelHub::start, "address"_a, "port"_a)
        .def("stop", &ModelHub::stop)
        .def("publish", &ModelHub::publish, "model_id"_a, "local_path"_a, "version"_a)
        .def_static("fetch", &ModelHub::fetch, "server_address"_a, "model_id"_a, "dest_path"_a, "version"_a);
    
    // -------------------------------------------------------------------------
    // Shard Placement API
    // -------------------------------------------------------------------------
    
    py::class_<NodeCapacity>(m, "NodeCapacity")
        .def(py::init<>())
        .def_readwrite("nodeId", &NodeCapacity::nodeId)
        .def_readwrite("vramFreeBytes", &NodeCapacity::vramFreeBytes)
        .def_readwrite("hostOffloadBytes", &NodeCapacity::hostOffloadBytes)
        .def_readwrite("diskCacheBytes", &NodeCapacity::diskCacheBytes)
        .def_readwrite("gpuCount", &NodeCapacity::gpuCount)
        .def_readwrite("nicCount", &NodeCapacity::nicCount)
        .def_readwrite("bandwidthGbps", &NodeCapacity::bandwidthGbps)
        .def_readwrite("hasRdma", &NodeCapacity::hasRdma);
    
    py::class_<ClusterCapacity>(m, "ClusterCapacity")
        .def(py::init<>())
        .def_readwrite("nodes", &ClusterCapacity::nodes)
        .def_readwrite("reservedActivationBytes", &ClusterCapacity::reservedActivationBytes);
    
    py::class_<ShardSpec>(m, "ShardSpec")
        .def(py::init<>())
        .def_readwrite("shardId", &ShardSpec::shardId)
        .def_readwrite("contentHash", &ShardSpec::contentHash)
        .def_readwrite("kind", &ShardSpec::kind)
        .def_readwrite("bytes", &ShardSpec::bytes)
        .def_readwrite("layerBegin", &ShardSpec::layerBegin)
        .def_readwrite("layerEnd", &ShardSpec::layerEnd)
        .def_readwrite("mustBeDeviceLocal", &ShardSpec::mustBeDeviceLocal)
        .def_readwrite("mustStayTogether", &ShardSpec::mustStayTogether);
    
    py::class_<ModelManifest>(m, "ModelManifest")
        .def(py::init<>())
        .def_readwrite("modelId", &ModelManifest::modelId)
        .def_readwrite("version", &ModelManifest::version)
        .def_readwrite("shards", &ModelManifest::shards)
        .def_readwrite("estimatedActivationBytes", &ModelManifest::estimatedActivationBytes);
    
    py::class_<ShardPlacement>(m, "ShardPlacement")
        .def_readonly("shardId", &ShardPlacement::shardId)
        .def_readonly("nodeId", &ShardPlacement::nodeId)
        .def_readonly("tier", &ShardPlacement::tier)
        .def_readonly("localDeviceIndex", &ShardPlacement::localDeviceIndex);
    
    py::class_<PlacementPolicy>(m, "PlacementPolicy")
        .def(py::init<>())
        .def_readwrite("allowHostOffload", &PlacementPolicy::allowHostOffload)
        .def_readwrite("allowDiskCache", &PlacementPolicy::allowDiskCache)
        .def_readwrite("preferContiguousLayers", &PlacementPolicy::preferContiguousLayers)
        .def_readwrite("packMode", &PlacementPolicy::packMode)
        .def_readwrite("failFast", &PlacementPolicy::failFast)
        .def_readwrite("transactionalNode", &PlacementPolicy::transactionalNode)
        .def_readwrite("revalidateCapacity", &PlacementPolicy::revalidateCapacity)
        .def_readwrite("bestEffort", &PlacementPolicy::bestEffort);
    
    py::class_<PlacementPlan>(m, "PlacementPlan")
        .def_readonly("assignments", &PlacementPlan::assignments)
        .def_readonly("status", &PlacementPlan::status);
    
    py::class_<ErrorDetail>(m, "ErrorDetail")
        .def_readonly("code", &ErrorDetail::code)
        .def_readonly("message", &ErrorDetail::message)
        .def_readonly("shardId", &ErrorDetail::shardId)
        .def_readonly("nodeId", &ErrorDetail::nodeId)
        .def_readonly("bytesNeeded", &ErrorDetail::bytesNeeded)
        .def_readonly("bytesAvailable", &ErrorDetail::bytesAvailable);
    
    py::class_<Status>(m, "Status")
        .def_readonly("code", &Status::code)
        .def_readonly("details", &Status::details)
        .def("__bool__", [](const Status& s) { return static_cast<bool>(s); });
    
    py::class_<ShardPlacer>(m, "ShardPlacer")
        .def_static("plan", &ShardPlacer::plan, "model"_a, "cluster"_a, "policy"_a);
    
    py::class_<PlacementExecutor>(m, "PlacementExecutor")
        .def(py::init<>())
        .def("execute_local", [](PlacementExecutor& exe, const ModelManifest& model, const PlacementPlan& plan, bool fetch_if_missing = true, bool verify_checksum = true) {
            ExecuteOptions opt;
            opt.fetchIfMissing = fetch_if_missing;
            opt.verifyChecksum = verify_checksum;
            return exe.executeLocal(model, plan, opt);
        }, "model"_a, "plan"_a, "fetch_if_missing"_a = true, "verify_checksum"_a = true);
    
    py::class_<ExecuteOptions>(m, "ExecuteOptions")
        .def(py::init<>())
        .def_readwrite("fetchIfMissing", &ExecuteOptions::fetchIfMissing)
        .def_readwrite("verifyChecksum", &ExecuteOptions::verifyChecksum)
        .def_readwrite("activationReserve", &ExecuteOptions::activationReserve);
    
    py::class_<ExecuteResult>(m, "ExecuteResult")
        .def_readonly("completedShardIds", &ExecuteResult::completedShardIds)
        .def_readonly("failedShardIds", &ExecuteResult::failedShardIds)
        .def_readonly("status", &ExecuteResult::status);
    
    // -------------------------------------------------------------------------
    // Utility functions
    // -------------------------------------------------------------------------
    
    m.def("enumerate_devices", &enumerateDevices, "instance"_a);
    m.def("select_best_device", [](const std::vector<DeviceInfo>& devices, bool prefer_discrete, int min_vram_mib) {
        auto best = selectBestDevice(devices, prefer_discrete, min_vram_mib);
        if (!best) throw std::runtime_error("No suitable device found");
        return *best;
    }, "devices"_a, "prefer_discrete"_a = true, "min_vram_mib"_a = 1024);
    m.def("find_queue_families", &findQueueFamilies, "physical_device"_a);
    m.def("detect_memory_topology", &detectMemoryTopology, "physical_device"_a);
    m.def("create_device", &createDevice, "physical_device"_a, "queues"_a);
    m.def("create_instance", &createInstance);
    m.def("duplicate_for_import", &duplicateForImport, "handle"_a);
    
    // Constants
    m.attr("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT") = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    m.attr("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT") = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    m.attr("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT") = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    m.attr("VK_BUFFER_USAGE_STORAGE_BUFFER_BIT") = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    m.attr("VK_BUFFER_USAGE_TRANSFER_SRC_BIT") = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    m.attr("VK_BUFFER_USAGE_TRANSFER_DST_BIT") = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m.attr("VK_BUFFER_USAGE_VERTEX_BUFFER_BIT") = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    m.attr("VK_BUFFER_USAGE_INDEX_BUFFER_BIT") = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    m.attr("VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT") = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    
    // ExternalHandleType
    py::enum_<ExternalHandleType>(m, "ExternalHandleType")
        .value("OpaqueFd", ExternalHandleType::OpaqueFd)
        .value("OpaqueWin32", ExternalHandleType::OpaqueWin32)
        .value("OpaqueWin32Kmt", ExternalHandleType::OpaqueWin32Kmt)
        .value("DmaBuf", ExternalHandleType::DmaBuf)
        .value("D3D12Heap", ExternalHandleType::D3D12Heap)
        .value("D3D11Texture", ExternalHandleType::D3D11Texture)
        .value("D3D11TextureKmt", ExternalHandleType::D3D11TextureKmt);
    
    // ShardKind already done
    // MemTier already done
    // ErrorCode already done
}