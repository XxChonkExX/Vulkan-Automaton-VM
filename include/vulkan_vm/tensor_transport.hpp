#pragma once

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"

#include <cstdint>
#include <vector>
#include <string>
#include <optional>
#include <functional>
#include <memory>

namespace vvm {
namespace tensor {

// ============================================================================
// Tensor Metadata
// ============================================================================

enum class DataType : uint32_t {
    Unknown = 0,
    Float32 = 1,
    Float16 = 2,
    BFloat16 = 3,
    Int8 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    UInt8 = 8,
    UInt16 = 9,
    UInt32 = 10,
    UInt64 = 11,
    Bool = 12,
    Float8_E4M3 = 13,   // FP8 E4M3
    Float8_E5M2 = 14,   // FP8 E5M2
};

inline size_t dataTypeSize(DataType dt) {
    switch (dt) {
        case DataType::Float32: return 4;
        case DataType::Float16:
        case DataType::BFloat16: return 2;
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool: return 1;
        case DataType::Int16:
        case DataType::UInt16: return 2;
        case DataType::Int32:
        case DataType::UInt32: return 4;
        case DataType::Int64:
        case DataType::UInt64: return 8;
        case DataType::Float8_E4M3:
        case DataType::Float8_E5M2: return 1;
        default: return 0;
    }
}

enum class MemoryLayout : uint32_t {
    Unknown = 0,
    Contiguous = 1,      // Dense, row-major (C-contiguous)
    ChannelsLast = 2,    // NHWC for 4D, channels-last for ND
    Blocked = 3,         // Blocked/tiling for tensor cores (e.g., 32x32x32)
    Strided = 4,         // Arbitrary strided (user-provided strides)
};

enum class ReduceOp : uint32_t {
    Sum = 0,
    Mean = 1,
    Min = 2,
    Max = 3,
    Product = 4,
    Band = 5,
    Bor = 6,
    Bxor = 7,
};

struct TensorShape {
    std::vector<uint64_t> dims;           // Dimensions [N, C, H, W] or [N, H, W, C]
    std::vector<uint64_t> strides;        // Strides in elements (empty = contiguous)
    
    uint64_t numel() const {
        uint64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    uint64_t bytes(DataType dt) const {
        return numel() * dataTypeSize(dims.empty() ? DataType::Float32 : DataType::Float32); // Will be overridden
    }
    
    bool isContiguous() const {
        if (strides.empty()) return true;
        uint64_t expected = 1;
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            if (strides[i] != expected) return false;
            expected *= dims[i];
        }
        return true;
    }
    
    static TensorShape makeContiguous(const std::vector<uint64_t>& dims) {
        TensorShape s;
        s.dims = dims;
        s.strides.resize(dims.size());
        uint64_t stride = 1;
        for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
            s.strides[i] = stride;
            stride *= dims[i];
        }
        return s;
    }
    
    static TensorShape makeChannelsLast(const std::vector<uint64_t>& dims) {
        // NHWC: [N, H, W, C] -> strides [H*W*C, W*C, C, 1]
        TensorShape s;
        s.dims = dims;
        s.strides.resize(dims.size());
        if (dims.size() == 4) {
            s.strides[0] = dims[1] * dims[2] * dims[3]; // N
            s.strides[1] = dims[2] * dims[3];           // H
            s.strides[2] = dims[3];                     // W
            s.strides[3] = 1;                           // C
        } else {
            // Generalized channels-last: last dim is contiguous
            s.strides = TensorShape::makeContiguous(dims).strides;
        }
        return s;
    }
};

struct TensorMetadata {
    DataType dtype = DataType::Float32;
    MemoryLayout layout = MemoryLayout::Contiguous;
    TensorShape shape;
    std::string name;                      // Debug name
    uint64_t version = 0;                  // For versioning/cache invalidation
    
    uint64_t numel() const { return shape.numel(); }
    uint64_t bytes() const { return shape.numel() * dataTypeSize(dtype); }
    
    bool isContiguous() const { return shape.isContiguous(); }
};

// ============================================================================
// Tensor Handle (Opaque)
// ============================================================================

class Tensor {
public:
    Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;
    
    // Accessors
    const TensorMetadata& meta() const { return meta_; }
    TensorMetadata& meta() { return meta_; }
    VkBuffer buffer() const { return buffer_; }
    VkDeviceAddress deviceAddress() const { return device_address_; }
    void* hostPtr() const { return host_ptr_; }
    uint32_t deviceIndex() const { return device_index_; }
    VkDeviceMemory memory() const { return memory_; }
    VkDeviceSize offset() const { return offset_; }
    VkDeviceSize size() const { return size_; }
    bool isValid() const { return buffer_ != VK_NULL_HANDLE; }
    
    // Conversion to Allocation for Vulkan operations (const)
    operator vvm::Allocation() const {
        vvm::Allocation alloc;
        alloc.buffer = buffer_;
        alloc.memory = memory_;
        alloc.offset = offset_;
        alloc.size = size_;
        alloc.deviceAddress = device_address_;
        alloc.hostPtr = host_ptr_;
        alloc.blockIndex = UINT32_MAX; // Dedicated allocation
        return alloc;
    }
    
    // Mutable access for transport internals
    vvm::Allocation asAllocation() {
        vvm::Allocation alloc;
        alloc.buffer = buffer_;
        alloc.memory = memory_;
        alloc.offset = offset_;
        alloc.size = size_;
        alloc.deviceAddress = device_address_;
        alloc.hostPtr = host_ptr_;
        alloc.blockIndex = UINT32_MAX;
        return alloc;
    }
    
    // Const version for const contexts
    vvm::Allocation asAllocation() const {
        vvm::Allocation alloc;
        alloc.buffer = buffer_;
        alloc.memory = memory_;
        alloc.offset = offset_;
        alloc.size = size_;
        alloc.deviceAddress = device_address_;
        alloc.hostPtr = host_ptr_;
        alloc.blockIndex = UINT32_MAX;
        return alloc;
    }
    
    // Factory (called by Transport)
    static std::unique_ptr<Tensor> create(
        const TensorMetadata& meta,
        VkBuffer buffer,
        VkDeviceAddress device_address,
        void* host_ptr,
        uint32_t device_index,
        VkDeviceMemory memory = VK_NULL_HANDLE,
        VkDeviceSize offset = 0,
        VkDeviceSize size = 0
    );
    
    // Layout conversion (returns new tensor with converted layout)
    std::unique_ptr<Tensor> toLayout(MemoryLayout target_layout) const;
    
    // Pin/unpin for async operations
    void pin();
    void unpin();
    bool isPinned() const { return pinned_; }

private:
    TensorMetadata meta_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceAddress device_address_ = 0;
    void* host_ptr_ = nullptr;
    uint32_t device_index_ = 0;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize offset_ = 0;
    VkDeviceSize size_ = 0;
    bool pinned_ = false;
};

// ============================================================================
// Transport Configuration
// ============================================================================

struct TransportConfig {
    // Transport priority (tried in order)
    enum class Preference : uint32_t {
        Auto = 0,           // Auto-select based on hardware
        P2POnly = 1,        // Vendor P2P only (VK_EXTERNAL_MEMORY)
        RDMAOnly = 2,       // GPU-Direct RDMA only
        HostStagedOnly = 3, // Host-staged fallback only
        NetworkOnly = 4,    // Network transport only
    };
    
    Preference preference = Preference::Auto;
    
    // Async pipeline
    bool enableAsyncPipeline = true;      // Overlap compute + transfer
    uint32_t maxInFlightTransfers = 4;    // Max concurrent async transfers
    uint32_t pipelineDepth = 2;           // Pipeline stages
    
    // Chunking
    VkDeviceSize hostStagedChunkSize = 4 * 1024 * 1024; // 4 MiB
    VkDeviceSize maxInlineTransferSize = 256 * 1024;    // 256 KiB inline
    
    // Compression (on wire)
    bool enableCompression = false;
    float compressionRatio = 0.5f;        // Target compression ratio
    
    // Network (if NetworkOnly or fallback)
    std::string networkInterface = "";    // NIC name (empty = auto)
    uint16_t networkPort = 51000;         // Base port
    bool enableTLS = false;
    std::string tlsCertPath = "";
    std::string tlsKeyPath = "";
    std::string tlsCaPath = "";
    
    // RDMA
    std::string rdmaNicName = "";         // e.g., "mlx5_0"
    bool enableGPUDirect = true;          // Use GPU-Direct RDMA
    
    // Layout optimization
    bool autoLayoutConversion = true;     // Auto-convert NHWC↔NCHW
    bool enableTensorCoreTiling = true;   // Use blocked layout for tensor cores
    uint32_t tileSize = 32;               // Tile size for blocked layout
    
    // Collective ops
    bool enableCollectives = true;
    uint32_t collectiveGroupSize = 0;     // 0 = auto (all devices)
};

// ============================================================================
// Transport Interface
// ============================================================================

class Transport {
public:
    using AsyncCallback = std::function<void(bool success, const std::string& error)>;
    
    // Default constructor for derived classes
    Transport() = default;
    
    // Factory
    static std::unique_ptr<Transport> create(
        const TransportConfig& config,
        const std::vector<vvm::DeviceConfig>& devices,
        const vvm::PoolConfig& poolConfig
    );
    
    virtual ~Transport() = default;
    
    // Non-copyable, movable
    Transport(const Transport&) = delete;
    Transport& operator=(const Transport&) = delete;
    Transport(Transport&&) noexcept = default;
    Transport& operator=(Transport&&) noexcept = default;
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool isReady() const = 0;
    
    // ========================================================================
    // Tensor Allocation
    // ========================================================================
    
    // Allocate tensor on specific device(s)
    virtual std::unique_ptr<Tensor> allocateTensor(
        const TensorMetadata& meta,
        uint32_t deviceIndex = 0
    ) = 0;
    
    // Allocate distributed tensor across multiple devices
    virtual std::vector<std::unique_ptr<Tensor>> allocateDistributed(
        const TensorMetadata& meta,
        const std::vector<uint32_t>& deviceIndices
    ) = 0;
    
    // Allocate tensor with specific layout on each device
    virtual std::vector<std::unique_ptr<Tensor>> allocateWithLayout(
        const TensorMetadata& meta,
        const std::vector<uint32_t>& deviceIndices,
        const std::vector<MemoryLayout>& layouts
    ) = 0;
    
    // Free tensor
    virtual void freeTensor(std::unique_ptr<Tensor>&& tensor) = 0;
    
    // ========================================================================
    // Tensor Transfer
    // ========================================================================
    
    // Synchronous copy
    virtual bool copyTensor(
        const Tensor& src,
        Tensor& dst,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // Async copy with callback
    virtual bool copyTensorAsync(
        const Tensor& src,
        Tensor& dst,
        AsyncCallback callback,
        VkSemaphore waitSemaphore = VK_NULL_HANDLE,
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT,
        VkSemaphore signalSemaphore = VK_NULL_HANDLE,
        VkPipelineStageFlags signalStage = VK_PIPELINE_STAGE_TRANSFER_BIT
    ) = 0;
    
    // Copy with layout conversion
    virtual bool copyWithLayoutConversion(
        const Tensor& src,
        Tensor& dst,
        MemoryLayout targetLayout,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // ========================================================================
    // Collective Operations
    // ========================================================================
    
    // All-reduce across devices in group
    virtual bool allReduce(
        const std::vector<Tensor*>& tensors,  // One per device
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // Async all-reduce
    virtual bool allReduceAsync(
        const std::vector<Tensor*>& tensors,
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        AsyncCallback callback
    ) = 0;
    
    // Broadcast from root to group
    virtual bool broadcast(
        Tensor* tensor,                       // Root tensor
        const std::vector<uint32_t>& deviceGroup,
        uint32_t rootDevice,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // All-gather
    virtual bool allGather(
        const std::vector<Tensor*>& inputs,   // One per device
        Tensor* output,                       // Concatenated output
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // Reduce-scatter
    virtual bool reduceScatter(
        const std::vector<Tensor*>& inputs,   // One per device
        Tensor* output,                       // Scattered output
        ReduceOp op,
        const std::vector<uint32_t>& deviceGroup,
        VkFence fence = VK_NULL_HANDLE
    ) = 0;
    
    // ========================================================================
    // Multi-Node Network
    // ========================================================================
    
    // Register with cluster (for network transport)
    virtual bool joinCluster(const std::string& clusterAddress) = 0;
    virtual void leaveCluster() = 0;
    
    // Remote tensor operations
    virtual bool sendTensor(
        const Tensor& tensor,
        const std::string& remoteNodeId,
        AsyncCallback callback = nullptr
    ) = 0;
    
    virtual bool recvTensor(
        Tensor& tensor,
        const std::string& remoteNodeId,
        AsyncCallback callback = nullptr
    ) = 0;
    
    // ========================================================================
    // Synchronization
    // ========================================================================
    
    virtual void flush() = 0;
    virtual void waitIdle() = 0;
    virtual size_t pollCompletions() = 0;
    
    // ========================================================================
    // Introspection
    // ========================================================================
    
    virtual std::string getActiveTransport() const = 0; // "P2P", "RDMA", "HostStaged", "Network"
    virtual bool supportsP2P(uint32_t src, uint32_t dst) const = 0;
    virtual bool supportsRDMA() const = 0;
    virtual std::string getTransportStats() const = 0;
    
    // Access to underlying pools
    virtual vvm::MultiGPUPoolManager* getPoolManager() = 0;
    virtual const std::vector<vvm::DeviceConfig>& getDevices() const = 0;
};

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<Transport> createTensorTransport(
    const TransportConfig& config,
    const std::vector<vvm::DeviceConfig>& devices,
    const vvm::PoolConfig& poolConfig
);

} // namespace tensor
} // namespace vvm