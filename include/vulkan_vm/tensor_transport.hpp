#pragma once

// Unified Tensor Transport
// Includes: TensorMetadata, TensorShape, DataType, MemoryLayout, TransportConfig,
//           Transport interface, collective operations

#include "vulkan_vm/core.hpp"
#include "vulkan_vm/cross_gpu.hpp"
#include "vulkan_vm/network.hpp"

namespace vvm {
namespace tensor {

// ============================================================================
// Data Types
// ============================================================================

enum class DataType {
    Float32 = 0,
    Float16,
    BFloat16,
    Int8,
    Int4,
    UInt8,
    Int32,
    Int64,
    Float8_E4M3,
    Float8_E5M2,
    Bool
};

inline size_t dataTypeSize(DataType dt) {
    switch (dt) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int8: return 1;
        case DataType::Int4: return 1;  // packed
        case DataType::UInt8: return 1;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::Float8_E4M3: return 1;
        case DataType::Float8_E5M2: return 1;
        case DataType::Bool: return 1;
    }
    return 4;
}

// ============================================================================
// Memory Layout
// ============================================================================

enum class MemoryLayout {
    Contiguous,     // Dense, row-major
    ChannelsLast,   // NHWC
    Blocked,        // Tiled for tensor cores
    Strided         // Custom strides
};

// ============================================================================
// Tensor Shape
// ============================================================================

struct TensorShape {
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;  // optional, empty = contiguous
    
    static TensorShape makeContiguous(const std::vector<int64_t>& dims);
    static TensorShape makeChannelsLast(const std::vector<int64_t>& dims);  // NHWC
    static TensorShape makeBlocked(const std::vector<int64_t>& dims, int blockSize);
    
    size_t numel() const {
        size_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
    
    size_t byteSize(DataType dtype) const {
        return numel() * dataTypeSize(dtype);
    }
};

// ============================================================================
// Tensor Metadata
// ============================================================================

struct TensorMetadata {
    DataType dtype = DataType::Float32;
    MemoryLayout layout = MemoryLayout::Contiguous;
    TensorShape shape;
    std::string name;           // Debug name, used for matching send/recv
    uint64_t contentHash = 0;   // Optional: for verification
};

// ============================================================================
// Transport Configuration
// ============================================================================

struct TransportConfig {
    enum class Preference {
        Auto,           // P2P -> RDMA -> HostStaged -> Network
        P2POnly,        // Local multi-GPU only
        RDMAOnly,       // GPU-Direct RDMA
        HostStagedOnly, // CPU-staged copies
        NetworkOnly     // Multi-node
    };
    
    Preference preference = Preference::Auto;
    bool enableAsyncPipeline = true;
    uint32_t maxInFlightTransfers = 4;
    VkDeviceSize hostStagedChunkSize = 4 * 1024 * 1024;  // 4 MB
    
    // RDMA
    std::string rdmaNicName;
    bool enableGPUDirect = true;
    
    // Network
    uint32_t networkPort = 51000;
    bool enableTLS = false;
    std::string tlsCertPath;
    std::string tlsKeyPath;
    std::string tlsCaPath;
};

// ============================================================================
// Tensor Transport Interface
// ============================================================================

struct TensorAllocation {
    Allocation allocation;
    TensorMetadata metadata;
};

using TensorHandle = std::shared_ptr<TensorAllocation>;

enum class ReduceOp {
    Sum, Mean, Min, Max, Product, Band, Bor, Bxor
};

using CompletionCallback = std::function<void(bool success, const std::string& error)>;

class Transport {
public:
    virtual ~Transport() = default;
    
    // Factory
    static std::unique_ptr<Transport> create(
        const TransportConfig& config,
        const std::vector<DeviceConfig>& devices,
        const PoolConfig& poolConfig);
    
    // Lifecycle
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    
    // Tensor allocation
    virtual TensorHandle allocateTensor(const TensorMetadata& meta, uint32_t deviceIndex) = 0;
    virtual std::vector<TensorHandle> allocateDistributed(
        const TensorMetadata& meta, const std::vector<uint32_t>& deviceIndices) = 0;
    
    // Copy with layout conversion
    virtual bool copyTensor(const TensorHandle& src, const TensorHandle& dst) = 0;
    virtual bool copyWithLayoutConversion(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) = 0;
    virtual bool copyTensorAsync(const TensorHandle& src, const TensorHandle& dst, CompletionCallback cb) = 0;
    
    // Collective operations
    virtual bool allReduce(const std::vector<TensorHandle>& tensors, ReduceOp op, const std::vector<uint32_t>& deviceIndices) = 0;
    virtual bool broadcast(const TensorHandle& root, const std::vector<uint32_t>& deviceIndices, uint32_t rootIndex) = 0;
    virtual bool allGather(const std::vector<TensorHandle>& inputs, TensorHandle output, const std::vector<uint32_t>& deviceIndices) = 0;
    virtual bool reduceScatter(const std::vector<TensorHandle>& inputs, TensorHandle output, ReduceOp op, const std::vector<uint32_t>& deviceIndices) = 0;
    
    // Multi-node network
    virtual bool joinCluster(const std::string& bootstrapAddress) = 0;
    virtual std::string getLocalNodeId() const = 0;
    
    // Send/recv by tensor name
    virtual void sendTensor(const TensorHandle& tensor, const std::string& targetNodeId, CompletionCallback cb) = 0;
    virtual void recvTensor(const TensorHandle& tensor, const std::string& sourceNodeId, CompletionCallback cb) = 0;
    
    // Capabilities
    virtual bool supportsP2P() const = 0;
    virtual bool supportsRDMA() const = 0;
    virtual bool supportsNetwork() const = 0;
};

} // namespace tensor
} // namespace vvm