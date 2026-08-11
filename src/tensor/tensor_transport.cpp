#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/core.hpp"
#include "vulkan_vm/cross_gpu.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/utils.hpp"

#if defined(VVM_HAS_UCX)
#include "vulkan_vm/ucx_transport.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <memory>

namespace vvm {
namespace tensor {

namespace {

// ============================================================================
// Host-side reduce primitives (CPU collective fallback path)
//
// The collective implementations below stage GPU buffers through host-visible
// memory when no native compute kernel is available. These helpers translate
// the on-wire scalar types to/from double so one reduction loop covers every
// DataType.
// ============================================================================

bool isIntegralType(DataType dt) {
    switch (dt) {
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Int32:
        case DataType::Int64:
            return true;
        default:
            return false;
    }
}

// Element width in bytes for the types we de/pack element-wise.
size_t packedElemSize(DataType dt) {
    switch (dt) {
        case DataType::Int4: return 1;  // two 4-bit elements per byte
        default: return dataTypeSize(dt);
    }
}

// Read one logical element. For packed Int4, index tells which nibble.
double readElem(DataType dt, const uint8_t* base, size_t index) {
    if (dt == DataType::Int4) {
        const uint8_t raw = base[index / 2];
        const int8_t lo = (raw & 0x8u) ? static_cast<int8_t>(raw | 0xF0u)
                                       : static_cast<int8_t>(raw & 0x0Fu);
        const int8_t hi = (raw & 0x80u) ? static_cast<int8_t>(raw >> 4)
                                        : static_cast<int8_t>(raw & 0xF0u);
        return static_cast<double>(index % 2 ? hi : lo);
    }
    const uint8_t* p = base + index * packedElemSize(dt);
    switch (dt) {
        case DataType::Float32: {
            float v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<double>(v);
        }
        case DataType::Float16: {
            uint16_t h;
            std::memcpy(&h, p, sizeof(h));
            const uint32_t sign = (h & 0x8000u) << 16;
            const uint32_t exp = (h >> 10) & 0x1Fu;
            const uint32_t man = h & 0x3FFu;
            uint32_t f;
            if (exp == 31) {
                f = sign | 0x7F800000u | (man << 13);
            } else if (exp == 0) {
                if (man == 0) {
                    f = sign;
                } else {
                    int e = -14;
                    uint32_t m = man;
                    while (!(m & 0x400u)) { m <<= 1; --e; }
                    f = sign | ((e + 127) << 23) | ((m & 0x3FFu) << 13);
                }
            } else {
                f = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return static_cast<double>(out);
        }
        case DataType::BFloat16: {
            uint16_t h;
            std::memcpy(&h, p, sizeof(h));
            const uint32_t f = static_cast<uint32_t>(h) << 16;
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return static_cast<double>(out);
        }
        case DataType::Float8_E4M3: {
            const uint8_t v = *p;
            const uint32_t sign = (v & 0x80u) << 24;
            const uint32_t exp = (v >> 3) & 0xFu;
            const uint32_t man = v & 0x7u;
            uint32_t f;
            if (exp == 15) {
                f = sign | 0x7F800000u | (man << 23);
            } else if (exp == 0) {
                if (man == 0) {
                    f = sign;
                } else {
                    int e = -6;
                    uint32_t m = man | 0x8u;
                    while (!(m & 0x8u)) { m <<= 1; --e; }
                    f = sign | ((e + 127) << 23) | ((m & 0x7u) << 20);
                }
            } else {
                f = sign | ((exp - 3 + 127) << 23) | (man << 20);
            }
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return static_cast<double>(out);
        }
        case DataType::Float8_E5M2: {
            const uint8_t v = *p;
            const uint32_t sign = (v & 0x80u) << 24;
            const uint32_t exp = (v >> 2) & 0x1Fu;
            const uint32_t man = v & 0x3u;
            uint32_t f;
            if (exp == 31) {
                f = sign | 0x7F800000u | (man << 23);
            } else if (exp == 0) {
                if (man == 0) {
                    f = sign;
                } else {
                    int e = -14;
                    uint32_t m = man;
                    while (!(m & 0x4u)) { m <<= 1; --e; }
                    f = sign | ((e + 127) << 23) | ((m & 0x3u) << 21);
                }
            } else {
                f = sign | ((exp - 15 + 127) << 23) | (man << 21);
            }
            float out;
            std::memcpy(&out, &f, sizeof(out));
            return static_cast<double>(out);
        }
        case DataType::Bool: {
            return (static_cast<int8_t>(*p) != 0) ? 1.0 : 0.0;
        }
        case DataType::Int8: return static_cast<double>(*reinterpret_cast<const int8_t*>(p));
        case DataType::UInt8: return static_cast<double>(*reinterpret_cast<const uint8_t*>(p));
        case DataType::Int32: return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
        case DataType::Int64: return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
        default: return 0.0;
    }
}

void writeElem(DataType dt, uint8_t* base, size_t index, double v) {
    if (dt == DataType::Int4) {
        // Clamp and sign-pack into the nibble: low nibble element 0, high element 1.
        int8_t n = static_cast<int8_t>(std::lround(v));
        if (n < -8) n = -8;
        if (n > 7) n = 7;
        const uint8_t nib = static_cast<uint8_t>(n & 0x0Fu);
        uint8_t& cell = base[index / 2];
        if (index % 2) { cell = static_cast<uint8_t>((cell & 0x0Fu) | (nib << 4)); }
        else { cell = static_cast<uint8_t>((cell & 0xF0u) | nib); }
        return;
    }
    uint8_t* p = base + index * packedElemSize(dt);
    switch (dt) {
        case DataType::Float32: {
            const float f = static_cast<float>(v);
            std::memcpy(p, &f, sizeof(f));
            break;
        }
        case DataType::Float16: {
            const double d = v;
            const uint32_t sign = (std::signbit(d) ? 1u : 0u) << 31;
            const double a = std::fabs(d);
            uint32_t f;
            uint32_t exp32;
            uint32_t man32;
            std::memcpy(&f, &a, sizeof(f));
            exp32 = (f >> 23) & 0xFFu;
            man32 = f & 0x7FFFFFu;
            uint16_t h;
            if (exp32 == 0xFFu) {
                h = static_cast<uint16_t>((sign >> 16) | 0x7C00u | (man32 ? 0x200u : 0u));
            } else {
                const int32_t e = static_cast<int32_t>(exp32) - 127 + 15;
                if (e >= 31) {
                    h = static_cast<uint16_t>((sign >> 16) | 0x7C00u);
                } else if (e <= 0) {
                    if (e < -10) {
                        h = static_cast<uint16_t>(sign >> 16);
                    } else {
                        const uint32_t m = man32 | 0x800000u;
                        const uint32_t shifted = m >> (13 - e);
                        if (e <= -25) h = static_cast<uint16_t>(sign >> 16);
                        else if (m & ((1u << (13 - e)) - 1)) {
                            h = static_cast<uint16_t>((sign >> 16) | (shifted >> 13) | 0x1u);
                        } else h = static_cast<uint16_t>((sign >> 16) | (shifted >> 13));
                    }
                } else {
                    h = static_cast<uint16_t>((sign >> 16) | (static_cast<uint32_t>(e) << 10) | (man32 >> 13));
                }
            }
            std::memcpy(p, &h, sizeof(h));
            break;
        }
        case DataType::BFloat16: {
            const float f = static_cast<float>(v);
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            // Round-to-nearest-even by adding half an ULP in the low 16 bits.
            const uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
            uint16_t out = static_cast<uint16_t>(rounded >> 16);
            std::memcpy(p, &out, sizeof(out));
            break;
        }
        case DataType::Float8_E4M3: {
            const double d = v;
            const uint32_t sign = (std::signbit(d) ? 1u : 0u) << 31;
            const double a = std::fabs(d);
            if (a == 0.0) { *p = static_cast<uint8_t>(sign >> 24); break; }
            uint32_t f;
            std::memcpy(&f, &a, sizeof(f));
            const int32_t e = static_cast<int32_t>((f >> 23) & 0xFFu) - 127;
            const uint32_t man = f & 0x7FFFFFu;
            uint8_t out;
            if (e > 15) { out = 0xFFu; }             // inf (sat max)
            else if (e >= 3) {
                const uint32_t m2 = man >> 20;
                out = static_cast<uint8_t>(sign | ((static_cast<uint32_t>(e + 3)) << 3) | (m2 & 0x7u));
            } else if (e >= -6) {
                const int shift = 20 - (e + 6);
                const uint32_t m = man | 0x800000u;
                const uint32_t m2 = shift >= 0 ? (m >> shift) : (m << (-shift));
                const uint32_t r = shift >= 0 ? (m & ((1u << shift) - 1u)) : 0;
                out = static_cast<uint8_t>(sign | ((m2 >> 3) & 0x7u) | (r ? 1u : 0u));
            } else {
                out = static_cast<uint8_t>(sign);
            }
            *p = out;
            break;
        }
        case DataType::Float8_E5M2: {
            const double d = v;
            const uint32_t sign = (std::signbit(d) ? 1u : 0u) << 31;
            if (d == 0.0) { *p = static_cast<uint8_t>(sign >> 24); break; }
            uint32_t f;
            const double a = std::fabs(d);
            std::memcpy(&f, &a, sizeof(f));
            const int32_t e = static_cast<int32_t>((f >> 23) & 0xFFu) - 127;
            const uint32_t man = f & 0x7FFFFFu;
            uint8_t out;
            if (e > 15) {
                out = static_cast<uint8_t>(0xFFu);  // saturate to inf (E5M2 max)
            } else if (e >= -14) {
                out = static_cast<uint8_t>(sign | ((static_cast<uint32_t>(e + 15)) << 2) | ((man >> 21) & 0x3u));
            } else {
                const int shift = -14 - e + 21;
                const uint32_t m = man | 0x800000u;
                const uint32_t m2 = m >> shift;
                out = static_cast<uint8_t>(sign | (m2 & 0x3u));
            }
            *p = out;
            break;
        }
        case DataType::Bool: {
            *p = static_cast<uint8_t>(v != 0.0 ? 1 : 0);
            break;
        }
        case DataType::Int8: *reinterpret_cast<int8_t*>(p) = static_cast<int8_t>(std::lround(v)); break;
        case DataType::UInt8: *reinterpret_cast<uint8_t*>(p) = static_cast<uint8_t>(std::lround(v)); break;
        case DataType::Int32: *reinterpret_cast<int32_t*>(p) = static_cast<int32_t>(std::llround(v)); break;
        case DataType::Int64: *reinterpret_cast<int64_t*>(p) = static_cast<int64_t>(std::llround(v)); break;
        default: break;
    }
}

// Accumulate `src` into `dst` element-wise: dst[i] = dst[i] OP src[i].
// Bitwise ops run over the raw byte representation of integer types.
bool combineInto(ReduceOp op, DataType dt, uint8_t* dst, const uint8_t* src,
                 size_t byteCount) {
    if (op == ReduceOp::Band || op == ReduceOp::Bor || op == ReduceOp::Bxor) {
        if (!isIntegralType(dt)) {
            VVM_LOG_WARN("Bitwise reduce ops require an integer tensor type");
            return false;
        }
        for (size_t i = 0; i < byteCount; ++i) {
            switch (op) {
                case ReduceOp::Band: dst[i] &= src[i]; break;
                case ReduceOp::Bor:  dst[i] |= src[i]; break;
                case ReduceOp::Bxor: dst[i] ^= src[i]; break;
                default: break;
            }
        }
        return true;
    }

    const size_t elemWidth = packedElemSize(dt);
    const size_t count = byteCount / elemWidth;
    for (size_t i = 0; i < count; ++i) {
        const double a = readElem(dt, dst, i);
        const double b = readElem(dt, src, i);
        double v = a;
        switch (op) {
            case ReduceOp::Sum:
            case ReduceOp::Mean:
                v = a + b;
                break;
            case ReduceOp::Product:
                v = a * b;
                break;
            case ReduceOp::Min:
                v = (a < b) ? a : b;
                break;
            case ReduceOp::Max:
                v = (a > b) ? a : b;
                break;
            default:
                break;
        }
        writeElem(dt, dst, i, v);
    }
    return true;
}

// Initialize the accumulator with the neutral value for `op`.
void initReduceOp(ReduceOp op, DataType dt, uint8_t* dst, size_t byteCount) {
    if (op == ReduceOp::Band) {
        std::fill(dst, dst + byteCount, static_cast<uint8_t>(0xFFu));
        return;
    }
    if (op == ReduceOp::Bor || op == ReduceOp::Bxor) {
        std::memset(dst, 0, byteCount);
        return;
    }
    const size_t elemWidth = packedElemSize(dt);
    const size_t count = byteCount / elemWidth;
    for (size_t i = 0; i < count; ++i) {
        double v = 0.0;
        if (op == ReduceOp::Min) v = std::numeric_limits<double>::infinity();
        else if (op == ReduceOp::Max) v = -std::numeric_limits<double>::infinity();
        else if (op == ReduceOp::Product) v = 1.0;
        writeElem(dt, dst, i, v);
    }
}

} // namespace

// ============================================================================
// Tensor Shape Static Methods
// ============================================================================

TensorShape TensorShape::makeContiguous(const std::vector<int64_t>& dims) {
    TensorShape shape;
    shape.dims = dims;
    if (!dims.empty()) {
        shape.strides.resize(dims.size());
        shape.strides.back() = 1;
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
            shape.strides[i] = shape.strides[i + 1] * dims[i + 1];
        }
    }
    return shape;
}

TensorShape TensorShape::makeChannelsLast(const std::vector<int64_t>& dims) {
    // NHWC: [N, H, W, C] -> strides = [H*W*C, W*C, C, 1]
    TensorShape shape;
    shape.dims = dims;
    if (dims.size() == 4) {
        shape.strides = {
            dims[1] * dims[2] * dims[3],  // N stride: H*W*C
            dims[2] * dims[3],            // H stride: W*C
            dims[3],                      // W stride: C
            1                             // C stride: 1
        };
    }
    return shape;
}

TensorShape TensorShape::makeBlocked(const std::vector<int64_t>& dims, int blockSize) {
    // Simple blocked layout - in practice would be more complex
    return makeContiguous(dims);
}

// ============================================================================
// TensorTransport Implementation
// ============================================================================

class TensorTransportImpl : public Transport {
public:
    TensorTransportImpl(
        const TransportConfig& config,
        const std::vector<vvm::DeviceConfig>& devices,
        const vvm::PoolConfig& poolConfig
    ) : config_(config), devices_(devices), poolConfig_(poolConfig) {}
    
    ~TensorTransportImpl() override {
        shutdown();
    }
    
    // ========================================================================
    // Lifecycle
    // ========================================================================
    
    bool initialize() override {
        if (initialized_) return true;
        
        // Create MultiGPUPoolManager
        auto poolOpt = vvm::MultiGPUPoolManager::create(devices_, poolConfig_, 0);
        if (!poolOpt) {
            VVM_LOG_ERROR("Failed to create MultiGPUPoolManager");
            return false;
        }
        poolManager_ = std::move(poolOpt);
        
        // Initialize UCX transport (if enabled)
#if defined(VVM_HAS_UCX)
        if (config_.enableUCX || 
            config_.preference == TransportConfig::Preference::UCXOnly ||
            config_.preference == TransportConfig::Preference::Auto) {
            if (!initUCXTransport()) {
                VVM_LOG_WARN("UCX transport initialization failed");
            }
        }
#endif
        
        // Initialize network transport
        if (config_.preference == TransportConfig::Preference::NetworkOnly ||
            config_.preference == TransportConfig::Preference::Auto) {
            if (!initNetworkTransport()) {
                VVM_LOG_WARN("Network transport initialization failed");
            }
        }

        // Wire UCX transport into the network manager so export/migration
        // paths can use UCX RMA when both peers support it.
#if defined(VVM_HAS_UCX)
        if (ucxTransport_ && networkManager_) {
            networkManager_->setUcxTransport(ucxTransport_.get());
        }
#endif

        // Start async processing thread
        if (config_.enableAsyncPipeline) {
            stopAsyncThread_ = false;
            asyncThread_ = std::thread(&TensorTransportImpl::asyncProcessingLoop, this);
            asyncThreadId_ = asyncThread_.get_id();
        }
        
        initialized_ = true;
        VVM_LOG_INFO("TensorTransport initialized with {} devices", devices_.size());
        return true;
    }
    
    void shutdown() override {
        if (!initialized_) return;
        
        stopAsyncThread_ = true;
        asyncCV_.notify_all();
        if (asyncThread_.joinable()) {
            asyncThread_.join();
        }
        
#if defined(VVM_HAS_UCX)
        if (ucxTransport_) {
            ucxTransport_->shutdown();
            ucxTransport_.reset();
        }
#endif
        
        if (networkManager_) {
            networkManager_->stop();
            networkManager_.reset();
        }
        
        poolManager_.reset();
        initialized_ = false;
        VVM_LOG_INFO("TensorTransport shut down");
    }
    
    bool isReady() const {
        return initialized_ && poolManager_.has_value();
    }
    
    // ========================================================================
    // Tensor Allocation
    // ========================================================================
    
    TensorHandle allocateTensor(const TensorMetadata& meta, uint32_t deviceIndex) override {
        if (!isReady() || deviceIndex >= devices_.size()) return nullptr;
        
        auto& pool = poolManager_->getPool(deviceIndex);
        
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        
        vvm::AllocDesc desc;
        desc.size = meta.bytes();
        desc.usage = usage;
        desc.memoryUsage = vvm::MemoryUsage::GpuOnly;
        desc.exportable = false;
        desc.mapped = false;
        desc.name = meta.name;
        
        auto alloc = pool.allocate(desc);
        if (!alloc) {
            VVM_LOG_ERROR("Failed to allocate tensor: {}", meta.name);
            return nullptr;
        }
        
        auto handle = std::make_shared<TensorAllocation>();
        handle->allocation = std::move(*alloc);
        handle->metadata = meta;
        handle->deviceIndex = deviceIndex;
        return handle;
    }
    
    std::vector<TensorHandle> allocateDistributed(
        const TensorMetadata& meta, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady()) return {};
        
        std::vector<TensorHandle> results;
        results.reserve(deviceIndices.size());
        
        if (deviceIndices.empty()) return results;
        
        // Allocate on first device (master)
        uint32_t masterIdx = deviceIndices[0];
        auto masterHandle = allocateTensor(meta, masterIdx);
        if (!masterHandle) return {};
        
        results.push_back(masterHandle);
        
        // Export from master and import on peers
        for (size_t i = 1; i < deviceIndices.size(); ++i) {
            uint32_t peerIdx = deviceIndices[i];
            
            // Get peer access info
            auto peerInfo = poolManager_->queryPeerAccess(deviceIndices[0], peerIdx);
            if (!peerInfo.canDirectCopy) {
                VVM_LOG_WARN("Cannot direct copy from {} to {}", deviceIndices[0], peerIdx);
                return {};
            }
            
            // Export from master
            auto exportInfo = poolManager_->getPool(deviceIndices[0]).exportMemory(
                masterHandle->allocation, peerInfo.recommendedType);
            if (!exportInfo) {
                VVM_LOG_ERROR("Failed to export tensor from master");
                return {};
            }
            
            // Import on peer
            auto importInfo = vvm::duplicateForImport(*exportInfo);
            importInfo.type = peerInfo.recommendedType;
            importInfo.size = exportInfo->size;
            importInfo.memoryTypeIndex = exportInfo->memoryTypeIndex;
            importInfo.dedicatedAllocation = exportInfo->dedicatedAllocation;
            
            auto& peerPool = poolManager_->getPool(peerIdx);
            auto peerAlloc = peerPool.importMemory(std::move(importInfo), 
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            
            if (!peerAlloc) {
                VVM_LOG_ERROR("Failed to import tensor on peer {}", peerIdx);
                return {};
            }
            
            // Create tensor wrapper
            auto peerHandle = std::make_shared<TensorAllocation>();
            peerHandle->allocation = std::move(*peerAlloc);
            peerHandle->metadata = results[0]->metadata;
            peerHandle->deviceIndex = peerIdx;
            results.push_back(peerHandle);
        }

        // Keep a per-device view of the distributed tensor so collectives can
        // address peers by device index.
        for (size_t i = 0; i < deviceIndices.size(); ++i) {
            distributedTensors_[deviceIndices[i]] = results[i];
        }

        return results;
    }
    
    // ========================================================================
    // Copy Operations
    // ========================================================================
    
    bool copyTensor(const TensorHandle& src, const TensorHandle& dst) override {
        if (!isReady() || !src || !dst) return false;
        if (src->metadata.bytes() != dst->metadata.bytes()) {
            VVM_LOG_ERROR("Tensor size mismatch in copy");
            return false;
        }
        
        auto& srcPool = poolManager_->getPool(src->deviceIndex);
        auto& dstPool = poolManager_->getPool(dst->deviceIndex);
        
        // Same device - direct copy
        if (src->deviceIndex == dst->deviceIndex) {
            return srcPool.copyBuffer(src->allocation, dst->allocation, 0, 0, src->metadata.bytes());
        }
        
        // Cross-device copy via MultiGPUPoolManager
        return poolManager_->copyDeviceToDevice(
            src->deviceIndex, dst->deviceIndex,
            src->allocation, dst->allocation, 0, 0, src->metadata.bytes());
    }
    
    // Copy a slice of a tensor
    bool copyTensorPartial(const TensorHandle& src, const TensorHandle& dst, 
                           size_t srcOffset, size_t dstOffset, size_t size) override {
        if (!isReady() || !src || !dst) return false;
        if (srcOffset + size > src->metadata.bytes() || dstOffset + size > dst->metadata.bytes()) {
            VVM_LOG_ERROR("copyTensorPartial: offset + size exceeds tensor bounds");
            return false;
        }
        
        auto& srcPool = poolManager_->getPool(src->deviceIndex);
        auto& dstPool = poolManager_->getPool(dst->deviceIndex);
        
        // Same device - direct copy
        if (src->deviceIndex == dst->deviceIndex) {
            return srcPool.copyBuffer(src->allocation, dst->allocation, srcOffset, dstOffset, size);
        }
        
        // Cross-device copy via MultiGPUPoolManager
        return poolManager_->copyDeviceToDevice(
            src->deviceIndex, dst->deviceIndex,
            src->allocation, dst->allocation, srcOffset, dstOffset, size);
    }
    
    bool copyWithLayoutConversion(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) override {
        if (!isReady() || !src || !dst) return false;
        
        // If layouts match, just do a plain copy
        if (src->metadata.layout == targetLayout) {
            return copyTensor(src, dst);
        }
        
        // Need layout conversion - use compute shader
        if (!convertLayoutShader(src, dst, targetLayout)) {
            VVM_LOG_WARN("Layout conversion shader failed for {}", static_cast<int>(targetLayout));
            // Fall back to an honest CPU permute so the conversion is applied
            // even when the SPIR-V shader is unavailable.
            return convertLayoutCpuFallback(src, dst, targetLayout);
        }
        
        return true;
    }
    
    // Host-side NHWC <-> NCHW permute. Also covers non-4D tensors with a flat
    // copy, where the layout flags have no byte-level meaning.
    bool convertLayoutCpuFallback(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) {
        const size_t bytes = src->metadata.bytes();
        if (dst->metadata.bytes() != bytes) {
            VVM_LOG_ERROR("convertLayoutCpuFallback: size mismatch (src={}, dst={})", bytes, dst->metadata.bytes());
            return false;
        }
        if (bytes == 0) return true;
        
        std::vector<uint8_t> hostSrc(bytes);
        std::vector<uint8_t> hostDst(bytes);
        if (!gpuToHost(src, hostSrc.data(), 0, bytes)) {
            VVM_LOG_ERROR("convertLayoutCpuFallback: gpuToHost failed");
            return false;
        }
        
        const TensorShape& shape = src->metadata.shape;
        const size_t elemSize = packedElemSize(src->metadata.dtype);
        if (shape.dims.size() < 4) {
            std::memcpy(hostDst.data(), hostSrc.data(), bytes);
        } else {
            const uint32_t N = static_cast<uint32_t>(shape.dims[0]);
            const uint32_t H = static_cast<uint32_t>(shape.dims[1]);
            const uint32_t W = static_cast<uint32_t>(shape.dims[2]);
            const uint32_t C = static_cast<uint32_t>(shape.dims[3]);
            const bool toNchw = (src->metadata.layout == MemoryLayout::ChannelsLast &&
                                 targetLayout == MemoryLayout::Contiguous);
            
            const uint64_t planes = static_cast<uint64_t>(N) * H * W;
            const uint8_t* s = hostSrc.data();
            uint8_t* d = hostDst.data();
            for (uint64_t i = 0; i < planes; ++i) {
                for (uint32_t c = 0; c < C; ++c) {
                    const uint64_t srcOffset = toNchw
                        ? (i * C + c) * elemSize
                        : (c * planes + i) * elemSize;
                    const uint64_t dstOffset = toNchw
                        ? (c * planes + i) * elemSize
                        : (i * C + c) * elemSize;
                    std::memcpy(d + dstOffset, s + srcOffset, elemSize);
                }
            }
        }
        
        if (!hostToGpu(dst, 0, hostDst.data(), bytes)) {
            VVM_LOG_ERROR("convertLayoutCpuFallback: hostToGpu failed");
            return false;
        }
        return true;
    }
    
    // ========================================================================
    // Layout Conversion Shaders
    // ========================================================================
    
    bool convertLayoutShader(const TensorHandle& src, const TensorHandle& dst, MemoryLayout targetLayout) {
        // For now, implement basic NHWC↔NCHW conversion
        // In production, would have a library of conversion shaders
        
        if (src->metadata.layout == MemoryLayout::ChannelsLast && targetLayout == MemoryLayout::Contiguous) {
            return runLayoutConversion(src, dst, "NHWC_to_NCHW");
        } else if (src->metadata.layout == MemoryLayout::Contiguous && targetLayout == MemoryLayout::ChannelsLast) {
            return runLayoutConversion(src, dst, "NCHW_to_NHWC");
        }
        
        VVM_LOG_WARN("Layout conversion from {} to {} not yet implemented",
                     static_cast<int>(src->metadata.layout), static_cast<int>(targetLayout));
        return false;
    }
    
    bool runLayoutConversion(const TensorHandle& src, const TensorHandle& dst, const std::string& shaderName) {
        // Get source and destination device pools
        auto& srcPool = poolManager_->getPool(src->allocation.blockIndex);
        auto& dstPool = poolManager_->getPool(dst->allocation.blockIndex);
        
        // Get device and queue for the source device
        uint32_t srcDeviceIdx = src->allocation.blockIndex;
        uint32_t dstDeviceIdx = dst->allocation.blockIndex;
        
        if (srcDeviceIdx >= devices_.size() || dstDeviceIdx >= devices_.size()) {
            return false;
        }
        
        // For cross-device conversion, we need to use the appropriate device's compute queue
        // For now, implement same-device conversion
        if (srcDeviceIdx != dstDeviceIdx) {
            VVM_LOG_WARN("Cross-device layout conversion not yet implemented");
            return false;
        }
        
        auto& pool = poolManager_->getPool(srcDeviceIdx);
        VkDevice device = pool.getDevice();
        VkQueue computeQueue = devices_[srcDeviceIdx].computeQueue;
        uint32_t computeQueueFamily = devices_[srcDeviceIdx].computeQueueFamily;
        
        // Create shader module for layout conversion
        VkShaderModule shaderModule = createLayoutConversionShader(device, shaderName);
        if (shaderModule == VK_NULL_HANDLE) {
            return false;
        }
        
        // Create pipeline layout
        VkPipelineLayout pipelineLayout;
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 0;
        layoutInfo.pushConstantRangeCount = 1;
        
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = 10 * sizeof(uint32_t); // N, H, W, C, layoutMode, blockSize, strides (x4)
        layoutInfo.pPushConstantRanges = &pushConstant;
        
        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create compute pipeline
        VkPipeline pipeline;
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS) {
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create command buffer
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = computeQueueFamily;
        
        VkCommandPool cmdPool;
        if (vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool) != VK_SUCCESS) {
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer cmdBuffer;
        if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuffer) != VK_SUCCESS) {
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmdBuffer, &beginInfo);
        
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        
        // Bind source and destination buffers as storage buffers
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
        setLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        setLayoutInfo.bindingCount = 2;
        setLayoutInfo.pBindings = bindings;
        
        VkDescriptorSetLayout setLayout;
        if (vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &setLayout) != VK_SUCCESS) {
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Create descriptor pool and set
        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;
        
        VkDescriptorPoolCreateInfo poolCreateInfo{};
        poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolCreateInfo.maxSets = 1;
        poolCreateInfo.poolSizeCount = 2;
        poolCreateInfo.pPoolSizes = poolSizes;
        
        VkDescriptorPool descPool;
        if (vkCreateDescriptorPool(device, &poolCreateInfo, nullptr, &descPool) != VK_SUCCESS) {
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        VkDescriptorSetAllocateInfo setAllocInfo{};
        setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        setAllocInfo.descriptorPool = descPool;
        setAllocInfo.descriptorSetCount = 1;
        setAllocInfo.pSetLayouts = &setLayout;
        
        VkDescriptorSet descSet;
        if (vkAllocateDescriptorSets(device, &setAllocInfo, &descSet) != VK_SUCCESS) {
            vkDestroyDescriptorPool(device, descPool, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        // Update descriptor set
        VkDescriptorBufferInfo srcBufferInfo{};
        srcBufferInfo.buffer = src->allocation.buffer;
        srcBufferInfo.offset = 0;
        srcBufferInfo.range = src->metadata.bytes();
        
        VkDescriptorBufferInfo dstBufferInfo{};
        dstBufferInfo.buffer = dst->allocation.buffer;
        dstBufferInfo.offset = 0;
        dstBufferInfo.range = dst->metadata.bytes();
        
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &srcBufferInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &dstBufferInfo;
        
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
        
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        
        // Push constants: N, H, W, C, layoutMode, blockSize, strides (x4)
        TensorShape shape = src->metadata.shape;
        uint32_t pushConstants[10] = {};
        if (shape.dims.size() >= 4) {
            pushConstants[0] = static_cast<uint32_t>(shape.dims[0]); // N
            pushConstants[1] = static_cast<uint32_t>(shape.dims[1]); // H
            pushConstants[2] = static_cast<uint32_t>(shape.dims[2]); // W
            pushConstants[3] = static_cast<uint32_t>(shape.dims[3]); // C
        } else {
            // Default to 1,1,1,size
            pushConstants[0] = 1;
            pushConstants[1] = 1;
            pushConstants[2] = 1;
            pushConstants[3] = static_cast<uint32_t>(
                src->metadata.bytes() / packedElemSize(src->metadata.dtype));
        }
        if (shaderName == "NHWC_to_NCHW") {
            pushConstants[4] = 0;
        } else if (shaderName == "NCHW_to_NHWC") {
            pushConstants[4] = 1;
        } else {
            pushConstants[4] = UINT32_MAX; // unsupported mode; shader writes nothing
        }
        pushConstants[5] = 1; // blockSize (unused for modes 0/1)
        // Strides (modes 0/1 are linear-to-linear; N strides only meaningful in mode 3)
        
        vkCmdPushConstants(cmdBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
        
        // Dispatch
        uint32_t totalElements = pushConstants[0] * pushConstants[1] * pushConstants[2] * pushConstants[3];
        uint32_t groupSize = 256;
        uint32_t numGroups = (totalElements + groupSize - 1) / groupSize;
        vkCmdDispatch(cmdBuffer, numGroups, 1, 1);
        
        vkEndCommandBuffer(cmdBuffer);
        
        // Submit
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;
        
        VkFence fence;
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        vkCreateFence(device, &fenceInfo, nullptr, &fence);
        
        VkResult result = vkQueueSubmit(computeQueue, 1, &submitInfo, fence);
        if (result != VK_SUCCESS) {
            vkDestroyFence(device, fence, nullptr);
            vkDestroyDescriptorPool(device, descPool, nullptr);
            vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
            vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
            vkDestroyCommandPool(device, cmdPool, nullptr);
            vkDestroyPipeline(device, pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyShaderModule(device, shaderModule, nullptr);
            return false;
        }
        
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        
        // Cleanup
        vkDestroyFence(device, fence, nullptr);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
        vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyShaderModule(device, shaderModule, nullptr);
        
        return true;
    }
    
    VkShaderModule createLayoutConversionShader(VkDevice device, const std::string& shaderName) {
        // Load compiled SPIR-V shader from file
        // The shader is compiled at build time by glslangValidator
        
        std::string shaderPath;
        if (shaderName == "NHWC_to_NCHW" || shaderName == "NCHW_to_NHWC") {
            shaderPath = "shaders/layout_conversion.spv";
        } else {
            VVM_LOG_WARN("Unknown layout conversion shader: {}", shaderName);
            return VK_NULL_HANDLE;
        }
        
        // Try to load from build directory first, then source directory
        std::vector<std::string> searchPaths = {
            "shaders/layout_conversion.spv",  // Build directory
            "../shaders/layout_conversion.spv",  // From build dir to source
            "shaders/layout_conversion.spv"   // Current directory
        };
        
        std::vector<uint32_t> spirvCode;
        bool loaded = false;
        
        for (const auto& path : searchPaths) {
            std::ifstream file(path, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                std::streamsize size = file.tellg();
                file.seekg(0, std::ios::beg);
                
                if (size > 0 && size % 4 == 0) {
                    spirvCode.resize(size / 4);
                    file.read(reinterpret_cast<char*>(spirvCode.data()), size);
                    if (file) {
                        VVM_LOG_INFO("Loaded layout conversion shader from: {}", path);
                        loaded = true;
                        break;
                    }
                }
                file.close();
            }
        }
        
if (!loaded) {
            // No synthetic fallback: a no-op module would let the caller
            // believe the layout was converted while the data stays in the
            // source order. Fail loudly; callers switch to the CPU fallback.
            VVM_LOG_ERROR("Failed to load layout conversion shader from any search path");
            return VK_NULL_HANDLE;
        }
        
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
        createInfo.pCode = spirvCode.data();
        
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            VVM_LOG_ERROR("Failed to create layout conversion shader module");
            return VK_NULL_HANDLE;
        }
        
        VVM_LOG_INFO("Created layout conversion shader module from SPIR-V ({} words)", spirvCode.size());
        return shaderModule;
    }
    
    bool copyTensorAsync(const TensorHandle& src, const TensorHandle& dst, CompletionCallback cb) override {
        enqueueAsync([this, src, dst, cb]() {
            bool ok = copyTensor(src, dst);
            if (cb) cb(ok, ok ? "" : "Copy failed");
        });
        return true;
    }
    
    // ========================================================================
    // Collective Operations
    // ========================================================================
    
    bool allReduce(const std::vector<TensorHandle>& tensors, ReduceOp op, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || tensors.size() < 2 || tensors.size() != deviceIndices.size()) return false;

        for (const auto& t : tensors) {
            if (!t) return false;
            if (t->metadata.dtype != tensors[0]->metadata.dtype) {
                VVM_LOG_ERROR("allReduce: mixed dtypes in one collective group");
                return false;
            }
        }
        const size_t bytes = tensors[0]->metadata.bytes();
        if (bytes == 0) return true;

        VVM_LOG_INFO("allReduce: {} participants, {} bytes, op {}", tensors.size(), bytes,
                     static_cast<int>(op));

        std::vector<uint8_t> acc;
        if (!reduceAllToHost(tensors, op, acc, bytes)) return false;

        // Result is identical on every participant.
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!hostToGpu(tensors[i], 0, acc.data(), bytes)) {
                VVM_LOG_ERROR("allReduce: write-back to device {} failed", deviceIndices[i]);
                return false;
            }
        }
        return true;
    }

    bool allReduceAsync(const std::vector<TensorHandle>& tensors, ReduceOp op, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, tensors, op, deviceIndices, cb]() {
            bool ok = allReduce(tensors, op, deviceIndices);
            if (cb) cb(ok, ok ? "" : "allReduce failed");
        });
        return true;
    }

    bool broadcast(const TensorHandle& root, const std::vector<uint32_t>& deviceIndices, uint32_t rootIndex) override {
        if (!isReady() || deviceIndices.empty() || !root) return false;
        if (rootIndex >= deviceIndices.size()) {
            VVM_LOG_ERROR("broadcast: rootIndex {} out of range", rootIndex);
            return false;
        }

        const uint32_t rootDevice = deviceIndices[rootIndex];
        if (root->deviceIndex != rootDevice) {
            VVM_LOG_ERROR("broadcast: root tensor lives on device {} but rootIndex maps to device {}",
                          root->deviceIndex, rootDevice);
            return false;
        }

        VVM_LOG_INFO("broadcast: root device {} -> {} participants", rootDevice, deviceIndices.size());
        for (size_t i = 0; i < deviceIndices.size(); ++i) {
            if (i == rootIndex) continue;
            auto it = distributedTensors_.find(deviceIndices[i]);
            if (it == distributedTensors_.end() || !it->second) {
                VVM_LOG_WARN("broadcast: no distributed peer registered for device {} (use allocateDistributed)",
                             deviceIndices[i]);
                return false;
            }
            if (!copyTensor(root, it->second)) {
                VVM_LOG_ERROR("broadcast: copy to device {} failed", deviceIndices[i]);
                return false;
            }
        }
        return true;
    }

    bool broadcastAsync(const TensorHandle& root, const std::vector<uint32_t>& deviceIndices, uint32_t rootIndex, CompletionCallback cb) override {
        enqueueAsync([this, root, deviceIndices, rootIndex, cb]() {
            bool ok = broadcast(root, deviceIndices, rootIndex);
            if (cb) cb(ok, ok ? "" : "broadcast failed");
        });
        return true;
    }

    bool allGather(const std::vector<TensorHandle>& inputs, TensorHandle output, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || inputs.empty() || inputs.size() != deviceIndices.size()) return false;
        for (const auto& in : inputs) {
            if (!in || in->metadata.bytes() != inputs[0]->metadata.bytes()) {
                VVM_LOG_ERROR("allGather: all inputs must have the same size");
                return false;
            }
        }
        if (!output) return false;

        const size_t totalBytes = inputs[0]->metadata.bytes() * inputs.size();
        if (output->metadata.bytes() != totalBytes) {
            VVM_LOG_ERROR("allGather: output size ({}) doesn't match total input size ({})",
                          output->metadata.bytes(), totalBytes);
            return false;
        }

        VVM_LOG_INFO("allGather: {} participants, concatenating into {} bytes",
                     inputs.size(), totalBytes);

        size_t offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (!copyTensorPartial(inputs[i], output, 0, offset, inputs[i]->metadata.bytes())) {
                VVM_LOG_ERROR("allGather: copy from input {} failed", i);
                return false;
            }
            offset += inputs[i]->metadata.bytes();
        }
        return true;
    }

    bool allGatherAsync(const std::vector<TensorHandle>& inputs, TensorHandle output, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, inputs, output, deviceIndices, cb]() {
            bool ok = allGather(inputs, output, deviceIndices);
            if (cb) cb(ok, ok ? "" : "allGather failed");
        });
        return true;
    }

    bool reduceScatter(const std::vector<TensorHandle>& inputs, TensorHandle output, ReduceOp op, const std::vector<uint32_t>& deviceIndices) override {
        if (!isReady() || inputs.empty() || inputs.size() != deviceIndices.size() || !output) return false;

        size_t inputBytes = inputs[0]->metadata.bytes();
        for (const auto& input : inputs) {
            if (!input || input->metadata.bytes() != inputBytes) {
                VVM_LOG_ERROR("reduceScatter: all inputs must have the same size");
                return false;
            }
        }

        const size_t chunkSize = inputBytes / inputs.size();
        if (chunkSize * inputs.size() != inputBytes) {
            VVM_LOG_ERROR("reduceScatter: input size {} not divisible by {} participants", inputBytes, inputs.size());
            return false;
        }
        if (output->metadata.bytes() != chunkSize) {
            VVM_LOG_ERROR("reduceScatter: output size ({}) doesn't match chunk size ({})",
                          output->metadata.bytes(), chunkSize);
            return false;
        }
        if (inputs[0]->metadata.dtype != output->metadata.dtype) {
            VVM_LOG_ERROR("reduceScatter: output dtype must match inputs");
            return false;
        }

        VVM_LOG_INFO("reduceScatter: {} participants, chunk {} bytes", inputs.size(), chunkSize);

        std::vector<uint8_t> acc;
        if (!reduceAllToHost(inputs, op, acc, inputBytes)) return false;

        // Each participant keeps the chunk of the reduced result that belongs to
        // its rank in this team. The caller's output device determines its rank.
        size_t rank = 0;
        for (size_t i = 0; i < deviceIndices.size(); ++i) {
            if (deviceIndices[i] == output->allocation.blockIndex) {
                rank = i;
                break;
            }
        }

        if (!hostToGpu(output, 0, acc.data() + rank * chunkSize, chunkSize)) {
            VVM_LOG_ERROR("reduceScatter: write chunk {} to device {} failed",
                          rank, output->allocation.blockIndex);
            return false;
        }
        return true;
    }
    
    bool reduceScatterAsync(const std::vector<TensorHandle>& inputs, TensorHandle output, ReduceOp op, const std::vector<uint32_t>& deviceIndices, CompletionCallback cb) override {
        enqueueAsync([this, inputs, output, op, deviceIndices, cb]() {
            bool ok = reduceScatter(inputs, output, op, deviceIndices);
            if (cb) cb(ok, ok ? "" : "reduceScatter failed");
        });
        return true;
    }
    
    // ========================================================================
    // UCX Transport
    // ========================================================================
    
#if defined(VVM_HAS_UCX)
    bool initUCXTransport() {
        if (!poolManager_) return false;
        
        ucxTransport_ = std::make_unique<vvm::tensor::UcxTransport>();
        
        vvm::tensor::UcxTransportConfig ucxConfig;
        ucxConfig.workerThreadCount = 1;
        ucxConfig.enableRndv = config_.ucxEnableRndv;
        ucxConfig.rndvThreshold = config_.ucxRndvThreshold;
        ucxConfig.enableGpuMemory = config_.ucxEnableGPUMem;
        ucxConfig.tls = config_.ucxTLS;
        ucxConfig.netDevices = config_.ucxNetDevices;
        ucxConfig.enableCudaIpc = config_.ucxEnableCudaIpc;
        
        if (!ucxTransport_->initialize(ucxConfig)) {
            VVM_LOG_ERROR("Failed to initialize UCX transport");
            ucxTransport_.reset();
            return false;
        }

        // Start background progress thread so UCX async ops advance without
        // requiring the caller to manually drive progress().
        if (config_.enableAsyncPipeline) {
            if (!ucxTransport_->startProgressThread()) {
                VVM_LOG_WARN("UCX progress thread failed to start");
            }
        }

        VVM_LOG_INFO("UCX transport initialized (GPU mem: {}, RNDV: {})",
                     config_.ucxEnableGPUMem ? "on" : "off",
                     config_.ucxEnableRndv ? "on" : "off");
        return true;
    }
#endif
    
    // ========================================================================
    // Multi-Node Network
    // ========================================================================
    
    bool initNetworkTransport() {
        if (!poolManager_) return false;
        
        vvm::network::NetworkConfig netConfig;
        netConfig.listenAddress = "0.0.0.0:" + std::to_string(config_.networkPort);
        netConfig.useTls = config_.enableTLS;
        
        if (config_.enableTLS) {
            netConfig.tlsCertPath = config_.tlsCertPath;
            netConfig.tlsKeyPath = config_.tlsKeyPath;
            netConfig.tlsCaPath = config_.tlsCaPath;
        }
        
        netConfig.seedNodes.clear(); // Bootstrap node
        
        auto poolOpt = vvm::network::MultiNodePoolManager::create(
            devices_, poolConfig_, netConfig);
        if (!poolOpt) {
            VVM_LOG_ERROR("Failed to create MultiNodePoolManager");
            return false;
        }
        networkManager_ = std::make_unique<vvm::network::MultiNodePoolManager>(std::move(*poolOpt));
        
        networkManager_->start();
        VVM_LOG_INFO("Network transport initialized on port {}", config_.networkPort);
        return true;
    }
    
    bool joinCluster(const std::string& bootstrapAddress) override {
        if (!networkManager_) {
            if (!initNetworkTransport()) return false;
        }
        
        // Parse bootstrap address and connect
        // For simplicity, assume bootstrapAddress is "host:port"
        vvm::network::NetworkConfig netConfig;
        netConfig.seedNodes = { bootstrapAddress };
        
        networkManager_->registerWithCluster();
        VVM_LOG_INFO("Joined cluster via {}", bootstrapAddress);
        return true;
    }
    
    std::string getLocalNodeId() const override {
        if (networkManager_) {
            return networkManager_->getLocalNodeId().toString();
        }
        return "";
    }
    
    // Send/recv by tensor name
    void sendTensor(const TensorHandle& tensor, const std::string& targetNodeId, CompletionCallback cb) override {
        if (!networkManager_ || !tensor) {
            if (cb) cb(false, "Network not ready or invalid tensor");
            return;
        }

        // Export tensor for remote access
        auto rdmaOk = config_.preference == TransportConfig::Preference::RDMAOnly ||
                      config_.enableGPUDirect;
        auto desc = networkManager_->exportForRemote(tensor->allocation, rdmaOk, !rdmaOk);
        if (!desc) {
            if (cb) cb(false, "exportForRemote failed");
            return;
        }

        // Augment with UCX transport info if available. This lets the peer
        // pull data over UCX (InfiniBand/RoCE) instead of verbs/TCP.
#if defined(VVM_HAS_UCX)
        if (ucxTransport_ && ucxTransport_->isInitialized()) {
            auto descCopy = *desc;
            if (networkManager_->exportForRemoteUcx(descCopy, tensor->allocation, tensor->deviceIndex)) {
                *desc = std::move(descCopy);
                VVM_LOG_INFO("sendTensor: UCX export enabled for tensor '{}'", tensor->metadata.name);
            }
        }
#endif

        // Parse target node ID
        auto targetNode = vvm::network::NodeId::fromString(targetNodeId);

        // Announce the tensor to target node
        if (!networkManager_->announceRemoteTensor(targetNode, tensor->metadata.name, *desc)) {
            if (cb) cb(false, "announceRemoteTensor failed");
            return;
        }

        if (cb) cb(true, "");
    }
    
    void recvTensor(const TensorHandle& tensor, const std::string& sourceNodeId, CompletionCallback cb) override {
        if (!networkManager_ || !tensor) {
            if (cb) cb(false, "Network not ready or invalid tensor");
            return;
        }

        // Parse source node ID
        auto sourceNode = vvm::network::NodeId::fromString(sourceNodeId);

        // Wait for tensor announcement from source
        const uint64_t kRecvTimeoutNs = 30ull * 1000 * 1000 * 1000; // 30s
        auto desc = networkManager_->waitRemoteTensor(sourceNode, tensor->metadata.name, kRecvTimeoutNs);
        if (!desc) {
            if (cb) cb(false, "waitRemoteTensor timed out");
            return;
        }

        // Prefer UCX path if both sides support it.
#if defined(VVM_HAS_UCX)
        if (ucxTransport_ && ucxTransport_->isInitialized() && desc->canUseUcx()) {
            VVM_LOG_INFO("recvTensor: using UCX path for tensor '{}'", tensor->metadata.name);
            // Use async pipeline to drive the UCX transfer and completion.
            enqueueAsync([this, desc = *desc, tensor, cb]() {
                bool ok = networkManager_->migrateFromRemoteUcx(
                    desc, tensor->allocation, tensor->deviceIndex,
                    [cb](bool success) {
                        if (cb) cb(success, success ? "" : "UCX migrateFromRemoteUcx failed");
                    });
                if (!ok && cb) cb(false, "migrateFromRemoteUcx initiation failed");
            });
            return;
        }
#endif

        // Fall back to RDMA/host-staged path
        auto rdmaOk = config_.preference == TransportConfig::Preference::RDMAOnly ||
                      config_.enableGPUDirect;
        auto op = networkManager_->migrateFromRemote(*desc, tensor->allocation, rdmaOk);
        if (!op) {
            if (cb) cb(false, "migrateFromRemote failed");
            return;
        }

        networkManager_->waitMigration(*op);

        // Verify content if checksum provided
        if (tensor->metadata.contentHash != 0) {
            // Would verify content here
        }

        if (cb) cb(true, "");
    }
    
    // ========================================================================
    // Capabilities
    // ========================================================================
    
    bool supportsP2P() const override {
        return true; // MultiGPUPoolManager handles P2P
    }
    
    bool supportsRDMA() const override {
        return config_.enableGPUDirect && poolManager_ && poolManager_->getInstances().size() > 0;
    }
    
    bool supportsNetwork() const override {
        return networkManager_ != nullptr;
    }

    bool supportsUCX() const override {
#if defined(VVM_HAS_UCX)
        return ucxTransport_ && ucxTransport_->isInitialized();
#else
        return false;
#endif
    }

private:
    // ========================================================================
    // Async Processing
    // ========================================================================
    
    void asyncProcessingLoop() {
        while (!stopAsyncThread_) {
            std::unique_lock<std::mutex> lock(asyncMutex_);
            if (asyncCV_.wait_for(lock, std::chrono::milliseconds(10), 
                                  [this] { return stopAsyncThread_ || !asyncQueue_.empty(); })) {
                if (stopAsyncThread_) break;
                
                while (!asyncQueue_.empty()) {
                    auto op = asyncQueue_.front();
                    asyncQueue_.pop();
                    lock.unlock();
                    
                    // Process async operation
                    try {
                        op();
                    } catch (const std::exception& e) {
                        VVM_LOG_ERROR("Async operation threw: {}", e.what());
                    }
                    
                    lock.lock();
                }
                // Queue drained: wake flushAsync() waiters.
                asyncCV_.notify_all();
            }
        }
    }
    
    void flushAsync() override {
        // If async thread is running, wait until queue is drained
        if (asyncThread_.joinable()) {
            if (std::this_thread::get_id() == asyncThreadId_) {
                // Called from the async worker itself; waiting here would
                // self-deadlock, and pending ops run after we return.
                return;
            }
            std::unique_lock<std::mutex> lock(asyncMutex_);
            asyncCV_.wait(lock, [this] { return asyncQueue_.empty() || stopAsyncThread_; });
        }
        // Without async thread, operations run inline so nothing to drain
    }
    
    vvm::MultiGPUPoolManager* getPoolManager() override {
        if (poolManager_) return &(*poolManager_);
        return nullptr;
    }
    
    void enqueueAsync(AsyncOperation op) override {
        {
            std::lock_guard<std::mutex> lock(asyncMutex_);
            asyncQueue_.push(std::move(op));
        }
        if (!asyncThread_.joinable()) {
            // No async thread running, process inline
            AsyncOperation next;
            while (true) {
                {
                    std::lock_guard<std::mutex> lock(asyncMutex_);
                    if (asyncQueue_.empty()) break;
                    next = std::move(asyncQueue_.front());
                    asyncQueue_.pop();
                }
                try {
                    next();
                } catch (const std::exception& e) {
                    VVM_LOG_ERROR("Async operation failed: {}", e.what());
                }
            }
        } else {
            asyncCV_.notify_one();
        }
    }
    
    // ========================================================================
    // Host staging helpers for collectives
    // ========================================================================

    // Copy `size` bytes at `offset` from a GPU tensor into `out` via a
    // host-visible staging allocation so CPU-side reduce kernels can see it.
    bool gpuToHost(const TensorHandle& t, uint8_t* out, VkDeviceSize offset, size_t size) {
        auto& pool = poolManager_->getPool(t->deviceIndex);

        vvm::AllocDesc desc;
        desc.size = size;
        desc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        desc.memoryUsage = vvm::MemoryUsage::GpuToCpu;
        desc.exportable = false;
        desc.mapped = true;
        desc.name = "collective-read-staging";

        auto stage = pool.allocate(desc);
        if (!stage || !stage->hostPtr) {
            VVM_LOG_ERROR("collectives: failed to allocate read-back staging");
            return false;
        }
        bool ok = pool.copyBuffer(t->allocation, *stage, offset, 0, size);
        if (ok) {
            std::memcpy(out, stage->hostPtr, size);
        }
        pool.deallocate(std::move(*stage));
        return ok;
    }

    // Upload `size` bytes into a GPU tensor (offset-relative) through staging.
    bool hostToGpu(const TensorHandle& t, VkDeviceSize offset, const uint8_t* in, size_t size) {
        auto& pool = poolManager_->getPool(t->deviceIndex);

        vvm::AllocDesc stage;
        stage.size = size;
        stage.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        stage.memoryUsage = vvm::MemoryUsage::CpuToGpu;
        stage.exportable = false;
        stage.mapped = true;
        stage.name = "collective-write-staging";

        auto stageAlloc = pool.allocate(stage);
        if (!stageAlloc || !stageAlloc->hostPtr) {
            VVM_LOG_ERROR("collectives: failed to allocate write staging");
            return false;
        }
        std::memcpy(stageAlloc->hostPtr, in, size);
        bool ok = pool.copyBuffer(*stageAlloc, t->allocation, 0, offset, size);
        pool.deallocate(std::move(*stageAlloc));
        return ok;
    }

    // Read all participants into host memory and apply `op` element-wise.
    // `acc` receives the full reduced result (per-tensor layout).
    bool reduceAllToHost(const std::vector<TensorHandle>& tensors, ReduceOp op,
                         std::vector<uint8_t>& acc, size_t bytes) {
        acc.resize(bytes);
        initReduceOp(op, tensors[0]->metadata.dtype, acc.data(), bytes);

        std::vector<uint8_t> buf(bytes);
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (!gpuToHost(tensors[i], buf.data(), 0, bytes)) {
                VVM_LOG_ERROR("collectives: read-back of participant {} failed", i);
                return false;
            }
            if (!combineInto(op, tensors[0]->metadata.dtype, acc.data(), buf.data(), bytes)) {
                return false;
            }
        }

        if (op == ReduceOp::Mean) {
            const size_t elemWidth = packedElemSize(tensors[0]->metadata.dtype);
            const size_t count = bytes / elemWidth;
            for (size_t i = 0; i < count; ++i) {
                double v = readElem(tensors[0]->metadata.dtype, acc.data(), i);
                writeElem(tensors[0]->metadata.dtype, acc.data(), i, v / tensors.size());
            }
        }
        return true;
    }

    // ========================================================================
    // Members
    // ========================================================================
    
    TransportConfig config_;
    std::vector<vvm::DeviceConfig> devices_;
    vvm::PoolConfig poolConfig_;
    bool initialized_ = false;
    
    std::optional<vvm::MultiGPUPoolManager> poolManager_;
    std::unique_ptr<vvm::network::MultiNodePoolManager> networkManager_;
    
#if defined(VVM_HAS_UCX)
    std::unique_ptr<vvm::tensor::UcxTransport> ucxTransport_;
#endif

    // Broadcast targets: device index -> nearest handle (populated by
    // allocateDistributed).
    std::unordered_map<uint32_t, TensorHandle> distributedTensors_;
    
    // Async processing
    std::thread asyncThread_;
    std::thread::id asyncThreadId_{};
    std::atomic<bool> stopAsyncThread_{false};
    std::mutex asyncMutex_;
    std::condition_variable asyncCV_;
    std::queue<std::function<void()>> asyncQueue_;
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<Transport> Transport::create(
    const TransportConfig& config,
    const std::vector<vvm::DeviceConfig>& devices,
    const vvm::PoolConfig& poolConfig) {
    
    return std::make_unique<TensorTransportImpl>(config, devices, poolConfig);
}

} // namespace tensor
} // namespace vvm