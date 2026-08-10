#include "vulkan_vm/ucx_transport.hpp"
#include "vulkan_vm/utils.hpp"

#if defined(VVM_HAS_UCX)
#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/type/status.h>
#include <ucs/debug/log.h>

#include <cstring>
#include <iostream>

namespace vvm {
namespace tensor {

// ============================================================================
// UCX Transport Implementation
// ============================================================================

UcxTransport::~UcxTransport() {
    shutdown();
}

bool UcxTransport::initialize(const UcxTransportConfig& config) {
    if (initialized_) return true;
    config_ = config;
    
    // UCX context configuration
    ucp_params_t params{};
    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
    params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA | UCP_FEATURE_AM;
    params.estimated_num_eps = 64;
    
    // Context configuration
    ucp_config_t* ucpConfig = nullptr;
    ucs_status_t status = ucp_config_read(nullptr, nullptr, &ucpConfig);
    if (status != UCS_OK) {
        VVM_LOG_ERROR("Failed to read UCX config: {}", ucs_status_string(status));
        return false;
    }
    
    // Apply custom config
    if (!config_.tls.empty()) {
        ucp_config_modify(ucpConfig, "TLS", config_.tls.c_str());
    }
    if (!config_.netDevices.empty()) {
        ucp_config_modify(ucpConfig, "NET_DEVICES", config_.netDevices.c_str());
    }
    if (config_.enableRndv) {
        ucp_config_modify(ucpConfig, "RNDV_THRESH", std::to_string(config_.rndvThreshold).c_str());
    }
    if (config_.enableCudaIpc) {
        ucp_config_modify(ucpConfig, "CUDA_IPC", "y");
    }
    
    // GPU memory registration support
    if (config_.enableGpuMemory) {
        ucp_config_modify(ucpConfig, "GPU_MEM_REG", "y");
    }
    
    // Create context
    status = ucp_init(&params, ucpConfig, &context_);
    ucp_config_release(ucpConfig);
    
    if (status != UCS_OK) {
        VVM_LOG_ERROR("ucp_init failed: {}", ucs_status_string(status));
        return false;
    }
    
    // Create worker
    ucp_worker_params_t workerParams{};
    workerParams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    workerParams.thread_mode = UCS_THREAD_MODE_MULTI;
    
    status = ucp_worker_create(context_, &workerParams, &worker_);
    if (status != UCS_OK) {
        VVM_LOG_ERROR("ucp_worker_create failed: {}", ucs_status_string(status));
        ucp_cleanup(context_);
        context_ = nullptr;
        return false;
    }
    
    initialized_ = true;
    VVM_LOG_INFO("UCX transport initialized (TLS: {}, RNDV: {})", 
                 config_.tls.empty() ? "auto" : config_.tls,
                 config_.enableRndv ? "on" : "off");
    return true;
}

void UcxTransport::shutdown() {
    if (!initialized_) return;
    
    // Deregister all memory
    {
        std::lock_guard<std::mutex> lock(memHandlesMutex_);
        for (auto& [key, handle] : memHandles_) {
            if (handle.memh) {
                ucp_mem_unmap(context_, handle.memh);
            }
        }
        memHandles_.clear();
    }
    
    // Close all endpoints
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        for (auto& [key, ep] : endpoints_) {
            if (ep.ep) {
                ucp_ep_close(ep.ep);
            }
        }
        endpoints_.clear();
    }
    
    if (worker_) {
        ucp_worker_destroy(worker_);
        worker_ = nullptr;
    }
    if (context_) {
        ucp_cleanup(context_);
        context_ = nullptr;
    }
    initialized_ = false;
    VVM_LOG_INFO("UCX transport shut down");
}

std::optional<UcxMemoryHandle> UcxTransport::registerGpuMemory(
    void* ptr, size_t size, uint32_t deviceIndex) {
    
    if (!initialized_ || !ptr || size == 0) return std::nullopt;
    
    ucp_mem_map_params_t mapParams{};
    mapParams.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | 
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                           UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mapParams.address = ptr;
    mapParams.length = size;
    mapParams.flags = UCP_MEM_MAP_ALLOCATE;
    
    // Try GPU memory registration first
    ucp_mem_h memh = nullptr;
    ucs_status_t status = ucp_mem_map(context_, &mapParams, &memh);
    
    if (status != UCS_OK) {
        VVM_LOG_WARN("GPU memory registration failed ({}), trying host fallback", 
                     ucs_status_string(status));
        // Fallback: register as host memory
        mapParams.flags = 0;  // Remove GPU flag
        status = ucp_mem_map(context_, &mapParams, &memh);
        if (status != UCS_OK) {
            VVM_LOG_ERROR("Memory registration failed: {}", ucs_status_string(status));
            return std::nullopt;
        }
    }
    
    UcxMemoryHandle handle;
    handle.memh = memh;
    handle.ptr = ptr;
    handle.size = size;
    handle.isGpuMemory = true;
    handle.deviceIndex = deviceIndex;
    
    {
        std::lock_guard<std::mutex> lock(memHandlesMutex_);
        memHandles_[reinterpret_cast<uintptr_t>(ptr)] = handle;
    }
    
    VVM_LOG_INFO("Registered GPU memory: ptr={}, size={}, device={}", 
                 ptr, size, deviceIndex);
    return handle;
}

std::optional<UcxMemoryHandle> UcxTransport::registerHostMemory(
    void* ptr, size_t size) {
    
    if (!initialized_ || !ptr || size == 0) return std::nullopt;
    
    ucp_mem_map_params_t mapParams{};
    mapParams.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | 
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mapParams.address = ptr;
    mapParams.length = size;
    
    ucp_mem_h memh = nullptr;
    ucs_status_t status = ucp_mem_map(context_, &mapParams, &memh);
    if (status != UCS_OK) {
        VVM_LOG_ERROR("Host memory registration failed: {}", ucs_status_string(status));
        return std::nullopt;
    }
    
    UcxMemoryHandle handle;
    handle.memh = memh;
    handle.ptr = ptr;
    handle.size = size;
    handle.isGpuMemory = false;
    handle.deviceIndex = 0;
    
    {
        std::lock_guard<std::mutex> lock(memHandlesMutex_);
        memHandles_[reinterpret_cast<uintptr_t>(ptr)] = handle;
    }
    
    VVM_LOG_INFO("Registered host memory: ptr={}, size={}", ptr, size);
    return handle;
}

void UcxTransport::deregisterMemory(const UcxMemoryHandle& handle) {
    if (!handle.memh) return;
    
    ucs_status_t status = ucp_mem_unmap(context_, handle.memh);
    if (status != UCS_OK) {
        VVM_LOG_WARN("Memory deregistration failed: {}", ucs_status_string(status));
    }
    
    {
        std::lock_guard<std::mutex> lock(memHandlesMutex_);
        memHandles_.erase(reinterpret_cast<uintptr_t>(handle.ptr));
    }
}

std::optional<UcxEndpoint> UcxTransport::createEndpoint(const std::string& remoteAddress) {
    if (!initialized_) return std::nullopt;
    
    // Parse remote address (format: "ip:port" or UCX address string)
    ucp_address_t* remoteAddr = nullptr;
    size_t addrLen = 0;
    
    // For now, use a simple approach - in production, you'd exchange addresses via bootstrap
    ucp_ep_params_t epParams{};
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_FLAGS;
    epParams.flags = UCP_EP_FLAG_NO_LOOPBACK;
    
    // Parse address - this is simplified; real implementation needs address exchange
    // For now, return empty and let higher layer handle connection
    UcxEndpoint endpoint;
    endpoint.remoteAddress = remoteAddress;
    endpoint.connected = false;
    
    return endpoint;
}

void UcxTransport::closeEndpoint(UcxEndpoint& endpoint) {
    if (endpoint.ep) {
        ucp_ep_close(endpoint.ep);
        endpoint.ep = nullptr;
        endpoint.connected = false;
    }
    
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        endpoints_.erase(endpoint.remoteAddress);
    }
}

bool UcxTransport::putAsync(const UcxEndpoint& endpoint,
                            const UcxMemoryHandle& localMem,
                            const UcxMemoryHandle& remoteMem,
                            size_t size,
                            std::function<void(bool)> callback) {
    
    if (!endpoint.ep || !localMem.memh || !remoteMem.memh) return false;
    
    auto* reqCtx = new RequestContext{std::move(callback)};
    void* request = ucp_put_nb(endpoint.ep, localMem.ptr, size, remoteMem.memh,
                               0, ucpRmaCallback, reqCtx);
    
    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_put_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }
    
    if (request != nullptr) {
        // Request completed immediately
        ucp_request_free(request);
        delete reqCtx;
    }
    return true;
}

bool UcxTransport::getAsync(const UcxEndpoint& endpoint,
                            const UcxMemoryHandle& localMem,
                            const UcxMemoryHandle& remoteMem,
                            size_t size,
                            std::function<void(bool)> callback) {
    
    if (!endpoint.ep || !localMem.memh || !remoteMem.memh) return false;
    
    auto* reqCtx = new RequestContext{std::move(callback)};
    void* request = ucp_get_nb(endpoint.ep, localMem.ptr, size, remoteMem.memh,
                               0, ucpRmaCallback, reqCtx);
    
    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_get_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }
    
    if (request != nullptr) {
        ucp_request_free(request);
        delete reqCtx;
    }
    return true;
}

bool UcxTransport::tagSendAsync(const UcxEndpoint& endpoint,
                                const void* buffer, size_t size,
                                uint64_t tag,
                                std::function<void(bool)> callback) {
    
    if (!endpoint.ep) return false;
    
    auto* reqCtx = new RequestContext{std::move(callback)};
    ucs_status_ptr_t request = ucp_tag_send_nb(endpoint.ep, buffer, size, 
                                               ucp_dt_make_contig(1), tag,
                                               ucpSendCallback, reqCtx);
    
    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_tag_send_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }
    
    if (request != nullptr) {
        ucp_request_free(request);
        delete reqCtx;
    }
    return true;
}

bool UcxTransport::tagRecvAsync(const UcxEndpoint& endpoint,
                                void* buffer, size_t size,
                                uint64_t tag,
                                std::function<void(bool)> callback) {
    
    if (!endpoint.ep) return false;
    
    auto* reqCtx = new RequestContext{std::move(callback)};
    ucs_status_ptr_t request = ucp_tag_recv_nb(worker_, buffer, size,
                                               ucp_dt_make_contig(1), tag, 0,
                                               ucpRecvCallback, reqCtx);
    
    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_tag_recv_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }
    
    if (request != nullptr) {
        ucp_request_free(request);
        delete reqCtx;
    }
    return true;
}

void UcxTransport::progress() {
    if (worker_) {
        ucp_worker_progress(worker_);
    }
}

void UcxTransport::ucpSendCallback(void* request, ucs_status_t status, void* userData) {
    auto* ctx = static_cast<RequestContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback(status == UCS_OK);
    }
    delete ctx;
}

void UcxTransport::ucpRecvCallback(void* request, ucs_status_t status,
                                   ucp_tag_recv_info_t* info, void* userData) {
    auto* ctx = static_cast<RequestContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback(status == UCS_OK);
    }
    delete ctx;
}

void UcxTransport::ucpRmaCallback(void* request, ucs_status_t status, void* userData) {
    auto* ctx = static_cast<RequestContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback(status == UCS_OK);
    }
    delete ctx;
}

} // namespace tensor
} // namespace vvm

#else // VVM_HAS_UCX

namespace vvm {
namespace tensor {

UcxTransport::~UcxTransport() = default;

bool UcxTransport::initialize(const UcxTransportConfig&) { return false; }
void UcxTransport::shutdown() {}
bool UcxTransport::isInitialized() const { return false; }

std::optional<UcxMemoryHandle> UcxTransport::registerGpuMemory(void*, size_t, uint32_t) { return std::nullopt; }
std::optional<UcxMemoryHandle> UcxTransport::registerHostMemory(void*, size_t) { return std::nullopt; }
void UcxTransport::deregisterMemory(const UcxMemoryHandle&) {}

std::optional<UcxEndpoint> UcxTransport::createEndpoint(const std::string&) { return std::nullopt; }
void UcxTransport::closeEndpoint(UcxEndpoint&) {}

bool UcxTransport::putAsync(const UcxEndpoint&, const UcxMemoryHandle&, const UcxMemoryHandle&, size_t, std::function<void(bool)>) { return false; }
bool UcxTransport::getAsync(const UcxEndpoint&, const UcxMemoryHandle&, const UcxMemoryHandle&, size_t, std::function<void(bool)>) { return false; }
bool UcxTransport::tagSendAsync(const UcxEndpoint&, const void*, size_t, uint64_t, std::function<void(bool)>) { return false; }
bool UcxTransport::tagRecvAsync(const UcxEndpoint&, void*, size_t, uint64_t, std::function<void(bool)>) { return false; }

void UcxTransport::progress() {}

void* UcxTransport::getContext() const { return nullptr; }
void* UcxTransport::getWorker() const { return nullptr; }

} // namespace tensor
} // namespace vvm

#endif // VVM_HAS_UCX