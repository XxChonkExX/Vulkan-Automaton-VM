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

    // Stop progress thread before tearing down UCX context. The thread touches
    // worker_ which ucp_worker_destroy invalidates.
    stopProgressThread();

    // Invalidate the cached local worker address -- the worker is about to be
    // destroyed and any stale blob would be a use-after-free if accidentally
    // exchanged by a peer.
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        localAddress_.bytes.clear();
        localAddressValid_ = false;
    }

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
    
    // Close all endpoints non-blockingly. Bare ucp_ep_close blocks on
    // outstanding operations; use ucp_ep_close_nb + worker_progress so we
    // don't wedge on slow / stalled peers during teardown.
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        for (auto& [key, ep] : endpoints_) {
            if (!ep.ep) continue;
            ucs_status_ptr_t req = ucp_ep_close_nb(ep.ep, UCP_EP_CLOSE_MODE_FLUSH);
            if (UCS_PTR_IS_PTR(req)) {
                ucs_status_t st;
                do {
                    ucp_worker_progress(worker_);
                    st = ucp_request_check_status(req);
                } while (st == UCS_INPROGRESS);
                ucp_request_free(req);
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
                           UCP_MEM_MAP_PARAM_FIELD_LENGTH;
                           // No flags field: mapping EXISTING memory (flags=0).
                           // UCP_MEM_MAP_ALLOCATE allocates NEW memory and is wrong
                           // for an existing GPU ptr. GPU recognition is done by
                           // UCX's CUDA/ROCm memory type detection via the TLS
                           // stack (UCX_TLS includes cuda/rocm, peer mem, dma-buf).
    mapParams.address = ptr;
    mapParams.length = size;
    
    ucp_mem_h memh = nullptr;
    ucs_status_t status = ucp_mem_map(context_, &mapParams, &memh);
    
    if (status != UCS_OK) {
        VVM_LOG_WARN("GPU memory registration failed ({}), trying host fallback", 
                     ucs_status_string(status));
        // Fallback: register as host memory (same params, UCX will detect
        // host pointer type). No flags change needed.
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
    if (handle.rkey) {
        ucp_rkey_destroy(handle.rkey);
    }
    if (handle.memh) {
        ucs_status_t status = ucp_mem_unmap(context_, handle.memh);
        if (status != UCS_OK) {
            VVM_LOG_WARN("Memory deregistration failed: {}", ucs_status_string(status));
        }
    }

    if (handle.ptr) {
        std::lock_guard<std::mutex> lock(memHandlesMutex_);
        memHandles_.erase(reinterpret_cast<uintptr_t>(handle.ptr));
    }
}

// ============================================================================
// Endpoint / connection surface
// ============================================================================
// UCX endpoints are created from an opaque worker-address blob obtained via
// ucp_worker_get_address (NOT from host:port strings). The bootstrap exchange
// happens over the existing TCP control plane using wire_format helpers
// (vulkan_vm/network/wire_format.hpp).

void UcxTransport::epErrorCallback(void* /*arg*/, ucp_ep_h /*ep*/, ucs_status_t status) {
    // We log here; the actual disconnect / cleanup is handled by closeEndpoint()
    // when the higher layer notices the failure or by shutdown(). This handler
    // fires asynchronously on the UCX progress thread.
    VVM_LOG_WARN("UCX endpoint error: {}", ucs_status_string(status));
}

std::optional<UcxWorkerAddress> UcxTransport::getLocalAddress() {
    if (!initialized_ || !worker_) return std::nullopt;

    // Cache once per init; UCX worker addresses are stable until the worker
    // is destroyed. Recompute on shutdown().
    if (localAddressValid_ && !localAddress_.empty()) {
        return localAddress_;
    }

    ucp_address_t* addr = nullptr;
    size_t addrLen = 0;
    ucs_status_t st = ucp_worker_get_address(worker_, &addr, &addrLen);
    if (st != UCS_OK || !addr || addrLen == 0) {
        VVM_LOG_ERROR("ucp_worker_get_address failed: {}", ucs_status_string(st));
        return std::nullopt;
    }

    if (addrLen > kMaxUcxAddrBytes) {
        VVM_LOG_ERROR("UCX worker address too large: {}", addrLen);
        ucp_worker_release_address(worker_, addr);
        return std::nullopt;
    }

    UcxWorkerAddress out;
    out.bytes.assign(addr, addr + addrLen);
    ucp_worker_release_address(worker_, addr);

    localAddress_ = std::move(out);
    localAddressValid_ = true;
    VVM_LOG_INFO("UCX local worker address ready ({} bytes)", localAddress_.size());
    return localAddress_;
}

std::optional<UcxEndpoint> UcxTransport::connectToAddress(
    const UcxWorkerAddress& peerAddress,
    const std::string& peerKey,
    uint32_t remoteNodeId) {

    if (!initialized_ || !worker_) return std::nullopt;
    if (peerAddress.empty() || peerKey.empty()) return std::nullopt;

    // Cache hit
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        auto it = endpoints_.find(peerKey);
        if (it != endpoints_.end() && it->second.connected && it->second.ep) {
            return it->second;
        }
    }

    ucp_ep_params_t epParams{};
    epParams.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                          UCP_EP_PARAM_FIELD_ERR_HANDLER |
                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    epParams.address = reinterpret_cast<const ucp_address_t*>(peerAddress.data());
    epParams.err_mode = UCP_ERR_HANDLING_MODE_PEER;
    epParams.err_handler.cb = &UcxTransport::epErrorCallback;
    epParams.err_handler.arg = this;

    ucp_ep_h ep = nullptr;
    ucs_status_t st = ucp_ep_create(worker_, &epParams, &ep);
    if (st != UCS_OK) {
        VVM_LOG_ERROR("ucp_ep_create failed for {}: {}", peerKey, ucs_status_string(st));
        return std::nullopt;
    }

    // Drain any pending connection-establishment callbacks. Until the EP is
    // up, sends can queue / fail. A bounded progress loop is sufficient here:
    // ucp_ep_create returns a valid handle immediately and resolves the
    // transport selection during progress.
    constexpr int kCreateProgressIters = 256;
    for (int i = 0; i < kCreateProgressIters; ++i) {
        ucp_worker_progress(worker_);
    }

    UcxEndpoint endpoint;
    endpoint.ep = ep;
    endpoint.peerKey = peerKey;
    endpoint.remoteNodeId = remoteNodeId;
    endpoint.connected = true;

    // Replace any prior entry (drop the old EP). We close the stale EP under
    // its own scope so we don't hold endpointsMutex_ while blocking on
    // ucp_ep_close_nb's progress loop.
    ucp_ep_h staleEp = nullptr;
    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        auto it = endpoints_.find(peerKey);
        if (it != endpoints_.end() && it->second.ep && it->second.ep != ep) {
            staleEp = it->second.ep;
            endpoints_.erase(it);
        }
        endpoints_[peerKey] = endpoint;
    }
    if (staleEp) {
        ucs_status_ptr_t req = ucp_ep_close_nb(staleEp, UCP_EP_CLOSE_MODE_FLUSH);
        if (UCS_PTR_IS_PTR(req)) {
            ucs_status_t st;
            do {
                ucp_worker_progress(worker_);
                st = ucp_request_check_status(req);
            } while (st == UCS_INPROGRESS);
            ucp_request_free(req);
        } else if (UCS_PTR_IS_ERR(req)) {
            VVM_LOG_WARN("ucp_ep_close_nb (stale): {}", ucs_status_string(UCS_PTR_STATUS(req)));
        }
    }

    VVM_LOG_INFO("UCX endpoint connected to {} (node {})", peerKey, remoteNodeId);
    return endpoint;
}

std::optional<UcxEndpoint> UcxTransport::exchangeAndConnect(
    const std::string& peerKey,
    uint32_t remoteNodeId,
    bool activeSide,
    const std::function<bool(const std::vector<uint8_t>&)>& sendFn,
    const std::function<bool(std::vector<uint8_t>&)>& recvFn) {

    if (!sendFn || !recvFn) return std::nullopt;
    auto local = getLocalAddress();
    if (!local) return std::nullopt;

    std::vector<uint8_t> peerBytes;

    if (activeSide) {
        // Active: send local first, then receive peer.
        if (!sendFn(local->bytes)) {
            VVM_LOG_ERROR("UCX address exchange: send local failed ({})", peerKey);
            return std::nullopt;
        }
        if (!recvFn(peerBytes) || peerBytes.empty()) {
            VVM_LOG_ERROR("UCX address exchange: recv peer failed ({})", peerKey);
            return std::nullopt;
        }
    } else {
        // Passive: receive peer first, then send local.
        if (!recvFn(peerBytes) || peerBytes.empty()) {
            VVM_LOG_ERROR("UCX address exchange: recv peer failed ({})", peerKey);
            return std::nullopt;
        }
        if (!sendFn(local->bytes)) {
            VVM_LOG_ERROR("UCX address exchange: send local failed ({})", peerKey);
            return std::nullopt;
        }
    }

    UcxWorkerAddress peerAddr;
    peerAddr.bytes = std::move(peerBytes);
    return connectToAddress(peerAddr, peerKey, remoteNodeId);
}

std::optional<UcxEndpoint> UcxTransport::getEndpoint(const std::string& peerKey) const {
    std::lock_guard<std::mutex> lock(endpointsMutex_);
    auto it = endpoints_.find(peerKey);
    if (it == endpoints_.end() || !it->second.connected || !it->second.ep) {
        return std::nullopt;
    }
    return it->second;
}

void UcxTransport::closeEndpoint(UcxEndpoint& endpoint) {
    if (!endpoint.ep) {
        endpoint.connected = false;
        {
            std::lock_guard<std::mutex> lock(endpointsMutex_);
            if (!endpoint.peerKey.empty()) endpoints_.erase(endpoint.peerKey);
        }
        return;
    }

    ucs_status_ptr_t req = ucp_ep_close_nb(endpoint.ep, UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_PTR(req)) {
        ucs_status_t st;
        do {
            ucp_worker_progress(worker_);
            st = ucp_request_check_status(req);
        } while (st == UCS_INPROGRESS);
        ucp_request_free(req);
    } else if (UCS_PTR_IS_ERR(req)) {
        VVM_LOG_WARN("ucp_ep_close_nb: {}", ucs_status_string(UCS_PTR_STATUS(req)));
    }

    {
        std::lock_guard<std::mutex> lock(endpointsMutex_);
        if (!endpoint.peerKey.empty()) endpoints_.erase(endpoint.peerKey);
    }
    endpoint.ep = nullptr;
    endpoint.connected = false;
}

// ============================================================================
// RMA (Remote Memory Access) support
// ============================================================================
// UCX RMA (ucp_put/ucp_get) requires a remote virtual address + rkey,
// NOT a local ucp_mem_h. The owner packs the rkey via ucp_rkey_pack,
// exchanges the packed rkey + remote address over the control plane,
// and the peer unpacks it via ucp_ep_rkey_unpack.

std::optional<UcxRmaKey> UcxTransport::packRmaKey(const UcxMemoryHandle& handle) {
    if (!initialized_ || !handle.memh || handle.size == 0) {
        VVM_LOG_WARN("packRmaKey: invalid handle");
        return std::nullopt;
    }

    // Standard UCX API: ucp_rkey_pack returns a packed buffer + length
    void* packedBuf = nullptr;
    size_t packedLen = 0;
    ucs_status_t st = ucp_rkey_pack(context_, handle.memh, &packedBuf, &packedLen);
    if (st != UCS_OK || !packedBuf || packedLen == 0) {
        VVM_LOG_ERROR("ucp_rkey_pack failed: {}", ucs_status_string(st));
        return std::nullopt;
    }

    UcxRmaKey rmaKey;
    rmaKey.packedRkey.assign(static_cast<const uint8_t*>(packedBuf),
                             static_cast<const uint8_t*>(packedBuf) + packedLen);
    rmaKey.remoteAddr = reinterpret_cast<uint64_t>(handle.ptr);
    rmaKey.size = handle.size;
    rmaKey.deviceIndex = handle.deviceIndex;

    ucp_rkey_buffer_release(packedBuf);

    VVM_LOG_DEBUG("Packed RMA key for handle ptr={}, size={}, rkey_len={}",
                  handle.ptr, handle.size, rmaKey.packedRkey.size());
    return rmaKey;
}

std::optional<UcxMemoryHandle> UcxTransport::unpackRmaKey(
    const UcxEndpoint& endpoint,
    const UcxRmaKey& rmaKey) {

    if (!initialized_ || !endpoint.ep || rmaKey.packedRkey.empty()) {
        VVM_LOG_WARN("unpackRmaKey: invalid parameters");
        return std::nullopt;
    }

    // Unpack the remote key
    ucp_rkey_h rkey = nullptr;
    ucs_status_t st = ucp_ep_rkey_unpack(endpoint.ep, rmaKey.packedRkey.data(), &rkey);
    if (st != UCS_OK) {
        VVM_LOG_ERROR("ucp_ep_rkey_unpack failed: {}", ucs_status_string(st));
        return std::nullopt;
    }

    UcxMemoryHandle handle;
    handle.memh = nullptr;          // peer does not own a local memh for this
    handle.rkey = rkey;             // correct type for ucp_put/get
    handle.ptr = reinterpret_cast<void*>(rmaKey.remoteAddr);
    handle.size = rmaKey.size;
    handle.isGpuMemory = (rmaKey.deviceIndex != 0);
    handle.deviceIndex = rmaKey.deviceIndex;
    handle.packedRkey = rmaKey.packedRkey;
    handle.remoteAddr = rmaKey.remoteAddr;
    handle.rkeyValid = true;

    VVM_LOG_DEBUG("Unpacked RMA key for remote addr={}, size={}",
                  rmaKey.remoteAddr, rmaKey.size);
    return handle;
}

std::optional<UcxMemoryHandle> UcxTransport::exchangeRmaKey(
    const UcxEndpoint& endpoint,
    const UcxMemoryHandle& localHandle,
    bool ownerSide,
    const std::function<bool(const UcxRmaKey&)>& sendFn,
    const std::function<bool(UcxRmaKey&)>& recvFn) {

    if (!sendFn || !recvFn) return std::nullopt;

    if (ownerSide) {
        // We own the memory: pack and send our RMA key
        auto rmaKeyOpt = packRmaKey(localHandle);
        if (!rmaKeyOpt) return std::nullopt;
        if (!sendFn(*rmaKeyOpt)) {
            VVM_LOG_ERROR("exchangeRmaKey: send failed");
            return std::nullopt;
        }
        // We don't need a remote handle for the owner side
        return std::nullopt;
    } else {
        // We are the peer: receive the owner's RMA key and unpack it
        UcxRmaKey rmaKey;
        if (!recvFn(rmaKey)) {
            VVM_LOG_ERROR("exchangeRmaKey: recv failed");
            return std::nullopt;
        }
        return unpackRmaKey(endpoint, rmaKey);
    }
}

bool UcxTransport::putAsync(const UcxEndpoint& endpoint,
                            const UcxMemoryHandle& localMem,
                            const UcxMemoryHandle& remoteMem,
                            size_t size,
                            std::function<void(bool)> callback) {

    if (!endpoint.ep || !localMem.ptr || !remoteMem.rkeyValid || !remoteMem.rkey) {
        VVM_LOG_WARN("putAsync: invalid parameters (ep={} local_ptr={} remote_rkeyValid={} remote_rkey={})",
                     endpoint.ep != nullptr, localMem.ptr != nullptr,
                     remoteMem.rkeyValid, remoteMem.rkey != nullptr);
        return false;
    }

    auto* reqCtx = new RequestContext{std::move(callback)};
    // ucp_put_nb: local buffer -> remote address using remote rkey
    void* request = ucp_put_nb(endpoint.ep, localMem.ptr, size,
                               remoteMem.remoteAddr, remoteMem.rkey,
                               ucpRmaCallback, reqCtx);

    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_put_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }

    if (request == nullptr) {
        // Completed synchronously
        if (reqCtx->callback) reqCtx->callback(true);
        delete reqCtx;
        return true;
    }

    // In progress — callback owns reqCtx and must free the request
    return true;
}

bool UcxTransport::getAsync(const UcxEndpoint& endpoint,
                            const UcxMemoryHandle& localMem,
                            const UcxMemoryHandle& remoteMem,
                            size_t size,
                            std::function<void(bool)> callback) {

    if (!endpoint.ep || !localMem.ptr || !remoteMem.rkeyValid || !remoteMem.rkey) {
        VVM_LOG_WARN("getAsync: invalid parameters (ep={} local_ptr={} remote_rkeyValid={} remote_rkey={})",
                     endpoint.ep != nullptr, localMem.ptr != nullptr,
                     remoteMem.rkeyValid, remoteMem.rkey != nullptr);
        return false;
    }

    auto* reqCtx = new RequestContext{std::move(callback)};
    // ucp_get_nb: remote address -> local buffer using remote rkey
    void* request = ucp_get_nb(endpoint.ep, localMem.ptr, size,
                               remoteMem.remoteAddr, remoteMem.rkey,
                               ucpRmaCallback, reqCtx);

    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_get_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }

    if (request == nullptr) {
        // Completed synchronously
        if (reqCtx->callback) reqCtx->callback(true);
        delete reqCtx;
        return true;
    }

    // In progress — callback owns reqCtx and must free the request
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

    if (request == nullptr) {
        // Completed synchronously
        if (reqCtx->callback) reqCtx->callback(true);
        delete reqCtx;
        return true;
    }

    // In progress — callback owns reqCtx and must free the request
    return true;
}

bool UcxTransport::tagRecvAsync(const UcxEndpoint& endpoint,
                                void* buffer, size_t size,
                                uint64_t tag,
                                std::function<void(bool)> callback) {

    if (!endpoint.ep || !worker_) return false;

    auto* reqCtx = new RequestContext{std::move(callback)};
    // Note: tag_recv is posted on the worker, not the endpoint.
    ucs_status_ptr_t request = ucp_tag_recv_nb(worker_, buffer, size,
                                               ucp_dt_make_contig(1), tag, 0,
                                               ucpRecvCallback, reqCtx);

    if (UCS_PTR_IS_ERR(request)) {
        VVM_LOG_ERROR("ucp_tag_recv_nb failed: {}", ucs_status_string(UCS_PTR_STATUS(request)));
        delete reqCtx;
        return false;
    }

    if (request == nullptr) {
        // Completed synchronously
        if (reqCtx->callback) reqCtx->callback(true);
        delete reqCtx;
        return true;
    }

    // In progress — callback owns reqCtx and must free the request
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
    if (request) ucp_request_free(request);
}

void UcxTransport::ucpRecvCallback(void* request, ucs_status_t status,
                                   ucp_tag_recv_info_t* /*info*/, void* userData) {
    auto* ctx = static_cast<RequestContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback(status == UCS_OK);
    }
    delete ctx;
    if (request) ucp_request_free(request);
}

void UcxTransport::ucpRmaCallback(void* request, ucs_status_t status, void* userData) {
    auto* ctx = static_cast<RequestContext*>(userData);
    if (ctx && ctx->callback) {
        ctx->callback(status == UCS_OK);
    }
    delete ctx;
    if (request) ucp_request_free(request);
}

// ============================================================================
// Progress Thread
// ============================================================================

bool UcxTransport::startProgressThread() {
    if (progressThreadRunning_) return false;
    if (!worker_) return false;

    progressThreadStop_.store(false, std::memory_order_release);
    progressThread_ = std::thread(progressThreadLoop, this);
    progressThreadRunning_ = true;
    return true;
}

void UcxTransport::stopProgressThread() {
    if (!progressThreadRunning_) return;

    progressThreadStop_.store(true, std::memory_order_release);
    if (progressThread_.joinable()) {
        progressThread_.join();
    }
    progressThreadRunning_ = false;
}

void UcxTransport::progressThreadLoop(UcxTransport* self) {
    while (!self->progressThreadStop_.load(std::memory_order_acquire)) {
        ucp_worker_progress(self->worker_);
        // Yield to avoid burning CPU when idle. UCX progress is edge-triggered
        // so a short sleep doesn't stall ops meaningfully.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
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

// Stub connection surface: every call returns empty / nullopt so callers
// that gate on isInitialized() never reach these.
std::optional<UcxWorkerAddress> UcxTransport::getLocalAddress() { return std::nullopt; }
std::optional<UcxEndpoint> UcxTransport::connectToAddress(const UcxWorkerAddress&,
                                                           const std::string&,
                                                           uint32_t) { return std::nullopt; }
std::optional<UcxEndpoint> UcxTransport::exchangeAndConnect(const std::string&,
                                                             uint32_t,
                                                             bool,
                                                             const std::function<bool(const std::vector<uint8_t>&)>&,
                                                             const std::function<bool(std::vector<uint8_t>&)>&) { return std::nullopt; }
void UcxTransport::closeEndpoint(UcxEndpoint&) {}
std::optional<UcxEndpoint> UcxTransport::getEndpoint(const std::string&) const { return std::nullopt; }

// Stub RMA surface
std::optional<UcxRmaKey> UcxTransport::packRmaKey(const UcxMemoryHandle&) { return std::nullopt; }
std::optional<UcxMemoryHandle> UcxTransport::unpackRmaKey(const UcxEndpoint&, const UcxRmaKey&) { return std::nullopt; }
std::optional<UcxMemoryHandle> UcxTransport::exchangeRmaKey(const UcxEndpoint&,
                                                             const UcxMemoryHandle&,
                                                             bool,
                                                             const std::function<bool(const UcxRmaKey&)>&,
                                                             const std::function<bool(UcxRmaKey&)>&) { return std::nullopt; }
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