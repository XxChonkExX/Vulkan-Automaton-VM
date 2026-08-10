#if defined(VVM_NETWORK_HAS_NDKPI)
// ============================================================================
// NdkRdmaTransport - Windows Network Direct (ND) transport
//
// Consumes the user-mode Network Direct SPI (IND2Provider/IND2Adapter and
// friends, see third_party/ndk) exposed by the Windows NDKPI stack (e.g. the
// NVIDIA mlx5nd2 ND provider on top of an NDK-capable miniport). This is the
// Windows counterpart of VerbsRdmaTransport: same RdmaTransport contract, one
// completion queue and one RC queue pair per connection, synchronous
// post-and-wait data path.
//
// GPU-direct: NVIDIA GPUDirect RDMA (nvidia-peermem / DMA-BUF) is Linux-only;
// on Windows registerGpuMemory() reports the constraint and returns nullopt so
// callers fall back to the existing host-staged RDMA export path.
// ============================================================================

#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/utils.hpp"

#include <initguid.h>  // must precede ndspi.h to materialize the IID_* globals

#include <winsock2.h>
#include <ws2spi.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <iphlpapi.h>
#include <ndspi.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")

namespace vvm {
namespace network {

namespace {

#ifndef PFL_NETWORKDIR_PROVIDER
#define PFL_NETWORKDIR_PROVIDER 0x00000010
#endif

constexpr auto kPollInterval = std::chrono::microseconds(200);
constexpr ULONG kMaxQueueDepth = 1024;

// DllGetClassObject as exported by ND providers (same signature as COM).
using DllGetClassObjectFn = HRESULT(WINAPI*)(REFCLSID, REFIID, void**);

// Parse the port out of a "host:port" listen address. Returns 0 if absent.
uint16_t parseTcpPort(const std::string& listen) {
    auto colon = listen.rfind(':');
    if (colon == std::string::npos) return 0;
    try {
        return static_cast<uint16_t>(std::stoul(listen.substr(colon + 1)));
    } catch (...) {
        return 0;
    }
}

bool toSockAddr(const std::string& ip, uint16_t port, sockaddr_in& out) {
    std::memset(&out, 0, sizeof(out));
    out.sin_family = AF_INET;
    out.sin_port = htons(port);
    if (ip.empty() || ip == "0.0.0.0" || ip == "*") {
        out.sin_addr.s_addr = INADDR_ANY;
        return true;
    }
    return inet_pton(AF_INET, ip.c_str(), &out.sin_addr) == 1;
}

std::string sockAddrToStr(const sockaddr_in& a) {
    char buf[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &a.sin_addr, buf, sizeof(buf)) == nullptr) return "";
    return buf;
}

bool isNdkV2Provider(const WSAPROTOCOL_INFO& info) {
    if (info.iVersion != 2) return false;
    if ((info.dwProviderFlags & PFL_NETWORKDIR_PROVIDER) == 0) return false;
    return info.iAddressFamily == AF_INET || info.iAddressFamily == AF_INET6;
}

// Block until an outstanding overlapped operation on `obj` completes.
HRESULT completeOverlapped(IND2Overlapped* obj, OVERLAPPED* ovl) {
    if (!obj) return E_FAIL;
    return obj->GetOverlappedResult(ovl, TRUE);
}

std::string hrToString(HRESULT hr) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "0x%08X", static_cast<unsigned>(hr));
    return buf;
}

} // namespace

// ============================================================================
// NdkRdmaTransport - Network Direct implementation
// ============================================================================

class NdkRdmaTransport : public RdmaTransport {
public:
    NdkRdmaTransport(const NetworkConfig& config, VkPhysicalDevice physicalDevice, VkDevice device)
        : config_(config), physicalDevice_(physicalDevice), device_(device) {
        uint16_t tcpPort = parseTcpPort(config_.listenAddress);
        rdmaPort_ = tcpPort != 0 ? static_cast<uint32_t>(tcpPort) + kRdmaPortOffset : 50052u;
    }

    ~NdkRdmaTransport() override {
        shutdown();
    }

    NdkRdmaTransport(const NdkRdmaTransport&) = delete;
    NdkRdmaTransport& operator=(const NdkRdmaTransport&) = delete;

    // ========================================================================
    // Lifecycle
    // ========================================================================

    bool initialize() override {
        if (shuttingDown_.load()) return false;

        // Partially-built state released on any failure below (mirrors shutdown
        // ordering: CQ/listener/overlapped file before adapter/provider/lib).
        auto failCleanup = [&]() {
            if (pCq_) { pCq_->Release(); pCq_ = nullptr; }
            if (pListener_) { pListener_->Release(); pListener_ = nullptr; }
            if (hOvlFile_ != INVALID_HANDLE_VALUE) { CloseHandle(hOvlFile_); hOvlFile_ = INVALID_HANDLE_VALUE; }
            if (adapter_) { adapter_->Release(); adapter_ = nullptr; }
            if (provider_) { provider_->Release(); provider_ = nullptr; }
            if (providerLib_) { FreeLibrary(providerLib_); providerLib_ = nullptr; }
        };

        HRESULT hr = openNdkAdapter(preferredLocalIp(), &provider_, &adapter_, &providerLib_);
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND: no usable Network Direct adapter (hr {})", hrToString(hr));
            return false;
        }

        adapterInfo_.InfoVersion = ND_VERSION_2;
        ULONG infoSize = sizeof(adapterInfo_);
        hr = adapter_->Query(&adapterInfo_, &infoSize);
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND: adapter query failed: {}", hrToString(hr));
            failCleanup();
            return false;
        }

        if (FAILED(adapter_->CreateOverlappedFile(&hOvlFile_))) {
            VVM_LOG_ERROR("ND: CreateOverlappedFile failed");
            failCleanup();
            return false;
        }

        ULONG cqDepth = adapterInfo_.MaxCompletionQueueDepth;
        if (cqDepth == 0 || cqDepth > kMaxQueueDepth) cqDepth = kMaxQueueDepth;
        hr = adapter_->CreateCompletionQueue(
            IID_IND2CompletionQueue, hOvlFile_, cqDepth, 0 /* group */, 0 /* affinity */,
            reinterpret_cast<void**>(&pCq_));
        if (FAILED(hr) || !pCq_) {
            VVM_LOG_ERROR("ND: CreateCompletionQueue failed: {}", hrToString(hr));
            failCleanup();
            return false;
        }

        sockaddr_in listenerAddr{};
        if (!toSockAddr("0.0.0.0", static_cast<uint16_t>(rdmaPort_), listenerAddr)) {
            VVM_LOG_ERROR("ND: invalid listener address");
            failCleanup();
            return false;
        }
        if (FAILED(adapter_->CreateListener(IID_IND2Listener, hOvlFile_,
                                            reinterpret_cast<void**>(&pListener_)))) {
            VVM_LOG_ERROR("ND: CreateListener failed");
            failCleanup();
            return false;
        }
        hr = pListener_->Bind(reinterpret_cast<const sockaddr*>(&listenerAddr),
                              static_cast<ULONG>(sizeof(listenerAddr)));
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND: failed to bind listener on port {}: {}", rdmaPort_, hrToString(hr));
            failCleanup();
            return false;
        }
        if (FAILED(pListener_->Listen(16))) {
            VVM_LOG_ERROR("ND: failed to listen on port {}", rdmaPort_);
            failCleanup();
            return false;
        }

        acceptThread_ = std::thread(&NdkRdmaTransport::acceptLoop, this);
        char devId[32];
        std::snprintf(devId, sizeof(devId), "%04X:%04X", adapterInfo_.VendorId, adapterInfo_.DeviceId);
        VVM_LOG_INFO("NdkRdmaTransport initialized on ND adapter {}, RDMA listener port {}",
                     devId, rdmaPort_);
        return true;
    }

    void shutdown() override {
        if (shuttingDown_.exchange(true)) return;

        // Wake the accept loop parked in GetConnectionRequest.
        if (pListener_) pListener_->CancelOverlappedRequests();
        if (acceptThread_.joinable()) acceptThread_.join();

        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            for (const auto& [key, ci] : connections_) {
                OVERLAPPED ovl{};
                ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (ovl.hEvent) {
                    HRESULT hr = ci->connector->Disconnect(&ovl);
                    if (hr == ND_PENDING) completeOverlapped(ci->connector, &ovl);
                    CloseHandle(ovl.hEvent);
                }
            }
            for (const auto& [key, ci] : connections_) destroyConnection(ci);
            connections_.clear();
        }

        // Deregister and release all memory regions.
        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            for (auto& [key, reg] : regions_) {
                if (reg.mr) {
                    OVERLAPPED ovl{};
                    ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                    if (ovl.hEvent) {
                        HRESULT hr = reg.mr->Deregister(&ovl);
                        if (hr == ND_PENDING) completeOverlapped(reg.mr, &ovl);
                        CloseHandle(ovl.hEvent);
                    }
                    reg.mr->Release();
                }
                if (reg.hOvlFile != INVALID_HANDLE_VALUE) CloseHandle(reg.hOvlFile);
            }
            regions_.clear();
        }

        if (pCq_) {
            pCq_->CancelOverlappedRequests();
            pCq_->Release();
            pCq_ = nullptr;
        }
        if (pListener_) {
            pListener_->Release();
            pListener_ = nullptr;
        }
        if (hOvlFile_ != INVALID_HANDLE_VALUE) {
            CloseHandle(hOvlFile_);
            hOvlFile_ = INVALID_HANDLE_VALUE;
        }
        if (adapter_) {
            adapter_->Release();
            adapter_ = nullptr;
        }
        if (provider_) {
            provider_->Release();
            provider_ = nullptr;
        }
        if (providerLib_) {
            FreeLibrary(providerLib_);
            providerLib_ = nullptr;
        }
    }

    bool isReady() const override {
        return adapter_ != nullptr && pCq_ != nullptr && !shuttingDown_.load();
    }

    // ========================================================================
    // Memory registration
    // ========================================================================

    std::optional<RdmaMemoryRegion> registerGpuMemory(
        VkDeviceMemory memory,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkBuffer buffer) override {
        (void)memory;
        (void)offset;
        (void)size;
        (void)buffer;
        // NVIDIA GPUDirect RDMA does not exist on Windows (peermem/DMA-BUF are
        // Linux kernel facilities). MultiNodePoolManager's host-staged export
        // path is the supported data route here.
        VVM_LOG_WARN("Windows ND transport: GPU-direct RDMA is not supported on Windows; "
                     "falling back to host-staged RDMA");
        return std::nullopt;
    }

    std::optional<RdmaMemoryRegion> registerHostMemory(void* ptr, size_t size) override {
        if (!isReady() || !ptr || size == 0) return std::nullopt;
        if (adapterInfo_.MaxRegistrationSize != 0 &&
            static_cast<SIZE_T>(size) > adapterInfo_.MaxRegistrationSize) {
            VVM_LOG_WARN("ND: registration of {} bytes exceeds adapter max {}",
                         size, adapterInfo_.MaxRegistrationSize);
            return std::nullopt;
        }

        // Check if we already have a persistent region for this address
        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            auto it = regions_.find(reinterpret_cast<uintptr_t>(ptr));
            if (it != regions_.end() && it->second.persistentlyPinned) {
                // Reuse existing persistent registration
                RegisteredRegion& reg = it->second;
                reg.refCount++;
                
                RdmaMemoryRegion region;
                region.addr = ptr;
                region.length = size;
                region.lkey = reg.mr->GetLocalToken();
                region.rkey = reg.mr->GetRemoteToken();
                region.rdmaAddr = reinterpret_cast<uint64_t>(ptr);
                region.ownsMemory = false;
                region.vkMemory = VK_NULL_HANDLE;
                region.vkBuffer = VK_NULL_HANDLE;
                VVM_LOG_DEBUG("ND: reused persistent host memory {} bytes (ref={})", 
                              size, reg.refCount);
                return region;
            }
        }

        // Create new registration
        HANDLE hOvlFile = INVALID_HANDLE_VALUE;
        IND2MemoryRegion* mr = nullptr;
        if (FAILED(adapter_->CreateOverlappedFile(&hOvlFile))) return std::nullopt;
        HRESULT hr = adapter_->CreateMemoryRegion(IID_IND2MemoryRegion, hOvlFile,
                                                  reinterpret_cast<void**>(&mr));
        if (FAILED(hr) || !mr) {
            if (hOvlFile != INVALID_HANDLE_VALUE) CloseHandle(hOvlFile);
            return std::nullopt;
        }

        OVERLAPPED ovl{};
        ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ovl.hEvent) {
            mr->Release();
            CloseHandle(hOvlFile);
            return std::nullopt;
        }
        hr = mr->Register(ptr, size,
                          ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ |
                              ND_MR_FLAG_ALLOW_REMOTE_WRITE,
                          &ovl);
        if (hr == ND_PENDING) hr = completeOverlapped(mr, &ovl);
        CloseHandle(ovl.hEvent);
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND: memory register failed: {}", hrToString(hr));
            mr->Release();
            CloseHandle(hOvlFile);
            return std::nullopt;
        }

        RegisteredRegion reg;
        reg.mr = mr;
        reg.hOvlFile = hOvlFile;
        reg.addr = ptr;
        reg.length = size;
        reg.refCount = 1;
        reg.persistentlyPinned = false;
        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            regions_[reinterpret_cast<uintptr_t>(ptr)] = reg;
        }

        RdmaMemoryRegion region;
        region.addr = ptr;
        region.length = size;
        region.lkey = mr->GetLocalToken();
        region.rkey = mr->GetRemoteToken();
        region.rdmaAddr = reinterpret_cast<uint64_t>(ptr);
        region.ownsMemory = false;
        region.vkMemory = VK_NULL_HANDLE;
        region.vkBuffer = VK_NULL_HANDLE;
        VVM_LOG_INFO("ND: registered host memory {} bytes (lkey={} rkey={})",
                     size, region.lkey, region.rkey);
        return region;
    }

    // Persistently pin host memory for reuse (GDRCopy-style)
    // Returns true if successful; the memory will stay registered until
    // releasePersistentHostMemory is called with the same address.
    bool pinPersistentHostMemory(void* ptr, size_t size) {
        auto regionOpt = registerHostMemory(ptr, size);
        if (!regionOpt) return false;
        
        std::lock_guard<std::mutex> lock(regionsMutex_);
        auto it = regions_.find(reinterpret_cast<uintptr_t>(ptr));
        if (it != regions_.end()) {
            it->second.persistentlyPinned = true;
            VVM_LOG_INFO("ND: persistently pinned host memory {} bytes at {}", size, ptr);
            return true;
        }
        return false;
    }

    // Release persistently pinned host memory
    // Decrements ref count; only actually deregisters when ref count reaches 0
    void releasePersistentHostMemory(void* ptr) {
        std::lock_guard<std::mutex> lock(regionsMutex_);
        auto it = regions_.find(reinterpret_cast<uintptr_t>(ptr));
        if (it == regions_.end()) return;
        
        RegisteredRegion& reg = it->second;
        if (!reg.persistentlyPinned) return;
        
        if (reg.refCount > 1) {
            reg.refCount--;
            VVM_LOG_DEBUG("ND: released persistent pin (ref={}) for {}", reg.refCount, ptr);
            return;
        }
        
        // Last reference - remove persistent flag but keep registered
        reg.persistentlyPinned = false;
        VVM_LOG_INFO("ND: removed persistent pin for {}", ptr);
    }

    void unregisterMemory(const RdmaMemoryRegion& region) override {
        if (!region.addr) return;
        std::unique_lock<std::mutex> lock(regionsMutex_);
        auto it = regions_.find(reinterpret_cast<uintptr_t>(region.addr));
        if (it == regions_.end()) return;
        
        RegisteredRegion& reg = it->second;
        
        // Don't unregister if persistently pinned
        if (reg.persistentlyPinned) {
            VVM_LOG_DEBUG("ND: skipping unregister for persistently pinned memory at {}", region.addr);
            return;
        }
        
        // Decrement ref count if > 0
        if (reg.refCount > 0) {
            reg.refCount--;
            if (reg.refCount > 0) {
                VVM_LOG_DEBUG("ND: deferred unregister (ref={}) for {}", reg.refCount, region.addr);
                return;
            }
        }
        
        // Actually unregister
        RegisteredRegion toUnreg = reg;
        regions_.erase(it);
        lock.unlock();
        
        if (toUnreg.mr) {
            OVERLAPPED ovl{};
            ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (ovl.hEvent) {
                HRESULT hr = toUnreg.mr->Deregister(&ovl);
                if (hr == ND_PENDING) completeOverlapped(toUnreg.mr, &ovl);
                CloseHandle(ovl.hEvent);
            }
            toUnreg.mr->Release();
        }
        if (toUnreg.hOvlFile != INVALID_HANDLE_VALUE) CloseHandle(toUnreg.hOvlFile);
    }

    // ========================================================================
    // Connections
    // ========================================================================

    std::optional<RdmaConnection> connect(
        const std::string& host,
        uint32_t port,
        uint32_t nodeIndex) override {
        if (!isReady()) return std::nullopt;
        std::lock_guard<std::mutex> lock(connectMutex_);

        sockaddr_in dst{};
        if (!toSockAddr(host, static_cast<uint16_t>(port), dst)) {
            VVM_LOG_ERROR("ND connect: invalid host: {}", host);
            return std::nullopt;
        }

        auto ci = std::make_shared<ConnectionInfo>();
        ci->remoteHost = host;
        ci->remotePort = port;
        ci->nodeIndex = nodeIndex;

        if (FAILED(adapter_->CreateOverlappedFile(&ci->hOvlFile))) return std::nullopt;
        if (FAILED(adapter_->CreateConnector(IID_IND2Connector, ci->hOvlFile,
                                             reinterpret_cast<void**>(&ci->connector)))) {
            destroyConnection(ci);
            return std::nullopt;
        }
        ci->qp = createQueuePair(ci);
        if (!ci->qp) {
            destroyConnection(ci);
            return std::nullopt;
        }

        OVERLAPPED ovl{};
        ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ovl.hEvent) {
            destroyConnection(ci);
            return std::nullopt;
        }
        HRESULT hr = ci->connector->Connect(
            static_cast<IUnknown*>(ci->qp), reinterpret_cast<const sockaddr*>(&dst),
            static_cast<ULONG>(sizeof(dst)), adapterInfo_.MaxInboundReadLimit,
            adapterInfo_.MaxOutboundReadLimit, nullptr, 0, &ovl);
        if (hr == ND_PENDING) hr = completeOverlapped(ci->connector, &ovl);
        CloseHandle(ovl.hEvent);
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND connect: connection to {}:{} failed: {}", host, port,
                          hrToString(hr));
            destroyConnection(ci);
            return std::nullopt;
        }

        ci->connected = true;
        {
            std::lock_guard<std::mutex> cLock(connectionsMutex_);
            connections_[reinterpret_cast<uintptr_t>(ci->connector)] = ci;
        }
        VVM_LOG_INFO("ND connect: established to {}:{} (node {})", host, port, nodeIndex);

        RdmaConnection conn;
        conn.remoteHost = host;
        conn.remotePort = port;
        conn.remoteNodeIndex = nodeIndex;
        conn.qpNum = 0;  // NDQPs are opaque; no exposed QP number.
        conn.connected = true;
        conn.gpuDirect = false;
        conn.internalId_ = ci->connector;  // connections_ is keyed by connector pointer
        return conn;
    }

    void disconnect(const RdmaConnection& conn) override {
        if (!conn.internalId_) return;
        std::shared_ptr<ConnectionInfo> ci;
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            auto it = connections_.find(reinterpret_cast<uintptr_t>(conn.internalId_));
            if (it == connections_.end()) return;
            ci = it->second;
            connections_.erase(it);
        }
        OVERLAPPED ovl{};
        ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (ovl.hEvent) {
            HRESULT hr = ci->connector->Disconnect(&ovl);
            if (hr == ND_PENDING) completeOverlapped(ci->connector, &ovl);
            CloseHandle(ovl.hEvent);
        }
        destroyConnection(ci);
    }

    std::vector<RdmaConnection> getConnections() const override {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        std::vector<RdmaConnection> result;
        for (const auto& [key, ci] : connections_) {
            if (!ci->connected) continue;
            RdmaConnection conn;
            conn.remoteHost = ci->remoteHost;
            conn.remotePort = ci->remotePort;
            conn.remoteNodeIndex = ci->nodeIndex;
            conn.connected = true;
            conn.gpuDirect = false;
            conn.internalId_ = ci->connector;
            result.push_back(conn);
        }
        return result;
    }

    // ========================================================================
    // Data transfer
    // ========================================================================

    bool rdmaWrite(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {
        return postRdma(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs, true);
    }

    bool rdmaRead(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {
        return postRdma(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs, false);
    }

    bool rdmaWriteAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs) override {
        // Same synchronous completion semantics as VerbsRdmaTransport.
        bool ok = rdmaWrite(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
        if (callback) callback(ok, ok ? "" : "RDMA write failed");
        return ok;
    }

    bool rdmaReadAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs) override {
        bool ok = rdmaRead(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
        if (callback) callback(ok, ok ? "" : "RDMA read failed");
        return ok;
    }

    void flush() override {
        // Synchronous post-and-wait semantics; nothing in flight.
    }

    size_t pollCompletions() override {
        return 0;  // Completions are drained by waitForCompletion.
    }

    // ========================================================================
    // Capabilities
    // ========================================================================

    bool supportsGpuDirect() const override {
        // NVIDIA GPUDirect RDMA is Linux-only; no Windows kernel path exists.
        return false;
    }

    bool supportsRdmaWrite() const override { return true; }
    bool supportsRdmaRead() const override { return true; }

    std::string getBackendName() const override { return "ndk"; }
    std::string getLocalNicName() const override { return config_.nicName; }
    uint32_t getLocalPort() const override { return rdmaPort_; }
    std::string getDeviceGuid() const override {
        if (!adapterInfo_.AdapterId) return "";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llX", static_cast<unsigned long long>(adapterInfo_.AdapterId));
        return buf;
    }

private:
    // ========================================================================
    // Internals
    // ========================================================================

    struct RegisteredRegion {
        IND2MemoryRegion* mr = nullptr;
        HANDLE hOvlFile = INVALID_HANDLE_VALUE;
        void* addr = nullptr;
        size_t length = 0;
        
        // Persistent pinning support (GDRCopy-style)
        uint32_t refCount = 0;          // Reference count for persistent pinning
        bool persistentlyPinned = false; // Whether this region is in the persistent pool
    };

    struct ConnectionInfo {
        IND2Connector* connector = nullptr;
        IND2QueuePair* qp = nullptr;
        HANDLE hOvlFile = INVALID_HANDLE_VALUE;
        bool serverSide = false;
        bool connected = false;
        std::string remoteHost;
        uint32_t remotePort = 0;
        uint32_t nodeIndex = 0;
    };

    std::shared_ptr<ConnectionInfo> findConnection(const RdmaConnection& conn) {
        if (!conn.internalId_) return nullptr;
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        auto it = connections_.find(reinterpret_cast<uintptr_t>(conn.internalId_));
        if (it == connections_.end()) return nullptr;
        return it->second;
    }

    struct PendingCtx {
        std::shared_ptr<ConnectionInfo> conn;
        uint64_t ctx = 0;
    };

    std::string preferredLocalIp() const {
        if (!config_.nicName.empty()) return config_.nicName;
        return config_.advertiseAddress;
    }

    // Enumerate WSC-registered Network Direct v2 providers, open the first one
    // whose address list matches the preferred IP (or any IPv4 address), and
    // open an IND2Adapter on it. On success the returned provider/adapter are
    // owned by the caller.
    static HRESULT openNdkAdapter(const std::string& preferIp,
                                  IND2Provider** ppProvider,
                                  IND2Adapter** ppAdapter,
                                  HMODULE* pLib) {
        // Test override: load a user-specified ND provider DLL directly (e.g., the
        // in-tree fake provider). This bypasses WSC catalog enumeration entirely.
        const char* fakeDllPath = std::getenv("VVM_ND_PROVIDER_DLL");
        if (fakeDllPath && *fakeDllPath) {
            HMODULE hLib = LoadLibraryA(fakeDllPath);
            if (!hLib) return HRESULT_FROM_WIN32(GetLastError());
            using GetClsidFn = HRESULT(WINAPI*)(CLSID*);
            auto pfnGetClsid = reinterpret_cast<GetClsidFn>(GetProcAddress(hLib, "NDFakeGetProviderClsid"));
            GUID providerGuid{};
            if (!pfnGetClsid || FAILED(pfnGetClsid(&providerGuid))) {
                FreeLibrary(hLib);
                return ND_DEVICE_NOT_READY;
            }
            auto pfnGetClassObject = reinterpret_cast<DllGetClassObjectFn>(
                GetProcAddress(hLib, "DllGetClassObject"));
            if (!pfnGetClassObject) {
                FreeLibrary(hLib);
                return ND_DEVICE_NOT_READY;
            }
            IND2Provider* pProvider = nullptr;
            HRESULT hr = pfnGetClassObject(providerGuid, IID_IND2Provider,
                                            reinterpret_cast<void**>(&pProvider));
            if (FAILED(hr) || !pProvider) {
                FreeLibrary(hLib);
                return ND_DEVICE_NOT_READY;
            }
            // Resolve local address 127.0.0.1 and open adapter (same flow as WSC path)
            sockaddr_in local{};
            local.sin_family = AF_INET;
            local.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            local.sin_port = 0;
            UINT64 adapterId = 0;
            hr = pProvider->ResolveAddress(reinterpret_cast<const sockaddr*>(&local),
                                            static_cast<ULONG>(sizeof(local)), &adapterId);
            if (FAILED(hr)) {
                pProvider->Release();
                FreeLibrary(hLib);
                return hr;
            }
            IND2Adapter* pAdapter = nullptr;
            hr = pProvider->OpenAdapter(IID_IND2Adapter, adapterId,
                                        reinterpret_cast<void**>(&pAdapter));
            if (FAILED(hr) || !pAdapter) {
                pProvider->Release();
                FreeLibrary(hLib);
                return hr;
            }
            *ppProvider = pProvider;
            *ppAdapter = pAdapter;
            *pLib = hLib;
            VVM_LOG_INFO("ND: loaded test provider from {}", fakeDllPath);
            return S_OK;
        }

        WSAPROTOCOL_INFO* pInfos = nullptr;
        DWORD len = 0;
        if (WSCEnumProtocols(nullptr, nullptr, &len, nullptr) == SOCKET_ERROR) {
            DWORD err = WSAGetLastError();
            if (err != WSAENOBUFS) return HRESULT_FROM_WIN32(err);
        }
        pInfos = static_cast<WSAPROTOCOL_INFO*>(HeapAlloc(GetProcessHeap(), 0, len));
        if (!pInfos) return ND_NO_MEMORY;
        int count = WSCEnumProtocols(nullptr, reinterpret_cast<LPWSAPROTOCOL_INFOW>(pInfos), &len, nullptr);
        if (count == SOCKET_ERROR) {
            int err = WSAGetLastError();
            HeapFree(GetProcessHeap(), 0, pInfos);
            return HRESULT_FROM_WIN32(err);
        }

        auto cleanupAll = [&]() {
            HeapFree(GetProcessHeap(), 0, pInfos);
        };

        for (int i = 0; i < count; ++i) {
            if (!isNdkV2Provider(pInfos[i])) continue;

            GUID providerGuid = pInfos[i].ProviderId;
            WCHAR path[MAX_PATH] = {0};
            INT pathLen = MAX_PATH;
            INT wscErr = 0;
            if (WSCGetProviderPath(&providerGuid, path, &pathLen, &wscErr) != 0) continue;
            WCHAR expanded[MAX_PATH] = {0};
            DWORD elen = ExpandEnvironmentStringsW(path, expanded, MAX_PATH);
            if (elen == 0 || elen >= MAX_PATH) continue;

            HMODULE hLib = LoadLibraryW(expanded);
            if (!hLib) continue;
            auto pfnGetClassObject = reinterpret_cast<DllGetClassObjectFn>(
                GetProcAddress(hLib, "DllGetClassObject"));
            if (!pfnGetClassObject) {
                FreeLibrary(hLib);
                continue;
            }

            IND2Provider* pProvider = nullptr;
            HRESULT hr = pfnGetClassObject(providerGuid, IID_IND2Provider,
                                           reinterpret_cast<void**>(&pProvider));
            if (FAILED(hr)) {
                FreeLibrary(hLib);
                continue;
            }

            // Pick the address to resolve: the preferred IP if parseable, else
            // the provider's first IPv4 address.
            sockaddr_in local{};
            bool haveLocal = false;
            if (!preferIp.empty()) {
                haveLocal = toSockAddr(preferIp, 0, local);
            }
            if (!haveLocal) {
                SOCKET_ADDRESS_LIST* pList = nullptr;
                ULONG cbList = 0;
                hr = pProvider->QueryAddressList(nullptr, &cbList);
                if (hr == ND_BUFFER_OVERFLOW && cbList > 0) {
                    pList = static_cast<SOCKET_ADDRESS_LIST*>(HeapAlloc(GetProcessHeap(), 0, cbList));
                    if (pList) {
                        hr = pProvider->QueryAddressList(pList, &cbList);
                        if (SUCCEEDED(hr)) {
                            for (ULONG a = 0; a < pList->iAddressCount; ++a) {
                                if (pList->Address[a].lpSockaddr->sa_family == AF_INET) {
                                    std::memcpy(&local, pList->Address[a].lpSockaddr,
                                                sizeof(sockaddr_in));
                                    haveLocal = true;
                                    break;
                                }
                            }
                        }
                        HeapFree(GetProcessHeap(), 0, pList);
                    }
                }
            }
            if (!haveLocal) {
                pProvider->Release();
                FreeLibrary(hLib);
                continue;
            }
            local.sin_port = 0;

            UINT64 adapterId = 0;
            hr = pProvider->ResolveAddress(reinterpret_cast<const sockaddr*>(&local),
                                           static_cast<ULONG>(sizeof(local)), &adapterId);
            if (FAILED(hr)) {
                pProvider->Release();
                FreeLibrary(hLib);
                continue;
            }

            IND2Adapter* pAdapter = nullptr;
            hr = pProvider->OpenAdapter(IID_IND2Adapter, adapterId,
                                        reinterpret_cast<void**>(&pAdapter));
            if (FAILED(hr) || !pAdapter) {
                pProvider->Release();
                FreeLibrary(hLib);
                continue;
            }

            *ppProvider = pProvider;
            *ppAdapter = pAdapter;
            *pLib = hLib;
            cleanupAll();
            return S_OK;
        }

        cleanupAll();
        VVM_LOG_WARN("ND: no Network Direct v2 provider found (install an RNIC ND provider, e.g. mlx5nd2)");
        return ND_DEVICE_NOT_READY;
    }

    // Create a queue pair attached to the transport's completion queue,
    // sized from the adapter's reported limits.
    IND2QueuePair* createQueuePair(const std::shared_ptr<ConnectionInfo>& ci) {
        ULONG rcv = adapterInfo_.MaxReceiveQueueDepth;
        if (rcv == 0 || rcv > kMaxQueueDepth) rcv = kMaxQueueDepth;
        ULONG init = adapterInfo_.MaxInitiatorQueueDepth;
        if (init == 0 || init > kMaxQueueDepth) init = kMaxQueueDepth;

        IND2QueuePair* qp = nullptr;
        HRESULT hr = adapter_->CreateQueuePair(
            IID_IND2QueuePair, static_cast<IUnknown*>(pCq_), static_cast<IUnknown*>(pCq_),
            ci.get(), rcv, init,
            1 /* maxReceiveRequestSge */, 1 /* maxInitiatorRequestSge */,
            adapterInfo_.InlineRequestThreshold, reinterpret_cast<void**>(&qp));
        if (FAILED(hr) || !qp) {
            VVM_LOG_ERROR("ND: CreateQueuePair failed: {}", hrToString(hr));
            return nullptr;
        }
        return qp;
    }

    static void destroyConnection(const std::shared_ptr<ConnectionInfo>& ci) {
        if (!ci) return;
        if (ci->qp) {
            ci->qp->Release();
            ci->qp = nullptr;
        }
        if (ci->connector) {
            ci->connector->Release();
            ci->connector = nullptr;
        }
        if (ci->hOvlFile != INVALID_HANDLE_VALUE) {
            CloseHandle(ci->hOvlFile);
            ci->hOvlFile = INVALID_HANDLE_VALUE;
        }
    }

    // Server side: accept incoming connection requests on the listening loop.
    void acceptLoop() {
        while (!shuttingDown_.load()) {
            HANDLE hOvl = INVALID_HANDLE_VALUE;
            IND2Connector* pending = nullptr;
            if (FAILED(adapter_->CreateOverlappedFile(&hOvl))) break;
            if (FAILED(adapter_->CreateConnector(IID_IND2Connector, hOvl,
                                                 reinterpret_cast<void**>(&pending)))) {
                CloseHandle(hOvl);
                break;
            }

            OVERLAPPED ovl{};
            ovl.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!ovl.hEvent) {
                pending->Release();
                CloseHandle(hOvl);
                break;
            }
            HRESULT hr = pListener_->GetConnectionRequest(static_cast<IUnknown*>(pending), &ovl);
            if (hr == ND_PENDING) hr = completeOverlapped(pListener_, &ovl);
            CloseHandle(ovl.hEvent);
            if (FAILED(hr) || shuttingDown_.load()) {
                pending->Release();
                CloseHandle(hOvl);
                break;
            }

            auto ci = std::make_shared<ConnectionInfo>();
            ci->serverSide = true;
            ci->hOvlFile = hOvl;
            ci->connector = pending;
            ci->qp = createQueuePair(ci);
            if (!ci->qp) {
                pending->Release();
                CloseHandle(hOvl);
                continue;
            }

            OVERLAPPED ovlAccept{};
            ovlAccept.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (!ovlAccept.hEvent) {
                destroyConnection(ci);
                continue;
            }
            hr = pending->Accept(static_cast<IUnknown*>(ci->qp),
                                 adapterInfo_.MaxInboundReadLimit,
                                 adapterInfo_.MaxOutboundReadLimit,
                                 nullptr, 0, &ovlAccept);
            if (hr == ND_PENDING) hr = completeOverlapped(pending, &ovlAccept);
            CloseHandle(ovlAccept.hEvent);
            if (FAILED(hr)) {
                VVM_LOG_WARN("ND accept: Accept failed: {}", hrToString(hr));
                destroyConnection(ci);
                continue;
            }

            sockaddr_in peer{};
            ULONG addrLen = sizeof(peer);
            if (SUCCEEDED(pending->GetPeerAddress(reinterpret_cast<sockaddr*>(&peer), &addrLen))) {
                ci->remoteHost = sockAddrToStr(peer);
                ci->remotePort = ntohs(peer.sin_port);
            }
            ci->connected = true;
            {
                std::lock_guard<std::mutex> lock(connectionsMutex_);
                connections_[reinterpret_cast<uintptr_t>(pending)] = ci;
            }
            VVM_LOG_INFO("ND accept: incoming connection from {}:{}", ci->remoteHost, ci->remotePort);
        }
    }

    // Post an RDMA_WRITE/RDMA_READ and wait for its completion on the CQ,
    // matching on the operation's request context.
    bool postRdma(const RdmaConnection& conn,
                  const RdmaMemoryRegion& localRegion,
                  uint64_t remoteAddr,
                  uint32_t remoteRkey,
                  VkDeviceSize size,
                  uint64_t timeoutNs,
                  bool isWrite) {
        auto ci = findConnection(conn);
        if (!ci || !ci->connected) {
            VVM_LOG_WARN("ND {}: connection not established to {}", isWrite ? "write" : "read",
                         conn.remoteHost);
            return false;
        }
        if (size == 0) return true;
        if (!localRegion.addr || localRegion.lkey == 0) {
            VVM_LOG_WARN("ND {}: no local SGE for the region (GPU-direct accepted only with BAR mapping)",
                         isWrite ? "write" : "read");
            return false;
        }
        if (size > 0xFFFFFFFFull) {
            VVM_LOG_WARN("ND {}: size exceeds ND2_SGE 32-bit limit", isWrite ? "write" : "read");
            return false;
        }

        // Serialize post+wait globally: the transport uses one shared completion
        // queue, so a concurrent draine on another connection could otherwise
        // consume this operation's ND2_RESULT.
        std::lock_guard<std::mutex> lock(transferMutex_);
        ND2_SGE sge{};
        sge.Buffer = localRegion.addr;
        sge.BufferLength = static_cast<ULONG>(size);
        sge.MemoryRegionToken = localRegion.lkey;

        uint64_t ctx = nextContextCounter_++;
        HRESULT hr = isWrite ? ci->qp->Write(reinterpret_cast<void*>(ctx), &sge, 1,
                                             remoteAddr, remoteRkey, 0 /* flags */)
                             : ci->qp->Read(reinterpret_cast<void*>(ctx), &sge, 1,
                                            remoteAddr, remoteRkey, 0 /* flags */);
        if (FAILED(hr)) {
            VVM_LOG_ERROR("ND {}: failed to post: {}", isWrite ? "write" : "read", hrToString(hr));
            return false;
        }
        return waitForCompletion(ci, reinterpret_cast<void*>(ctx), timeoutNs);
    }

    // Drain the completion queue synchronously until the operation carrying
    // `ctx` completes (or the deadline elapses).
    bool waitForCompletion(const std::shared_ptr<ConnectionInfo>& ci,
                           void* ctx,
                           uint64_t timeoutNs) {
        bool useDeadline = (timeoutNs != UINT64_MAX);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(timeoutNs);

        ND2_RESULT results[64] = {};
        for (;;) {
            ULONG num = pCq_->GetResults(results, 64);
            for (ULONG i = 0; i < num; ++i) {
                if (results[i].QueuePairContext == ci.get() &&
                    results[i].RequestContext == ctx) {
                    if (FAILED(results[i].Status)) {
                        VVM_LOG_WARN("ND completion error: {}", hrToString(results[i].Status));
                        return false;
                    }
                    return true;
                }
            }
            if (useDeadline && std::chrono::steady_clock::now() >= deadline) {
                VVM_LOG_WARN("ND operation timed out");
                return false;
            }
            std::this_thread::sleep_for(kPollInterval);
        }
    }

    // ========================================================================
    // Members
    // ========================================================================

    NetworkConfig config_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    uint32_t rdmaPort_ = 50052;
    std::atomic<bool> shuttingDown_{false};

    HMODULE providerLib_ = nullptr;
    IND2Provider* provider_ = nullptr;
    IND2Adapter* adapter_ = nullptr;
    ND2_ADAPTER_INFO adapterInfo_ = {};
    IND2CompletionQueue* pCq_ = nullptr;
    IND2Listener* pListener_ = nullptr;
    HANDLE hOvlFile_ = INVALID_HANDLE_VALUE;

    std::thread acceptThread_;
    std::unordered_map<uintptr_t, std::shared_ptr<ConnectionInfo>> connections_;
    mutable std::mutex connectionsMutex_;

    std::unordered_map<uintptr_t, RegisteredRegion> regions_;
    mutable std::mutex regionsMutex_;

    std::atomic<uint64_t> nextContextCounter_{1};
    std::mutex connectMutex_;
    std::mutex transferMutex_;  // serializes post+wait on the shared CQ
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<RdmaTransport> RdmaTransport::create(
    const NetworkConfig& config,
    VkPhysicalDevice physicalDevice,
    VkDevice device) {
    return std::make_unique<NdkRdmaTransport>(config, physicalDevice, device);
}

} // namespace network
} // namespace vvm

#endif // VVM_NETWORK_HAS_NDKPI