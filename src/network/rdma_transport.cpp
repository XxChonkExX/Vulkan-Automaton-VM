#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/network/gpu_direct_registration.hpp"

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <sys/mman.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <vector>
#include <unordered_map>

namespace vvm {
namespace network {

namespace {

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

} // namespace

// ============================================================================
// VerbsRdmaTransport - libibverbs implementation
//
// Uses the rdma_cm connection manager (RDMA_PS_TCP), one completion queue and
// one libverbs RC QP per connection, and a dedicated event thread per outbound
// connection so the listener's event loop never shares a channel with clients.
// ============================================================================

class VerbsRdmaTransport : public RdmaTransport {
public:
    VerbsRdmaTransport(const NetworkConfig& config, VkPhysicalDevice physicalDevice, VkDevice device)
        : config_(config), physicalDevice_(physicalDevice), device_(device) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        vendorId_ = props.vendorID;

        uint16_t tcpPort = parseTcpPort(config_.listenAddress);
        rdmaPort_ = tcpPort != 0 ? static_cast<uint32_t>(tcpPort) + kRdmaPortOffset : 50052u;
    }

    ~VerbsRdmaTransport() override {
        shutdown();
    }

    bool initialize() override {
        if (shuttingDown_.load()) return false;

        // Find and open RDMA device
        int numDevices = 0;
        struct ibv_device** devList = ibv_get_device_list(&numDevices);
        if (!devList || numDevices == 0) {
            VVM_LOG_ERROR("No RDMA devices found");
            return false;
        }

        struct ibv_device* chosenDev = nullptr;
        if (!config_.nicName.empty()) {
            for (int i = 0; i < numDevices; ++i) {
                if (std::string(ibv_get_device_name(devList[i])) == config_.nicName) {
                    chosenDev = devList[i];
                    break;
                }
            }
        } else {
            chosenDev = devList[0];
        }

        if (!chosenDev) {
            VVM_LOG_ERROR("RDMA device not found: {}", config_.nicName);
            ibv_free_device_list(devList);
            return false;
        }

        context_ = ibv_open_device(chosenDev);
        ibv_free_device_list(devList);
        if (!context_) {
            VVM_LOG_ERROR("Failed to open RDMA device");
            return false;
        }

        pd_ = ibv_alloc_pd(context_);
        if (!pd_) {
            VVM_LOG_ERROR("Failed to allocate protection domain");
            return false;
        }

        // Listener: separate event channel owned by the CM thread.
        struct rdma_event_channel* ec = rdma_create_event_channel();
        if (!ec) {
            VVM_LOG_ERROR("Failed to create RDMA event channel");
            return false;
        }

        struct rdma_cm_id* listener = nullptr;
        if (rdma_create_id(ec, &listener, this, RDMA_PS_TCP)) {
            VVM_LOG_ERROR("Failed to create RDMA CM ID");
            rdma_destroy_event_channel(ec);
            return false;
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(rdmaPort_));
        addr.sin_addr.s_addr = INADDR_ANY;

        if (rdma_bind_addr(listener, reinterpret_cast<struct sockaddr*>(&addr))) {
            VVM_LOG_ERROR("Failed to bind RDMA listener on port {}: {}",
                          rdmaPort_, strerror(errno));
            rdma_destroy_id(listener);
            rdma_destroy_event_channel(ec);
            return false;
        }

        if (rdma_listen(listener, 16)) {
            VVM_LOG_ERROR("Failed to listen on RDMA port {}: {}", rdmaPort_, strerror(errno));
            rdma_destroy_id(listener);
            rdma_destroy_event_channel(ec);
            return false;
        }

        listener_ = listener;
        eventChannel_ = ec;
        cmThread_ = std::thread(&VerbsRdmaTransport::listenerEventLoop, this);

        VVM_LOG_INFO("VerbsRdmaTransport initialized on device '{}', RDMA listener port {}",
                     ibv_get_device_name(chosenDev), rdmaPort_);
        return true;
    }

    void shutdown() override {
        if (shuttingDown_.exchange(true)) return;

        // Kick every connection so its event thread observes DISCONNECTED.
        std::vector<std::shared_ptr<ConnectionInfo>> conns;
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            for (const auto& [key, info] : connections_) conns.push_back(info);
        }
        for (const auto& info : conns) {
            if (info->id) rdma_disconnect(info->id);
        }

        if (cmThread_.joinable()) {
            // Destroying the event channel wakes the listener thread blocked in
            // rdma_get_cm_event (returns ECANCELED), so the join below completes.
            if (eventChannel_) {
                rdma_destroy_event_channel(eventChannel_);
                eventChannel_ = nullptr;
            }
            cmThread_.join();
        }
        if (listener_) {
            rdma_destroy_id(listener_);
            listener_ = nullptr;
        }

        for (auto& t : connThreads_) {
            if (t.joinable()) t.join();
        }
        connThreads_.clear();

        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            for (const auto& [key, info] : connections_) destroyConnection(info);
            connections_.clear();
        }

        registeredRegions_.clear();
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        if (context_) {
            ibv_close_device(context_);
            context_ = nullptr;
        }
    }

    bool isReady() const override {
        return context_ != nullptr && pd_ != nullptr && !shuttingDown_.load();
    }

    // ========================================================================
    // Memory registration
    // ========================================================================

    std::optional<RdmaMemoryRegion> registerGpuMemory(
        VkDeviceMemory memory,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkBuffer buffer) override {

        if (!pd_ || !memory || size == 0) return std::nullopt;

        if (vendorId_ == 0x10DE) {
            // NVIDIA: VK_NV_external_memory_rdma gives the remote address.
            // A usable local MR additionally requires nvidia-peermem registration,
            // which is not wired yet; the returned region carries rdmaAddr only.
            VkQueue transferQueue = VK_NULL_HANDLE;
            uint32_t transferQueueFamily = UINT32_MAX;
            auto reg = registerGpuMemoryForRdmaVendor(
                device_, physicalDevice_, memory, offset, size,
                transferQueue, transferQueueFamily, config_.nicName, vendorId_);
            if (!reg || !reg->valid) {
                VVM_LOG_WARN("NVIDIA GPUDirect registration failed");
                return std::nullopt;
            }

            RdmaMemoryRegion region;
            region.addr = nullptr;
            region.length = size;
            region.lkey = 0;
            region.rkey = 0;  // Set once an ibv_mr has been registered.
            region.rdmaAddr = reinterpret_cast<uint64_t>(reg->remoteAddress);
            region.ownsMemory = false;
            region.vkMemory = memory;
            region.vkBuffer = buffer;
            VVM_LOG_INFO("NVIDIA GPUDirect registered: rdmaAddr=0x%llx", region.rdmaAddr);
            return region;
        } else if (vendorId_ == 0x1002 || vendorId_ == 0x8086) {
            // AMD/Intel: export a DMA-BUF then register it with ibv_reg_dmabuf_mr.
            VkMemoryGetFdInfoKHR getFdInfo{};
            getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
            getFdInfo.memory = memory;
            getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

            PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR =
                (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR");
            if (!vkGetMemoryFdKHR) {
                VVM_LOG_ERROR("vkGetMemoryFdKHR not available");
                return std::nullopt;
            }

            int dmaBufFd = -1;
            VkResult result = vkGetMemoryFdKHR(device_, &getFdInfo, &dmaBufFd);
            if (result != VK_SUCCESS || dmaBufFd < 0) {
                VVM_LOG_ERROR("vkGetMemoryFdKHR failed: %s", vkResultToString(result).c_str());
                return std::nullopt;
            }

            const int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ;

            struct ibv_mr* mr = ibv_reg_dmabuf_mr(pd_, offset, size, 0 /* iova */, dmaBufFd, accessFlags);
            if (!mr) {
                VVM_LOG_WARN("ibv_reg_dmabuf_mr failed (%s), falling back to mmap+ibv_reg_mr",
                             strerror(errno));
                void* mappedVa = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmaBufFd, 0);
                if (mappedVa == MAP_FAILED) {
                    VVM_LOG_ERROR("mmap on DMA-BUF fd failed: %s", strerror(errno));
                    close(dmaBufFd);
                    return std::nullopt;
                }
                mr = ibv_reg_mr(pd_, mappedVa, size, accessFlags);
                if (!mr) {
                    VVM_LOG_ERROR("ibv_reg_mr on mapped DMA-BUF failed: %s", strerror(errno));
                    munmap(mappedVa, size);
                    close(dmaBufFd);
                    return std::nullopt;
                }
                RdmaMemoryRegion region;
                region.addr = mappedVa;
                region.length = size;
                region.lkey = mr->lkey;
                region.rkey = mr->rkey;
                region.rdmaAddr = 0;
                region.ownsMemory = false;
                region.vkMemory = memory;
                region.vkBuffer = buffer;
                {
                    std::lock_guard<std::mutex> lock(regionsMutex_);
                    registeredRegions_[mr] = region;
                }
                VVM_LOG_INFO("AMD/Intel GPUDirect via DMA-BUF mmap fallback: lkey=%u, rkey=%u",
                             mr->lkey, mr->rkey);
                return region;
            }

            RdmaMemoryRegion region;
            region.addr = nullptr;
            region.length = size;
            region.lkey = mr->lkey;
            region.rkey = mr->rkey;
            region.rdmaAddr = 0;  // remote DMA address needs a vendor query
            region.ownsMemory = false;
            region.vkMemory = memory;
            region.vkBuffer = buffer;
            {
                std::lock_guard<std::mutex> lock(regionsMutex_);
                registeredRegions_[mr] = region;
            }
            VVM_LOG_INFO("AMD/Intel GPUDirect via ibv_reg_dmabuf_mr: lkey=%u, rkey=%u",
                         mr->lkey, mr->rkey);
            return region;
        }

        VVM_LOG_WARN("registerGpuMemory: unsupported vendor 0x%x", vendorId_);
        return std::nullopt;
    }

    std::optional<RdmaMemoryRegion> registerHostMemory(void* ptr, size_t size) override {
        if (!pd_ || !ptr || size == 0) return std::nullopt;

        int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ;

        struct ibv_mr* mr = ibv_reg_mr(pd_, ptr, size, accessFlags);
        if (!mr) {
            VVM_LOG_ERROR("Failed to register host memory: {}", strerror(errno));
            return std::nullopt;
        }

        RdmaMemoryRegion region;
        region.addr = ptr;
        region.length = size;
        region.lkey = mr->lkey;
        region.rkey = mr->rkey;
        region.ownsMemory = false;
        region.vkMemory = VK_NULL_HANDLE;
        region.vkBuffer = VK_NULL_HANDLE;

        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            registeredRegions_[mr] = region;
        }
        return region;
    }

    void unregisterMemory(const RdmaMemoryRegion& region) override {
        if (!region.addr || region.length == 0) return;
        struct ibv_mr* mr = nullptr;
        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            for (auto& [key, val] : registeredRegions_) {
                if (val.addr == region.addr && val.length == region.length) {
                    mr = key;
                    break;
                }
            }
            if (mr) registeredRegions_.erase(mr);
        }
        if (mr) ibv_dereg_mr(mr);
    }

    // ------------------------------------------------------------------------
    // Connections
    // ------------------------------------------------------------------------

    std::optional<RdmaConnection> connect(
        const std::string& host,
        uint32_t port,
        uint32_t nodeIndex) override {

        if (!isReady()) return std::nullopt;
        std::lock_guard<std::mutex> lock(connCreateMutex_);

        struct rdma_event_channel* ec = rdma_create_event_channel();
        if (!ec) {
            VVM_LOG_ERROR("connect: failed to create event channel");
            return std::nullopt;
        }

        struct rdma_cm_id* id = nullptr;
        void* ctx = this;
        if (rdma_create_id(ec, &id, ctx, RDMA_PS_TCP)) {
            VVM_LOG_ERROR("connect: failed to create RDMA CM ID");
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(port));
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            VVM_LOG_ERROR("connect: invalid host address: {}", host);
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        if (rdma_resolve_addr(id, nullptr, reinterpret_cast<struct sockaddr*>(&addr), 5000)) {
            VVM_LOG_ERROR("connect: rdma_resolve_addr failed for {}: {}",
                          host, strerror(errno));
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }
        if (rdma_resolve_route(id, 5000)) {
            VVM_LOG_ERROR("connect: rdma_resolve_route failed for {}: {}",
                          host, strerror(errno));
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        struct ibv_cq* cq = ibv_create_cq(context_, 1024, nullptr, nullptr, 0);
        if (!cq) {
            VVM_LOG_ERROR("connect: failed to create CQ");
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        struct ibv_qp_init_attr qpAttr{};
        qpAttr.cap.max_send_wr = 1024;
        qpAttr.cap.max_recv_wr = 1024;
        qpAttr.cap.max_send_sge = 1;
        qpAttr.cap.max_recv_sge = 1;
        qpAttr.qp_type = IBV_QPT_RC;
        qpAttr.send_cq = cq;
        qpAttr.recv_cq = cq;
        qpAttr.sq_sig_all = 1;

        if (rdma_create_qp(id, pd_, &qpAttr)) {
            VVM_LOG_ERROR("connect: failed to create QP: {}", strerror(errno));
            ibv_destroy_cq(cq);
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        struct rdma_conn_param connParam{};
        connParam.initiator_depth = 1;
        connParam.responder_resources = 1;
        connParam.retry_count = 7;
        connParam.rnr_retry_count = 7;

        if (rdma_connect(id, &connParam)) {
            VVM_LOG_ERROR("connect: rdma_connect to {}:{} failed: {}", host, port, strerror(errno));
            rdma_destroy_qp(id);
            ibv_destroy_cq(cq);
            rdma_destroy_id(id);
            rdma_destroy_event_channel(ec);
            return std::nullopt;
        }

        auto info = std::make_shared<ConnectionInfo>();
        info->id = id;
        info->qp = id->qp;
        info->cq = cq;
        info->ec = ec;
        info->ownsEc = true;
        info->remoteHost = host;
        info->remotePort = port;
        info->nodeIndex = nodeIndex;

        {
            std::lock_guard<std::mutex> cLock(connectionsMutex_);
            connections_[reinterpret_cast<uintptr_t>(id)] = info;
        }

        {
            std::lock_guard<std::mutex> tLock(connThreadsMutex_);
            connThreads_.emplace_back(&VerbsRdmaTransport::connectionEventLoop, this, info);
        }

        // Block until ESTABLISHED (or a definitive failure).
        {
            std::unique_lock<std::mutex> wLock(info->mutex);
            if (!info->cv.wait_for(wLock, std::chrono::seconds(15),
                                   [&] { return info->connected || info->failed; })) {
                VVM_LOG_ERROR("connect: timed out establishing RDMA connection to {}:{}",
                              host, port);
                rdma_disconnect(id);
                // Let the event loop clean up asynchronously.
                return std::nullopt;
            }
            if (!info->connected) {
                VVM_LOG_ERROR("connect: RDMA connection to {}:{} failed", host, port);
                return std::nullopt;
            }
        }

        VVM_LOG_INFO("connect: RDMA established to {}:{} (ports {} qp {} rkey-ready)",
                     host, port, nodeIndex, info->qp->qp_num);
        RdmaConnection conn;
        conn.remoteHost = host;
        conn.remotePort = port;
        conn.remoteNodeIndex = nodeIndex;
        conn.qpNum = info->qp->qp_num;
        conn.connected = true;
        conn.gpuDirect = false;
        conn.internalId_ = id;
        return conn;
    }

    void disconnect(const RdmaConnection& conn) override {
        if (conn.internalId_) rdma_disconnect(static_cast<rdma_cm_id*>(conn.internalId_));
    }

    std::vector<RdmaConnection> getConnections() const override {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        std::vector<RdmaConnection> result;
        for (const auto& [key, info] : connections_) {
            if (!info->connected) continue;
            RdmaConnection conn;
            conn.remoteHost = info->remoteHost;
            conn.remotePort = info->remotePort;
            conn.remoteNodeIndex = info->nodeIndex;
            conn.qpNum = info->qp ? info->qp->qp_num : 0;
            conn.connected = info->connected;
            conn.gpuDirect = info->gpuDirect;
            conn.internalId_ = info->id;
            result.push_back(conn);
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Data transfer
    // ------------------------------------------------------------------------

    bool rdmaWrite(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {

        auto info = findConnection(conn);
        if (!info || !info->connected) {
            VVM_LOG_WARN("rdmaWrite: connection not established to {}", conn.remoteHost);
            return false;
        }
        if (size == 0) return true;
        if (!localRegion.addr) {
            VVM_LOG_WARN("rdmaWrite: no CPU VA for local region (GPU-direct SGE needs an exported BAR mapping)");
            return false;
        }
        return postRdma(info, IBV_WR_RDMA_WRITE, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
    }

    bool rdmaRead(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {

        auto info = findConnection(static_cast<struct rdma_cm_id*>(conn.internalId_));
        if (!info || !info->connected) {
            VVM_LOG_WARN("rdmaRead: connection not established to {}", conn.remoteHost);
            return false;
        }
        if (size == 0) return true;
        if (!localRegion.addr) {
            VVM_LOG_WARN("rdmaRead: no SGE address for local region (GPU-direct accepted only with BAR mapping)");
            return false;
        }
        return postRdma(info, IBV_WR_RDMA_READ, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
    }

    bool rdmaWriteAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs) override {
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
        return 0; // Completions are consumed inline by rdmaRead/rdmaWrite.
    }

    bool supportsGpuDirect() const override {
        // Check if we have the capability for GPU-direct
        if (vendorId_ == 0x10DE) {
            // NVIDIA needs VK_NV_external_memory_rdma + nvidia-peermem
            return false; // Not fully wired yet
        } else if (vendorId_ == 0x1002 || vendorId_ == 0x8086) {
            // AMD/Intel: ibv_reg_dmabuf_mr is available on modern kernels
            return true;
        }
        return false;
    }

    bool supportsRdmaWrite() const override { return true; }
    bool supportsRdmaRead() const override { return true; }

    std::string getBackendName() const override { return "verbs"; }
    std::string getLocalNicName() const override { return config_.nicName; }
    uint32_t getLocalPort() const override { return rdmaPort_; }
    std::string getDeviceGuid() const override { return ""; }

private:
    // ========================================================================
    // Connection internals
    // ========================================================================

    struct ConnectionInfo {
        struct rdma_cm_id* id = nullptr;
        struct ibv_qp* qp = nullptr;
        struct ibv_cq* cq = nullptr;
        struct rdma_event_channel* ec = nullptr;
        bool ownsEc = false;
        bool serverSide = false;
        bool connected = false;
        bool failed = false;
        bool gpuDirect = false;
        std::string remoteHost;
        uint32_t remotePort = 0;
        uint32_t nodeIndex = 0;
        std::mutex mutex;
        std::condition_variable cv;
    };

    std::shared_ptr<ConnectionInfo> findConnection(const RdmaConnection& conn) {
        if (!conn.internalId_) return nullptr;
        return findConnection(static_cast<rdma_cm_id*>(conn.internalId_));
    }

    std::shared_ptr<ConnectionInfo> findConnection(struct rdma_cm_id* id) {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        auto it = connections_.find(reinterpret_cast<uintptr_t>(id));
        if (it == connections_.end()) return nullptr;
        return it->second;
    }

    static void destroyConnection(const std::shared_ptr<ConnectionInfo>& info) {
        if (!info) return;
        if (info->qp) rdma_destroy_qp(info->id);
        if (info->cq) ibv_destroy_cq(info->cq);
        if (info->id) rdma_destroy_id(info->id);
        if (info->ownsEc && info->ec) rdma_destroy_event_channel(info->ec);
        info->id = nullptr;
        info->qp = nullptr;
        info->cq = nullptr;
        info->ec = nullptr;
    }

    // Handles the events for ONE outbound connection. Runs on the connection's
    // own thread so it never contends with the listener's event loop.
    void connectionEventLoop(const std::shared_ptr<ConnectionInfo>& info) {
        if (!info || !info->ec) return;
        struct rdma_cm_event* event = nullptr;
        while (rdma_get_cm_event(info->ec, &event) == 0) {
            struct rdma_cm_event evt = *event;
            rdma_ack_cm_event(event);

            switch (evt.event) {
                case RDMA_CM_EVENT_ESTABLISHED: {
                    std::lock_guard<std::mutex> lock(info->mutex);
                    info->connected = true;
                }
                    info->cv.notify_all();
                    break;
                case RDMA_CM_EVENT_ADDR_RESOLVED:
                case RDMA_CM_EVENT_ROUTE_RESOLVED:
                case RDMA_CM_EVENT_TIMEWAIT_EXIT:
                    break;
                case RDMA_CM_EVENT_REJECTED:
                case RDMA_CM_EVENT_ADDR_ERROR:
                case RDMA_CM_EVENT_ROUTE_ERROR:
                case RDMA_CM_EVENT_CONNECT_ERROR:
                case RDMA_CM_EVENT_UNREACHABLE: {
                    VVM_LOG_WARN("RDMA connection to {}:{} failed (event {})",
                                 info->remoteHost, info->remotePort, static_cast<int>(evt.event));
                    std::lock_guard<std::mutex> lock(info->mutex);
                    info->failed = true;
                }
                    info->cv.notify_all();
                    break;
                case RDMA_CM_EVENT_DISCONNECTED:
                case RDMA_CM_EVENT_DEVICE_REMOVAL: {
                    {
                        std::lock_guard<std::mutex> lock(connectionsMutex_);
                        connections_.erase(reinterpret_cast<uintptr_t>(info->id));
                    }
                    destroyConnection(info);
                    return;
                }
                default:
                    break;
            }
        }
        // Event channel closed without DISCONNECTED.
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            connections_.erase(reinterpret_cast<uintptr_t>(info->id));
        }
        destroyConnection(info);
    }

    // ---- Listener (server) event loop ----
    void listenerEventLoop() {
        struct rdma_cm_event* event = nullptr;
        while (!shuttingDown_.load() && rdma_get_cm_event(eventChannel_, &event) == 0) {
            struct rdma_cm_event evt = *event;
            rdma_ack_cm_event(event);

            switch (evt.event) {
                case RDMA_CM_EVENT_CONNECT_REQUEST:
                    handleConnectRequest(evt.id);
                    break;
                case RDMA_CM_EVENT_ESTABLISHED:
                    markEstablished(evt.id);
                    break;
                case RDMA_CM_EVENT_DISCONNECTED:
                case RDMA_CM_EVENT_DEVICE_REMOVAL:
                    handleDisconnected(evt.id);
                    break;
                default:
                    break;
            }
        }
    }

    void handleConnectRequest(struct rdma_cm_id* id) {
        struct ibv_cq* cq = ibv_create_cq(context_, 1024, nullptr, nullptr, 0);
        if (!cq) {
            VVM_LOG_ERROR("accept: failed to create CQ");
            rdma_reject(id, nullptr, 0);
            return;
        }

        struct ibv_qp_init_attr qpAttr{};
        qpAttr.cap.max_send_wr = 1024;
        qpAttr.cap.max_recv_wr = 1024;
        qpAttr.cap.max_send_sge = 1;
        qpAttr.cap.max_recv_sge = 1;
        qpAttr.qp_type = IBV_QPT_RC;
        qpAttr.send_cq = cq;
        qpAttr.recv_cq = cq;
        qpAttr.sq_sig_all = 1;

        if (rdma_create_qp(id, pd_, &qpAttr)) {
            VVM_LOG_ERROR("accept: failed to create QP: {}", strerror(errno));
            ibv_destroy_cq(cq);
            rdma_reject(id, nullptr, 0);
            return;
        }

        struct rdma_conn_param connParam{};
        connParam.initiator_depth = 1;
        connParam.responder_resources = 1;
        connParam.retry_count = 7;
        connParam.rnr_retry_count = 7;

        if (rdma_accept(id, &connParam)) {
            VVM_LOG_ERROR("accept: rdma_accept failed: {}", strerror(errno));
            rdma_destroy_qp(id);
            ibv_destroy_cq(cq);
            rdma_reject(id, nullptr, 0);
            return;
        }

        auto info = std::make_shared<ConnectionInfo>();
        info->id = id;
        info->qp = id->qp;
        info->cq = cq;
        info->ec = nullptr;      // events arrive on the listener's channel
        info->ownsEc = false;
        info->serverSide = true;
        info->connected = false;

        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            connections_[reinterpret_cast<uintptr_t>(id)] = info;
        }
        VVM_LOG_INFO("RDMA accepted incoming connection (qp {})", id->qp->qp_num);
    }

    void markEstablished(struct rdma_cm_id* id) {
        auto info = findConnection(id);
        if (!info) return;
        {
            std::lock_guard<std::mutex> lock(info->mutex);
            info->connected = true;
        }
        info->cv.notify_all();
    }

    void handleDisconnected(struct rdma_cm_id* id) {
        std::shared_ptr<ConnectionInfo> info;
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            auto it = connections_.find(reinterpret_cast<uintptr_t>(id));
            if (it == connections_.end()) {
                rdma_destroy_qp(id);
                rdma_destroy_id(id);
                return;
            }
            info = it->second;
            connections_.erase(it);
        }
        VVM_LOG_INFO("RDMA connection to {}:{} disconnected",
                     info->remoteHost, info->remotePort);
        if (info->serverSide) {
            if (info->id) {
                rdma_destroy_qp(info->id);
                ibv_destroy_cq(info->cq);
                rdma_destroy_id(info->id);
            }
        } else if (info->ownsEc && info->ec) {
            // Outbound connections are handled by their own event loop; the
            // DISCONNECTED event for them arrives on their own channel. If it
            // somehow lands here, still tear the resources down.
            if (info->id) {
                rdma_destroy_qp(info->id);
                ibv_destroy_cq(info->cq);
                rdma_destroy_id(info->id);
            }
            rdma_destroy_event_channel(info->ec);
        }
    }

    bool postRdma(const std::shared_ptr<ConnectionInfo>& info,
                  enum ibv_wr_opcode opcode,
                  const RdmaMemoryRegion& localRegion,
                  uint64_t remoteAddr,
                  uint32_t remoteRkey,
                  VkDeviceSize size,
                  uint64_t timeoutNs) {

        struct ibv_sge sge{};
        sge.addr = reinterpret_cast<uintptr_t>(localRegion.addr);
        sge.length = static_cast<uint32_t>(size);
        sge.lkey = localRegion.lkey;

        struct ibv_send_wr wr{};
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = opcode;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey = remoteRkey;

        struct ibv_send_wr* badWr = nullptr;
        if (ibv_post_send(info->qp, &wr, &badWr)) {
            VVM_LOG_ERROR("Failed to post {}: {}", opcodeToString(opcode), strerror(errno));
            return false;
        }
        return waitForCompletion(info->cq, timeoutNs);
    }

    bool waitForCompletion(struct ibv_cq* cq, uint64_t timeoutNs) {
        struct ibv_wc wc{};
        auto start = std::chrono::steady_clock::now();
        bool useDeadline = (timeoutNs != UINT64_MAX);
        auto deadline = start + std::chrono::nanoseconds(timeoutNs);

        for (;;) {
            int num = ibv_poll_cq(cq, 1, &wc);
            if (num < 0) return false;
            if (num == 1) {
                if (wc.status != IBV_WC_SUCCESS) {
                    VVM_LOG_WARN("RDMA completion error: {}", ibv_wc_status_str(wc.status));
                    return false;
                }
                return true;
            }
            if (useDeadline && std::chrono::steady_clock::now() >= deadline) {
                VVM_LOG_WARN("RDMA operation timed out");
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    static const char* opcodeToString(enum ibv_wr_opcode op) {
        switch (op) {
            case IBV_WR_RDMA_WRITE: return "RDMA_WRITE";
            case IBV_WR_RDMA_READ:  return "RDMA_READ";
            default: return "OTHER";
        }
    }

    // ========================================================================
    // Members
    // ========================================================================

    NetworkConfig config_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    uint32_t vendorId_ = 0;
    uint32_t rdmaPort_ = 50052;

    struct ibv_context* context_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct rdma_event_channel* eventChannel_ = nullptr;
    struct rdma_cm_id* listener_ = nullptr;

    std::thread cmThread_;
    std::vector<std::thread> connThreads_;
    mutable std::mutex connThreadsMutex_;
    std::atomic<bool> shuttingDown_{false};

    std::unordered_map<uintptr_t, std::shared_ptr<ConnectionInfo>> connections_;
    mutable std::mutex connectionsMutex_;

    std::unordered_map<struct ibv_mr*, RdmaMemoryRegion> registeredRegions_;
    mutable std::mutex regionsMutex_;

    std::mutex connCreateMutex_;
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<RdmaTransport> RdmaTransport::create(
    const NetworkConfig& config,
    VkPhysicalDevice physicalDevice,
    VkDevice device) {

    return std::make_unique<VerbsRdmaTransport>(config, physicalDevice, device);
}

} // namespace network
} // namespace vvm