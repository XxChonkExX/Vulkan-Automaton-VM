#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"
#include "vulkan_vm/network/gpu_direct_registration.hpp"

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#include <cstring>
#include <memory>
#include <mutex>

namespace vvm {
namespace network {

// ============================================================================
// VerbsRdmaTransport - libibverbs implementation
// ============================================================================

class VerbsRdmaTransport : public RdmaTransport {
public:
    VerbsRdmaTransport(const NetworkConfig& config, VkPhysicalDevice physicalDevice, VkDevice device)
        : config_(config), physicalDevice_(physicalDevice), device_(device) {
        // Query vendor ID
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &props);
        vendorId_ = props.vendorID;
    }
    
    ~VerbsRdmaTransport() override {
        shutdown();
    }
    
    bool initialize() override {
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
            // Auto-select first device
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
        
        // Query device attributes
        struct ibv_device_attr devAttr{};
        if (ibv_query_device(context_, &devAttr)) {
            VVM_LOG_ERROR("Failed to query device attributes");
            return false;
        }
        
        // Create protection domain
        pd_ = ibv_alloc_pd(context_);
        if (!pd_) {
            VVM_LOG_ERROR("Failed to allocate protection domain");
            return false;
        }
        
        // Create completion channel and CQ
        compChannel_ = ibv_create_comp_channel(context_);
        if (!compChannel_) {
            VVM_LOG_ERROR("Failed to create completion channel");
            return false;
        }
        
        cq_ = ibv_create_cq(context_, 1024, nullptr, compChannel_, 0);
        if (!cq_) {
            VVM_LOG_ERROR("Failed to create completion queue");
            return false;
        }
        
        if (ibv_req_notify_cq(cq_, 0)) {
            VVM_LOG_ERROR("Failed to request CQ notification");
            return false;
        }
        
        // Start completion polling thread
        pollingThread_ = std::thread(&VerbsRdmaTransport::pollCompletionsLoop, this);
        
        // Initialize RDMA CM
        struct rdma_event_channel* ec = rdma_create_event_channel();
        if (!ec) {
            VVM_LOG_ERROR("Failed to create RDMA event channel");
            return false;
        }
        
        struct rdma_cm_id* listener = nullptr;
        if (rdma_create_id(ec, &listener, this, RDMA_PS_TCP)) {
            VVM_LOG_ERROR("Failed to create RDMA CM ID");
            return false;
        }
        
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(50052);  // RDMA port
        addr.sin_addr.s_addr = INADDR_ANY;
        
        if (rdma_bind_addr(listener, (struct sockaddr*)&addr)) {
            VVM_LOG_ERROR("Failed to bind RDMA address");
            return false;
        }
        
        if (rdma_listen(listener, 10)) {
            VVM_LOG_ERROR("Failed to listen on RDMA");
            return false;
        }
        
        listener_ = listener;
        eventChannel_ = ec;
        cmThread_ = std::thread(&VerbsRdmaTransport::cmEventLoop, this);
        
        VVM_LOG_INFO("VerbsRdmaTransport initialized on device: {}", ibv_get_device_name(chosenDev));
        return true;
    }
    
    void shutdown() override {
        if (shuttingDown_) return;
        shuttingDown_ = true;
        
        // Stop threads
        if (cmThread_.joinable()) cmThread_.join();
        if (pollingThread_.joinable()) pollingThread_.join();
        
        // Cleanup connections
        for (auto& [id, conn] : connections_) {
            if (conn.id) rdma_destroy_qp(conn.id);
            if (conn.id) rdma_destroy_id(conn.id);
        }
        connections_.clear();
        
        if (listener_) {
            rdma_destroy_id(listener_);
            listener_ = nullptr;
        }
        
        if (eventChannel_) {
            rdma_destroy_event_channel(eventChannel_);
            eventChannel_ = nullptr;
        }
        
        if (cq_) {
            ibv_destroy_cq(cq_);
            cq_ = nullptr;
        }
        
        if (compChannel_) {
            ibv_destroy_comp_channel(compChannel_);
            compChannel_ = nullptr;
        }
        
        if (pd_) {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        
        if (context_) {
            ibv_close_device(context_);
            context_ = nullptr;
        }
        
        registeredRegions_.clear();
    }
    
    bool isReady() const override {
        return context_ != nullptr && pd_ != nullptr;
    }
    
    std::optional<RdmaMemoryRegion> registerGpuMemory(
        VkDeviceMemory memory,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkBuffer buffer) override {
        
        if (!pd_ || !memory || size == 0) return std::nullopt;
        
        // On Linux with ibverbs, we need to:
        // 1. Export Vulkan memory to a DMA-BUF fd (Linux) or use the vendor-specific path
        // 2. Register the DMA-BUF fd with ibv_reg_dmabuf_mr (or ibv_reg_mr on mapped VA)
        
        #ifdef VVM_PLATFORM_LINUX
        // Get vendor-specific registration
        VkQueue transferQueue = VK_NULL_HANDLE;
        uint32_t transferQueueFamily = UINT32_MAX;
        
        if (vendorId_ == 0x10DE) {
            // NVIDIA: Use VK_NV_external_memory_rdma
            auto reg = registerGpuMemoryForRdmaVendor(
                device_, physicalDevice_, memory, offset, size,
                transferQueue, transferQueueFamily,
                config_.nicName, vendorId_);
            
            if (!reg || !reg->valid) {
                VVM_LOG_WARN("NVIDIA GPUDirect registration failed");
                return std::nullopt;
            }
            
            // On Linux with nvidia-peermem, we can register the GPU BAR memory
            // using the remote address from vkGetMemoryRemoteAddressNV
            // For now, return region with the remote address
            RdmaMemoryRegion region;
            region.addr = nullptr;
            region.length = size;
            region.lkey = 0;
            region.rkey = 0;  // Will be set after ibv_reg_mr
            region.rdmaAddr = reg->remoteAddress.address;
            region.ownsMemory = false;
            region.vkMemory = memory;
            region.vkBuffer = buffer;
            
            VVM_LOG_INFO("NVIDIA GPUDirect registered: rdmaAddr=0x%llx", region.rdmaAddr);
            return region;
        } else if (vendorId_ == 0x1002 || vendorId_ == 0x8086) {
            // AMD/Intel: Export as DMA-BUF fd, then register with ibv_reg_dmabuf_mr
            // Export Vulkan memory as DMA-BUF fd
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
            
            // Register DMA-BUF with ibverbs
            // Use ibv_reg_dmabuf_mr (requires kernel 5.10+ and MLNX_OFED 5.5+)
            struct ibv_reg_dmabuf_mr_attr attr{};
            attr.pd = pd_;
            attr.fd = dmaBufFd;
            attr.offset = offset;
            attr.length = size;
            attr.access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                                IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_READ;
            
            struct ibv_mr* mr = ibv_reg_dmabuf_mr(&attr);
            if (!mr) {
                VVM_LOG_WARN("ibv_reg_dmabuf_mr failed (fallback to ibv_reg_mr on mapped VA): %s", strerror(errno));
                close(dmaBufFd);
                // Fallback: try to map the DMA-BUF and use ibv_reg_mr
                // This requires the DMA-BUF to be mappable
                void* mappedVa = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmaBufFd, 0);
                if (mappedVa == MAP_FAILED) {
                    VVM_LOG_ERROR("mmap on DMA-BUF fd failed: %s", strerror(errno));
                    return std::nullopt;
                }
                
                mr = ibv_reg_mr(pd_, mappedVa, size, attr.access_flags);
                if (!mr) {
                    VVM_LOG_ERROR("ibv_reg_mr on mapped DMA-BUF failed: %s", strerror(errno));
                    munmap(mappedVa, size);
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
                
                // Store for cleanup (we need to track the mapped VA and DMA-BUF fd)
                {
                    std::lock_guard<std::mutex> lock(regionsMutex_);
                    registeredRegions_[mr] = region;
                }
                
                VVM_LOG_INFO("AMD/Intel GPUDirect registered via DMA-BUF mmap fallback: lkey=%u, rkey=%u", mr->lkey, mr->rkey);
                return region;
            }
            
            RdmaMemoryRegion region;
            region.addr = nullptr;  // GPU memory - no CPU VA
            region.length = size;
            region.lkey = mr->lkey;
            region.rkey = mr->rkey;
            region.rdmaAddr = 0;
            region.ownsMemory = false;
            region.vkMemory = memory;
            region.vkBuffer = buffer;
            
            // Store for cleanup
            {
                std::lock_guard<std::mutex> lock(regionsMutex_);
                registeredRegions_[mr] = region;
            }
            
            VVM_LOG_INFO("AMD/Intel GPUDirect registered via ibv_reg_dmabuf_mr: lkey=%u, rkey=%u", mr->lkey, mr->rkey);
            return region;
        }
        
        VVM_LOG_WARN("registerGpuMemory: unsupported vendor 0x%x", vendorId_);
        return std::nullopt;
        
        #else
        // Windows: RDMA/verbs not available, would need NDKPI-based implementation
        VVM_LOG_WARN("GPU-direct RDMA registration not implemented on Windows (requires NDKPI)");
        return std::nullopt;
        #endif
    }
    
    std::optional<RdmaMemoryRegion> registerHostMemory(void* ptr, size_t size) override {
        if (!pd_ || !ptr || size == 0) return std::nullopt;
        
        int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | 
                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_READ;
        
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
        
        // Store for cleanup
        {
            std::lock_guard<std::mutex> lock(regionsMutex_);
            registeredRegions_[mr] = region;
        }
        
        return region;
    }
    
    void unregisterMemory(const RdmaMemoryRegion& region) override {
        if (region.addr && region.length > 0) {
            // Find and deregister
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
            
            if (mr) {
                ibv_dereg_mr(mr);
            }
        }
    }
    
    std::optional<RdmaConnection> connect(
        const std::string& host,
        uint32_t port,
        uint32_t nodeIndex) override {
        
        // RDMA CM connection
        struct rdma_cm_id* id = nullptr;
        if (rdma_create_id(eventChannel_, &id, this, RDMA_PS_TCP)) {
            VVM_LOG_ERROR("Failed to create RDMA CM ID for connection");
            return std::nullopt;
        }
        
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            VVM_LOG_ERROR("Invalid host address: {}", host);
            rdma_destroy_id(id);
            return std::nullopt;
        }
        
        struct rdma_conn_param connParam{};
        connParam.initiator_depth = 1;
        connParam.responder_resources = 1;
        connParam.retry_count = 7;
        connParam.rnr_retry_count = 7;
        
        if (rdma_connect(id, &connParam)) {
            VVM_LOG_ERROR("Failed to initiate RDMA connection to {}:{}", host, port);
            rdma_destroy_id(id);
            return std::nullopt;
        }
        
        // Wait for connection establishment (simplified)
        // Real implementation would wait for RDMA_CM_EVENT_ESTABLISHED
        
        RdmaConnection conn;
        conn.remoteHost = host;
        conn.remotePort = port;
        conn.remoteNodeIndex = nodeIndex;
        conn.id = id;
        conn.connected = false;  // Will be set to true on ESTABLISHED event
        conn.gpuDirect = false;
        
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            connections_[reinterpret_cast<uintptr_t>(id)] = conn;
        }
        
        return conn;
    }
    
    void disconnect(const RdmaConnection& conn) override {
        if (conn.id) {
            rdma_disconnect(conn.id);
            // Wait for DISCONNECTED event
        }
    }
    
    std::vector<RdmaConnection> getConnections() const override {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        std::vector<RdmaConnection> result;
        for (const auto& [key, conn] : connections_) {
            result.push_back(conn);
        }
        return result;
    }
    
    bool rdmaWrite(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {
        
        if (!conn.id || !conn.connected) return false;
        
        struct ibv_send_wr wr{};
        struct ibv_sge sge{};
        
        sge.addr = reinterpret_cast<uintptr_t>(localRegion.addr);
        sge.length = static_cast<uint32_t>(size);
        sge.lkey = localRegion.lkey;
        
        wr.wr_id = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey = remoteRkey;
        
        struct ibv_send_wr* badWr = nullptr;
        if (ibv_post_send(conn.id->qp, &wr, &badWr)) {
            VVM_LOG_ERROR("Failed to post RDMA write: {}", strerror(errno));
            return false;
        }
        
        // Wait for completion (simplified)
        return waitForCompletion(conn.id->qp, timeoutNs);
    }
    
    bool rdmaRead(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        uint64_t timeoutNs) override {
        
        if (!conn.id || !conn.connected) return false;
        
        struct ibv_send_wr wr{};
        struct ibv_sge sge{};
        
        sge.addr = reinterpret_cast<uintptr_t>(localRegion.addr);
        sge.length = static_cast<uint32_t>(size);
        sge.lkey = localRegion.lkey;
        
        wr.wr_id = 0;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_READ;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey = remoteRkey;
        
        struct ibv_send_wr* badWr = nullptr;
        if (ibv_post_send(conn.id->qp, &wr, &badWr)) {
            VVM_LOG_ERROR("Failed to post RDMA read: {}", strerror(errno));
            return false;
        }
        
        return waitForCompletion(conn.id->qp, timeoutNs);
    }
    
    bool rdmaWriteAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs) override {
        // Post async and store callback
        // Simplified: just call sync version for now
        bool result = rdmaWrite(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
        if (callback) callback(result, result ? "" : "RDMA write failed");
        return result;
    }
    
    bool rdmaReadAsync(
        const RdmaConnection& conn,
        const RdmaMemoryRegion& localRegion,
        uint64_t remoteAddr,
        uint32_t remoteRkey,
        VkDeviceSize size,
        CompletionCallback callback,
        uint64_t timeoutNs) override {
        bool result = rdmaRead(conn, localRegion, remoteAddr, remoteRkey, size, timeoutNs);
        if (callback) callback(result, result ? "" : "RDMA read failed");
        return result;
    }
    
    void flush() override {
        // Wait for all pending operations
        // Simplified
    }
    
    size_t pollCompletions() override {
        // Called by polling thread
        return 0;
    }
    
    bool supportsGpuDirect() const override {
        // Check if device supports GPU-direct
        // NVIDIA (0x10DE): VK_NV_external_memory_rdma
        // AMD (0x1002): OPAQUE_WIN32 / DMA-BUF
        // Intel (0x8086): OPAQUE_WIN32 / DMA-BUF
        return (vendorId_ == 0x10DE || vendorId_ == 0x1002 || vendorId_ == 0x8086);
    }
    
    bool supportsRdmaWrite() const override { return true; }
    bool supportsRdmaRead() const override { return true; }
    
    std::string getBackendName() const override { return "verbs"; }
    
    std::string getLocalNicName() const override { return config_.nicName; }
    uint32_t getLocalPort() const override { return 50052; }
    std::string getDeviceGuid() const override { return ""; }
    
private:
    // CM event loop
    void cmEventLoop() {
        struct rdma_cm_event* event = nullptr;
        while (!shuttingDown_ && rdma_get_cm_event(eventChannel_, &event) == 0) {
            struct rdma_cm_event eventCopy = *event;
            rdma_ack_cm_event(event);
            
            // Handle events
            switch (eventCopy.event) {
                case RDMA_CM_EVENT_CONNECT_REQUEST:
                    handleConnectRequest(eventCopy.id);
                    break;
                case RDMA_CM_EVENT_ESTABLISHED:
                    handleEstablished(eventCopy.id);
                    break;
                case RDMA_CM_EVENT_DISCONNECTED:
                    handleDisconnected(eventCopy.id);
                    break;
                default:
                    break;
            }
        }
    }
    
    void handleConnectRequest(struct rdma_cm_id* id) {
        // Accept incoming connection
        struct rdma_conn_param connParam{};
        connParam.initiator_depth = 1;
        connParam.responder_resources = 1;
        connParam.retry_count = 7;
        connParam.rnr_retry_count = 7;
        
        if (rdma_accept(id, &connParam)) {
            VVM_LOG_ERROR("Failed to accept RDMA connection");
            rdma_reject(id, nullptr, 0);
        }
    }
    
    void handleEstablished(struct rdma_cm_id* id) {
        // Create QP
        struct ibv_qp_init_attr qpAttr{};
        qpAttr.cap.max_send_wr = 128;
        qpAttr.cap.max_recv_wr = 128;
        qpAttr.cap.max_send_sge = 1;
        qpAttr.cap.max_recv_sge = 1;
        qpAttr.qp_type = IBV_QPT_RC;
        qpAttr.send_cq = cq_;
        qpAttr.recv_cq = cq_;
        qpAttr.sq_sig_all = 1;
        
        if (rdma_create_qp(id, pd_, &qpAttr)) {
            VVM_LOG_ERROR("Failed to create QP for connection");
            return;
        }
        
        RdmaConnection conn;
        conn.id = id;
        conn.connected = true;
        conn.gpuDirect = false;
        
        {
            std::lock_guard<std::mutex> lock(connectionsMutex_);
            connections_[reinterpret_cast<uintptr_t>(id)] = conn;
        }
        
        VVM_LOG_INFO("RDMA connection established");
    }
    
    void handleDisconnected(struct rdma_cm_id* id) {
        std::lock_guard<std::mutex> lock(connectionsMutex_);
        connections_.erase(reinterpret_cast<uintptr_t>(id));
        VVM_LOG_INFO("RDMA connection disconnected");
    }
    
    // Completion polling loop
    void pollCompletionsLoop() {
        while (!shuttingDown_) {
            struct ibv_cq* cq = nullptr;
            void* ctx = nullptr;
            if (ibv_get_cq_event(compChannel_, &cq, &ctx)) {
                break;
            }
            
            ibv_ack_cq_events(cq, 1);
            if (ibv_req_notify_cq(cq, 0)) break;
            
            struct ibv_wc wc[32];
            int num = ibv_poll_cq(cq, 32, wc);
            for (int i = 0; i < num; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                    VVM_LOG_WARN("RDMA completion error: {}", ibv_wc_status_str(wc[i].status));
                }
                // Handle completion callbacks
            }
        }
    }
    
    bool waitForCompletion(struct ibv_qp* qp, uint64_t timeoutNs) {
        // Simplified: poll CQ until completion
        struct ibv_wc wc;
        auto start = std::chrono::steady_clock::now();
        
        while (true) {
            int num = ibv_poll_cq(cq_, 1, &wc);
            if (num == 1) {
                return wc.status == IBV_WC_SUCCESS;
            } else if (num < 0) {
                return false;
            }
            
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed > std::chrono::nanoseconds(timeoutNs)) {
                return false;
            }
            
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
    }
    
// Member variables
    NetworkConfig config_;
    VkPhysicalDevice physicalDevice_;
    VkDevice device_;
    uint32_t vendorId_ = 0;

    struct ibv_context* context_ = nullptr;
    struct ibv_pd* pd_ = nullptr;
    struct ibv_cq* cq_ = nullptr;
    struct ibv_comp_channel* compChannel_ = nullptr;
    struct rdma_event_channel* eventChannel_ = nullptr;
    struct rdma_cm_id* listener_ = nullptr;
    
    std::thread pollingThread_;
    std::thread cmThread_;
    std::atomic<bool> shuttingDown_{false};
    
    // Registered memory regions
    std::unordered_map<struct ibv_mr*, RdmaMemoryRegion> registeredRegions_;
    mutable std::mutex regionsMutex_;
    
    // Connections
    struct ConnectionInfo {
        struct rdma_cm_id* id = nullptr;
        bool connected = false;
        bool gpuDirect = false;
    };
    std::unordered_map<uintptr_t, ConnectionInfo> connections_;
    mutable std::mutex connectionsMutex_;
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