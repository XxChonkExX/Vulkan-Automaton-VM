#include "vulkan_vm/network/cluster_server.hpp"
#include "vulkan_vm/network/tcp_transport.hpp"
#include "vulkan_vm/network/network_types.hpp"
#include "vulkan_vm/utils.hpp"

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <functional>
#include <thread>

namespace vvm {
namespace network {

// ============================================================================
// TCP-based ClusterServer Implementation
// ============================================================================

class ClusterServerImpl : public ClusterServer {
public:
    ClusterServerImpl(const NetworkConfig& config, MultiNodePoolManager* poolManager)
        : config_(config), poolManager_(poolManager) {}

    ~ClusterServerImpl() override {
        stop();
    }

    bool start() override {
        if (running_.load()) return true;

        transport_ = TcpTransport::create();
        if (!transport_) {
            VVM_LOG_ERROR("Failed to create TCP transport for cluster server");
            return false;
        }

        uint16_t port = 51010; // default control port
        size_t colonPos = config_.listenAddress.rfind(':');
        if (colonPos != std::string::npos) {
            try {
                port = static_cast<uint16_t>(std::stoul(config_.listenAddress.substr(colonPos + 1)));
            } catch (...) {
                VVM_LOG_WARN("Invalid port in listen address, using default 51010");
            }
        }

        if (!transport_->start(config_.listenAddress.substr(0, config_.listenAddress.rfind(':')),
                               port,
                               [this](TcpMessage& req, TcpMessage& resp) {
            handleRequest(req, resp);
        })) {
            VVM_LOG_ERROR("Failed to start TCP transport for cluster server");
            return false;
        }

        running_.store(true);
        VVM_LOG_INFO("Cluster server started on port {}", port);
        return true;
    }

    void stop() override {
        if (!running_.load()) return;

        running_.store(false);
        
        if (transport_) {
            transport_->stop();
            transport_.reset();
        }

        VVM_LOG_INFO("Cluster server stopped");
    }

    bool isRunning() const override {
        return running_.load();
    }

    void setAllocateHandler(AllocateHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        allocateHandler_ = std::move(handler);
    }

    void setExportHandler(ExportHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        exportHandler_ = std::move(handler);
    }

    void setImportHandler(ImportHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        importHandler_ = std::move(handler);
    }

    void setMigrateHandler(MigrateHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        migrateHandler_ = std::move(handler);
    }

    void setRegisterHandler(RegisterHandler handler) override {
        std::lock_guard<std::mutex> lock(handlerMutex_);
        registerHandler_ = std::move(handler);
    }

    void updateClusterView(const std::vector<NodeInfo>& view) override {
        std::lock_guard<std::mutex> lock(viewMutex_);
        clusterView_ = view;
    }

    std::vector<NodeInfo> getClusterView() const override {
        std::lock_guard<std::mutex> lock(viewMutex_);
        return clusterView_;
    }

    ServerStats getStats() const override {
        std::lock_guard<std::mutex> lock(statsMutex_);
        return stats_;
    }

private:
    void handleRequest(TcpMessage& req, TcpMessage& resp) {
        if (!running_.load()) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(0)}; // error
            return;
        }

        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.rpcCallsReceived++;

        try {
            switch (req.type) {
                case MsgAllocate:
                    handleAllocate(req, resp);
                    break;
                case MsgExport:
                    handleExport(req, resp);
                    break;
                case MsgImport:
                    handleImport(req, resp);
                    break;
                case MsgMigrate:
                    handleMigrate(req, resp);
                    break;
                case MsgRegisterNode:
                    handleRegisterNode(req, resp);
                    break;
                case MsgHeartbeat:
                    handleHeartbeat(req, resp);
                    break;
                case MsgGetClusterView:
                    handleGetClusterView(req, resp);
                    break;
                case MsgLeaveCluster:
                    handleLeaveCluster(req, resp);
                    break;
                default:
                    resp.type = MsgError;
                    resp.body = {static_cast<uint8_t>(1)}; // unknown message type
                    break;
            }
        } catch (const std::exception& e) {
            VVM_LOG_ERROR("Exception handling request: {}", e.what());
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(2)}; // internal error
        }

        std::lock_guard<std::mutex> lock(statsMutex_);
        if (resp.type == MsgError) {
            stats_.rpcCallsFailed++;
        } else {
            stats_.rpcCallsSucceeded++;
        }
    }

    void handleAllocate(const TcpMessage& req, TcpMessage& resp) {
        if (!allocateHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)}; // handler not set
            return;
        }

        // Deserialize request
        // For now, placeholder
        VVM_LOG_DEBUG("Allocate request received");
        resp.type = MsgAllocate;
        resp.body = {};
    }

    void handleExport(const TcpMessage& req, TcpMessage& resp) {
        if (!exportHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }
        VVM_LOG_DEBUG("Export request received");
        resp.type = MsgExport;
        resp.body = {};
    }

    void handleImport(const TcpMessage& req, TcpMessage& resp) {
        if (!importHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }
        VVM_LOG_DEBUG("Import request received");
        resp.type = MsgImport;
        resp.body = {};
    }

    void handleMigrate(const TcpMessage& req, TcpMessage& resp) {
        if (!migrateHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }
        VVM_LOG_DEBUG("Migrate request received");
        resp.type = MsgMigrate;
        resp.body = {};
    }

    void handleRegisterNode(const TcpMessage& req, TcpMessage& resp) {
        if (!registerHandler_) {
            resp.type = MsgError;
            resp.body = {static_cast<uint8_t>(3)};
            return;
        }
        VVM_LOG_DEBUG("RegisterNode request received");
        resp.type = MsgRegisterNode;
        resp.body = {};
    }

    void handleHeartbeat(const TcpMessage& req, TcpMessage& resp) {
        // Heartbeat just returns current cluster view
        std::lock_guard<std::mutex> lock(viewMutex_);
        resp.type = MsgHeartbeat;
        resp.body = serializeNodeList(clusterView_);
    }

    void handleGetClusterView(const TcpMessage& req, TcpMessage& resp) {
        std::lock_guard<std::mutex> lock(viewMutex_);
        resp.type = MsgGetClusterView;
        resp.body = serializeNodeList(clusterView_);
    }

    void handleLeaveCluster(const TcpMessage& req, TcpMessage& resp) {
        // Node is leaving, remove from cluster view
        // For now, just acknowledge
        resp.type = MsgLeaveCluster;
        resp.body = {static_cast<uint8_t>(0)}; // success
    }

    std::vector<uint8_t> serializeNodeList(const std::vector<NodeInfo>& nodes) {
        std::vector<uint8_t> data;
        // Simplified serialization
        return data;
    }

    NetworkConfig config_;
    MultiNodePoolManager* poolManager_ = nullptr;
    std::unique_ptr<TcpTransport> transport_;
    std::atomic<bool> running_{false};

    // Handlers
    std::mutex handlerMutex_;
    AllocateHandler allocateHandler_;
    ExportHandler exportHandler_;
    ImportHandler importHandler_;
    MigrateHandler migrateHandler_;
    RegisterHandler registerHandler_;

    // Cluster view
    mutable std::mutex viewMutex_;
    std::vector<NodeInfo> clusterView_;

    // Stats
    mutable std::mutex statsMutex_;
    ServerStats stats_;
};

std::unique_ptr<ClusterServer> ClusterServer::create(
    const NetworkConfig& config,
    MultiNodePoolManager* poolManager) {
    return std::make_unique<ClusterServerImpl>(config, poolManager);
}

} // namespace network
} // namespace vvm