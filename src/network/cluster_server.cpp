#include "vulkan_vm/network/cluster_server.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <grpcpp/grpcpp.h>
#include <memory>
#include <thread>

namespace vvm {
namespace network {

// ============================================================================
// ClusterServer Stub Implementation
// ============================================================================

class ClusterServerImpl : public ClusterServer {
public:
    ClusterServerImpl(const NetworkConfig& config, MultiNodePoolManager* poolManager)
        : config_(config), poolManager_(poolManager) {}
    
    ~ClusterServerImpl() override {
        stop();
    }
    
    bool start() override {
        if (server_) return true;
        
        std::string serverAddress = config_.listenAddress;
        
        grpc::ServerBuilder builder;
        builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
        builder.SetMaxReceiveMessageSize(1024 * 1024 * 1024);
        builder.SetMaxSendMessageSize(1024 * 1024 * 1024);
        
        // Register service (would use generated code)
        // builder.RegisterService(&service_);
        
        server_ = builder.BuildAndStart();
        if (!server_) {
            VVM_LOG_ERROR("Failed to start gRPC server on {}", serverAddress);
            return false;
        }
        
        VVM_LOG_INFO("Cluster server listening on {}", serverAddress);
        
        // Start serving thread
        servingThread_ = std::thread([this]() {
            // server_->Wait();
        });
        
        return true;
    }
    
    void stop() override {
        if (server_) {
            server_->Shutdown();
            server_.reset();
        }
        
        if (servingThread_.joinable()) {
            servingThread_.join();
        }
    }
    
    bool isRunning() const override {
        return server_ != nullptr;
    }
    
    void setAllocateHandler(AllocateHandler handler) override {
        allocateHandler_ = std::move(handler);
    }
    
    void setExportHandler(ExportHandler handler) override {
        exportHandler_ = std::move(handler);
    }
    
    void setImportHandler(ImportHandler handler) override {
        importHandler_ = std::move(handler);
    }
    
    void setMigrateHandler(MigrateHandler handler) override {
        migrateHandler_ = std::move(handler);
    }
    
    void setRegisterHandler(RegisterHandler handler) override {
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
        return stats_;
    }
    
private:
    NetworkConfig config_;
    MultiNodePoolManager* poolManager_;
    
    std::unique_ptr<grpc::Server> server_;
    std::thread servingThread_;
    
    // Handlers
    AllocateHandler allocateHandler_;
    ExportHandler exportHandler_;
    ImportHandler importHandler_;
    MigrateHandler migrateHandler_;
    RegisterHandler registerHandler_;
    
    // Cluster view
    mutable std::mutex viewMutex_;
    std::vector<NodeInfo> clusterView_;
    
    // Stats
    ServerStats stats_;
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<ClusterServer> ClusterServer::create(
    const NetworkConfig& config,
    MultiNodePoolManager* poolManager) {
    
    return std::make_unique<ClusterServerImpl>(config, poolManager);
}

} // namespace network
} // namespace vvm