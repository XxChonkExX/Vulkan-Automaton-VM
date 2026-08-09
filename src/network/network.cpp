// Network subsystem implementation

#include "vulkan_vm/network.hpp"
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#if defined(VVM_NETWORK_HAS_VERBS)
#include "vulkan_vm/network/rdma_transport.hpp"
#include <infiniband/verbs.h>
#endif

#include <mutex>

#if defined(VVM_PLATFORM_WINDOWS)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace vvm {
namespace network {

static std::once_flag networkInitFlag;
static bool networkInitialized = false;

bool initializeNetwork() {
    std::call_once(networkInitFlag, []() {
        // Initialize gRPC
        // grpc_init();
        
        networkInitialized = true;
        VVM_LOG_INFO("VulkanVM Network subsystem initialized");
    });
    
    return networkInitialized;
}

void shutdownNetwork() {
    if (networkInitialized) {
        // grpc_shutdown();
        networkInitialized = false;
        VVM_LOG_INFO("VulkanVM Network subsystem shutdown");
    }
}

#if defined(VVM_NETWORK_HAS_VERBS)
bool isRdmaAvailable() {
    // Check for RDMA devices
    int numDevices = 0;
    struct ibv_device** devList = ibv_get_device_list(&numDevices);
    if (devList) {
        ibv_free_device_list(devList);
        return numDevices > 0;
    }
    return false;
}

std::optional<std::string> getRecommendedRdmaNic() {
    int numDevices = 0;
    struct ibv_device** devList = ibv_get_device_list(&numDevices);
    if (!devList || numDevices == 0) return std::nullopt;

    // Prefer devices with "mlx" prefix (Mellanox/NVIDIA)
    for (int i = 0; i < numDevices; ++i) {
        std::string name = ibv_get_device_name(devList[i]);
        if (name.rfind("mlx", 0) == 0) {
            ibv_free_device_list(devList);
            return name;
        }
    }

    // Return first available
    std::string name = ibv_get_device_name(devList[0]);
    ibv_free_device_list(devList);
    return name;
}
#else
bool isRdmaAvailable() {
    return false;
}

std::optional<std::string> getRecommendedRdmaNic() {
    return std::nullopt;
}
#endif

} // namespace network
} // namespace vvm