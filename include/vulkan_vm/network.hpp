#pragma once

// Network module main header
#include "vulkan_vm/network/network_config.hpp"
#include "vulkan_vm/network/multi_node_manager.hpp"
#include "vulkan_vm/network/rdma_transport.hpp"
#include "vulkan_vm/network/cluster_client.hpp"
#include "vulkan_vm/network/cluster_server.hpp"

namespace vvm {
namespace network {

// Version info
inline constexpr uint32_t NETWORK_API_VERSION = 1;
inline constexpr const char* NETWORK_VERSION_STRING = "0.1.0";

// Initialize network subsystem (call once at startup)
bool initializeNetwork();

// Shutdown network subsystem
void shutdownNetwork();

// Check if RDMA is available on this system
bool isRdmaAvailable();

// Get recommended NIC for RDMA
std::optional<std::string> getRecommendedRdmaNic();

} // namespace network
} // namespace vvm