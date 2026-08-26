#pragma once
// Internal backend factory declarations - not part of the public API.
// rdma_transport.cpp / ndk_transport.cpp expose their implementations here so
// network_factory.cpp can dispatch on VVM_RDMA_BACKEND.

#include "vulkan_vm/network/rdma_transport.hpp"

namespace vvm {
namespace network {

#if defined(VVM_NETWORK_HAS_VERBS)
std::unique_ptr<RdmaTransport> createVerbsRdmaTransport(
    const NetworkConfig& config, VkPhysicalDevice physicalDevice,
    VkDevice device);
#endif

#if defined(VVM_NETWORK_HAS_NDKPI)
std::unique_ptr<RdmaTransport> createNdkRdmaTransport(
    const NetworkConfig& config, VkPhysicalDevice physicalDevice,
    VkDevice device);
#endif

}  // namespace network
}  // namespace vvm
