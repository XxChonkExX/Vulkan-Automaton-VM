// network_factory.cpp - sole definition of RdmaTransport::create. Dispatches
// to the platform backend selected by VVM_RDMA_BACKEND:
//   auto (default) | verbs | ndk | udp
// "auto" prefers verbs when compiled in, then NDKPI on Windows, and always
// offers "udp" as a fallback so every platform can join the fabric.

#include "vulkan_vm/network/rdma_transport.hpp"
#include "backends.hpp"
#include "vulkan_vm/network/udp_verb_transport.hpp"

#include <cstdlib>
#include <cstring>

namespace vvm {
namespace network {

enum class Backend { Auto, Verbs, Ndk, Udp };

static Backend selectedBackend() {
    static const Backend b = [] {
        if (const char* e = std::getenv("VVM_RDMA_BACKEND")) {
            if (!std::strcmp(e, "verbs")) return Backend::Verbs;
            if (!std::strcmp(e, "ndk")) return Backend::Ndk;
            if (!std::strcmp(e, "udp")) return Backend::Udp;
        }
        return Backend::Auto;
    }();
    return b;
}

#if defined(VVM_NETWORK_HAS_VERBS)
bool verbsBackendAvailable();
#endif

std::unique_ptr<RdmaTransport> RdmaTransport::create(
    const NetworkConfig& config,
    VkPhysicalDevice physicalDevice,
    VkDevice device) {

    const Backend b = selectedBackend();
    // No Vulkan context = no hardware backends possible; serve the software
    // fabric instead of crashing inside a vendor constructor.
    if (physicalDevice == VK_NULL_HANDLE &&
        (b == Backend::Auto || b == Backend::Verbs)) {
        return createUdpVerbRdmaTransport(config, physicalDevice, device);
    }

    switch (b) {
        case Backend::Udp:
            return createUdpVerbRdmaTransport(config, physicalDevice, device);
#if defined(VVM_NETWORK_HAS_VERBS)
        case Backend::Verbs:
            return createVerbsRdmaTransport(config, physicalDevice, device);
#endif
#if defined(VVM_NETWORK_HAS_NDKPI)
        case Backend::Ndk:
            return createNdkRdmaTransport(config, physicalDevice, device);
#endif
        case Backend::Auto:
            break;
    }

    // Auto: prefer verbs, then NDKPI, then UDP fallback.
#if defined(VVM_NETWORK_HAS_VERBS)
    return createVerbsRdmaTransport(config, physicalDevice, device);
#elif defined(VVM_NETWORK_HAS_NDKPI)
    return createNdkRdmaTransport(config, physicalDevice, device);
#else
    VVM_LOG_INFO("network factory: no hardware backend compiled in - using "
                 "udp-mini-verb transport");
    return createUdpVerbRdmaTransport(config, physicalDevice, device);
#endif
}

}  // namespace network
}  // namespace vvm
