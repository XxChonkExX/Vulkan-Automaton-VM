#pragma once

// UdpVerbTransport - RDMA-shaped semantics over plain UDP datagrams.
// See docs/udp_mini_verbs.md for the wire protocol. Dependency-free by
// design: this is the transport that lets locked-down devices (Android,
// unprivileged shells, any OS with a UDP stack) join a VulkanVM cluster.

#include "vulkan_vm/network/rdma_transport.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(VVM_PLATFORM_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace vvm {
namespace network {

class VVM_API UdpVerbTransport : public RdmaTransport {
public:
    explicit UdpVerbTransport(const NetworkConfig& config);
    ~UdpVerbTransport() override;

    // RdmaTransport surface
    bool initialize() override;
    void shutdown() override;
    bool isReady() const override;

    std::optional<RdmaMemoryRegion> registerGpuMemory(
        VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
        VkBuffer buffer = VK_NULL_HANDLE) override;
    std::optional<RdmaMemoryRegion> registerHostMemory(
        void* ptr, size_t size) override;
    void unregisterMemory(const RdmaMemoryRegion& region) override;

    std::optional<RdmaConnection> connect(
        const std::string& host, uint32_t port, uint32_t nodeIndex = 0) override;
    void disconnect(const RdmaConnection& conn) override;
    std::vector<RdmaConnection> getConnections() const override;

    bool rdmaWrite(const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
                   uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
                   uint64_t timeoutNs = UINT64_MAX) override;
    bool rdmaRead(const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
                  uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
                  uint64_t timeoutNs = UINT64_MAX) override;

    bool rdmaWriteAsync(const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
                        uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
                        CompletionCallback callback,
                        uint64_t timeoutNs = UINT64_MAX) override;
    bool rdmaReadAsync(const RdmaConnection& conn, const RdmaMemoryRegion& localRegion,
                       uint64_t remoteAddr, uint32_t remoteRkey, VkDeviceSize size,
                       CompletionCallback callback,
                       uint64_t timeoutNs = UINT64_MAX) override;

    void flush() override;
    size_t pollCompletions() override;

    bool supportsGpuDirect() const override { return false; }
    bool supportsRdmaWrite() const override { return true; }
    bool supportsRdmaRead() const override { return true; }
    std::string getBackendName() const override { return "udp-mini-verb"; }
    std::string getLocalNicName() const override { return "udp"; }
    uint32_t getLocalPort() const override { return localPort_; }
    std::string getDeviceGuid() const override;

private:
    struct Impl;
    struct Xfer;   // per-transfer shared state (defined in the .cpp)
    std::unique_ptr<Impl> impl_;
    uint16_t localPort_ = 0;
};

// Factory hooks used by the shared backend dispatcher (rdma_transport factory).
VVM_API std::unique_ptr<RdmaTransport> createUdpVerbRdmaTransport(
    const NetworkConfig& config, VkPhysicalDevice physicalDevice, VkDevice device);

}  // namespace network
}  // namespace vvm
