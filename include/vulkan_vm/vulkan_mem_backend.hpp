#pragma once
// VulkanMemoryBackend: IDeviceMemoryBackend over VkDeviceMemory/VkBuffer.
// See mem_backend.hpp for the seam contract.

#include "vulkan_vm/mem_backend.hpp"

#include <vulkan/vulkan.h>

namespace vvm {

class VVM_API VulkanMemoryBackend final : public IDeviceMemoryBackend {
public:
    explicit VulkanMemoryBackend(const DeviceConfig& cfg);

    const char* name() const override;

    std::vector<BackendHeap>    heaps() const override;
    std::vector<BackendMemType> memoryTypes() const override;
    BackendBudget heapBudget(uint32_t heapIndex) const override;
    bool  type_is_host_visible(uint32_t typeIndex) const override;

    BackendMemory allocate(const BackendAllocRequest& req,
                           int* resultError) override;
    void free(BackendMemory mem) override;
    void* map(BackendMemory mem, bool* ok) override;
    void  unmap(BackendMemory mem) override;

    BackendBuffer create_buffer(BackendMemory mem, uint64_t offset,
                                uint64_t size, uint64_t usageBits,
                                bool exportable, int* resultError) override;
    bool bind_buffer(BackendBuffer buf, BackendMemory mem) override;
    void destroy_buffer(BackendBuffer buf) override;
    uint64_t buffer_device_address(BackendBuffer buf) const override;

    bool has_device_address() const override { return deviceAddressFeature_; }
    bool supports_export(ExternalHandleType type) const override;

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memProps_{};
    PFN_vkGetBufferDeviceAddress pfnGetBufferDeviceAddress = nullptr;
    bool deviceAddressFeature_ = false;
};

} // namespace vvm
