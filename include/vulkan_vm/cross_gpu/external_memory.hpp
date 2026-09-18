#pragma once

#include "vulkan_vm/vulkan_vm.hpp"

namespace vvm {

struct ExternalMemoryCaps {
    bool supportsOpaqueFd = false;
    bool supportsOpaqueWin32 = false;
    bool supportsD3D12Heap = false;
    bool supportsDmaBuf = false;
    bool supportsZirconHandle = false;
    bool supportsAndroidHardwareBuffer = false;
    VkExternalMemoryHandleTypeFlags supportedHandleTypes = 0;
    VkExternalMemoryFeatureFlags supportedFeatures = 0;
};

VVM_API ExternalMemoryCaps queryExternalMemoryCaps(VkPhysicalDevice physicalDevice);
VVM_API VkPhysicalDeviceProperties getVendorProperties(VkPhysicalDevice physicalDevice);

struct VendorPairCaps {
    bool nvidiaToAmd = false;
    bool nvidiaToIntel = false;
    bool amdToIntel = false;
    bool amdToNvidia = false;
    bool sameVendor = true;
    ExternalHandleType recommendedType = ExternalHandleType::OpaqueFd;
    std::string notes;
};

VVM_API VendorPairCaps getCrossVendorCaps(VkPhysicalDevice src, VkPhysicalDevice dst);

} // namespace vvm