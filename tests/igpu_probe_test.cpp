// iGPU memory probe: what can the Radeon(TM) Graphics (gfx1036, UMA) ACTUALLY
// serve? The Vulkan heap report says ~32 GB DEVICE_LOCAL (AMD UMA driver
// exposes ~half of system RAM as DEVICE_LOCAL with ReBAR), but the BIOS
// carveout is ~512 MB and Windows limits GPU-accessible locked pages. This
// probe measures the REAL working limits: allocation ladders from the
// DEVICE_LOCAL type and from HOST_VISIBLE types (pinned), each freed after
// every step so limits don't compound. Requires instance 1.3.
#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <vector>

using namespace vvm;

namespace {

struct Dev {
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
};

bool makeDev(const DeviceScore& score, Dev& out) {
    out.phys = score.device;
    auto queues = findQueueFamilies(score.device);
    if (!queues.transfer && !queues.graphics) return false;
    out.queueFamily = queues.transfer.value_or(queues.graphics.value());
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = out.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(score.device, &dci, nullptr, &out.device) != VK_SUCCESS) return false;
    return true;
}

// One allocation of `size` from a type matching `required` (min flags).
// Frees before returning. Returns true on success.
bool tryAlloc(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
              VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & required) != required) continue;
        if (preferred != 0 && (f & preferred) != preferred) continue;
        mt = i;
        break;
    }
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = size;
    ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS) return false;
    vkFreeMemory(device, mem, nullptr);
    return true;
}

// Live budget of the heap containing a type matching `required`.
VkDeviceSize budgetOf(VkPhysicalDevice phys, VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
    bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = &bud;
    vkGetPhysicalDeviceMemoryProperties2(phys, &mp2);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((mp.memoryTypes[i].propertyFlags & required) != required) continue;
        const uint32_t h = mp.memoryTypes[i].heapIndex;
        if (h < mp.memoryHeapCount) {
            return bud.heapBudget[h] ? bud.heapBudget[h] : mp.memoryHeaps[h].size;
        }
    }
    return 0;
}

} // namespace

int main() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM iGPU Probe";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "FAIL: vkCreateInstance\n";
        return 1;
    }
    auto devices = enumerateDevices(instance);
    int idx = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i].vendorID == 0x1002 &&
            devices[i].props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            idx = static_cast<int>(i);
            break;
        }
    }
    if (idx < 0) {
        std::cout << "SKIP: no AMD iGPU\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    std::cout << "iGPU: " << devices[idx].props.deviceName << "\n";
    Dev dev;
    if (!makeDev(devices[idx], dev)) {
        std::cerr << "FAIL: device creation\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Heap/type dump.
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(devices[idx].device, &mp);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bud{};
    bud.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = &bud;
    vkGetPhysicalDeviceMemoryProperties2(devices[idx].device, &mp2);
    for (uint32_t h = 0; h < mp.memoryHeapCount; ++h) {
        std::cout << "  heap " << h << ": " << mp.memoryHeaps[h].size / (1024*1024)
                  << " MB heap, budget " << bud.heapBudget[h] / (1024*1024) << " MB"
                  << ((mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                      ? " DEVICE_LOCAL" : "") << "\n";
    }
    for (uint32_t t = 0; t < mp.memoryTypeCount; ++t) {
        std::cout << "  type " << t << ": heap " << mp.memoryTypes[t].heapIndex
                  << " flags 0x" << std::hex << mp.memoryTypes[t].propertyFlags << std::dec << "\n";
    }

    // Allocation ladders (freed after each step).
    struct Ladder { const char* name; VkMemoryPropertyFlags req; VkMemoryPropertyFlags pref; };
    const Ladder ladders[] = {
        { "DEVICE_LOCAL (pure)", VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT },
        { "DEVICE_LOCAL|HOST_VISIBLE|COHERENT (UMA)", VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT },
        { "HOST_VISIBLE|COHERENT|CACHED (pinned)", VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT },
    };
    const VkDeviceSize steps[] = {
        512ull * 1024 * 1024, 1024ull * 1024 * 1024, 2048ull * 1024 * 1024,
        4096ull * 1024 * 1024, 8192ull * 1024 * 1024, 16384ull * 1024 * 1024,
    };
    // Aggregate limit: many LIVE 1 GiB blocks (simulates pool growth with
    // all blocks resident - the llama failure was a 1GB block failing AFTER
    // ~11 GB was committed, contradicting the freed-per-step ladder above).
    {
        VkPhysicalDeviceMemoryProperties mp2{};
        vkGetPhysicalDeviceMemoryProperties(devices[idx].device, &mp2);
        uint32_t mt = UINT32_MAX;
        for (uint32_t i = 0; i < mp2.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags f = mp2.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) { mt = i; break; }
        }
        std::cout << "  AGGREGATE DEVICE_LOCAL (pure), 1 GiB blocks kept live:\n";
        std::vector<VkDeviceMemory> live;
        const VkDeviceSize kBlock = 1024ull * 1024 * 1024;
        for (int n = 1; n <= 24; ++n) {
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = kBlock;
            ai.memoryTypeIndex = mt;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (vkAllocateMemory(dev.device, &ai, nullptr, &mem) != VK_SUCCESS) {
                std::cout << "    FAIL at block " << n << " ("
                          << live.size() << " GiB live)\n";
                break;
            }
            live.push_back(mem);
            std::cout << "    block " << n << ": ok (" << live.size() << " GiB live)\n";
        }
        for (auto m : live) vkFreeMemory(dev.device, m, nullptr);
    }

    // Interaction: does committing DEVICE_LOCAL squeeze the pinned limit?
    {
        uint32_t mtDev = UINT32_MAX, mtPin = UINT32_MAX;
        VkPhysicalDeviceMemoryProperties mp2{};
        vkGetPhysicalDeviceMemoryProperties(devices[idx].device, &mp2);
        for (uint32_t i = 0; i < mp2.memoryTypeCount; ++i) {
            const VkMemoryPropertyFlags f = mp2.memoryTypes[i].propertyFlags;
            if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && mtDev == UINT32_MAX) mtDev = i;
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) && mtPin == UINT32_MAX) mtPin = i;
        }
        std::cout << "  INTERACTION: 12 GiB DEVICE_LOCAL live, then pinned ladder:\n";
        std::vector<VkDeviceMemory> live;
        for (int n = 0; n < 12; ++n) {
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = 1024ull * 1024 * 1024;
            ai.memoryTypeIndex = mtDev;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            if (vkAllocateMemory(dev.device, &ai, nullptr, &mem) == VK_SUCCESS) live.push_back(mem);
            else break;
        }
        std::cout << "    " << live.size() << " GiB DEVICE_LOCAL committed\n";
        for (VkDeviceSize s : {512ull*1024*1024, 1024ull*1024*1024, 2048ull*1024*1024}) {
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = s;
            ai.memoryTypeIndex = mtPin;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            const bool ok = vkAllocateMemory(dev.device, &ai, nullptr, &mem) == VK_SUCCESS;
            if (ok) vkFreeMemory(dev.device, mem, nullptr);
            std::cout << "    pinned " << s / (1024*1024) << " MB: " << (ok ? "ok" : "FAIL") << "\n";
        }
        for (auto m : live) vkFreeMemory(dev.device, m, nullptr);
    }

    vkDestroyDevice(dev.device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
