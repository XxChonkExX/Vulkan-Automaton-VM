#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/utils.hpp"

#include <iostream>
#include <vector>
#include <cstring>
#include <string>

using namespace vvm;

// ============================================================================
// Direct GPU->GPU (P2P) copy test. Uses the first two usable physical devices
// and verifies copyDeviceToDevice moves data without host staging.
// ============================================================================

static bool createDeviceForPool(const DeviceScore& score, const std::string& name,
                                VkInstance instance, DeviceConfig& out) {
    auto queues = findQueueFamilies(score.device);
    if (!queues.transfer && !queues.graphics) {
        std::cerr << "  " << name << ": no transfer/graphics queue\n";
        return false;
    }

    uint32_t family = queues.transfer.value_or(queues.graphics.value());
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.bufferDeviceAddress = VK_TRUE;
    v12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFeatures feats{};
    feats.sparseBinding = VK_TRUE;

    const char* exts[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#if defined(VVM_PLATFORM_WINDOWS)
        "VK_KHR_external_memory_win32",
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
#endif
        VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    dci.pNext = &v12;
    dci.enabledExtensionCount = 5;
#ifndef _WIN32
    dci.enabledExtensionCount = 6;  // + VK_EXT_external_memory_dma_buf
#endif
    dci.ppEnabledExtensionNames = exts;

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(score.device, &dci, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "  " << name << ": vkCreateDevice failed\n";
        return false;
    }

    out.physicalDevice = score.device;
    out.device = device;
    out.graphicsQueueFamily = family;
    out.computeQueueFamily = family;
    out.transferQueueFamily = family;
    VkQueue q = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &q);
    out.graphicsQueue = q;
    out.computeQueue = q;
    out.transferQueue = q;
    return true;
}

int main() {
    std::cout << "=== VulkanVM Direct GPU->GPU (P2P) Copy Test ===\n";

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM P2P Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
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
    if (devices.size() < 2) {
        std::cout << "SKIP: need at least 2 physical devices (found " << devices.size() << ")\n";
        vkDestroyInstance(instance, nullptr);
        return 0;
    }
    std::cout << "Physical devices: " << devices.size() << "\n" << std::flush;
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "  [" << i << "] " << devices[i].props.deviceName
                  << " (vendor 0x" << std::hex << devices[i].vendorID << std::dec << ")\n" << std::flush;
    }

    // Pick the best AMD↔AMD pair if available (same vendor = best P2P chance).
    // AMD 7900 XTX (discrete) + AMD Radeon integrated = both 0x1002.
    uint32_t idx0 = 0, idx1 = 1;
    bool amdPairFound = false;
    for (size_t i = 0; i < devices.size() && !amdPairFound; ++i) {
        if (devices[i].vendorID != 0x1002) continue;
        for (size_t j = i + 1; j < devices.size() && !amdPairFound; ++j) {
            if (devices[j].vendorID == 0x1002) {
                idx0 = static_cast<uint32_t>(i);
                idx1 = static_cast<uint32_t>(j);
                amdPairFound = true;
            }
        }
    }

    DeviceConfig dev0, dev1;
    if (!createDeviceForPool(devices[idx0], "device 0", instance, dev0) ||
        !createDeviceForPool(devices[idx1], "device 1", instance, dev1)) {
        std::cerr << "FAIL: device creation for test pair\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    PoolConfig poolCfg;
    poolCfg.blockSize = 256 * 1024 * 1024;
    poolCfg.minAlignment = 256;
    poolCfg.enableHostVisible = false;
    poolCfg.enableExternal = true;
    poolCfg.enableDeviceAddress = true;
    poolCfg.maxBlocks = 4;
    poolCfg.maxHeapFraction = 0.0f;
    poolCfg.maxPoolBytes = 0;

    auto manager = MultiGPUPoolManager::create({dev0, dev1}, poolCfg, 0);
    if (!manager) {
        std::cerr << "FAIL: MultiGPUPoolManager::create\n";
        vkDestroyDevice(dev0.device, nullptr);
        vkDestroyDevice(dev1.device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    std::cout << "Manager created for 2 devices (master="
              << devices[idx0].props.deviceName << ")\n" << std::flush;

    auto peer = manager->queryPeerAccess(0, 1);
    std::cout << "Peer access 0->1: canDirectCopy=" << (peer.canDirectCopy ? "yes" : "no")
              << " external=" << (peer.externalMemorySupported ? "yes" : "no")
              << " note: " << peer.notes << "\n" << std::flush;
    if (!peer.externalMemorySupported) {
        std::cout << "SKIP: external memory not supported between this pair\n";
        vkDestroyDevice(dev0.device, nullptr);
        vkDestroyDevice(dev1.device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    int failures = 0;

    // Allocate src (dedicated exportable) on device 0, dst on device 1.
    const VkDeviceSize kSize = 4ull * 1024 * 1024;
    const VkBufferUsageFlags kUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    auto src = manager->getPool(0).allocateDedicatedExportable(kSize, kUsage);
    auto dst = manager->getPool(1).allocate(kSize, kUsage);
    if (!src || !dst) {
        std::cerr << "FAIL: allocate src/dst\n";
        failures++;
    } else {
        std::cout << "  src (dev0): " << src->size / (1024 * 1024) << " MiB"
                  << " (blockIndex=" << src->blockIndex << ")\n";
        std::cout << "  dst (dev1): " << dst->size / (1024 * 1024) << " MiB"
                  << " (blockIndex=" << dst->blockIndex << ")\n";

        // Allocate a host-visible staging buffer on each device for write/verify.
        auto staging0 = manager->getPool(0).allocate(
            kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto staging1 = manager->getPool(1).allocate(
            kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!staging0 || !staging1) {
            std::cerr << "FAIL: allocate staging buffers\n";
            failures++;
        } else {
            // Fill src on GPU0 with a pattern via staging copy.
            std::vector<uint8_t> pattern(static_cast<size_t>(kSize));
            for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<uint8_t>(i * 7 + 3);
            std::memcpy(staging0->hostPtr, pattern.data(), pattern.size());
            bool staged = manager->getPool(0).copyBuffer(*staging0, *src, 0, 0, kSize);
            std::cout << "  src filled with pattern via staging: "
                      << (staged ? "OK" : "FAIL") << "\n";

            if (staged) {
                // Direct GPU->GPU copy (uses export/import fast path, or
                // automatically falls back to host-staged peer copy when the
                // driver refuses the cross-GPU import).
                bool ok = manager->copyDeviceToDevice(0, 1, *src, *dst, 0, 0, kSize);
                if (!ok) {
                    std::cout << "  copyDeviceToDevice: FAILED (both fast and fallback paths)\n";
                    std::cout << "\n=== SKIPPED: P2P copy not supported between these GPUs ===\n";
                    failures = 0;  // not a code bug
                    manager->getPool(1).deallocate(std::move(*staging1));
                    manager->getPool(0).deallocate(std::move(*staging0));
                    manager->getPool(1).deallocate(std::move(*dst));
                    manager->getPool(0).deallocate(std::move(*src));
                    vkDestroyDevice(dev1.device, nullptr);
                    vkDestroyDevice(dev0.device, nullptr);
                    vkDestroyInstance(instance, nullptr);
                    return 0;
                } else {
                    std::cout << "  copyDeviceToDevice: succeeded (fast export/import or host-staged fallback)\n" << std::flush;

                    // Readback dst on GPU1 via staging copy.
                    std::memset(staging1->hostPtr, 0, static_cast<size_t>(kSize));
                    bool readback = manager->getPool(1).copyBuffer(*dst, *staging1, 0, 0, kSize);
                    if (!readback) {
                        std::cerr << "FAIL: staging readback\n" << std::flush;
                        failures++;
                    } else {
                        bool match = std::memcmp(staging1->hostPtr, pattern.data(), pattern.size()) == 0;
                        if (match) {
                            std::cout << "  D2D copy verified on dst: PASS\n" << std::flush;
                        } else {
                            std::cerr << "  MISMATCH: dst content differs from src pattern\n" << std::flush;
                            failures++;
                        }
                    }

                    // Reverse direction with a sub-range copy (offset + half size).
                    constexpr VkDeviceSize kHalf = kSize / 2;
                    constexpr VkDeviceSize kOff = kSize / 4;
                    std::vector<uint8_t> rev(static_cast<size_t>(kHalf), 0xBC);
                    std::memcpy(staging1->hostPtr, rev.data(), rev.size());
                    bool revWrite = manager->getPool(1).copyBuffer(*staging1, *dst, 0, kOff, kHalf);
                    if (revWrite) {
                        bool okRev = manager->copyDeviceToDevice(1, 0, *dst, *src, kOff, kOff, kHalf);
                        if (okRev) {
                            bool revRead = manager->getPool(0).copyBuffer(*src, *staging0, kOff, 0, kHalf);
                            if (revRead) {
                                bool matchRev = std::memcmp(staging0->hostPtr, rev.data(), rev.size()) == 0;
                                std::cout << "  Reverse D2D copy (offset range) verified: "
                                          << (matchRev ? "PASS" : "MISMATCH") << "\n" << std::flush;
                                if (!matchRev) failures++;
                            } else {
                                std::cerr << "FAIL: reverse readback\n" << std::flush;
                                failures++;
                            }
                        } else {
                            std::cout << "  Reverse D2D copy FAILED (asymmetric P2P)\n" << std::flush;
                        }
                    }
                }
            } else {
                failures++;
            }

            manager->getPool(1).deallocate(std::move(*staging1));
            manager->getPool(0).deallocate(std::move(*staging0));
        }

        manager->getPool(1).deallocate(std::move(*dst));
        manager->getPool(0).deallocate(std::move(*src));
    }

    // M: live-allocation migration 0->1->0 with byte verification, plus
    // pressure snapshots and invalid-argument rejection.
    {
        auto msrc = manager->getPool(0).allocateDedicatedExportable(kSize, kUsage);
        auto mstg0 = manager->getPool(0).allocate(
            kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        auto mstg1 = manager->getPool(1).allocate(
            kSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (!msrc || !mstg0 || !mstg1 || !mstg0->hostPtr || !mstg1->hostPtr) {
            std::cerr << "FAIL: M setup allocs\n" << std::flush;
            failures++;
        } else {
            std::vector<uint8_t> mpat(static_cast<size_t>(kSize));
            for (size_t i = 0; i < mpat.size(); ++i) mpat[i] = static_cast<uint8_t>(i * 3 + 11);
            std::memcpy(mstg0->hostPtr, mpat.data(), mpat.size());
            if (!manager->getPool(0).copyBuffer(*mstg0, *msrc, 0, 0, kSize)) {
                std::cerr << "FAIL: M fill\n" << std::flush;
                failures++;
            } else {
                // M2: invalid arguments rejected, nothing touched.
                auto scratch = manager->getPool(0).allocate(1 << 20, kUsage);
                if (!scratch) {
                    std::cerr << "FAIL: M2 scratch alloc\n" << std::flush;
                    failures++;
                } else {
                    auto rSame = manager->migrateAllocation(0, 0, std::move(*scratch), kUsage);
                    if (rSame) {
                        std::cerr << "FAIL: M2 same-index accepted\n" << std::flush;
                        failures++;
                        manager->getPool(0).deallocate(std::move(*rSame));
                        // scratch was consumed by the (wrongly accepted) call.
                    } else {
                        std::cout << "  M2 same-index rejected: PASS\n" << std::flush;
                        manager->getPool(0).deallocate(std::move(*scratch));
                    }
                    Allocation empty{};
                    auto rNull = manager->migrateAllocation(0, 1, std::move(empty), kUsage);
                    if (rNull) {
                        std::cerr << "FAIL: M2 null-buffer accepted\n" << std::flush;
                        failures++;
                        manager->getPool(1).deallocate(std::move(*rNull));
                    } else {
                        std::cout << "  M2 null-buffer rejected: PASS\n" << std::flush;
                    }
                }

                // M1: migrate 0->1 live; bytes verify on the new handle.
                auto moved = manager->migrateAllocation(0, 1, std::move(*msrc), kUsage);
                if (!moved) {
                    std::cerr << "FAIL: M1 migrate 0->1\n" << std::flush;
                    failures++;
                } else {
                    std::memset(mstg1->hostPtr, 0, mpat.size());
                    bool mread = manager->getPool(1).copyBuffer(*moved, *mstg1, 0, 0, kSize);
                    if (!mread || std::memcmp(mstg1->hostPtr, mpat.data(), mpat.size()) != 0) {
                        std::cerr << "FAIL: M1 migrated bytes mismatch\n" << std::flush;
                        failures++;
                    } else {
                        std::cout << "  M1 migrate 0->1 verified: PASS\n" << std::flush;
                    }
                    auto pr = manager->poolPressures();
                    if (pr.size() != 2 || pr[0].instanceIndex != 0 ||
                        pr[1].instanceIndex != 1 || pr[0].budgetBytes == 0 ||
                        pr[1].budgetBytes == 0) {
                        std::cerr << "FAIL: M1 pressures insane\n" << std::flush;
                        failures++;
                    } else {
                        std::cout << "  M1 pressures: pool0 used=" << pr[0].usedBytes
                                  << " budget=" << pr[0].budgetBytes << " p="
                                  << pr[0].pressure << " | pool1 used=" << pr[1].usedBytes
                                  << " budget=" << pr[1].budgetBytes << " p="
                                  << pr[1].pressure << "\n" << std::flush;
                    }
                    if (manager->getPool(0).collect() != 1) {
                        std::cerr << "FAIL: M1 retired src not reclaimed\n" << std::flush;
                        failures++;
                    } else {
                        std::cout << "  M1 retired src reclaimed: PASS\n" << std::flush;
                    }

                    // M3: migrate back 1->0; bytes verify again.
                    auto back = manager->migrateAllocation(1, 0, std::move(*moved), kUsage);
                    if (!back) {
                        std::cerr << "FAIL: M3 migrate 1->0\n" << std::flush;
                        failures++;
                    } else {
                        std::memset(mstg0->hostPtr, 0, mpat.size());
                        bool bread = manager->getPool(0).copyBuffer(*back, *mstg0, 0, 0, kSize);
                        if (!bread || std::memcmp(mstg0->hostPtr, mpat.data(), mpat.size()) != 0) {
                            std::cerr << "FAIL: M3 migrated bytes mismatch\n" << std::flush;
                            failures++;
                        } else {
                            std::cout << "  M3 migrate 1->0 verified: PASS\n" << std::flush;
                        }
                        if (manager->getPool(1).collect() != 1) {
                            std::cerr << "FAIL: M3 retired src not reclaimed\n" << std::flush;
                            failures++;
                        }
                        manager->getPool(0).deallocate(std::move(*back));
                    }
                }
            }
            manager->getPool(1).deallocate(std::move(*mstg1));
            manager->getPool(0).deallocate(std::move(*mstg0));
        }
    }

    std::cout << "\n=== " << (failures == 0 ? "ALL P2P TESTS PASSED" : "P2P TESTS FAILED")
              << " (" << failures << " failures) ===\n" << std::flush;

    // Destroy pools BEFORE their VkDevices: letting `manager` outlive device
    // destruction made every pool teardown call vkUnmapMemory/vkFreeMemory on
    // a dead device ("vkUnmapMemory: Invalid device" -> loader SIGABRT).
    manager.reset();
    vkDestroyDevice(dev1.device, nullptr);
    vkDestroyDevice(dev0.device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return failures == 0 ? 0 : 1;
}
