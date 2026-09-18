// migration_token_test: MigrationEngine submits carry a completion token.
//
// E1: a submitted op's token is a VulkanTimeline (engine semaphore, value
//     >= 1, device set) - never the born-signaled 0.
// E2: after waitMigration, pollMigration AND isTokenComplete(token) agree
//     the op is done (one sync language, two consults).
// E3: host -> device -> host data round-trip through the engine verifies
//     the copies the token gates actually ran.
//
// Skips (exit 0) with no device, no transfer queue, or no timeline feature.

#include "vulkan_vm/vulkan_vm.hpp"
#include "vulkan_vm/offload.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace vvm;

static int failures = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL: %s (line %d)\n", msg, __LINE__);               \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

#define SKIP(msg)                                                             \
    do {                                                                      \
        std::printf("SKIP: %s\n", msg);                                       \
        return 0;                                                              \
    } while (0)

namespace {

struct RawBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
};

uint32_t findMemory(VkPhysicalDevice phys, uint32_t bits,
                    VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(phys, &props);
    for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (props.memoryTypes[i].propertyFlags & flags) == flags) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool makeBuffer(VkPhysicalDevice phys, VkDevice device, VkDeviceSize size,
                VkBufferUsageFlags usage, VkMemoryPropertyFlags memFlags,
                RawBuffer& out) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &out.buffer) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, out.buffer, &req);
    const uint32_t mt = findMemory(phys, req.memoryTypeBits, memFlags);
    if (mt == UINT32_MAX) return false;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = mt;
    if (vkAllocateMemory(device, &ai, nullptr, &out.memory) != VK_SUCCESS) {
        return false;
    }
    if (vkBindBufferMemory(device, out.buffer, out.memory, 0) != VK_SUCCESS) {
        return false;
    }
    if (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        if (vkMapMemory(device, out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS) {
            return false;
        }
    }
    return true;
}

void freeBuffer(VkDevice device, RawBuffer& b) {
    if (b.mapped) vkUnmapMemory(device, b.memory);
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = RawBuffer{};
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("== migration_token_test ==\n");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVM MigrationToken Test";
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        SKIP("vkCreateInstance failed");
    }
    auto devices = enumerateDevices(instance);
    if (devices.empty()) SKIP("no physical devices");
    const VkPhysicalDevice phys = devices[0].device;

    auto queues = findQueueFamilies(phys);
    if (!queues.transfer && !queues.graphics) SKIP("no transfer/graphics queue");
    const uint32_t qf = queues.transfer.value_or(queues.graphics.value());
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features v12{};
    v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    v12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &v12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        SKIP("vkCreateDevice (timeline feature) failed");
    }
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, qf, 0, &queue);

    constexpr VkDeviceSize kSize = 4ull * 1024ull * 1024ull;
    const VkBufferUsageFlags kUsage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const VkMemoryPropertyFlags kHostFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    RawBuffer dev{}, hostA{}, hostB{};
    if (!makeBuffer(phys, device, kSize, kUsage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dev) ||
        !makeBuffer(phys, device, kSize, kUsage, kHostFlags, hostA) ||
        !makeBuffer(phys, device, kSize, kUsage, kHostFlags, hostB)) {
        std::printf("SKIP: buffer setup failed\n");
        freeBuffer(device, dev);
        freeBuffer(device, hostA);
        freeBuffer(device, hostB);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    // Deterministic pattern in hostA.
    std::vector<uint8_t> pattern(static_cast<size_t>(kSize));
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    std::memcpy(hostA.mapped, pattern.data(), pattern.size());
    std::memset(hostB.mapped, 0, pattern.size());

    // Shell Allocation over the device buffer for the engine request.
    Allocation shell{};
    shell.buffer = dev.buffer;
    shell.offset = 0;
    shell.size = kSize;

    MigrationEngine engine(device, queue, qf);

    // E1+E2 (leg 1): host -> device.
    MigrationEngine::MigrationRequest up{};
    up.allocation = &shell;
    up.srcOffset = 0;
    up.dstOffset = 0;
    up.size = kSize;
    up.toHost = false;
    up.hostShadowBuffer = hostA.buffer;
    auto opUp = engine.submitMigration(up);
    CHECK(opUp.has_value(), "E1: submit host->device accepted");
    if (opUp.has_value()) {
        const CompletionToken& tok = opUp->completionToken;
        CHECK(tok.kind == CompletionToken::Kind::VulkanTimeline,
              "E1: token is VulkanTimeline kind");
        CHECK(tok.timeline != VK_NULL_HANDLE, "E1: token semaphore set");
        CHECK(tok.device == device, "E1: token device set");
        CHECK(tok.value >= 1, "E1: token value never born-signaled 0");
        engine.waitMigration(*opUp);
        CHECK(engine.pollMigration(*opUp),
              "E2: poll agrees done after wait");
        CHECK(isTokenComplete(tok, device),
              "E2: token consult agrees done after wait");
    }

    // E3 (leg 2): device -> hostB, then byte-verify the round trip.
    MigrationEngine::MigrationRequest down{};
    down.allocation = &shell;
    down.srcOffset = 0;
    down.dstOffset = 0;
    down.size = kSize;
    down.toHost = true;
    down.hostShadowBuffer = hostB.buffer;
    auto opDown = engine.submitMigration(down);
    CHECK(opDown.has_value(), "E3: submit device->host accepted");
    if (opDown.has_value()) {
        engine.waitMigration(*opDown);
        CHECK(engine.pollMigration(*opDown), "E3: poll agrees done");
        CHECK(isTokenComplete(opDown->completionToken, device),
              "E3: token consult agrees done");
        CHECK(std::memcmp(hostB.mapped, pattern.data(), pattern.size()) == 0,
              "E3: host->device->host pattern intact");
    }

    freeBuffer(device, dev);
    freeBuffer(device, hostA);
    freeBuffer(device, hostB);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::printf("migration_token_test: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
