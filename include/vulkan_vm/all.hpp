#pragma once

// VulkanVM "everything" umbrella: core memory pool + transport layer.
// For core-only usage (Chonk Buffer standalone), include
// "vulkan_vm/vulkan_vm.hpp" instead - it has zero network dependencies.

#include "vulkan_vm/vulkan_vm.hpp"      // core: pool, allocator, offload,
                                        // cross-GPU, sparse, placement planning
#include "vulkan_vm/transport.hpp"      // optional: network, tensor transport
