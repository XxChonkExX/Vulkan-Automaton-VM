#pragma once

// VulkanVM Transport layer (OPTIONAL - not part of the core memory pool).
// Includes: cluster networking, multi-node management, RDMA/UCX/TCP
// transports, unified tensor transport, placement execution.
//
// The core Chonk Buffer (vulkan_vm.hpp) builds and ships WITHOUT this layer.
#include "vulkan_vm/network.hpp"
#include "vulkan_vm/tensor_transport.hpp"
#include "vulkan_vm/ucx_transport.hpp"
