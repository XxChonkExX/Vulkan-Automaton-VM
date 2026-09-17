#pragma once

// Re-export shim: BuddyAllocator declarations. Includes only what it needs
// (NOT the vulkan_vm.hpp umbrella - keep individual systems independently
// includable).
#include "vulkan_vm/buddy_allocator.hpp"

namespace vvm {

// ============================================================================
// Buddy Allocator (power-of-2, for large tensor allocations)
// ============================================================================

// All declarations are in buddy_allocator.hpp

} // namespace vvm
