# Tensor Transport Module

set(TENSOR_TRANSPORT_SRCS
    tensor_transport.cpp
)

add_library(vulkan_vm_tensor ${TENSOR_TRANSPORT_SRCS})

target_include_directories(vulkan_vm_tensor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(vulkan_vm_tensor PRIVATE
    vulkan_vm
    vulkan_vm_network
    Threads::Threads
)

# Install
install(TARGETS vulkan_vm_tensor
    EXPORT VulkanVMTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/vulkan_vm/tensor_transport.hpp
    DESTINATION include/vulkan_vm
)