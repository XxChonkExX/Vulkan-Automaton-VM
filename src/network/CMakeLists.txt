# Network module CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Find gRPC and Protobuf
find_package(Protobuf QUIET)
find_package(gRPC QUIET)

if(NOT Protobuf_FOUND OR NOT gRPC_FOUND)
    message(WARNING "Protobuf/gRPC not found, network module disabled")
    return()
endif()

# Protobuf generation
set(PROTO_FILES
    ${CMAKE_SOURCE_DIR}/proto/control.proto
)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})

# Protobuf target for internal use
add_library(vulkan_vm_proto INTERFACE)
target_include_directories(vulkan_vm_proto INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)
target_link_libraries(vulkan_vm_proto INTERFACE
    ${Protobuf_LIBRARIES}
    ${gRPC_LIBRARIES}
    gRPC::grpc++
    gRPC::grpc++_reflection
)

# Find RDMA libraries (optional but recommended)
find_package(ibverbs QUIET)
find_package(rdma_cm QUIET)

# Network module library
set(NETWORK_SRCS
    src/network/cluster_client.cpp
    src/network/cluster_server.cpp
    src/network/rdma_transport.cpp
    src/network/multi_node_manager.cpp
    src/network/gpu_direct_registration.cpp
    src/network/network.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

add_library(vulkan_vm_network ${NETWORK_SRCS})
target_include_directories(vulkan_vm_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${gRPC_INCLUDE_DIRS}
)
if(IBVERBS_FOUND)
    target_include_directories(vulkan_vm_network PRIVATE ${IBVERBS_INCLUDE_DIRS})
endif()
if(RDMA_CM_FOUND)
    target_include_directories(vulkan_vm_network PRIVATE ${RDMA_CM_INCLUDE_DIRS})
endif()
target_link_libraries(vulkan_vm_network PRIVATE
    vulkan_vm
    vulkan_vm_proto
    gRPC::grpc++
    gRPC::grpc++_reflection
)
if(IBVERBS_FOUND)
    target_link_libraries(vulkan_vm_network PRIVATE ${IBVERBS_LIBRARIES})
endif()
if(RDMA_CM_FOUND)
    target_link_libraries(vulkan_vm_network PRIVATE ${RDMA_CM_LIBRARIES})
endif()

# Compile definitions
target_compile_definitions(vulkan_vm_network PRIVATE
    VVM_NETWORK_EXPORTS
    VVM_BUILD_SHARED=${VVM_BUILD_SHARED}
)

# Install
install(TARGETS vulkan_vm_network
    EXPORT VulkanVMTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    include/vulkan_vm/network.hpp
    include/vulkan_vm/network/network_config.hpp
    include/vulkan_vm/network/multi_node_manager.hpp
    include/vulkan_vm/network/rdma_transport.hpp
    include/vulkan_vm/network/cluster_client.hpp
    include/vulkan_vm/network/cluster_server.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vulkan_vm/network
)

# Tests
if(VVM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(VVM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()