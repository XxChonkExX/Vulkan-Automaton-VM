# Network module CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# The core network module (TCP control plane + host-staged migration) builds
# without any external dependencies. gRPC and RDMA/verbs are optional extras.
find_package(Protobuf QUIET)
find_package(gRPC QUIET)
find_package(ibverbs QUIET)
find_package(rdma_cm QUIET)

set(NETWORK_CORE_SRCS
    tcp_transport.cpp
    multi_node_manager.cpp
    network.cpp
)

set(NETWORK_DEFS
    VVM_NETWORK_EXPORTS
    VVM_BUILD_SHARED=${VVM_BUILD_SHARED}
)

# gRPC control plane (optional): cluster client/server
if(Protobuf_FOUND AND gRPC_FOUND)
    message(STATUS "VulkanVM network: gRPC control plane enabled")
    list(APPEND NETWORK_CORE_SRCS
        cluster_client.cpp
        cluster_server.cpp
    )
    list(APPEND NETWORK_DEFS VVM_NETWORK_HAS_GRPC=1)
    set(VVM_NETWORK_GRPC_LIBS ${Protobuf_LIBRARIES} ${gRPC_LIBRARIES} gRPC::grpc++)
else()
    message(STATUS "VulkanVM network: gRPC control plane disabled (TCP only)")
endif()

# RDMA transport (optional): verbs-backed GPU-direct path
if(IBVERBS_FOUND AND RDMA_CM_FOUND)
    message(STATUS "VulkanVM network: RDMA/verbs transport enabled")
    list(APPEND NETWORK_CORE_SRCS
        rdma_transport.cpp
        gpu_direct_registration.cpp
    )
    list(APPEND NETWORK_DEFS VVM_NETWORK_HAS_VERBS=1)
    set(VVM_NETWORK_VERBS_LIBS ${IBVERBS_LIBRARIES} ${RDMA_CM_LIBRARIES})
    set(VVM_NETWORK_VERBS_INCS ${IBVERBS_INCLUDE_DIRS} ${RDMA_CM_INCLUDE_DIRS})
else()
    message(STATUS "VulkanVM network: RDMA/verbs transport disabled (host-staged TCP only)")
endif()

# TLS support via OpenSSL
if(OPENSSL_FOUND)
    message(STATUS "VulkanVM network: TLS support enabled via OpenSSL")
    list(APPEND NETWORK_DEFS VVM_NETWORK_HAS_TLS=1)
    set(VVM_NETWORK_TLS_LIBS ${OPENSSL_LIBRARIES})
else()
    message(STATUS "VulkanVM network: TLS support disabled (OpenSSL not found)")
endif()

add_library(vulkan_vm_network ${NETWORK_CORE_SRCS})

target_include_directories(vulkan_vm_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
    ${VVM_NETWORK_VERBS_INCS}
    ${WINDOWS_SDK_INCLUDE_PATH}
    ${WINDOWS_SDK_INCLUDE_PATH_SHARED}
)

target_link_libraries(vulkan_vm_network PUBLIC vulkan_vm)
target_link_libraries(vulkan_vm_network PRIVATE
    Threads::Threads
    ${VVM_NETWORK_GRPC_LIBS}
    ${VVM_NETWORK_VERBS_LIBS}
    ${VVM_NETWORK_TLS_LIBS}
)

# Winsock is required by the TCP transport on Windows.
if(WIN32)
    target_link_libraries(vulkan_vm_network PRIVATE ws2_32)
    target_compile_definitions(vulkan_vm_network PRIVATE _WINSOCKAPI_)
endif()

target_compile_definitions(vulkan_vm_network PRIVATE ${NETWORK_DEFS})

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
    include/vulkan_vm/network/tcp_transport.hpp
    include/vulkan_vm/network/multi_node_manager.hpp
    include/vulkan_vm/network/rdma_transport.hpp
    include/vulkan_vm/network/cluster_client.hpp
    include/vulkan_vm/network/cluster_server.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vulkan_vm/network
)

# Tests
if(VVM_BUILD_TESTS)
    enable_testing()
    # Tests are in the top-level tests/ directory, not here
endif()

# Examples
if(VVM_BUILD_EXAMPLES)
    # Examples are in the top-level examples/ directory, not here
endif()
