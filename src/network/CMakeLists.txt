# Network module CMakeLists.txt
# The core network module (TCP control plane + host-staged migration) builds
# without any external dependencies. gRPC and RDMA/verbs are optional extras.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)
find_package(Protobuf QUIET)
find_package(gRPC QUIET)
find_package(ibverbs QUIET)
find_package(rdma_cm QUIET)

# Level Zero (Intel GPU-direct on Windows)
if(WIN32)
    # Try to find Level Zero SDK
    set(LEVEL_ZERO_ROOT_PATHS
        "C:/Program Files/LevelZeroSDK"
        "C:/Program Files (x86)/LevelZeroSDK"
        "$ENV{LEVEL_ZERO_SDK_PATH}"
    )
    foreach(path ${LEVEL_ZERO_ROOT_PATHS})
        if(EXISTS "${path}/include/level_zero/ze_api.h")
            set(LEVEL_ZERO_INCLUDE_DIR "${path}/include")
            set(LEVEL_ZERO_LIB_DIR "${path}/lib")
            message(STATUS "Found Level Zero SDK at ${path}")
            break()
        endif()
    endforeach()

    if(LEVEL_ZERO_INCLUDE_DIR)
        find_library(LEVEL_ZERO_LIB loader
            PATHS "${LEVEL_ZERO_LIB_DIR}"
            NO_DEFAULT_PATH)
        if(LEVEL_ZERO_LIB)
            message(STATUS "VulkanVM network: Level Zero (Intel GPU-direct) enabled")
            list(APPEND NETWORK_ABI_DEFS VVM_HAS_LEVEL_ZERO=1)
            set(VVM_NETWORK_L0_LIBS ${LEVEL_ZERO_LIB})
            set(VVM_NETWORK_L0_INCS ${LEVEL_ZERO_INCLUDE_DIR})
        else()
            message(WARNING "Level Zero headers found but loader.lib not found in ${LEVEL_ZERO_LIB_DIR}")
        endif()
    else()
        message(STATUS "VulkanVM network: Level Zero SDK not found (Intel GPU-direct disabled)")
    endif()
endif()

set(NETWORK_CORE_SRCS
    tcp_transport.cpp
    multi_node_manager.cpp
    network.cpp
    model_registry.cpp
)

set(NETWORK_PRIV_DEFS
    VVM_NETWORK_EXPORTS
    VVM_BUILD_SHARED=${VVM_BUILD_SHARED}
)

# These alter the layout of public classes (guarded members in the headers),
# so every TU that includes the network headers must see the same value.
set(NETWORK_ABI_DEFS)

# gRPC control plane (optional): cluster client/server
if(Protobuf_FOUND AND gRPC_FOUND)
    message(STATUS "VulkanVM network: gRPC control plane enabled")
    list(APPEND NETWORK_CORE_SRCS
        cluster_client.cpp
        cluster_server.cpp
    )
    list(APPEND NETWORK_ABI_DEFS VVM_NETWORK_HAS_GRPC=1)
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
    list(APPEND NETWORK_ABI_DEFS VVM_NETWORK_HAS_VERBS=1)
    set(VVM_NETWORK_VERBS_LIBS ${IBVERBS_LIBRARIES} ${RDMA_CM_LIBRARIES})
    set(VVM_NETWORK_VERBS_INCS ${IBVERBS_INCLUDE_DIRS} ${RDMA_CM_INCLUDE_DIRS})
else()
    message(STATUS "VulkanVM network: RDMA/verbs transport disabled (host-staged TCP only)")
endif()

# UCX transport (optional): Unified Communication X
if(UCX_FOUND)
    message(STATUS "VulkanVM network: UCX transport enabled")
    list(APPEND NETWORK_CORE_SRCS
        ucx_transport.cpp
    )
    list(APPEND NETWORK_ABI_DEFS VVM_HAS_UCX=1)
    set(VVM_NETWORK_UCX_LIBS ${UCX_LIBRARIES})
    set(VVM_NETWORK_UCX_INCS ${UCX_INCLUDE_DIRS})
else()
    message(STATUS "VulkanVM network: UCX transport disabled")
endif()

# Windows: Level Zero GPU-direct (Intel)
if(WIN32 AND LevelZero_FOUND)
    message(STATUS "VulkanVM network: Level Zero GPU-direct enabled for Intel")
    list(APPEND NETWORK_CORE_SRCS gpu_direct_registration.cpp)
    set(VVM_NETWORK_L0_LIBS ${LevelZero_LIBRARIES})
    set(VVM_NETWORK_L0_INCS ${LevelZero_INCLUDE_DIRS})
endif()

# Windows: Network Direct (ND) transport. NDKPI is the kernel-mode contract
# RNIC miniports implement; the user-mode consumer this library talks to is the
# ND SPI (IND2*, headers vendored under third_party/ndk).
if(WIN32)
    message(STATUS "VulkanVM network: ND (Network Direct) transport enabled on Windows")
    list(APPEND NETWORK_CORE_SRCS ndk_transport.cpp)
    list(APPEND NETWORK_ABI_DEFS VVM_NETWORK_HAS_NDKPI=1)
    set(VVM_NETWORK_NDK_INCS ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/ndk)
endif()

# TLS support via OpenSSL
if(OPENSSL_FOUND)
    message(STATUS "VulkanVM network: TLS support enabled via OpenSSL")
    list(APPEND NETWORK_ABI_DEFS VVM_NETWORK_HAS_TLS=1)
    set(VVM_NETWORK_TLS_LIBS ${OPENSSL_LIBRARIES})
else()
    message(STATUS "VulkanVM network: TLS support disabled (OpenSSL not found)")
endif()

if(VVM_BUILD_SHARED)
    add_library(vulkan_vm_network SHARED ${NETWORK_CORE_SRCS})
else()
    add_library(vulkan_vm_network STATIC ${NETWORK_CORE_SRCS})
endif()

target_include_directories(vulkan_vm_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cross_gpu
    ${Vulkan_INCLUDE_DIRS}
    ${VVM_NETWORK_VERBS_INCS}
    ${VVM_NETWORK_L0_INCS}
    ${VVM_NETWORK_UCX_INCS}
    ${VVM_NETWORK_NDK_INCS}
)

target_link_libraries(vulkan_vm_network PUBLIC vulkan_vm)
target_link_libraries(vulkan_vm_network PRIVATE
    Threads::Threads
    ${VVM_NETWORK_GRPC_LIBS}
    ${VVM_NETWORK_VERBS_LIBS}
    ${VVM_NETWORK_UCX_LIBS}
    ${VVM_NETWORK_L0_LIBS}
    ${VVM_NETWORK_TLS_LIBS}
)

# Winsock is required by the TCP transport on Windows. The ND transport also
# uses iphlpapi helpers.
if(WIN32)
    target_link_libraries(vulkan_vm_network PRIVATE ws2_32 iphlpapi)
    target_compile_definitions(vulkan_vm_network PRIVATE _WINSOCKAPI_)
endif()

target_compile_definitions(vulkan_vm_network
    PRIVATE ${NETWORK_PRIV_DEFS}
    PUBLIC ${NETWORK_ABI_DEFS}
)

# Consumer targets include the network headers too; the verbs/grpc/tls macros
# change public class layout (e.g. MultiNodePoolManager members are guarded by
# VVM_NETWORK_HAS_VERBS), so the include dirs and ABI-affecting defines must be
# visible to any TU that uses the headers.
target_include_directories(vulkan_vm_network PUBLIC
    ${VVM_NETWORK_VERBS_INCS}
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
    include/vulkan_vm/network/tcp_transport.hpp
    include/vulkan_vm/network/multi_node_manager.hpp
    include/vulkan_vm/network/rdma_transport.hpp
    include/vulkan_vm/network/gpu_direct_registration.hpp
    include/vulkan_vm/network/cluster_client.hpp
    include/vulkan_vm/network/cluster_server.hpp
    include/vulkan_vm/network/model_registry.hpp
    include/vulkan_vm/network/model_types.hpp
    include/vulkan_vm/network/sha256.hpp
    include/vulkan_vm/ucx_transport.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vulkan_vm/network
)


