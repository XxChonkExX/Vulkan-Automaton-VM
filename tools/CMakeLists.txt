# vvm-daemon: local control daemon (docs/vvm_daemon_spec.md)
add_executable(vvm_daemon vvm_daemon.cpp)
target_link_libraries(vvm_daemon PRIVATE
    vulkan_vm_network
    vulkan_vm
    ${CMAKE_DL_LIBS}
)
target_include_directories(vvm_daemon PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)

# vvm-info: read-only diagnostics bundle (devices, heaps, caps, P2P policy)
add_executable(vvm_info vvm_info.cpp)
target_link_libraries(vvm_info PRIVATE vulkan_vm ${CMAKE_DL_LIBS})
target_include_directories(vvm_info PRIVATE ${CMAKE_SOURCE_DIR}/include ${Vulkan_INCLUDE_DIRS})

# Backend smoke tests (HIP / Level Zero / CUDA / unified registry) - the
# backends load their runtimes dynamically, so these need no extra link deps.
foreach(t hip_backend_test l0_backend_test cuda_backend_test unified_set_test auto_place_test)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
        add_executable(${t} ${t}.cpp)
        target_link_libraries(${t} PRIVATE vulkan_vm ${CMAKE_DL_LIBS})
        target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include ${Vulkan_INCLUDE_DIRS})
    endif()
endforeach()
