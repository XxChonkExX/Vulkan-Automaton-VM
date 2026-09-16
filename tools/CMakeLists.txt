# vvm-daemon: local control daemon (docs/vvm_daemon_spec.md)
add_executable(vvm_daemon vvm_daemon.cpp)
target_link_libraries(vvm_daemon PRIVATE
    vulkan_vm_network
    vulkan_vm
    ${CMAKE_DL_LIBS}
)
target_include_directories(vvm_daemon PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Backend smoke tests (HIP / Level Zero / unified registry) — the backends
# load their runtimes dynamically, so these need no extra link deps.
foreach(t hip_backend_test l0_backend_test unified_set_test auto_place_test)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp)
        add_executable(${t} ${t}.cpp)
        target_link_libraries(${t} PRIVATE vulkan_vm ${CMAKE_DL_LIBS})
        target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    endif()
endforeach()
