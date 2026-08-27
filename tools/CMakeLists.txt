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
