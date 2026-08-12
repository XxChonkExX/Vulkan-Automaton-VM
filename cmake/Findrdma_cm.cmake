# Findrdma_cm.cmake - locate librdmacm (RDMA connection manager)
#
# Defines:
#   RDMA_CM_FOUND        - true if found
#   RDMA_CM_INCLUDE_DIRS - include directory
#   RDMA_CM_LIBRARIES    - the library to link
#
# Falls back to pkg-config (librdmacm.pc) when no CMake config is shipped.

if(RDMA_CM_INCLUDE_DIRS AND RDMA_CM_LIBRARIES)
    set(RDMA_CM_FOUND TRUE)
    return()
endif()

find_path(RDMA_CM_INCLUDE_DIR
    NAMES rdma/rdma_cma.h
    PATH_SUFFIXES include)

find_library(RDMA_CM_LIBRARY
    NAMES rdmacm
    PATH_SUFFIXES lib lib64)

if(NOT RDMA_CM_INCLUDE_DIR OR NOT RDMA_CM_LIBRARY)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_RDMA_CM QUIET librdmacm)
        if(PC_RDMA_CM_FOUND)
            if(NOT RDMA_CM_INCLUDE_DIR)
                set(RDMA_CM_INCLUDE_DIR ${PC_RDMA_CM_INCLUDE_DIRS})
            endif()
            if(NOT RDMA_CM_LIBRARY)
                set(RDMA_CM_LIBRARY ${PC_RDMA_CM_LIBRARIES})
            endif()
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(rdma_cm DEFAULT_MSG
    RDMA_CM_INCLUDE_DIR RDMA_CM_LIBRARY)

if(RDMA_CM_FOUND)
    set(RDMA_CM_INCLUDE_DIRS ${RDMA_CM_INCLUDE_DIR})
    set(RDMA_CM_LIBRARIES ${RDMA_CM_LIBRARY})
    mark_as_advanced(RDMA_CM_INCLUDE_DIR RDMA_CM_LIBRARY)
endif()
