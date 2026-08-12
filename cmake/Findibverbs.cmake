# Findibverbs.cmake - locate libibverbs (RDMA verbs)
#
# Defines:
#   IBVERBS_FOUND        - true if found
#   IBVERBS_INCLUDE_DIRS - include directory
#   IBVERBS_LIBRARIES    - the library to link
#
# Falls back to pkg-config (libibverbs.pc) when no CMake config is shipped.

if(IBVERBS_INCLUDE_DIRS AND IBVERBS_LIBRARIES)
    set(IBVERBS_FOUND TRUE)
    return()
endif()

find_path(IBVERBS_INCLUDE_DIR
    NAMES infiniband/verbs.h
    PATH_SUFFIXES include)

find_library(IBVERBS_LIBRARY
    NAMES ibverbs
    PATH_SUFFIXES lib lib64)

if(NOT IBVERBS_INCLUDE_DIR OR NOT IBVERBS_LIBRARY)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(PC_IBVERBS QUIET libibverbs)
        if(PC_IBVERBS_FOUND)
            if(NOT IBVERBS_INCLUDE_DIR)
                set(IBVERBS_INCLUDE_DIR ${PC_IBVERBS_INCLUDE_DIRS})
            endif()
            if(NOT IBVERBS_LIBRARY)
                set(IBVERBS_LIBRARY ${PC_IBVERBS_LIBRARIES})
            endif()
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ibverbs DEFAULT_MSG
    IBVERBS_INCLUDE_DIR IBVERBS_LIBRARY)

if(IBVERBS_FOUND)
    set(IBVERBS_INCLUDE_DIRS ${IBVERBS_INCLUDE_DIR})
    set(IBVERBS_LIBRARIES ${IBVERBS_LIBRARY})
    mark_as_advanced(IBVERBS_INCLUDE_DIR IBVERBS_LIBRARY)
endif()
