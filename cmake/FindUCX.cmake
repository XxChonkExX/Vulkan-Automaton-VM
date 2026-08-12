# FindUCX.cmake - Find Unified Communication X (UCX) library
# 
# Sets:
#   UCX_FOUND - whether UCX was found
#   UCX_INCLUDE_DIRS - include directories
#   UCX_LIBRARIES - libraries to link
#   UCX_VERSION - version string

find_path(UCX_INCLUDE_DIR ucp/api/ucp.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/ucx/include
        $ENV{UCX_DIR}/include
        "C:/Program Files/UCX/include"
        "C:/Program Files (x86)/UCX/include"
    DOC "UCX include directory"
)

find_library(UCX_UCP_LIB ucp
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/ucx/lib
        $ENV{UCX_DIR}/lib
        "C:/Program Files/UCX/lib"
        "C:/Program Files (x86)/UCX/lib"
    DOC "UCX UCP library"
)

find_library(UCX_UCT_LIB uct
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/ucx/lib
        $ENV{UCX_DIR}/lib
        "C:/Program Files/UCX/lib"
        "C:/Program Files (x86)/UCX/lib"
    DOC "UCX UCT library"
)

find_library(UCX_UCS_LIB ucs
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/ucx/lib
        $ENV{UCX_DIR}/lib
        "C:/Program Files/UCX/lib"
        "C:/Program Files (x86)/UCX/lib"
    DOC "UCX UCS library"
)

find_library(UCX_UMM_LIB umm
    PATHS
        /usr/lib
        /usr/lib64
        /usr/local/lib
        /opt/ucx/lib
        $ENV{UCX_DIR}/lib
        "C:/Program Files/UCX/lib"
        "C:/Program Files (x86)/UCX/lib"
    DOC "UCX UMM library"
)

set(UCX_LIBRARIES ${UCX_UCP_LIB} ${UCX_UCT_LIB} ${UCX_UCS_LIB} ${UCX_UMM_LIB})

if(UCX_INCLUDE_DIR AND UCX_UCP_LIB AND UCX_UCT_LIB AND UCX_UCS_LIB)
    set(UCX_FOUND TRUE)
    set(UCX_INCLUDE_DIRS ${UCX_INCLUDE_DIR})
    
    # Try to get version from ucx_version.h
    find_file(UCX_VERSION_H ucp/version.h
        PATHS ${UCX_INCLUDE_DIR}
    )
    if(UCX_VERSION_H)
        file(STRINGS ${UCX_VERSION_H} UCX_VERSION_STR REGEX "^#define UCX_VERSION ")
        if(UCX_VERSION_STR)
            string(REGEX REPLACE "^#define UCX_VERSION \"([^\"]+)\"" "\\1" UCX_VERSION "${UCX_VERSION_STR}")
        endif()
    endif()
    
    message(STATUS "Found UCX: ${UCX_VERSION} at ${UCX_INCLUDE_DIR}")
else()
    set(UCX_FOUND FALSE)
    if(UCX_FIND_REQUIRED)
        message(FATAL_ERROR "UCX not found")
    endif()
endif()

mark_as_advanced(
    UCX_INCLUDE_DIR
    UCX_UCP_LIB
    UCX_UCT_LIB
    UCX_UCS_LIB
    UCX_UMM_LIB
)