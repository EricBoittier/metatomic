# CUDA 12.x only officially supports GCC up to 13. find_package(metatensor_torch)
# and find_package(Torch) enable the CUDA language, so this must run before either
# call.
#
# Include once before project() and once after project():
# - before project(): set CMAKE_CUDA_FLAGS_INIT
# - after project(): set CMAKE_CUDA_FLAGS in the cache for compiler identification

function(_metatomic_add_cuda_allow_unsupported_compiler_flag)
    if(NOT "${CMAKE_CUDA_FLAGS_INIT}" MATCHES "allow-unsupported-compiler")
        if(CMAKE_CUDA_FLAGS_INIT)
            set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler" PARENT_SCOPE)
        else()
            set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler" PARENT_SCOPE)
        endif()
    endif()

    if(NOT "${CMAKE_CUDA_FLAGS}" MATCHES "allow-unsupported-compiler")
        if(CMAKE_CUDA_FLAGS)
            set(_metatomic_cuda_flags_ "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
        else()
            set(_metatomic_cuda_flags_ "-allow-unsupported-compiler")
        endif()
        set(CMAKE_CUDA_FLAGS "${_metatomic_cuda_flags_}" CACHE STRING "Flags used by the CUDA compiler" FORCE)
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(REGEX MATCH "^([0-9]+)" _metatomic_gcc_major_ "${CMAKE_CXX_COMPILER_VERSION}")
    if(_metatomic_gcc_major_ GREATER 13)
        _metatomic_add_cuda_allow_unsupported_compiler_flag()
        message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} is newer than CUDA officially supports; adding -allow-unsupported-compiler for nvcc")
    endif()
elseif(NOT _metatomic_cuda_host_compiler_checked)
    set(_metatomic_cuda_host_compiler_checked TRUE)

    if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
        set(_metatomic_cxx_compiler "$ENV{CXX}")
    elseif(CMAKE_CXX_COMPILER)
        set(_metatomic_cxx_compiler "${CMAKE_CXX_COMPILER}")
    else()
        set(_metatomic_cxx_compiler "c++")
    endif()

    execute_process(
        COMMAND ${_metatomic_cxx_compiler} -dumpversion
        OUTPUT_VARIABLE _metatomic_gcc_version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(_metatomic_gcc_version MATCHES "^([0-9]+)")
        if(CMAKE_MATCH_1 GREATER 13)
            _metatomic_add_cuda_allow_unsupported_compiler_flag()
        endif()
    endif()
endif()
