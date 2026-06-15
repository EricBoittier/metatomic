# CUDA 12.x only supports GCC up to 13 as nvcc's host compiler. When the main C++
# compiler is newer (common on HPC clusters), point nvcc at an older g++ instead.
#
# Include once before project() and once after project(), before find_package(Torch)
# or find_package(metatensor_torch) enable CUDA.
#
# Override detection with METATOMIC_CUDA_HOST_COMPILER or CUDAHOSTCXX.

function(_metatomic_gcc_major_version _compiler _out_major)
    execute_process(
        COMMAND "${_compiler}" -dumpversion
        OUTPUT_VARIABLE _version
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _result
    )
    if(NOT _result EQUAL 0)
        set(${_out_major} -1 PARENT_SCOPE)
        return()
    endif()
    if(_version MATCHES "^([0-9]+)")
        set(${_out_major} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
        set(${_out_major} -1 PARENT_SCOPE)
    endif()
endfunction()

function(_metatomic_try_cuda_host_compiler _compiler _out_var)
    if(NOT EXISTS "${_compiler}")
        return()
    endif()
    _metatomic_gcc_major_version("${_compiler}" _major)
    if(_major GREATER 0 AND _major LESS_EQUAL 13)
        set(${_out_var} "${_compiler}" PARENT_SCOPE)
    endif()
endfunction()

function(_metatomic_find_cuda_host_compiler _out_var)
    set(_explicit "")
    if(DEFINED ENV{METATOMIC_CUDA_HOST_COMPILER} AND NOT "$ENV{METATOMIC_CUDA_HOST_COMPILER}" STREQUAL "")
        set(_explicit "$ENV{METATOMIC_CUDA_HOST_COMPILER}")
    elseif(DEFINED ENV{CUDAHOSTCXX} AND NOT "$ENV{CUDAHOSTCXX}" STREQUAL "")
        set(_explicit "$ENV{CUDAHOSTCXX}")
    endif()
    if(_explicit)
        _metatomic_try_cuda_host_compiler("${_explicit}" _found)
        if(_found)
            set(${_out_var} "${_found}" PARENT_SCOPE)
            return()
        endif()
        message(WARNING "Explicit CUDA host compiler '${_explicit}' is missing or uses GCC > 13")
    endif()

    if(CMAKE_CUDA_HOST_COMPILER)
        _metatomic_try_cuda_host_compiler("${CMAKE_CUDA_HOST_COMPILER}" _found)
        if(_found)
            set(${_out_var} "${_found}" PARENT_SCOPE)
            return()
        endif()
    endif()

    foreach(_name IN ITEMS g++-13 gcc-13 g++-12 gcc-12 g++-11 gcc-11)
        find_program(_candidate NAMES ${_name})
        if(_candidate)
            _metatomic_try_cuda_host_compiler("${_candidate}" _found)
            if(_found)
                set(${_out_var} "${_found}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()

    file(GLOB _opt_gcc_dirs /opt/gcc-13* /opt/gcc-12* /opt/gcc-11*)
    list(SORT _opt_gcc_dirs COMPARE NATURAL ORDER DESCENDING)
    foreach(_gcc_dir IN LISTS _opt_gcc_dirs)
        foreach(_candidate IN ITEMS "${_gcc_dir}/build/bin/g++" "${_gcc_dir}/bin/g++")
            _metatomic_try_cuda_host_compiler("${_candidate}" _found)
            if(_found)
                set(${_out_var} "${_found}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endforeach()

    foreach(_candidate IN ITEMS /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11)
        _metatomic_try_cuda_host_compiler("${_candidate}" _found)
        if(_found)
            set(${_out_var} "${_found}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
endfunction()

function(_metatomic_set_cuda_host_compiler_if_needed _cxx_compiler _cxx_major)
    if(NOT _cxx_major GREATER 13)
        return()
    endif()

    if(CMAKE_CUDA_HOST_COMPILER)
        _metatomic_gcc_major_version("${CMAKE_CUDA_HOST_COMPILER}" _cuda_host_major)
        if(_cuda_host_major GREATER 0 AND _cuda_host_major LESS_EQUAL 13)
            message(STATUS "Using CMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER} for nvcc (main CXX is GCC ${_cxx_major})")
            return()
        endif()
    endif()

    _metatomic_find_cuda_host_compiler(_cuda_host)
    if(_cuda_host)
        get_filename_component(_cuda_host "${_cuda_host}" REALPATH)
        set(CMAKE_CUDA_HOST_COMPILER "${_cuda_host}" CACHE FILEPATH "CUDA host compiler" FORCE)
        message(STATUS "Main CXX compiler ${_cxx_compiler} is GCC ${_cxx_major}; using ${_cuda_host} as CMAKE_CUDA_HOST_COMPILER for nvcc")
    else()
        message(FATAL_ERROR
            "The main C++ compiler (${_cxx_compiler}, GCC ${_cxx_major}) cannot be used as nvcc's host compiler with CUDA 12.x.\n"
            "Set METATOMIC_CUDA_HOST_COMPILER or CUDAHOSTCXX to a GCC 13 (or older) g++, "
            "for example: export METATOMIC_CUDA_HOST_COMPILER=/usr/bin/g++-13")
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(REGEX MATCH "^([0-9]+)" _metatomic_cxx_major_ "${CMAKE_CXX_COMPILER_VERSION}")
    _metatomic_set_cuda_host_compiler_if_needed("${CMAKE_CXX_COMPILER}" "${_metatomic_cxx_major_}")
elseif(NOT _metatomic_cuda_host_compiler_checked)
    set(_metatomic_cuda_host_compiler_checked TRUE)

    if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
        set(_metatomic_cxx_compiler "$ENV{CXX}")
    elseif(CMAKE_CXX_COMPILER)
        set(_metatomic_cxx_compiler "${CMAKE_CXX_COMPILER}")
    else()
        set(_metatomic_cxx_compiler "c++")
    endif()

    _metatomic_gcc_major_version("${_metatomic_cxx_compiler}" _metatomic_cxx_major_)
    if(_metatomic_cxx_major_ GREATER 13)
        _metatomic_set_cuda_host_compiler_if_needed("${_metatomic_cxx_compiler}" "${_metatomic_cxx_major_}")
    endif()
endif()
