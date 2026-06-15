# CUDA 12.x only officially supports GCC up to 13. find_package(metatensor_torch)
# and find_package(Torch) enable the CUDA language, so this must run before either
# call. CMAKE_CUDA_FLAGS_INIT is used when CMake first enables CUDA, including
# during compiler identification.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(REGEX MATCH "^([0-9]+)" _metatomic_gcc_major_ "${CMAKE_CXX_COMPILER_VERSION}")
    if(_metatomic_gcc_major_ GREATER 13)
        if(NOT "${CMAKE_CUDA_FLAGS_INIT}" MATCHES "allow-unsupported-compiler")
            if(CMAKE_CUDA_FLAGS_INIT)
                set(CMAKE_CUDA_FLAGS_INIT "${CMAKE_CUDA_FLAGS_INIT} -allow-unsupported-compiler")
            else()
                set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
            endif()
        endif()
        message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} is newer than CUDA officially supports; adding -allow-unsupported-compiler for nvcc")
    endif()
endif()
