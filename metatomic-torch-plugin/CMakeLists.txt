# metatomic-torch plugin for metatomic-core: runs TorchScript `AtomisticModel` files
# (*.pt, exported with metatomic-torch) through the metatomic C API.
cmake_minimum_required(VERSION 3.22)
project(metatomic-torch-plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# One metatensor-core for everything in the process: metatomic-core and
# metatensor-torch must be built against the same one.
find_package(Torch REQUIRED)
find_package(metatensor REQUIRED CONFIG)
find_package(metatensor_torch REQUIRED CONFIG)
find_package(metatomic_torch REQUIRED CONFIG)
find_package(metatomic REQUIRED CONFIG)

add_library(metatomic_torch_plugin SHARED src/torch_plugin.cpp)
target_link_libraries(metatomic_torch_plugin PRIVATE metatomic::shared metatomic_torch metatensor_torch torch)
set_target_properties(metatomic_torch_plugin PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH_USE_LINK_PATH ON
)
install(TARGETS metatomic_torch_plugin LIBRARY DESTINATION lib)
