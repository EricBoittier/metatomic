add_library(lj-plugin SHARED lennard_jones.cpp)
target_link_libraries(lj-plugin PRIVATE metatomic)
target_compile_features(lj-plugin PRIVATE cxx_std_17)

# Same as tests/test-plugins: PREFIX "" + .so on every OS. dlopen/LoadLibrary
# don't care about the suffix, so engines only have to look for one name.
set_target_properties(lj-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)

# plugin lives in libexec/metatomic/; libmetatomic is in lib/ (or lib/<triplet>)
if(APPLE)
    set_target_properties(lj-plugin PROPERTIES
        INSTALL_RPATH "@loader_path/../../${CMAKE_INSTALL_LIBDIR}"
    )
elseif(UNIX)
    set_target_properties(lj-plugin PROPERTIES
        INSTALL_RPATH "$ORIGIN/../../${CMAKE_INSTALL_LIBDIR}"
    )
endif()

if(MINGW)
    # tests/ only sets -static-libstdc++ on executables
    target_link_options(lj-plugin PRIVATE -static-libstdc++)
endif()

# Windows has no rpath, so sit next to metatomic.dll. Unix: libexec + rpath above.
if(WIN32)
    set(_lj_plugin_dest ${CMAKE_INSTALL_BINDIR})
else()
    set(_lj_plugin_dest ${CMAKE_INSTALL_LIBEXECDIR}/metatomic)
endif()

install(
    TARGETS lj-plugin
    LIBRARY DESTINATION ${_lj_plugin_dest}
    RUNTIME DESTINATION ${_lj_plugin_dest}
)
