# Shifted Lennard-Jones C plugin used when developing / testing engines.
#
# Built with the C API tests (`cargo test --package=metatomic-core`) and
# installed with the Python package (`pip install python/metatomic_core`).
# The library is named `lj-plugin.so` on every platform. Engines can discover
# it with ``metatomic.utils.lj_plugin_path()``, then
# ``mta_load_plugin`` / ``mta_load_model("lennard-jones", options, "lj-plugin")``.

# python/metatomic_core is `project(... NONE)`; enable C when we are added
# against an already-installed metatomic (conda / METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB).
enable_language(C)

add_library(lj-plugin SHARED lennard_jones.c plugin.c)
target_link_libraries(lj-plugin metatomic)

if(APPLE)
    set(_lj_plugin_rpath "@loader_path")
else()
    set(_lj_plugin_rpath "$ORIGIN")
endif()

set_target_properties(lj-plugin PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LINKER_LANGUAGE C
    INSTALL_RPATH "${_lj_plugin_rpath}"
)

# Installed next to libmetatomic so ``metatomic.utils.lj_plugin_path()`` can
# find it after ``pip install metatomic-core``.
install(
    TARGETS lj-plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
