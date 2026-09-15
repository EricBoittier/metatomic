import os


_HERE = os.path.dirname(os.path.abspath(__file__))


try:
    from ._external import EXTERNAL_METATOMIC_PREFIX

    cmake_prefix_path = EXTERNAL_METATOMIC_PREFIX
    """
    Path containing the CMake configuration files for the underlying C library
    """

except ImportError:
    cmake_prefix_path = os.path.join(_HERE, "lib", "cmake")
    """
    Path containing the CMake configuration files for the underlying C library
    """


def lj_plugin_path():
    """Absolute path of the shifted Lennard-Jones test plugin.

    The shared library is a C metatomic plugin that simulation engines can load
    in their own test suites (LAMMPS, i-PI, ASE, …)::

        python -c "import metatomic; print(metatomic.utils.lj_plugin_path())"

    Load it with :c:func:`mta_load_plugin`, then
    ``mta_load_model("lennard-jones", options, "lj-plugin")``.
    """
    candidates = [os.path.join(_HERE, "lib", "lj-plugin.so")]
    try:
        from ._external import EXTERNAL_METATOMIC_PREFIX

        candidates.append(
            os.path.join(EXTERNAL_METATOMIC_PREFIX, "lib", "lj-plugin.so")
        )
    except ImportError:
        pass

    for path in candidates:
        if os.path.isfile(path):
            return path

    searched = ", ".join(candidates)
    raise FileNotFoundError(
        f"Lennard-Jones test plugin not found (looked in: {searched}). "
        "Rebuild metatomic-core with METATOMIC_BUILD_LJ_PLUGIN=ON; this is "
        "the default when installing the Python package."
    )
