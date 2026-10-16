Python backend integration tests
===============================

Run from the parent example directory::

    sh test-backends.sh -q

The suite has 35 tests and runs on CPU in float64. It calls the PLUMED Python
API directly, through the source-built METATOMIC action and pinned core
``b02b9ff3df6458cca21fa8690d229bbda97e6f78``, into NumPy, PyTorch and JAX
models. No mock PLUMED calculation or TorchScript model is used.

This core revision predates the production Python model loader.
``python_bridge.py`` is a test-only adapter registering Python callbacks with
the pinned public C ABI. These tests validate that route, not a production
Python plugin loader. ``backend_models.py`` implements an anisotropic quadratic
CV: NumPy uses analytic gradients, while PyTorch and JAX differentiate positions
and homogeneous strain. Native System arrays reach each backend through the
pinned bindings. TensorMaps return through the C ABI to PLUMED.

Coverage
--------

* CV values, harmonic bias energies, forces and virials for all three backends.
* All nine position and nine strain derivatives checked by finite differences;
  a non-symmetric strain gradient detects index transposition mistakes.
* Angstrom/nanometer conversion and independent MD energy-unit scaling.
* Zero bias and opposing bias-force directions, repeated evaluations, garbage
  collection, model unloading, backend dtype and empty-system probing.
* Python model errors propagated through PLUMED's nested exception chain.
* Exact core revision and shared native library handles.
* The pinned core's known rejection of feature gradients with
  ``CHECK_CONSISTENCY`` enabled. This is an explicit expected-error test;
  successful gradient calculations omit that optional flag and are checked
  independently against analytic results and finite differences.

The small three-atom fixture deliberately isolates API behavior. It is not a
water potential, and does not test neighbor lists or GPU execution.

Environment
-----------

The runner defaults to ``python3`` on PATH; activate the dependency environment first.
Override its Python with ``HEXAMER_PYTHON`` and its native dependency prefix
with ``HEXAMER_DEPS`` (default: the Python environment prefix). ``PLUMED_TEST_KERNEL`` can override the source kernel
path. The current native-library configuration uses macOS ``.dylib`` names.

Build the pinned native libraries first using the example's ``build.sh``.
``requirements.txt`` records the Python versions used for the passing run.
If preparing a fresh environment, install those dependencies and the Python
package metadata from this same pinned checkout (commands from the example
directory)::

    python -m pip install -r tests/requirements.txt
    METATOMIC_CORE_PYTHON_USE_EXTERNAL_LIB=ON \
        CMAKE_PREFIX_PATH="$(cd ../../../install && pwd)" \
        python -m pip install --no-build-isolation ../../../metatomic-pinned/python/metatomic_core

``conftest.py`` selects the checkout's Python bindings and the same external
metatomic/metatensor libraries used by PLUMED. Loading a separate wheel-bundled
metatensor library would create a second registry and invalidate this test.

PLUMED input energies use kJ/mol to avoid conflating API conversion errors with
small differences between ASE and PLUMED's electronvolt conversion constants.
An optional machine-readable report can be generated with::

    sh test-backends.sh -q --junitxml=tests/report.xml
