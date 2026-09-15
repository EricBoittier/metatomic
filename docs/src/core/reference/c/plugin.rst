Plugin system
=============

.. doxygenstruct:: mta_plugin_t
    :members:

The following functions operate on :c:type:`mta_plugin_t`:

- :c:func:`mta_register_plugin`: TODO summary
- :c:func:`mta_load_plugin`: TODO summary

--------------------------------------------------------------------------------

.. doxygenfunction:: mta_register_plugin

.. doxygenfunction:: mta_load_plugin

Lennard-Jones test plugin
-------------------------

Metatomic installations include a shifted Lennard-Jones plugin for testing
simulation-engine integrations. The Python package exposes its absolute path:

.. code-block:: bash

    python -c "import metatomic; print(metatomic.utils.lj_plugin_path())"

Load this path with :c:func:`mta_load_plugin`, then load the model named
``lennard-jones`` from the ``lj-plugin`` plugin. Model options are a JSON object
with string values. The supported options and defaults are:

================= ============
Option            Default
================= ============
``sigma``         ``"1.0"``
``epsilon``       ``"1.0"``
``cutoff``        ``"3.0"``
``atomic_type``   ``"1"``
``length_unit``   ``"Angstrom"``
``energy_unit``   ``"eV"``
================= ============

The model runs on the CPU with ``float64`` data. It supports one system at a
time, the ``energy`` output, and an optional gradient with respect to
positions. It does not support selected-atom calculations.
