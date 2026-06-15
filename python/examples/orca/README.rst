ORCA external optimizer example
================================

This directory contains a minimal wrapper that lets `ORCA`_ drive geometry
optimization (and other workflows using the external-tool interface) with a
metatomic machine-learning potential.

The wrapper implements the file protocol documented in `orca-external-tools`_.
On each ORCA step it reads ``*.extinp.tmp`` and the accompanying XYZ geometry,
evaluates energy and gradient with :py:class:`metatomic_ase.MetatomicCalculator`,
and writes ``*.engrad`` back for ORCA.

.. _ORCA: https://www.faccts.de/orca/
.. _orca-external-tools: https://github.com/faccts/orca-external-tools#interface

Prerequisites
-------------

- ORCA 6 or newer (``ProgExt`` / ``Ext_Params`` in the input file)
- Python packages ``metatomic``, ``metatomic-ase``, and their dependencies
- An exported metatomic model (``.pt``), plus an ``extensions/`` directory if the
  model requires compiled extensions

Files
-----

``metatomic-orca-external``
    Standalone script invoked by ORCA via ``%method ProgExt``.

``water_opt/water.xyz``
    Starting water geometry for a test optimization.

``water_opt/water_opt.inp``
    ORCA input template. Edit the absolute paths before running.

Setup
-----

1. Install metatomic and metatomic-ase in the Python environment ORCA will use.

2. Make the wrapper executable (optional if you call it with ``python``)::

       chmod +x /path/to/metatomic/python/examples/orca/metatomic-orca-external

3. Edit ``water_opt/water_opt.inp`` and replace the placeholder paths:

   - ``ProgExt`` must point to ``metatomic-orca-external`` (absolute path)
   - ``Ext_Params`` must pass ``--model`` and, if needed,
     ``--extensions-directory`` (absolute paths)

   Example::

       %method
         ProgExt "/home/user/metatomic/python/examples/orca/metatomic-orca-external"
         Ext_Params "--model /home/user/models/model-md.pt --extensions-directory /home/user/models/extensions"
       end

   You can also set defaults with environment variables ``METATOMIC_MODEL``,
   ``METATOMIC_EXTENSIONS``, and ``METATOMIC_DEVICE``.

Run
---

From the example directory::

    cd water_opt
    orca water_opt.inp > job.out

Expected outputs include:

- ``water_opt.engrad`` — energy and gradient written by the wrapper each step
- ``water_opt.xyz`` — final optimized geometry
- ``water_opt_trj.xyz`` — optimization trajectory (if ORCA writes it)

Standalone test (without ORCA)
------------------------------

You can smoke-test the script directly if ORCA has already created an
``*.extinp.tmp`` file, or craft one following the `interface specification`_.
With a model available::

    ./metatomic-orca-external water_opt_EXT.extinp.tmp \
        --model /path/to/model-md.pt \
        --extensions-directory /path/to/extensions

.. _interface specification: https://github.com/faccts/orca-external-tools#interface

Troubleshooting
---------------

**ORCA cannot find the script**
    Use an absolute path in ``ProgExt``. ORCA's working directory may differ
    from where you launch the job.

**Model or extensions not found**
    Pass absolute paths in ``Ext_Params``, or set ``METATOMIC_MODEL`` and
    ``METATOMIC_EXTENSIONS``.

**Slow optimization**
    Each ORCA call starts a new Python process and reloads the model. For
    production workflows (NEB, GOAT, long optimizations), consider a persistent
    server/client pattern (not included in this minimal example).

**Point charges**
    ORCA point-charge files (``pointcharges.pc``) are not supported in this
    version.

Related
-------

- `metatomic issue #228`_
- `ORCA external optimizer tutorial`_
- `orca-external-tools`_

.. _metatomic issue #228: https://github.com/metatensor/metatomic/issues/228
.. _ORCA external optimizer tutorial: https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html
