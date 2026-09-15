System
======

.. currentmodule:: metatomic

The :py:class:`System` class stores the atomic types, positions, unit cell, and
periodic boundary conditions passed to a metatomic model. For example, the
following creates a system containing a water molecule in a non-periodic cell:

.. code-block:: python

    import numpy as np
    from metatomic import System

    system = System(
        length_unit="angstrom",
        types=np.array([8, 1, 1], dtype=np.int32),
        positions=np.array(
            [
                [0.000, 0.000, 0.000],
                [0.757, 0.586, 0.000],
                [-0.757, 0.586, 0.000],
            ],
            dtype=np.float64,
        ),
        cell=np.zeros((3, 3), dtype=np.float64),
        pbc=np.array([False, False, False]),
    )

The arrays returned by :py:attr:`System.types`, :py:attr:`System.positions`,
:py:attr:`System.cell`, and :py:attr:`System.pbc` are read-only views. They keep
the underlying data alive even if the original :py:class:`System` is deleted.

Pair lists can be attached to a system using :py:meth:`System.add_pairs`. Each
pair list is identified by a :py:class:`PairListOptions` object. Arbitrary
per-system data stored as a :py:class:`metatensor.TensorMap` can be attached
with :py:meth:`System.add_custom_data`.

.. autoclass:: PairListOptions
   :members:

.. autoclass:: System
   :members:
