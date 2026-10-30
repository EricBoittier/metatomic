r"""
.. _jax-tutorial-pet-jax:

A JAX model: PET with ``pet-jax``
=================================

.. py:currentmodule:: metatomic

This tutorial runs `pet-jax <https://github.com/lab-cosmo/pet-jax>`_, a JAX/Flax
implementation of the PET architecture, as a metatomic model. ``pet-jax`` ships a
:py:class:`JaxModel` wrapper, ``petjax.metatomic.UPETModel``, which:

- requests a single full pair list at the model cutoff from the engine;
- runs the adaptive neighbour selection and the PET forward on the pair vectors
  given by the engine, in one :py:func:`jax.jit`-compiled function;
- lets :py:class:`JaxModel` compute the forces and stress with JAX autodiff,
  through these pair vectors.

To keep this example self-contained, we use a small, randomly initialized PET
model. A real checkpoint (e.g. PET-MAD, converted with ``petjax-convert``) is
loaded with ``UPETModel.from_checkpoint(folder)`` instead, everything else stays
the same.
"""

# sphinx_gallery_thumbnail_number = 1

import ase.build
import jax
import matplotlib.pyplot as plt
import numpy as np
from petjax import UPET, to_structure, truncate
from petjax.metatomic import UPETModel, system_from_atoms

from metatomic import Quantity, execute_model


jax.config.update("jax_enable_x64", True)

# %%
#
# Defining the model
# ------------------
#
# The hyper-parameters are the ones stored in a ``pet-jax`` checkpoint, here for a
# much smaller model than the released ones. The parameters are initialized on a
# structure dict built by ``pet-jax`` itself.

config = {
    "cutoff": 4.5,
    "cutoff_width": 0.5,
    "cutoff_width_adaptive": 0.5,
    "adaptive_cutoff_method": "solver",
    "num_neighbors_adaptive": 8,
    "d_pet": 16,
    "d_node": 32,
    "d_head": 16,
    "d_feedforward": 32,
    "num_heads": 2,
    "num_attention_layers": 1,
    "num_gnn_layers": 2,
    "max_atomic_number": 14,
    "attention_temperature": 1.0,
}

atoms = ase.build.bulk("Si", "diamond", a=5.43, cubic=True)
structure = to_structure(atoms, config["cutoff"], skin=0.0)
structure["k_sel_sizer"] = np.zeros(16, dtype=bool)
truncated, _ = truncate(
    structure,
    config["num_neighbors_adaptive"],
    config["cutoff"],
    config["cutoff_width_adaptive"],
    method=config["adaptive_cutoff_method"],
)

upet = UPET(**config)
params = upet.init(jax.random.key(0), **truncated)

# the per-species energy shifts, added to the per-atom energies
metadata = {"config": config, "shifts": {14: -5.0}}
model = UPETModel(upet, params, metadata, dtype="float64")

print(model.capabilities().outputs)
print(model.requested_pair_lists(model.capabilities().outputs))

# %%
#
# Running the model
# -----------------
#
# Simulation engines build a :py:class:`System` with the pair lists requested by
# the model. ``system_from_atoms`` does the same for an :py:class:`ase.Atoms`,
# with `vesin <https://github.com/Luthaf/vesin>`_.

atoms = atoms.repeat(2)
atoms.rattle(0.05, seed=0)
(options,) = model.requested_pair_lists([])
system = system_from_atoms(atoms, options)

energy = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions", "strain"]
)
(result,) = execute_model(model, [system], [energy], check_consistency=True)
block = result.block()
forces = -np.asarray(block.gradient("positions").values).reshape(-1, 3)
print("energy:", block.values[0, 0], "eV")

# %%
#
# The forces come from :py:func:`jax.vjp` through the pair vectors. A finite
# difference of the energy along one coordinate checks them:

delta = 1e-4


def energy_at(shift):
    displaced = atoms.copy()
    displaced.positions[0, 0] += shift
    (out,) = execute_model(
        model,
        [system_from_atoms(displaced, options)],
        [Quantity(name="energy", unit="eV", sample_kind="system")],
    )
    return out.block().values[0, 0]


finite_difference = -(energy_at(delta) - energy_at(-delta)) / (2 * delta)
print("autodiff:", forces[0, 0], "finite difference:", finite_difference)

# %%
#
# Energy and stress under strain
# ------------------------------
#
# The ``strain`` gradient gives the virial. Along a uniaxial strain, its ``xx``
# component divided by the volume is the derivative of the energy per volume.

strains = np.linspace(-0.04, 0.04, 9)
energies, stresses = [], []
for eps in strains:
    strained = atoms.copy()
    strained.set_cell(atoms.cell[:] @ np.diag([1 + eps, 1, 1]), scale_atoms=True)
    (out,) = execute_model(model, [system_from_atoms(strained, options)], [energy])
    energies.append(out.block().values[0, 0])
    virial = np.asarray(out.block().gradient("strain").values).reshape(3, 3)
    stresses.append(virial[0, 0] / strained.get_volume())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), layout="constrained")
ax1.plot(strains, energies, "o-")
ax1.set(xlabel=r"strain $\varepsilon_{xx}$", ylabel="energy / eV")
ax2.plot(strains, stresses, "o-")
ax2.set(xlabel=r"strain $\varepsilon_{xx}$", ylabel=r"$\sigma_{xx}$ / eV Å$^{-3}$")
plt.show()
