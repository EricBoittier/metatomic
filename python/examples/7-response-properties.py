r"""
.. _atomistic-tutorial-response-properties:

Response properties as derivatives of the energy
================================================

.. py:currentmodule:: metatomic.torch

Forces, stress, dipoles, charges and magnetic moments are all first derivatives of one
energy :math:`E(\mathbf{R}, \varepsilon, \mathcal{E}, \varphi, \mathbf{B})` with
respect to something an experiment can push on: the atomic positions, the cell strain,
an electric field, the electrostatic potential at each atom and a magnetic field.
Second derivatives give response tensors (force constants, elastic constants, Born
effective charges, polarizability, ...), and third derivatives give Raman tensors and
hyperpolarizabilities.

This tutorial builds a small, fully analytic model whose energy depends on all five of
these perturbations, exports it as a metatomic model, and computes every first and
second derivative in a single Hessian. The parameters are made up: the model exists to
exercise the derivative machinery, not to describe a real material. This tutorial
assumes you already know how to export a model and use neighbor lists (see
:ref:`atomistic-tutorial-export` and :ref:`atomistic-tutorial-nl-md`).
"""

# sphinx_gallery_thumbnail_number = 1

import math
from typing import Dict, List, Optional

import ase.build
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import torch
import vesin.metatomic
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from metatomic_ase import MetatomicCalculator


DTYPE = torch.float64

# %%
#
# The toy energy
# --------------
#
# The model adds up five terms, each chosen so that a different part of the derivative
# table is non-zero:
#
# - **Morse pairs** between all atoms and a **bond-angle** term give forces, stress,
#   force constants and elastic constants.
# - **Split-charge equilibration** (SQE, `Nistor et al. 2006
#   <https://doi.org/10.1063/1.2346671>`_) gives atomic charges. Charges move along
#   bonds, :math:`q_I = Q/N + \sum_b T_{Ib} p_b`, and the split charges :math:`p_b`
#   minimize
#
#   .. math::
#
#       E_q = \sum_I \left(\chi_I q_I + \tfrac{1}{2} \eta_I q_I^2 \right)
#           + \sum_{I<J} J(r_{IJ}) q_I q_J + \sum_b \tfrac{1}{2} \kappa_b p_b^2
#           + \sum_I q_I \varphi_I - \mathcal{E} \cdot \boldsymbol{\mu}
#
#   with a dipole :math:`\boldsymbol{\mu} = -\sum_b p_b \mathbf{r}_b` built from bond
#   vectors, so it stays well defined in periodic cells. The electronegativity
#   :math:`\chi_I` depends on the environment, and the bond hardness :math:`\kappa_b`
#   drops as bonds get longer and more ionic (as in Harrison's bond-orbital picture).
#   Charges then flow when atoms move, which makes Born effective charges larger than
#   the static charges. Since :math:`E_q` is quadratic, the minimum is a single linear
#   solve and autograd differentiates through it exactly.
# - **Atomic and bond polarizabilities** add :math:`-\frac{1}{2} \alpha_I
#   |\mathcal{E}|^2` per atom and :math:`-\frac{1}{6} \beta_b (\mathcal{E} \cdot
#   \hat{\mathbf{r}}_b)^3` per bond, for a non-zero polarizability and
#   hyperpolarizability.
# - **Exchange** :math:`J_\text{ex}(r_{IJ}) \, \mathbf{s}_I \cdot \mathbf{s}_J` between
#   the input magnetic moments couples magnetic order to the lattice.
# - **Zeeman** :math:`-\mu_B \sum_I \mathbf{m}_I \cdot \mathbf{B}`, where the size of
#   each moment depends on the environment and on the atomic charge,
#   :math:`\mathbf{m}_I = \mathbf{s}_I (1 + c^q_I q_I + c^\text{env}_I n_I)`. The charge
#   dependence couples the magnetic field to the electric field and to the potential.
#
# The perturbations reach the model as positions and cell (strain is applied to both),
# and as inputs requested from the engine: the total ``charge`` (a standard metatomic
# quantity), the electric field and magnetic field from ``atoms.info``, the potential at
# each atom from ``atoms.arrays`` and the moments :math:`\mathbf{s}_I` from the initial
# magnetic moments. Units are eV, Å, e, V/Å for the electric field, V for the potential,
# Tesla for the magnetic field and Bohr magnetons for the moments. metatomic's unit
# parser does not know volts or teslas, so the electric field is declared in the
# equivalent ``eV/(e*Angstrom)`` and the magnetic field without a unit.

MU_B = 5.7883818060e-5  # Bohr magneton, eV/T
MU_B_UNIT = "e * hbar / (2 * m_e)"
COULOMB = 14.399645  # e^2 / (4 pi eps_0), eV Å / e^2

INPUTS = {
    "charge": ModelOutput(unit="e", sample_kind="system"),
    "ase::info::electric_field": ModelOutput(
        unit="eV/(e*Angstrom)", sample_kind="system"
    ),
    "ase::info::magnetic_field": ModelOutput(unit="", sample_kind="system"),
    "ase::arrays::potential": ModelOutput(unit="eV/e", sample_kind="atom"),
    "ase::initial_magmoms": ModelOutput(unit=MU_B_UNIT, sample_kind="atom"),
}


def _table(values: Dict[int, float], index: Dict[int, int]) -> torch.Tensor:
    return torch.tensor([values[t] for t in sorted(index, key=index.get)], dtype=DTYPE)


def _pair_table(
    values: Dict[str, float], index: Dict[int, int], sign: float
) -> torch.Tensor:
    """Pair parameters from ``"A-B"`` keys; ``sign=-1`` for antisymmetric ones."""
    table = torch.zeros(len(index), len(index), dtype=DTYPE)
    for key, value in values.items():
        a, b = (index[int(t)] for t in key.split("-"))
        table[a, b] = value
        table[b, a] = sign * value if a != b else value
    return table


def zero(name: str) -> Labels:
    return Labels([name], torch.tensor([[0]]))


class ToyResponseModel(torch.nn.Module):
    """Analytic energy of positions, strain, electric field, potential and B field."""

    _requested_inputs: Dict[str, ModelOutput]

    def __init__(
        self, atom: Dict[str, Dict[int, float]], pair: Dict[str, Dict], hypers
    ):
        super().__init__()
        types = sorted(atom["chi"])
        index = {t: n for n, t in enumerate(types)}
        lookup = torch.full((max(types) + 1,), -1, dtype=torch.long)
        lookup[types] = torch.arange(len(types))
        self.register_buffer("type_index", lookup)

        for name, values in atom.items():
            self.register_buffer(name, _table(values, index))
        for name, values in pair.items():
            # bond hyperpolarizabilities flip sign with the bond direction
            sign = -1.0 if name == "beta" else 1.0
            self.register_buffer(name, _pair_table(values, index, sign))

        self.r_on = float(hypers["r_on"])
        self.r_cut = float(hypers["r_cut"])
        self.r_bond = float(hypers["r_bond"])
        self.env_length = float(hypers["env_length"])
        self.screening = float(hypers["screening"])
        self.coulomb_width = float(hypers["coulomb_width"])
        self.exchange_length = float(hypers["exchange_length"])
        self.kappa_length = float(hypers["kappa_length"])
        self.nl_options = NeighborListOptions(
            cutoff=self.r_cut, full_list=False, strict=True
        )
        self._requested_inputs = INPUTS
        self.mu_b = MU_B
        self.coulomb = COULOMB

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.nl_options]

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        return self._requested_inputs

    def taper(self, r: torch.Tensor) -> torch.Tensor:
        x = ((r - self.r_on) / (self.r_cut - self.r_on)).clamp(0.0, 1.0)
        return 0.5 * (1.0 + torch.cos(math.pi * x))

    def energy_and_charges(self, system: System):
        n_atoms = len(system)
        species = self.type_index[system.types]
        positions = system.positions

        # pair vectors are recomputed from positions and cell so that autograd can take
        # derivatives of any order with respect to both
        nl = system.get_neighbor_list(self.nl_options)
        i = nl.samples.column("first_atom")
        j = nl.samples.column("second_atom")
        shifts = nl.samples.values[:, 2:5].to(positions.dtype)
        vectors = positions[j] - positions[i] + shifts @ system.cell
        r = torch.linalg.vector_norm(vectors, dim=1)
        cut = self.taper(r)
        si, sj = species[i], species[j]

        env_pair = torch.exp(-r / self.env_length) * cut
        env = (
            positions.new_zeros(n_atoms)
            .index_add(0, i, env_pair)
            .index_add(0, j, env_pair)
        )

        # Morse pairs
        stretch = 1.0 - torch.exp(-self.morse_a[si, sj] * (r - self.morse_r0[si, sj]))
        energy = torch.sum(self.morse_d[si, sj] * (stretch**2 - 1.0) * cut)

        # Keating-style bond angles keep the tetrahedra tetrahedral
        bonds = torch.nonzero(r.detach() < self.r_bond).reshape(-1)
        bond_vectors = vectors[bonds]
        centers = torch.cat([i[bonds], j[bonds]])
        arms = torch.cat([bond_vectors, -bond_vectors])
        arms = arms / torch.linalg.vector_norm(arms, dim=1, keepdim=True)
        angle_pairs = torch.triu_indices(len(arms), len(arms), 1, device=arms.device)
        first, second = angle_pairs[0], angle_pairs[1]
        same = centers[first] == centers[second]
        first, second = first[same], second[same]
        cosine = (arms[first] * arms[second]).sum(dim=1)
        energy = energy + torch.sum(
            self.angle[species[centers[first]]] * (cosine + 1.0 / 3.0) ** 2
        )

        # inputs
        total_charge = system.get_data("charge").block().values.sum()
        field = system.get_data("ase::info::electric_field").block().values.reshape(3)
        b_field = system.get_data("ase::info::magnetic_field").block().values.reshape(3)
        potential = system.get_data("ase::arrays::potential").block().values
        potential = potential.reshape(n_atoms)
        magmoms = system.get_data("ase::initial_magmoms").block().values
        if magmoms.shape[1] == 3:
            spins = magmoms.reshape(n_atoms, 3)
        else:
            spins = positions.new_zeros(n_atoms, 3)
            spins[:, 2] = magmoms.reshape(n_atoms)
        zeeman = self.mu_b * (spins @ b_field)

        # split-charge equilibration, with split charges on bonds shorter than r_bond
        coulomb = (
            self.screening
            * self.coulomb
            * torch.erf(r / (math.sqrt(2.0) * self.coulomb_width))
            / r
            * torch.exp(-r / self.r_cut)
            * cut
        )
        hardness = (
            torch.diag(self.eta[species])
            .reshape(-1)
            .index_add(0, i * n_atoms + j, coulomb)
            .index_add(0, j * n_atoms + i, coulomb)
            .reshape(n_atoms, n_atoms)
        )
        n_bonds = bonds.shape[0]
        b_range = torch.arange(n_bonds, device=positions.device)
        ones = positions.new_ones(n_bonds)
        transfer = (
            positions.new_zeros(n_atoms * n_bonds)
            .index_add(0, i[bonds] * n_bonds + b_range, ones)
            .index_add(0, j[bonds] * n_bonds + b_range, -ones)
            .reshape(n_atoms, n_bonds)
        )
        bi, bj = si[bonds], sj[bonds]
        # longer bonds are more ionic, so charge moves across them more easily
        stretch_b = (r[bonds] - self.morse_r0[bi, bj]) / self.kappa_length
        bond_hardness = self.kappa[bi, bj] * torch.exp(-stretch_b)

        chi = self.chi[species] + self.chi_env[species] * env
        chi = chi + potential - self.moment_q[species] * zeeman
        base = (total_charge / n_atoms) * positions.new_ones(n_atoms)
        lhs = transfer.T @ hardness @ transfer + torch.diag(bond_hardness)
        rhs = -(transfer.T @ (chi + hardness @ base) + bond_vectors @ field)
        split = torch.linalg.solve(lhs, rhs)
        charges = base + transfer @ split

        energy = energy + chi @ charges + 0.5 * charges @ hardness @ charges
        energy = energy + 0.5 * torch.sum(bond_hardness * split**2)
        energy = energy + (bond_vectors @ field) @ split - base @ (positions @ field)

        # atomic and bond polarizabilities
        alpha = self.alpha[species] * (1.0 + self.alpha_env[species] * env)
        energy = energy - 0.5 * (field @ field) * alpha.sum()
        bond_r = r[bonds].unsqueeze(1)
        along = (bond_vectors / bond_r) @ field
        energy = energy - torch.sum(self.beta[bi, bj] * along**3) / 6.0

        # exchange and Zeeman
        exchange = self.exchange[si, sj] * torch.exp(
            -(r - self.morse_r0[si, sj]) / self.exchange_length
        )
        energy = energy + torch.sum(exchange * cut * (spins[i] * spins[j]).sum(dim=1))
        energy = energy - torch.sum(zeeman * (1.0 + self.moment_env[species] * env))

        return energy, charges

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("selected_atoms is not implemented")

        energies: List[torch.Tensor] = []
        charges: List[torch.Tensor] = []
        for system in systems:
            energy, q = self.energy_and_charges(system)
            energies.append(energy)
            charges.append(q)

        results: Dict[str, TensorMap] = {}
        if "energy" in outputs:
            if outputs["energy"].sample_kind != "system":
                raise NotImplementedError("per-atom energy is not implemented")
            block = TensorBlock(
                values=torch.stack(energies).reshape(-1, 1),
                samples=Labels(["system"], torch.arange(len(systems)).reshape(-1, 1)),
                components=torch.jit.annotate(List[Labels], []),
                properties=zero("energy"),
            )
            results["energy"] = TensorMap(zero("_"), [block])

        if "charge" in outputs:
            system_index = torch.cat(
                [
                    torch.full((len(q),), s, dtype=torch.int32)
                    for s, q in enumerate(charges)
                ]
            )
            atom_index = torch.cat(
                [torch.arange(len(q), dtype=torch.int32) for q in charges]
            )
            block = TensorBlock(
                values=torch.cat(charges).reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"], torch.stack([system_index, atom_index], dim=1)
                ),
                components=torch.jit.annotate(List[Labels], []),
                properties=zero("charge"),
            )
            results["charge"] = TensorMap(zero("_"), [block])

        return results


# %%
#
# The material
# ------------
#
# We use a wurtzite crystal with four atoms per cell, labelled Mn and S after
# γ-MnS. Wurtzite is polar: it has no inversion centre and a unique axis, so it carries
# a spontaneous dipole, is piezoelectric, and a field along the axis moves charge
# between atoms. In a cubic crystal such as zinc blende, symmetry would zero out several
# of the mixed charge and field cells. A ferromagnetic moment of 5 µB on each Mn
# switches on the magnetic-field row.

S, MN = 16, 25
model = ToyResponseModel(
    atom={
        "chi": {S: 6.5, MN: 2.5},  # eV/e
        "eta": {S: 9.0, MN: 8.0},  # eV/e^2
        "chi_env": {S: 0.4, MN: -0.4},  # eV/e
        "alpha": {S: 0.25, MN: 0.30},  # e^2 Å^2/eV
        "alpha_env": {S: 0.05, MN: 0.05},
        "angle": {S: 1.0, MN: 1.0},  # eV
        "moment_q": {S: 0.0, MN: -0.3},  # 1/e
        "moment_env": {S: 0.0, MN: -0.05},
    },
    pair={
        "morse_d": {"16-25": 1.2, "16-16": 0.05, "25-25": 0.05},  # eV
        "morse_a": {"16-25": 1.4, "16-16": 1.2, "25-25": 1.2},  # 1/Å
        "morse_r0": {"16-25": 2.43, "16-16": 3.96, "25-25": 3.96},  # Å
        "kappa": {"16-25": 6.0},  # eV/e^2
        "beta": {"25-16": 0.2},  # e^3 Å^3/eV^2, for the Mn -> S direction
        "exchange": {"25-25": 0.002},  # eV/µB^2
    },
    hypers={
        "r_on": 4.2,
        "r_cut": 4.5,
        "r_bond": 3.0,
        "env_length": 1.5,
        "screening": 0.3,
        "coulomb_width": 1.0,
        "exchange_length": 0.5,
        "kappa_length": 0.3,
    },
)

capabilities = ModelCapabilities(
    outputs={
        "energy": ModelOutput(unit="eV", sample_kind="system"),
        "charge": ModelOutput(unit="e", sample_kind="atom"),
    },
    atomic_types=[S, MN],
    interaction_range=4.5,
    length_unit="Angstrom",
    supported_devices=["cpu"],
    dtype="float64",
)
atomistic = AtomisticModel(model.eval(), ModelMetadata(), capabilities)

initial = ase.build.bulk("MnS", "wurtzite", a=3.98, covera=1.62)
initial.set_initial_magnetic_moments([5.0 if z == MN else 0.0 for z in initial.numbers])

# %%
#
# All perturbations as one vector
# -------------------------------
#
# To get every derivative at once, we put all perturbations in a single vector
# :math:`x = (\mathbf{u}, \gamma, \mathcal{E}, \varphi, \mathbf{B})`: atomic
# displacements, the six Voigt strains (engineering shear), the electric field, the
# potential on each atom and the magnetic field. ``energy(x)`` builds a
# :py:class:`System` from :math:`x` and runs the exported model on it.
#
# A simulation engine would normally compute the neighbor list and the inputs. Here we
# compute the neighbor list once with ``vesin`` and rebuild its distance vectors from
# the strained positions, the same way the model does.

n_atoms = len(initial)
VARIABLES = {
    "R": 3 * n_atoms,
    "ε": 6,
    "ℰ": 3,
    "φ": n_atoms,
    "B": 3,
}
SLICES = {}
start = 0
for name, size in VARIABLES.items():
    SLICES[name] = slice(start, start + size)
    start += size


def voigt_to_strain(gamma):
    xx, yy, zz, yz, xz, xy = gamma
    return torch.stack(
        [
            torch.stack([xx, xy / 2, xz / 2]),
            torch.stack([xy / 2, yy, yz / 2]),
            torch.stack([xz / 2, yz / 2, zz]),
        ]
    )


def make_input(values, per_atom, unit, name):
    """Wrap ``values`` the way the ASE calculator wraps model inputs."""
    values = values.reshape(len(values) if per_atom else 1, -1)
    n = values.shape[0]
    if per_atom:
        index = torch.arange(n, dtype=torch.int32)
        samples = Labels(["system", "atom"], torch.stack([0 * index, index], dim=1))
    else:
        samples = Labels(["system"], torch.tensor([[0]], dtype=torch.int32))
    vector = values.shape[1] == 3
    block = TensorBlock(
        values=values.reshape(n, 3, 1) if vector else values,
        samples=samples,
        components=[Labels.range("xyz", 3)] if vector else [],
        properties=Labels.range(name, 1),
    )
    tensor = TensorMap(Labels.range("_", 1), [block])
    tensor.set_info("unit", unit)
    return tensor


def neighbor_samples(atoms):
    """Pairs, computed once for the undeformed structure."""
    system = System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32),
        positions=torch.tensor(atoms.positions),
        cell=torch.tensor(atoms.cell.array),
        pbc=torch.tensor(atoms.pbc),
    )
    vesin.metatomic.compute_requested_neighbors(
        system, system_length_unit="angstrom", model=atomistic
    )
    return system.get_neighbor_list(model.nl_options).samples


def make_system(atoms, x, samples):
    deform = torch.eye(3, dtype=DTYPE) + voigt_to_strain(x[SLICES["ε"]])
    reference = torch.tensor(atoms.positions)
    positions = (reference + x[SLICES["R"]].reshape(-1, 3)) @ deform
    cell = torch.tensor(atoms.cell.array) @ deform
    system = System(
        types=torch.tensor(atoms.numbers, dtype=torch.int32),
        positions=positions,
        cell=cell,
        pbc=torch.tensor(atoms.pbc),
    )

    i = samples.column("first_atom")
    j = samples.column("second_atom")
    shifts = samples.values[:, 2:5].to(positions.dtype)
    vectors = positions[j] - positions[i] + shifts @ cell
    system.add_neighbor_list(
        model.nl_options,
        TensorBlock(
            values=vectors.reshape(-1, 3, 1),
            samples=samples,
            components=[Labels.range("xyz", 3)],
            properties=Labels.range("distance", 1),
        ),
    )

    magmoms = torch.tensor(atoms.get_initial_magnetic_moments())
    charge = torch.tensor([float(atoms.info.get("charge", 0.0))], dtype=DTYPE)
    for name, values, per_atom, unit, prop in [
        ("charge", charge, False, "e", "charge"),
        ("ase::info::electric_field", x[SLICES["ℰ"]], False, "", "_"),
        ("ase::info::magnetic_field", x[SLICES["B"]], False, "", "_"),
        ("ase::arrays::potential", x[SLICES["φ"]], True, "", "_"),
        ("ase::initial_magmoms", magmoms, True, MU_B_UNIT, "magnetic_moment"),
    ]:
        system.add_data(name, make_input(values, per_atom, unit, prop))
    return system


evaluation = ModelEvaluationOptions(
    length_unit="Angstrom",
    outputs={"energy": ModelOutput(unit="eV", sample_kind="system")},
)


def energy(x, atoms):
    system = make_system(atoms, x, neighbor_samples(atoms))
    outputs = atomistic([system], evaluation, check_consistency=True)
    return outputs["energy"].block().values.sum()


x0 = torch.zeros(sum(VARIABLES.values()), dtype=DTYPE)


def gradient(x, atoms):
    x = x.detach().requires_grad_()
    return torch.autograd.grad(energy(x, atoms), x)[0]


def hessian(atoms):
    return torch.autograd.functional.hessian(lambda x: energy(x, atoms), x0)


# %%
#
# Relaxing the structure
# ----------------------
#
# Derivatives are easiest to read at a minimum, where forces and stress vanish. We
# relax the positions and the cell together with BFGS, using the force and stress parts
# of the gradient.

MECHANICAL = torch.cat(
    [torch.arange(SLICES[n].start, SLICES[n].stop) for n in ("R", "ε")]
)


def objective(values):
    x = x0.clone()
    x[MECHANICAL] = torch.tensor(values)
    x.requires_grad_()
    e = energy(x, initial)
    return e.item(), torch.autograd.grad(e, x)[0][MECHANICAL].numpy()


result = scipy.optimize.minimize(
    objective,
    np.zeros(len(MECHANICAL)),
    jac=True,
    method="BFGS",
    options={"gtol": 1e-9},
)
x_min = x0.clone()
x_min[MECHANICAL] = torch.tensor(result.x)
deform = (torch.eye(3, dtype=DTYPE) + voigt_to_strain(x_min[SLICES["ε"]])).numpy()

atoms = initial.copy()
atoms.positions = (initial.positions + result.x[: 3 * n_atoms].reshape(-1, 3)) @ deform
atoms.cell = initial.cell.array @ deform
a, _, c = atoms.cell.cellpar()[:3]
print(f"a = {a:.3f} Å, c = {c:.3f} Å, c/a = {c / a:.3f}")

# %%
#
# Every second derivative from one Hessian
# ----------------------------------------
#
# With all perturbations in one vector, the gradient holds every first derivative and
# the Hessian every second derivative. Each block of the Hessian is one cell of the
# table. To compare blocks with very different units, we plot the dimensionless
# coupling :math:`H_{ij} / \sqrt{|H_{ii} H_{jj}|}`.

g = gradient(x0, atoms)
H = hessian(atoms)

diagonal = H.diagonal().abs()
scale = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal)).sqrt()
coupling = (H / scale[:, None] / scale[None, :]).numpy()

fig, ax = plt.subplots(figsize=(6.4, 5.6))
image = ax.imshow(coupling, cmap="RdBu_r", vmin=-1, vmax=1)
edges = np.cumsum(list(VARIABLES.values()))[:-1] - 0.5
centers = [(SLICES[n].start + SLICES[n].stop - 1) / 2 for n in VARIABLES]
for edge in edges:
    ax.axhline(edge, color="0.3", linewidth=0.8)
    ax.axvline(edge, color="0.3", linewidth=0.8)
ax.set_xticks(centers, list(VARIABLES))
ax.set_yticks(centers, list(VARIABLES))
ax.tick_params(length=0)
fig.colorbar(image, ax=ax, label=r"$H_{ij} / \sqrt{|H_{ii} H_{jj}|}$")
ax.set_title("Second derivatives of the toy energy")
fig.tight_layout()
plt.show()

# %%
#
# Reading off the table
# ---------------------
#
# The blocks become physical quantities once we fix signs and units. Elastic and
# piezoelectric constants come in two flavours: *clamped-ion* values are the bare
# blocks, *relaxed-ion* values let the atoms relax under strain through the internal
# strain :math:`\Lambda` and the force constants :math:`\Phi`.

TO_GPA = 160.21766  # eV/Å^3
TO_C_PER_M2 = 16.021766  # e/Å^2
TO_CM = 521.4709  # sqrt(eV / (Å^2 u)) to cm^-1

volume = atoms.get_volume()
symbols = atoms.get_chemical_symbols()


def block(row, col):
    return H[SLICES[row], SLICES[col]]


forces = -g[SLICES["R"]]
stress = g[SLICES["ε"]] / volume
dipole = -g[SLICES["ℰ"]]
charges = g[SLICES["φ"]]
moment = -g[SLICES["B"]] / MU_B

force_constants = block("R", "R")
internal_strain = -block("R", "ε")
dipole_derivative = -block("ℰ", "R")
born = dipole_derivative.reshape(3, n_atoms, 3).transpose(0, 1)  # Z*_I[field, disp]
polarizability = -block("ℰ", "ℰ")
epsilon_inf = 1 + 4 * math.pi * COULOMB * polarizability / volume

pinv = torch.linalg.pinv(force_constants, rtol=1e-8)
elastic = block("ε", "ε") / volume
elastic_relaxed = elastic - internal_strain.T @ pinv @ internal_strain / volume
piezo = -block("ℰ", "ε") / volume
piezo_relaxed = piezo + dipole_derivative @ pinv @ internal_strain / volume


def fmt(values, digits=3):
    return np.array2string(
        np.asarray(values), precision=digits, suppress_small=True, separator=", "
    )


print("First derivatives")
print(f"  force       max |F| = {forces.abs().max():.1e} eV/Å")
print(f"  stress      max |σ| = {stress.abs().max() * TO_GPA:.1e} GPa")
print(f"  dipole      μ = {fmt(dipole)} e·Å, spontaneous and along c")
print(f"  charge      q = {fmt(charges)} e")
print(f"  moment      m = {fmt(moment)} µB")
print()
print("Second derivatives")
print(f"  Born charge         Z*({symbols[0]}) diagonal = {fmt(born[0].diagonal())} e")
print(f"  static charge       q({symbols[0]}) = {charges[0]:.3f} e")
print(f"  polarizability      ε∞ diagonal = {fmt(epsilon_inf.diagonal())}")
C = elastic_relaxed * TO_GPA
print(
    f"  elastic, relaxed    C11 = {C[0, 0]:.1f}, C12 = {C[0, 1]:.1f},"
    f" C13 = {C[0, 2]:.1f}, C33 = {C[2, 2]:.1f}, C44 = {C[3, 3]:.1f} GPa"
)
print(f"  elastic, clamped    C44 = {elastic[3, 3] * TO_GPA:.1f} GPa")
for name, e in [("clamped", piezo), ("relaxed", piezo_relaxed)]:
    e = e * TO_C_PER_M2
    print(
        f"  piezo, {name}     e31 = {e[2, 0]:.3f}, e33 = {e[2, 2]:.3f},"
        f" e15 = {e[0, 4]:.3f} C/m²"
    )
print(f"  charge flux         max |∂q/∂R| = {block('φ', 'R').abs().max():.3f} e/Å")
print(f"  charge-field        ∂q/∂ℰz = {fmt(block('φ', 'ℰ')[:, 2])} e·Å/V")
print(f"  response kernel     χ diagonal = {fmt(block('φ', 'φ').diagonal())} e/V")
print(
    f"  spin-lattice        max |∂m/∂R| = {block('B', 'R').abs().max() / MU_B:.4f} µB/Å"
)
print(f"  magnetostriction    ∂mz/∂ε = {fmt(-block('B', 'ε')[2] / MU_B)} µB")
print(f"  magnetoelectric     ∂mz/∂ℰz = {-block('B', 'ℰ')[2, 2] / MU_B:.4f} µB·Å/V")
print(f"  magnetizability     ξzz = {-block('B', 'B')[2, 2] / MU_B:.2e} µB/T")

# %%
#
# Checking the table
# ------------------
#
# Every off-diagonal block can be computed two ways, since mixed derivatives commute.
# For the Born charges, ``block("R", "ℰ")`` differentiates the forces by the field and
# ``block("ℰ", "R")`` differentiates the dipole by the positions; autograd computes them
# in separate backward passes. The sum rules follow from invariances: translations
# (forces and Born charges) and charge conservation (the charge response kernel). The
# charges the model outputs must also match :math:`\partial E / \partial \varphi`.
# Finally, central finite differences of the gradient check the whole Hessian.

mirror = (H - H.T).abs().max()
print(f"mirror readings:  max |∂F/∂ℰ - ∂μ/∂R| etc. = {mirror:.1e}")

translations = force_constants.reshape(n_atoms, 3, n_atoms, 3).sum(dim=2)
print(f"sum rules:        Σ_J Φ_IJ = {translations.abs().max():.1e},", end=" ")
print(f"Σ_I Z*_I = {born.sum(dim=0).abs().max():.1e},", end=" ")
print(f"Σ_I χ_IJ = {block('φ', 'φ').sum(dim=0).abs().max():.1e} ")
print(f"χ eigenvalues:    {fmt(torch.linalg.eigvalsh(block('φ', 'φ')))} (≤ 0)")

output = atomistic(
    [make_system(atoms, x0, neighbor_samples(atoms))],
    ModelEvaluationOptions(
        length_unit="Angstrom", outputs={"charge": ModelOutput(sample_kind="atom")}
    ),
    check_consistency=True,
)
model_charges = output["charge"].block().values.reshape(-1)
print(
    f"charge output:    max |q - ∂E/∂φ| = {(model_charges - charges).abs().max():.1e}"
)

steps = torch.full_like(x0, 1e-5)
steps[SLICES["B"]] = 1.0  # Tesla; magnetic energies are small
finite = torch.stack(
    [
        (gradient(x0 + h * e, atoms) - gradient(x0 - h * e, atoms)) / (2 * h)
        for h, e in zip(steps, torch.eye(len(x0), dtype=DTYPE), strict=True)
    ],
    dim=1,
)
error = (finite - H).abs().max(dim=0).values / H.abs().max(dim=0).values.clamp(
    min=1e-30
)
print(f"finite differences: worst relative error per column = {error.max():.1e}")

# %%
#
# Third derivatives: Raman and hyperpolarizability
# ------------------------------------------------
#
# Differentiating the polarizability once more gives the Raman tensor
# :math:`\partial \alpha / \partial \mathbf{R}` and the first hyperpolarizability
# :math:`\beta = \partial \alpha / \partial \mathcal{E}`. Projected on the phonons,
# the Born charges give infrared intensities and the Raman tensor gives Raman
# activities, so both spectra come out of the same energy.

x = x0.clone().requires_grad_()
first = torch.autograd.grad(energy(x, atoms), x, create_graph=True)[0]
field = SLICES["ℰ"]
alpha = -torch.stack(
    [
        torch.autograd.grad(first[field][k], x, create_graph=True)[0][field]
        for k in range(3)
    ]
)
third = torch.stack(
    [torch.autograd.grad(a, x, retain_graph=True)[0] for a in alpha.reshape(-1)]
).reshape(3, 3, -1)
raman = third[:, :, SLICES["R"]]
beta = third[:, :, SLICES["ℰ"]]

masses = torch.tensor(atoms.get_masses(), dtype=DTYPE).repeat_interleave(3)


def phonons(force_constants):
    """Γ-point frequencies in cm⁻¹ and mass-scaled displacement patterns."""
    dynamical = force_constants / torch.sqrt(masses[:, None] * masses[None, :])
    eigenvalues, modes = torch.linalg.eigh(dynamical)
    frequencies = TO_CM * torch.sign(eigenvalues) * eigenvalues.abs().sqrt()
    return frequencies, modes / torch.sqrt(masses)[:, None]


frequencies, displacements = phonons(force_constants)
print(" mode   ω (cm⁻¹)   IR (e²/u)   Raman (Å⁴/u)")
for m in range(3, 3 * n_atoms):
    ir = (dipole_derivative @ displacements[:, m]).square().sum()
    activity = (raman @ displacements[:, m] * COULOMB).square().sum()
    print(f"{m - 2:5d} {frequencies[m]:10.1f} {ir:11.4f} {activity:13.4f}")

axes = "xyz"
print("non-zero β components (e³Å³/eV²):")
for index in torch.nonzero(beta.abs() > 1e-6 * beta.abs().max()).tolist():
    index = tuple(index)
    print(f"  β_{''.join(axes[k] for k in index)} = {beta[index]:+.4f}")

# %%
#
# The selection rules of wurtzite come out without being imposed: two silent
# :math:`B_1` modes, Raman-only :math:`E_2` modes, and :math:`A_1` and :math:`E_1` modes
# active in both spectra. Likewise, point group 6mm only allows :math:`\beta_{zzz}` and
# components with a single :math:`z`, with :math:`\beta_{zxx} = \beta_{zyy}`.
#
# Switching off magnetism
# -----------------------
#
# Without magnetic moments, every derivative with respect to :math:`\mathbf{B}`
# vanishes: these are the faded cells of the table for closed-shell, non-magnetic
# systems. The exchange term also disappears, which shifts the phonons
# (spin–phonon coupling).

nonmagnetic = atoms.copy()
nonmagnetic.set_initial_magnetic_moments(np.zeros(n_atoms))
H_nonmagnetic = hessian(nonmagnetic)
print(f"magnetic row, ferromagnet:  max |∂²E/∂B∂x| = {H[SLICES['B']].abs().max():.1e}")
print(
    "magnetic row, non-magnetic: max |∂²E/∂B∂x| = "
    f"{H_nonmagnetic[SLICES['B']].abs().max():.1e}"
)
shift = phonons(H_nonmagnetic[SLICES["R"], SLICES["R"]])[0] - frequencies
print(f"optical phonons shift by up to {shift[3:].abs().max():.1f} cm⁻¹")

# %%
#
# Running the exported model in ASE
# ---------------------------------
#
# The model is a regular metatomic model, so any engine that can provide its inputs can
# run it. With the ASE calculator, the fields and potential go in ``atoms.info`` and
# ``atoms.arrays``. In a small electric field, the forces should match the Born charges:
# :math:`F_{I\beta} = \sum_\alpha Z^*_{I,\alpha\beta} \mathcal{E}_\alpha`.

atomistic.save("toy-response-model.pt")

in_field = atoms.copy()
in_field.info["electric_field"] = [0.0, 0.0, 1e-3]  # V/Å
in_field.info["magnetic_field"] = [0.0, 0.0, 0.0]  # T
in_field.arrays["potential"] = np.zeros(n_atoms)  # V
in_field.calc = MetatomicCalculator("toy-response-model.pt", check_consistency=True)

applied = torch.tensor(in_field.info["electric_field"], dtype=DTYPE)
predicted = torch.einsum("iab,a->ib", born, applied).numpy()
computed = in_field.get_forces()
for symbol, f, p in zip(symbols, computed, predicted, strict=True):
    print(f"{symbol:2s}  ASE: {fmt(f, 6)}   Z*·ℰ: {fmt(p, 6)}  eV/Å")
