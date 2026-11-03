"""Tests for CoreModel, running metatomic-core models (BaseModel) in TorchSim.

Uses the numpy ``metatomic.testing.LennardJonesModel``, and the native
``metatomic-lj-plugin`` when it is installed.
"""

import ase.build
import numpy as np
import pytest
import torch
import torch_sim as ts

import metatomic
from metatomic import Quantity
from metatomic.testing import LennardJonesModel
from metatomic_torchsim import CoreModel, MetatomicModel


DTYPE = torch.float64
CPU = torch.device("cpu")
LJ = dict(atomic_type=28, cutoff=5.0, sigma=1.5808, epsilon=0.1729)


@pytest.fixture
def lj():
    return LennardJonesModel(**LJ, length_unit="angstrom", energy_unit="eV")


def nickel(seed=0, size=2):
    atoms = ase.build.bulk("Ni", "fcc", a=3.6, cubic=True).repeat(size)
    atoms.positions += 0.05 * np.random.default_rng(seed).standard_normal(
        atoms.positions.shape
    )
    return atoms


def run(model, atoms):
    return model(ts.initialize_state(atoms, CPU, DTYPE))


def energy(model, atoms):
    return float(run(model, atoms)["energy"].sum())


def test_dispatch(lj):
    model = MetatomicModel(lj, uncertainty_threshold=None)
    assert isinstance(model, CoreModel)
    assert model.model() is lj
    assert model.dtype == torch.float64 and model.device == CPU
    assert model.compute_forces and model.compute_stress


def test_forces_and_stress_match_finite_differences(lj):
    model = CoreModel(lj, uncertainty_threshold=None)
    atoms = nickel()
    out = run(model, atoms)
    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (len(atoms), 3)
    assert out["stress"].shape == (1, 3, 3)

    h = 1e-5
    for atom, axis in [(0, 0), (3, 2)]:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom, axis] += h
        minus.positions[atom, axis] -= h
        fd = -(energy(model, plus) - energy(model, minus)) / (2 * h)
        assert abs(float(out["forces"][atom, axis]) - fd) < 1e-6

    strained = []
    for s in (h, -h):
        strain = np.eye(3)
        strain[0, 1] = strain[1, 0] = s / 2
        moved = atoms.copy()
        moved.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        strained.append(energy(model, moved))
    fd = (strained[0] - strained[1]) / (2 * h) / atoms.cell.volume
    assert abs(float(out["stress"][0, 0, 1]) - fd) < 1e-6


def test_batched_matches_separate(lj):
    model = CoreModel(lj, uncertainty_threshold=None)
    systems = [nickel(0), nickel(1, size=1), nickel(2)]
    batched = model(ts.initialize_state(systems, CPU, DTYPE))
    separate = [run(model, atoms) for atoms in systems]
    for key in ("energy", "stress"):
        torch.testing.assert_close(batched[key], torch.cat([s[key] for s in separate]))
    torch.testing.assert_close(
        batched["forces"], torch.cat([s["forces"] for s in separate])
    )


def test_variants_and_non_conservative(lj):
    atoms = nickel()
    base = run(CoreModel(lj, uncertainty_threshold=None), atoms)
    doubled = run(
        CoreModel(lj, uncertainty_threshold=None, variants={"energy": "doubled"}), atoms
    )
    torch.testing.assert_close(doubled["energy"], 2 * base["energy"])
    torch.testing.assert_close(doubled["forces"], 2 * base["forces"])

    nc = run(CoreModel(lj, uncertainty_threshold=None, non_conservative=True), atoms)
    assert nc["forces"].shape == base["forces"].shape
    torch.testing.assert_close(nc["forces"].sum(0), torch.zeros(3, dtype=DTYPE))
    assert nc["stress"].shape == (1, 3, 3)


def test_uncertainty_warning(lj):
    model, atoms = CoreModel(lj), nickel()
    with pytest.warns(UserWarning, match="atomic energy uncertainties are larger"):
        run(model, atoms)


def test_non_periodic_stress(lj):
    molecule = ase.Atoms("Ni2", positions=[[0, 0, 0], [2.2, 0, 0]])
    out = run(CoreModel(lj, uncertainty_threshold=None), molecule)
    assert "stress" not in out and out["forces"].shape == (2, 3)
    with pytest.raises(ValueError, match="stress requires periodic"):
        run(CoreModel(lj, uncertainty_threshold=None, compute_stress=True), molecule)


def test_extra_outputs(lj):
    model = CoreModel(
        lj,
        uncertainty_threshold=None,
        additional_outputs={
            "ensemble": Quantity(
                name="energy_ensemble", unit="eV", sample_kind="system"
            )
        },
    )
    run(model, nickel())
    assert set(model.additional_outputs) == {"ensemble"}


def test_nve_conserves_energy(lj):
    model = CoreModel(lj, uncertainty_threshold=None)
    state = ts.initialize_state(nickel(), CPU, DTYPE)
    state.rng = 0
    state = ts.nve_init(state, model, kT=torch.tensor(0.01, dtype=DTYPE))
    dt = torch.tensor(1e-3 * ts.units.MetalUnits.time, dtype=DTYPE)  # 1 fs
    total = []
    for _ in range(50):
        state = ts.nve_step(state, model, dt=dt)
        kinetic = ts.calc_kinetic_energy(masses=state.masses, momenta=state.momenta)
        total.append(float((state.energy + kinetic).sum()))
    assert np.ptp(total) < 1e-3


def test_native_plugin(lj):
    try:
        metatomic.load_plugin(metatomic.testing.plugin_path("lj"))
    except Exception:
        pytest.skip("metatomic-lj-plugin is not available")
    options = {
        **{k: str(v) for k, v in LJ.items()},
        "length_unit": "angstrom",
        "energy_unit": "eV",
    }
    plugin = MetatomicModel("metatomic-lj-model", model_options=options)
    assert isinstance(plugin, CoreModel)
    assert not plugin.compute_stress  # the plugin has no strain gradients

    atoms = nickel()
    out = run(plugin, atoms)
    ref = run(CoreModel(lj, uncertainty_threshold=None, compute_stress=False), atoms)
    torch.testing.assert_close(out["energy"], ref["energy"])
    torch.testing.assert_close(out["forces"], ref["forces"])
