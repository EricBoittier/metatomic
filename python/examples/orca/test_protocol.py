"""Smoke tests for the ORCA external-tool file protocol (no ORCA required)."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import MagicMock

import ase.units
import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

SCRIPT = Path(__file__).resolve().parent / "metatomic-orca-external"


def load_orca_module():
    mock_metatomic_ase = MagicMock()
    sys.modules.setdefault("metatomic_ase", mock_metatomic_ase)

    loader = SourceFileLoader("metatomic_orca_external", str(SCRIPT))
    spec = importlib.util.spec_from_loader("metatomic_orca_external", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


@pytest.fixture
def orca():
    module = load_orca_module()
    module.clear_calculator_cache()
    yield module
    module.clear_calculator_cache()


class FixedEnergyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, energy_ev: float, forces_ev_angstrom: np.ndarray):
        super().__init__()
        self.energy_ev = energy_ev
        self.forces_ev_angstrom = np.asarray(forces_ev_angstrom, dtype=float)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        self.results = {
            "energy": self.energy_ev,
            "forces": self.forces_ev_angstrom.copy(),
        }


def test_read_extinp_and_xyz(tmp_path, orca):
    xyz_path = tmp_path / "water.xyz"
    xyz_path.write_text(
        "3\n"
        "water\n"
        "O 0.0 0.0 0.1173\n"
        "H 0.0 0.7572 -0.4692\n"
        "H 0.0 -0.7572 -0.4692\n"
    )
    extinp_path = tmp_path / "job_EXT.extinp.tmp"
    extinp_path.write_text(
        f"{xyz_path.name}\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
    )

    extinp = orca.read_extinp(extinp_path)
    assert extinp.charge == 0
    assert extinp.multiplicity == 1
    assert extinp.do_gradient is True
    assert extinp.xyz_path == xyz_path.resolve()

    symbols, coords = orca.read_xyz(xyz_path)
    assert symbols == ["O", "H", "H"]
    assert len(coords) == 3


def test_write_engrad_roundtrip(tmp_path, orca):
    gradient = [0.0, 0.0, 0.1, -0.05, 0.0, 0.0, -0.05, 0.0, 0.0]
    engrad_path = tmp_path / "job.engrad"
    orca.write_engrad(engrad_path, natoms=3, energy_hartree=-0.5, gradient_hartree_bohr=gradient)

    text = engrad_path.read_text()
    assert "3" in text
    assert "-5.000000000000e-01" in text
    assert "1.000000000000e-01" in text


def test_forces_to_orca_gradient(orca):
    forces = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    gradient = orca.forces_to_orca_gradient(forces)
    expected = -forces * ase.units.Bohr / ase.units.Hartree
    np.testing.assert_allclose(gradient, expected.reshape(-1))


def test_run_orca_job_writes_engrad(tmp_path, orca, monkeypatch):
    xyz_path = tmp_path / "water.xyz"
    xyz_path.write_text(
        "3\n"
        "water\n"
        "O 0.0 0.0 0.1173\n"
        "H 0.0 0.7572 -0.4692\n"
        "H 0.0 -0.7572 -0.4692\n"
    )
    extinp_path = tmp_path / "water_EXT.extinp.tmp"
    extinp_path.write_text(
        f"{xyz_path.name}\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
    )

    energy_ev = -27.2
    forces = np.zeros((3, 3))
    forces[0, 2] = 0.01
    fake_calc = FixedEnergyCalculator(energy_ev, forces)

    settings = orca.MetatomicOrcaSettings(model=tmp_path / "fake.pt")
    settings.model.write_text("placeholder")

    monkeypatch.setattr(orca, "get_calculator", lambda _settings: fake_calc)

    engrad_path = orca.run_orca_job(extinp_path, settings)
    assert engrad_path == tmp_path / "water.engrad"
    assert engrad_path.is_file()

    extinp = orca.read_extinp(extinp_path)
    atoms = orca.atoms_from_extinp(extinp)
    assert atoms.info["charge"] == 0
    assert atoms.info["spin_multiplicity"] == 1

    expected_energy = energy_ev / ase.units.Hartree
    expected_gradient = orca.forces_to_orca_gradient(forces).tolist()
    content = engrad_path.read_text()
    assert f"{expected_energy:.12e}" in content
    for value in expected_gradient:
        assert f"{value: .12e}" in content
