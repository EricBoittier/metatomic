"""Analysis checks run without the native PLUMED kernel."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.io import write


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_sampling import analyze, distance_features, project_pca
from conformations import load_conformations
from scan_cv import oxygen_rmsd


# ASE 3.29 assigns ndarray.shape; NumPy 2.5 deprecates this upstream operation.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning:ase.*"
)


def test_source_integrity_and_water_order():
    refs = load_conformations()
    assert len(refs) == 9
    assert {"ring", "boat-1", "boat-2", "bag", "chair", "book-2"} <= refs.keys()
    root = Path(__file__).resolve().parents[1] / "references"
    for item in json.loads((root / "sources.json").read_text()):
        source = root / item["url"].split("/")[-1]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
        coords = np.array(
            [
                [float(v) for v in line.split()[1:4]]
                for line in source.read_text().splitlines()[2:]
            ]
        )
        np.testing.assert_array_equal(
            refs[item["name"]].positions, coords[item["atom_order"]]
        )
    for atoms in refs.values():
        for o in range(0, 18, 3):
            assert np.all(
                np.linalg.norm(
                    atoms.positions[o + 1 : o + 3] - atoms.positions[o], axis=1
                )
                < 1.2
            )


def test_distance_features_and_rmsd_are_rigid_motion_and_permutation_invariant():
    atoms = load_conformations()["boat-1"]
    changed = atoms.copy()
    changed.rotate(37, "x")
    changed.translate([2, -3, 0.7])
    order = (np.array([4, 0, 3, 5, 1, 2])[:, None] * 3 + np.arange(3)).ravel()
    changed = changed[order]
    np.testing.assert_allclose(
        distance_features([atoms]), distance_features([changed]), atol=1e-12
    )
    assert oxygen_rmsd(atoms, changed) < 1e-12


def test_pca_fits_only_trajectory():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(30, 15))
    ref = rng.normal(size=(9, 15))
    scores, projected, variance, mean, components = project_pca(x, ref)
    other = project_pca(x, ref * 100)
    np.testing.assert_allclose(scores, other[0])
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(projected, (ref - mean) @ components.T)
    assert 0 < variance.sum() <= 1


def test_sampling_counts_and_reference_dimensions(tmp_path):
    refs = load_conformations()
    frames = [refs[n].copy() for n in ("book", "prism", "boat-1", "ring")]
    for i, a in enumerate(frames):
        a.info["time_fs"] = i * 10
    write(tmp_path / "trajectory.traj", frames)
    result = analyze(tmp_path)
    assert result["rmsd"].shape == (4, 9)
    assert result["hist"].sum() == 4
    np.testing.assert_allclose(result["fractions"].sum(), 1)
    assert result["fractions"][-1] == 0
    assert (tmp_path / "sampling-pca.csv").exists()


def test_tour_template_mapping_preserves_geometry_and_rigid_water():
    from ase.calculators.tip3p import rOH
    from tour_cv import ORDER, full_rmsd, map_reference, rigid_reference

    refs = load_conformations()
    previous = rigid_reference(refs[ORDER[0]])
    for name in ORDER:
        target = rigid_reference(refs[name])
        mapped = map_reference(target, previous)
        assert oxygen_rmsd(mapped, target) < 1e-10
        p = mapped.positions.reshape(6, 3, 3)
        np.testing.assert_allclose(np.linalg.norm(p[:, 1:] - p[:, :1], axis=2), rOH)
        changed = mapped.copy()
        changed.rotate(43, "y")
        changed.translate([1, -2, 3])
        assert full_rmsd(changed, mapped) < 1e-10
        previous = mapped
    # Proper rotations must not accept the mirror image of this labeled target.
    mirrored = previous.copy()
    mirrored.positions[:, 0] *= -1
    assert full_rmsd(mirrored, previous) > 0.15


def test_exact_water_permutation_and_inversion_matching():
    from symmetry import match

    original = load_conformations()["prism"].copy()
    original.positions += np.random.default_rng(7).normal(0, 0.15, (18, 3))
    order = (3 * np.array([2, 0, 5, 1, 4, 3])[:, None] + np.arange(3)).ravel()
    order[[1, 2]] = order[[2, 1]]
    moved = original[order]
    moved.rotate(73, "x")
    moved.translate([3, -2, 1])
    fits = match(moved, original)
    assert fits["proper"]["rmsd_A"] < 1e-10
    assert fits["improper"]["rmsd_A"] > 0.05
    np.testing.assert_array_equal(fits["proper"]["atom_map_0based"], order)
    moved.positions[:, 0] *= -1
    fits = match(moved, original)
    assert fits["improper"]["rmsd_A"] < 1e-10
    assert fits["proper"]["rmsd_A"] > 0.05


def test_relaxed_minima_library_and_duplicate_families():
    from ase.calculators.tip3p import TIP3P
    from ase.constraints import FixBondLengths, FixCom
    from scan_cv import PAIRS
    from symmetry import match

    root = Path(__file__).resolve().parents[1]
    seeds = load_conformations(root / "relaxed-seeds.xyz")
    minima = load_conformations(root / "minima.xyz")
    report = json.loads((root / "minima.json").read_text())
    assert len(seeds) == 9 and len(minima) == 6
    assert set(sum(report["families"].values(), [])) == set(seeds)
    for name, atoms in seeds.items():
        atoms.set_constraint([FixBondLengths(PAIRS, tolerance=1e-12), FixCom()])
        atoms.calc = TIP3P(rc=20, width=1)
        # The archived extended XYZ coordinates were rounded to eight decimals.
        assert abs(atoms.get_potential_energy() - report["energies_eV"][name]) < 1e-7
        assert np.linalg.norm(atoms.get_forces(), axis=1).max() < 1.2e-4
    for representative, members in report["families"].items():
        for name in members:
            assert match(seeds[name], minima[representative])["proper"]["rmsd_A"] < 0.01
