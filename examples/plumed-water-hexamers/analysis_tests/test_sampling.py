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
