"""Check all 54 position and nine strain derivatives via the pinned PLUMED API."""

import numpy as np
from ase.io import read, write
from bias_utils import HERE, BiasProbe
from conformations import load_conformations
from tour_cv import rotation


out = HERE / "gradient-checks"
out.mkdir(exist_ok=True)
refs = load_conformations()
worst = 0.0
for name, atoms in refs.items():
    path = out / f"{name}-target.xyz"
    write(path, atoms, format="xyz")
    atoms = read(path, format="xyz")
    line = (
        f"q: METATOMIC MODEL=rmsd-target:{path} PLUGIN=water-hexamer"
        " SPECIES1=1,4,7,10,13,16"
        " SPECIES2=2,3,5,6,8,9,11,12,14,15,17,18 SPECIES_TO_TYPES=8,1"
    )
    probe = BiasProbe("q", out / f"rmsd-{name}.log", extra_input=[line])
    p = atoms.positions + np.random.default_rng(42).normal(0, 0.08, (18, 3))
    energy, force, virial = probe.evaluate(p)
    h = 1e-5
    fd = np.zeros_like(p)
    for i in range(18):
        for k in range(3):
            delta = np.zeros_like(p)
            delta[i, k] = h
            fd[i, k] = -(
                probe.evaluate(p + delta)[0] - probe.evaluate(p - delta)[0]
            ) / (2 * h)
    strain = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            delta = np.zeros((3, 3))
            delta[i, j] = h
            strain[i, j] = (
                probe.evaluate(p @ (np.eye(3) + delta))[0]
                - probe.evaluate(p @ (np.eye(3) - delta))[0]
            ) / (2 * h)
    error = max(np.max(abs(fd - force)), np.max(abs(strain - virial)))
    assert error < 2e-7, (name, error)
    assert np.linalg.norm(force.sum(0)) < 1e-12
    assert np.linalg.norm(np.cross(p, force).sum(0)) < 1e-12
    # Independently verify the normalization and the regularized minimum.
    centered = p - p.mean(0)
    target = atoms.positions - atoms.positions.mean(0)
    expected = 0.5 * (
        ((centered @ rotation(centered, target) - target) ** 2).sum() / 18 + 1e-16
    )
    assert abs(energy - expected) < 1e-8, (energy, expected)
    assert probe.evaluate(atoms.positions)[0] < 1e-14
    # A large rigid motion must preserve the CV; an improper reflection must
    # agree with the independent *proper*-rotation SVD metric, not fit away.
    moved = atoms.copy()
    moved.positions = p
    moved.rotate(113, "z")
    moved.translate([2, -3, 4])
    assert abs(probe.evaluate(moved.positions)[0] - energy) < 1e-10
    reflected = p.copy()
    reflected[:, 0] *= -1
    centered = reflected - reflected.mean(0)
    expected_mirror = 0.5 * (
        ((centered @ rotation(centered, target) - target) ** 2).sum() / 18 + 1e-16
    )
    observed_mirror = probe.evaluate(reflected)[0]
    # PLUMED and ASE use slightly different eV/kJ/mol conversion constants.
    np.testing.assert_allclose(observed_mirror, expected_mirror, rtol=3e-7, atol=1e-10)
    other = BiasProbe(
        "q", out / f"rmsd-{name}-nm.log", length_unit="nm", extra_input=[line]
    )
    for x, y in zip((energy, force, virial), other.evaluate(p), strict=False):
        assert np.max(abs(np.asarray(x) - np.asarray(y))) < 1e-11
    other.close()
    probe.close()
    worst = max(worst, error)
print(f"PASS: nine targets, positions/strain, A/nm; maximum error {worst:.3e}")
