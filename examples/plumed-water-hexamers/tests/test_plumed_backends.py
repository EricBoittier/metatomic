"""End-to-end tests: PLUMED API -> pinned core -> Python model -> forces/virial."""

import gc

import numpy as np
import pytest
from conftest import KERNEL
from plumed import Plumed, PlumedError
from python_bridge import MODELS, register


BACKENDS = ["numpy", "torch", "jax"]
POSITIONS = np.array(
    [[0.3, -0.4, 0.2], [1.1, 0.3, -0.7], [1.4, 0.8, 1.6]], dtype=np.float64
)


class Engine:
    def __init__(
        self,
        backend,
        directory,
        length="A",
        kappa=0.7,
        target=0.4,
        consistency=False,
        md_energy=1.0,
    ):
        register()
        self.p = Plumed(kernel=str(KERNEL))
        self.kappa, self.target = kappa, target
        self.step = 0
        self.output = directory / "COLVAR"
        for cmd, value in [
            ("setMDEnergyUnits", md_energy),
            ("setMDLengthUnits", 0.1),
            ("setMDTimeUnits", 0.001),
            ("setMDMassUnits", 1.0),
            ("setMDChargeUnits", 1.0),
            ("setNatoms", 3),
            ("setMDEngine", "pytest"),
            ("setTimestep", 1.0),
            ("setLogFile", str(directory / "plumed.log")),
        ]:
            self.p.cmd(cmd, value)
        self.p.cmd("init")
        try:
            lines = [
                f"UNITS LENGTH={length} ENERGY=kj/mol TIME=fs",
                f"q: METATOMIC PLUGIN=pytest-python-backends MODEL={backend} "
                "SPECIES1=1,3 SPECIES2=2 SPECIES_TO_TYPES=8,1"
                + (" CHECK_CONSISTENCY" if consistency else ""),
                f"b: RESTRAINT ARG=q AT={target} KAPPA={kappa}",
                f"PRINT ARG=q,b.bias FILE={self.output} STRIDE=1 FMT=%.15f",
                "FLUSH STRIDE=1",
            ]
            for line in lines:
                self.p.cmd("readInputLine", line)
        except BaseException:
            self.p.finalize()
            raise

    def evaluate(self, positions):
        positions = np.ascontiguousarray(positions, dtype=np.float64)
        forces = np.zeros((3, 3))
        virial = np.zeros((3, 3))
        box = np.zeros((3, 3))
        masses = np.array([16.0, 1.0, 16.0])
        for cmd, value in [
            ("setStep", self.step),
            ("setPositions", positions),
            ("setBox", box),
            ("setMasses", masses),
            ("setEnergy", 0.0),
            ("setForces", forces),
            ("setVirial", virial),
        ]:
            self.p.cmd(cmd, value)
        self.p.cmd("calc")
        energy = np.zeros(1)
        self.p.cmd("getBias", energy)
        self.step += 1
        return energy[0], forces, virial

    def close(self):
        self.p.finalize()


def reference(p, kappa, target):
    d = p[2] - p[0]
    g = d * np.array([1.0, 2.0, 3.0])
    q = 0.5 * np.dot(d, g)
    factor = kappa * (q - target)
    forces = np.zeros((3, 3))
    forces[0] = factor * g
    forces[2] = -factor * g
    return q, 0.5 * kappa * (q - target) ** 2, forces, factor * np.outer(d, g)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", ["A", "nm"])
@pytest.mark.parametrize("kappa,target", [(0.0, 0.0), (0.7, 0.4), (1.3, 12.0)])
def test_values_forces_virial_and_lifetime(backend, length, kappa, target, tmp_path):
    """Includes zero bias, opposite force signs, empty probe and repeated updates."""
    engine = Engine(backend, tmp_path, length, kappa, target)
    try:
        expected = []
        for scale in [1.0, 0.9, 1.2]:
            p = POSITIONS * scale + [2.1, -3.2, 0.7]
            gc.collect()  # callbacks must remain alive across force calls
            energy, forces, virial = engine.evaluate(p)
            q, e, f, v = reference(p, kappa, target)
            np.testing.assert_allclose(energy, e, rtol=1e-12, atol=2e-12)
            np.testing.assert_allclose(forces, f, rtol=1e-12, atol=2e-12)
            np.testing.assert_allclose(virial, v, rtol=1e-12, atol=2e-12)
            np.testing.assert_allclose(forces.sum(0), 0, atol=1e-13)
            np.testing.assert_array_equal(forces[1], np.zeros(3))
            expected.append([q, e])
        calls = MODELS[backend].calls
        assert calls[0][0] == 0  # PLUMED's empty-system shape probe
        assert sum(n == 3 for n, *_ in calls) == 3
        for _, module, dtype, gradients in calls:
            assert (
                "torch" in module
                if backend == "torch"
                else "jax" in module
                if backend == "jax"
                else module == "numpy"
            )
            assert "float64" in dtype
            assert set(gradients) == {"positions", "strain"}
    finally:
        engine.close()
    assert MODELS[backend].unloads == 1
    np.testing.assert_allclose(
        np.loadtxt(engine.output)[:, 1:], expected, rtol=1e-12, atol=2e-12
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("length", ["A", "nm"])
def test_all_position_and_strain_components_finite_difference(
    backend, length, tmp_path
):
    engine = Engine(backend, tmp_path, length)
    try:
        _, force, virial = engine.evaluate(POSITIONS)
        h = 1e-5
        fd_force = np.zeros((3, 3))
        fd_strain = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                delta = np.zeros((3, 3))
                delta[a, b] = h
                fd_force[a, b] = -(
                    engine.evaluate(POSITIONS + delta)[0]
                    - engine.evaluate(POSITIONS - delta)[0]
                ) / (2 * h)
                fd_strain[a, b] = (
                    engine.evaluate(POSITIONS @ (np.eye(3) + delta))[0]
                    - engine.evaluate(POSITIONS @ (np.eye(3) - delta))[0]
                ) / (2 * h)
        # Unlike isotropic CVs this catches accidental strain transposition.
        assert np.max(np.abs(virial - virial.T)) > 1.0
        np.testing.assert_allclose(force, fd_force, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(virial, fd_strain, rtol=1e-8, atol=1e-8)
    finally:
        engine.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_python_model_error_reaches_plumed(backend, tmp_path):
    engine = Engine(backend, tmp_path)
    MODELS[backend].fail = True
    try:
        with pytest.raises(
            PlumedError, match="An error happened while calculating q"
        ) as caught:
            engine.evaluate(POSITIONS)
        assert f"intentional {backend} model failure" in str(caught.value.__cause__)
    finally:
        engine.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_pinned_feature_consistency_limitation_is_explicit(backend, tmp_path):
    # This is a passing regression test for a known limitation, not a hidden xfail.
    with pytest.raises(PlumedError, match="expected no gradients"):
        Engine(backend, tmp_path, consistency=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_md_energy_unit_conversion(backend, tmp_path):
    # One engine energy unit = 2 kJ/mol; PLUMED input/output stays kJ/mol.
    engine = Engine(backend, tmp_path, md_energy=2.0)
    try:
        energy, forces, virial = engine.evaluate(POSITIONS)
        _, e, f, v = reference(POSITIONS, 0.7, 0.4)
        np.testing.assert_allclose(energy, e / 2, rtol=1e-12)
        np.testing.assert_allclose(forces, f / 2, atol=1e-12)
        np.testing.assert_allclose(virial, v / 2, atol=1e-12)
    finally:
        engine.close()


def test_exact_core_revision():
    from metatomic._c_lib import _get_library

    assert "b02b9ff" in _get_library().mta_version().decode()


def test_python_and_plumed_share_native_libraries():
    from conftest import _NATIVE
    from metatensor._c_lib import _get_library as tensors

    from metatomic._c_lib import _get_library as core

    assert tensors()._handle == _NATIVE[0]._handle
    assert core()._handle == _NATIVE[1]._handle
