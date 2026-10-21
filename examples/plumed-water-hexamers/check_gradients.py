"""Finite differences through PLUMED, testing plugin gradients AND force/virial propagation."""
import json
from pathlib import Path
import numpy as np
from conformations import load_conformations
from bias_utils import HERE, BiasProbe

out = HERE / 'gradient-checks'
out.mkdir(exist_ok=True)
refs = load_conformations()
frames = list(refs.values())
report = []
h = 1e-5
for cv in ('rg', 'cn'):
    probe = BiasProbe(cv, out / f'{cv}.log')
    for name, atoms in refs.items():
        p = atoms.positions
        energy, force, virial = probe.evaluate(p)
        fd_force = np.zeros_like(p)
        fd_strain = np.zeros((3, 3))
        for i in range(18):
            for k in range(3):
                delta = np.zeros_like(p)
                delta[i, k] = h
                fd_force[i, k] = -(probe.evaluate(p + delta)[0] - probe.evaluate(p - delta)[0]) / (2*h)
        for a in range(3):
            for b in range(3):
                delta = np.zeros((3, 3))
                delta[a, b] = h
                fd_strain[a, b] = (probe.evaluate(p @ (np.eye(3) + delta))[0] - probe.evaluate(p @ (np.eye(3) - delta))[0]) / (2*h)
        # Bias E=CV^2/2, hence dE/dCV=CV: these also check CV derivatives.
        value = np.sqrt(2*energy)
        pos_error = np.max(np.abs(force - fd_force))
        strain_error = np.max(np.abs(virial - fd_strain))
        assert pos_error < 2e-7, (cv, name, pos_error)
        assert strain_error < 2e-7, (cv, name, strain_error)
        assert np.max(np.abs(force[atoms.numbers == 1])) < 1e-14
        assert np.linalg.norm(force.sum(axis=0)) < 1e-12
        assert np.linalg.norm(np.cross(p, force).sum(axis=0)) < 1e-11
        shifted = probe.evaluate(p + [3.4, -1.3, 0.7])
        assert abs(shifted[0] - energy) < 1e-12
        assert np.max(np.abs(shifted[2] - virial)) < 1e-11
        item = dict(cv=cv, isomer=name, value=value, max_force_error=float(pos_error),
                    max_strain_error=float(strain_error),
                    max_position_cv_gradient_error=float(pos_error/value),
                    max_strain_cv_gradient_error=float(strain_error/value))
        report.append(item)
        print(f'{cv:2} {name:5}: force error {pos_error:.3e}, strain error {strain_error:.3e}')
    # Repeat in PLUMED nanometer units; MD positions/forces remain A and eV/A.
    other = BiasProbe(cv, out / f'{cv}-nm.log', length_unit='nm')
    reference = probe.evaluate(frames[0].positions)
    converted = other.evaluate(frames[0].positions)
    for x, y in zip(reference, converted):
        assert np.max(np.abs(np.asarray(x)-np.asarray(y))) < 1e-11
    print(f'{cv}: angstrom/nanometer force and virial conversion PASS')
    other.close()
    probe.close()
(out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS: both CVs, all reference isomers, all 54 coordinates and all 9 strain components.')
