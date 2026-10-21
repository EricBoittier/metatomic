"""Short matched biased/unbiased rigid-TIP3P trajectories using pinned core CVs."""
import argparse
import json
from pathlib import Path
import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.calculators.plumed import Plumed
from ase.constraints import FixBondLengths, FixCom
from ase.optimize import BFGS
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from bias_utils import HERE, KERNEL, cv_input


def descriptors(atoms):
    r = atoms.positions[::3]
    rg = np.sqrt(((r - r.mean(axis=0))**2).sum()/6)
    d = np.linalg.norm(r[:, None] - r[None, :], axis=2)
    cn = (2/6) * np.sum(1/(1+(d[np.triu_indices(6, 1)]/3.2)**6))
    return float(rg), float(cn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=HERE/'md-run')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)  # preserve previous trajectories
    atoms = read(HERE/'hexamers.xyz', index=0)
    atoms.set_pbc(False)
    atoms.set_cell(np.zeros((3, 3)))
    # Set exact TIP3P geometry while preserving each molecular plane/bisector.
    for i in range(0, 18, 3):
        u = atoms.positions[i+1] - atoms.positions[i]
        v = atoms.positions[i+2] - atoms.positions[i]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        bisector = (u+v)/np.linalg.norm(u+v)
        lateral = (u-v)/np.linalg.norm(u-v)
        theta = np.deg2rad(angleHOH/2)
        atoms.positions[i+1] = atoms.positions[i] + rOH*(np.cos(theta)*bisector + np.sin(theta)*lateral)
        atoms.positions[i+2] = atoms.positions[i] + rOH*(np.cos(theta)*bisector - np.sin(theta)*lateral)
    pairs = [(i+a, i+b) for i in range(0, 18, 3) for a,b in ((0,1),(0,2),(1,2))]
    atoms.set_constraint([FixBondLengths(pairs, tolerance=1e-12), FixCom()])
    atoms.calc = TIP3P(rc=20, width=1)
    opt = BFGS(atoms, logfile=str(out/'relax.log'), maxstep=0.05)
    converged = opt.run(fmax=0.02, steps=500)
    if not converged:
        raise RuntimeError('Rigid TIP3P relaxation did not converge')
    initial_rg, initial_cn = descriptors(atoms)
    target = initial_rg + 0.15
    kappa = 10.0  # eV / A^2 for the numerical Rg feature in angstrom
    temperature = 100.0
    dt = 0.5 * units.fs
    thermalize_momenta(atoms, temperature_K=temperature, rng=np.random.default_rng(20261016))
    write(out/'initial.extxyz', atoms)
    ref_lengths = np.array([atoms.get_distance(a,b) for a,b in pairs])
    report = dict(metatomic_commit='b02b9ff3df6458cca21fa8690d229bbda97e6f78',
                  plumed_commit='8226d8f682372750f46b3ed859c37e3dc0657803',
                  kernel=str(KERNEL), steps=args.steps, timestep_fs=0.5,
                  temperature_K=temperature, thermostat='Langevin, friction 0.01/fs',
                  potential='ASE rigid TIP3P, nonperiodic, cutoff 20 A (switch starts at 19 A)',
                  initial_rg_A=initial_rg, initial_cn=initial_cn,
                  target_rg_A=target, kappa_eV_per_A2=kappa, runs={})
    for name, strength in [('unbiased', 0.0), ('biased', kappa)]:
        case = out/name
        case.mkdir()
        replica = atoms.copy()
        lines = cv_input() + [f'bias: RESTRAINT ARG=rg AT={target:.16g} KAPPA={strength}',
                             f'PRINT ARG=rg,cn,bias.bias STRIDE=1 FILE={case/"COLVAR"} FMT=%.12f']
        # Standalone input includes the library even for the second in-process instance.
        saved_lines = list(lines)
        if 'PLUGIN_LIBRARY=' not in saved_lines[1]:
            saved_lines[1] += f' PLUGIN_LIBRARY={HERE/"hexamer.dylib"}'
        (case/'plumed.dat').write_text('\n'.join(saved_lines)+'\n')
        calc = Plumed(TIP3P(rc=20, width=1), input=lines, timestep=dt,
                      atoms=replica, kT=units.kB*temperature, log=str(case/'plumed.log'))
        replica.calc = calc
        data = []
        max_bond_error = 0.0
        with Trajectory(str(case/'trajectory.traj'), 'w', replica) as traj:
            with Langevin(replica, dt, temperature_K=temperature, friction=0.01/units.fs, fixcm=False,
                          rng=np.random.default_rng(9173)) as dyn:
                def record():
                    nonlocal max_bond_error
                    rg, cn = descriptors(replica)
                    potential = replica.get_potential_energy()
                    kinetic = replica.get_kinetic_energy()
                    bias = 0.5*strength*(rg-target)**2
                    row = [dyn.nsteps*0.5, rg, cn, potential-bias, bias, kinetic,
                           replica.get_temperature()]
                    assert np.isfinite(row).all()
                    oxygen = replica.positions[::3]
                    diameter = np.linalg.norm(oxygen[:,None]-oxygen[None,:], axis=2).max()
                    assert diameter < 19, 'Cluster entered TIP3P switching region'
                    lengths = np.array([replica.get_distance(a,b) for a,b in pairs])
                    max_bond_error = max(max_bond_error, float(np.max(np.abs(lengths-ref_lengths))))
                    assert max_bond_error < 1e-8
                    data.append(row)
                    if dyn.nsteps % 10 == 0 or dyn.nsteps == args.steps:
                        traj.write()
                dyn.attach(record, interval=1)
                dyn.run(args.steps)
        calc.plumed.finalize()
        a = np.array(data)
        np.savetxt(case/'observables.csv', a, delimiter=',',
                   header='time_fs,rg_A,cn,tip3p_eV,bias_eV,kinetic_eV,temperature_K', comments='')
        colvar = np.loadtxt(case/'COLVAR')
        assert len(colvar) == len(a) == args.steps+1
        assert np.max(np.abs(colvar[:,0]-a[:,0])) < 1e-9
        assert np.max(np.abs(colvar[:,1:3]-a[:,1:3])) < 1e-9
        assert np.max(np.abs(colvar[:,3]-a[:,4])) < 1e-9
        write(case/'trajectory.extxyz', read(case/'trajectory.traj', ':'))
        report['runs'][name] = dict(final_rg_A=float(a[-1,1]), final_cn=float(a[-1,2]),
                                   mean_rg_last_half_A=float(a[len(a)//2:,1].mean()),
                                   mean_temperature_K=float(a[:,6].mean()),
                                   max_bond_error_A=max_bond_error,
                                   final_bias_eV=float(a[-1,4]), frames=len(a))
        print(name, report['runs'][name], flush=True)
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout='constrained')
    for name in report['runs']:
        a = np.loadtxt(out/name/'observables.csv', delimiter=',', skiprows=1)
        for ax, column in zip(axes, (1, 2)):
            ax.plot(a[:,0], a[:,column], label=name, linewidth=1.3)
    axes[0].axhline(target, color='black', linestyle='--', linewidth=1, label='restraint target')
    axes[0].set_ylabel('Oxygen Rg (Å)')
    axes[1].set_ylabel('Mean O coordination')
    axes[1].set_xlabel('Time (fs)')
    axes[0].legend(frameon=False)
    fig.suptitle('Water hexamer • rigid TIP3P • 100 K\nPinned metatomic core CV plugin + PLUMED')
    fig.savefig(out/'comparison.png', dpi=180)
    plt.close(fig)
    print('PASS: finite trajectories, rigid-water geometry, and PLUMED CV/bias agreement.')


if __name__ == '__main__':
    main()
