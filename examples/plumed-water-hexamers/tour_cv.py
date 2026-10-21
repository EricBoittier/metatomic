"""Continuous driven tours of source templates or relaxed representatives."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from analyze_scan_endpoints import hb_graph
from ase import units
from ase.calculators.plumed import Plumed
from ase.calculators.tip3p import TIP3P, angleHOH, rOH
from ase.constraints import FixBondLengths, FixCom
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import thermalize_momenta
from bias_utils import HERE, KERNEL, cv_input
from conformations import load_conformations
from run_md import descriptors
from scan_cv import PAIRS, PERMUTATIONS, oxygen_rmsd


ORDER = ["book", "book-2", "bag", "cage", "prism", "boat-1", "boat-2", "chair", "ring"]


def rigid_reference(atoms):
    atoms = atoms.copy()
    atoms.set_cell(np.zeros((3, 3)))
    atoms.pbc = False
    for i in range(0, 18, 3):
        u, v = atoms.positions[i + 1 : i + 3] - atoms.positions[i]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        x, y = (u + v) / np.linalg.norm(u + v), (u - v) / np.linalg.norm(u - v)
        theta = np.deg2rad(angleHOH / 2)
        atoms.positions[i + 1] = atoms.positions[i] + rOH * (
            np.cos(theta) * x + np.sin(theta) * y
        )
        atoms.positions[i + 2] = atoms.positions[i] + rOH * (
            np.cos(theta) * x - np.sin(theta) * y
        )
    return atoms


def rotation(p, q):
    u, _, vh = np.linalg.svd(p.T @ q)
    return (u * np.array([1, 1, np.linalg.det(u @ vh)])) @ vh


def map_reference(target, previous):
    """Choose fixed molecule/H labels before MD, with proper rotations only."""
    p = previous.positions[::3] - previous.positions[::3].mean(0)
    q = target.positions[::3] - target.positions[::3].mean(0)
    candidates = q[PERMUTATIONS]
    u, _, vh = np.linalg.svd(np.einsum("pia,ib->pab", candidates, p))
    signs = np.ones((720, 3))
    signs[:, -1] = np.linalg.det(u @ vh)
    rotations = (u * signs[:, None, :]) @ vh
    # Score whole waters, not just oxygens: cyclic O skeletons have near ties
    # with very different proton arrangements and unnecessary rotation barriers.
    indices = (3 * PERMUTATIONS[:, :, None] + np.arange(3)).reshape(720, 18)
    positions = target.positions[indices] - target.positions[::3].mean(0)
    positions = positions @ rotations + previous.positions[::3].mean(0)
    for i in range(0, 18, 3):
        h = positions[:, i + 1 : i + 3].copy()
        old = previous.positions[i + 1 : i + 3]
        swap = ((h[:, ::-1] - old) ** 2).sum((1, 2)) < ((h - old) ** 2).sum((1, 2))
        positions[swap, i + 1 : i + 3] = h[swap, ::-1]
    errors = ((positions - previous.positions) ** 2).sum((1, 2))
    best = np.argmin(errors)
    result = target.copy()
    result.positions = positions[best]
    return result


def full_rmsd(atoms, target):
    p = atoms.positions - atoms.positions.mean(0)
    q = target.positions - target.positions.mean(0)
    return float(np.sqrt(((p @ rotation(p, q) - q) ** 2).sum() / 18))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minima",
        action="store_true",
        help="Closed loop over the relaxed minima library",
    )
    parser.add_argument("--output", type=Path, default=HERE / "cv-tour")
    parser.add_argument("--ramp", type=int, default=400)
    parser.add_argument("--hold", type=int, default=400)
    parser.add_argument("--kappa", type=float, default=2000.0)
    parser.add_argument("--temperature", type=float, default=200.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--max-seconds", type=float, default=400.0)
    args = parser.parse_args()
    if args.ramp <= 0 or args.hold <= 0 or args.ramp % 10 or args.hold % 10:
        parser.error("ramp and hold must be positive multiples of ten")
    if not (0 < args.dt <= 0.5 and 0 < args.max_seconds <= 460):
        parser.error("dt must be in (0, 0.5] fs; max-seconds in (0, 460]")
    if not (args.kappa > 0 and args.temperature > 0):
        parser.error("kappa and temperature must be positive")
    started = time.monotonic()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = load_conformations(HERE / "minima.xyz" if args.minima else None)
    order = list(raw) + ["book-return"] if args.minima else ORDER
    if args.minima:
        (out / "reference-library.xyz").write_text((HERE / "minima.xyz").read_text())
        raw["book-return"] = raw["book"].copy()
    count = len(order)
    refs = []
    for name in order:
        ref = rigid_reference(raw[name])
        if refs:
            ref = map_reference(ref, refs[-1])
        refs.append(ref)
        write(out / f"{name}-target.xyz", ref, format="xyz")
    lines = cv_input()
    species = (
        " SPECIES1=1,4,7,10,13,16"
        " SPECIES2=2,3,5,6,8,9,11,12,14,15,17,18 SPECIES_TO_TYPES=8,1"
    )
    for i, name in enumerate(order):
        lines.append(
            f"q{i}: METATOMIC MODEL=rmsd-target:{out / (name + '-target.xyz')}"
            " PLUGIN=water-hexamer" + species
        )
    stages = [0, args.hold]
    weights = [np.eye(count)[0], np.eye(count)[0]]
    for i in range(1, count):
        stages.extend([stages[-1] + args.ramp, stages[-1] + args.ramp + args.hold])
        weights.extend([np.eye(count)[i], np.eye(count)[i]])
    lines.append(
        "tour: MOVINGRESTRAINT ARG="
        + ",".join(f"q{i}" for i in range(count))
        + " "
        + " ".join(
            f"STEP{i}={step} AT{i}="
            + ",".join(["0"] * count)
            + f" KAPPA{i}="
            + ",".join(str(args.kappa * w) for w in weight)
            for i, (step, weight) in enumerate(zip(stages, weights, strict=False))
        )
    )
    lines.append(
        "PRINT ARG=rg,cn,"
        + ",".join(f"q{i}" for i in range(count))
        + f",tour.bias STRIDE=10 FILE={out / 'COLVAR'} FMT=%.12f"
    )
    (out / "plumed.dat").write_text("\n".join(lines) + "\n")
    atoms = refs[0].copy()
    atoms.set_constraint([FixBondLengths(PAIRS, tolerance=1e-12), FixCom()])
    thermalize_momenta(
        atoms, temperature_K=args.temperature, rng=np.random.default_rng(20261021)
    )
    calc = Plumed(
        TIP3P(rc=20, width=1),
        input=lines,
        timestep=args.dt * units.fs,
        atoms=atoms,
        kT=args.temperature * units.kB,
        log=str(out / "plumed.log"),
    )
    atoms.calc = calc
    rows, frames, visits = [], [], []
    with Trajectory(str(out / "trajectory.traj"), "w", atoms) as traj:
        with Langevin(
            atoms,
            args.dt * units.fs,
            temperature_K=args.temperature,
            friction=0.05 / units.fs,
            fixcm=False,
            rng=np.random.default_rng(1729),
        ) as dyn:

            def record():
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("Tour exceeded its wall-clock budget")
                step = dyn.nsteps
                rg, cn = descriptors(atoms)
                energy = atoms.get_potential_energy()
                assert np.isfinite(energy)
                if step % 10 == 0:
                    atoms.info.update(time_fs=step * args.dt, rg=rg, cn=cn)
                    traj.write()
                    frames.append(atoms.copy())
                    rows.append(
                        [step * args.dt, rg, cn, atoms.get_temperature(), energy]
                    )
                if step in stages[1::2]:
                    i = stages[1::2].index(step)
                    mismatch = int(np.abs(hb_graph(atoms) - hb_graph(refs[i])).sum())
                    visit = dict(
                        name=order[i],
                        step=step,
                        time_fs=step * args.dt,
                        all_atom_rmsd_A=full_rmsd(atoms, refs[i]),
                        oxygen_rmsd_A=oxygen_rmsd(atoms, raw[order[i]]),
                        directed_hbond_mismatches=mismatch,
                    )
                    visit["visited"] = (
                        visit["all_atom_rmsd_A"] < 0.15
                        and visit["oxygen_rmsd_A"] < 0.15
                        and mismatch == 0
                    )
                    visits.append(visit)
                    print(json.dumps(visit), flush=True)

            dyn.attach(record, interval=1)
            try:
                dyn.run(stages[-1])
            finally:
                calc.plumed.finalize()
    write(out / "trajectory.extxyz", frames)
    np.savetxt(
        out / "observables.csv",
        rows,
        delimiter=",",
        comments="",
        header="time_fs,rg_A,cn,temperature_K,energy_eV",
    )
    report = dict(
        metatomic_commit="b02b9ff3df6458cca21fa8690d229bbda97e6f78",
        kernel=str(KERNEL),
        parameters=vars(args) | {"output": str(out)},
        visits=visits,
        reference_kind="TIP3P-relaxed minima"
        if args.minima
        else "rigidified source templates",
        all_visited=all(v["visited"] for v in visits),
        elapsed_seconds=time.monotonic() - started,
        interpretation=(
            "Driven template tour, not equilibrium sampling or spontaneous "
            "transitions. Fixed molecule labels; proper-rotation all-atom RMSD "
            "to fixed targets, permutation-invariant O RMSD to the selected "
            "reference library, directed hydrogen bonds to fixed targets."
        ),
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_visited"]:
        raise RuntimeError(
            "Tour finished but one or more templates failed the visit criteria"
        )


if __name__ == "__main__":
    main()
