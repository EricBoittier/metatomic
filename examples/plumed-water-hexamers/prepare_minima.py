"""Quench tour holds with rigid TIP3P and group equivalent relaxed structures."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ase.calculators.tip3p import TIP3P
from ase.constraints import FixBondLengths, FixCom
from ase.io import read, write
from ase.optimize import BFGS
from scan_cv import PAIRS
from symmetry import match


def classify(names, frames, energies):
    comparisons, representatives, families = [], [], {}
    for i, name in enumerate(names):
        equivalent = None
        for j in representatives:
            fits = match(frames[i], frames[j])
            delta = abs(energies[i] - energies[j])
            same = fits["proper"]["rmsd_A"] < 0.01 and delta < 1e-5
            comparisons.append(
                dict(
                    first=name,
                    second=names[j],
                    energy_difference_eV=delta,
                    fits=fits,
                    same_relaxed_structure=same,
                )
            )
            if same:
                equivalent = names[j]
                break
        if equivalent is None:
            representatives.append(i)
            equivalent = name
        families.setdefault(equivalent, []).append(name)
    return representatives, families, comparisons


def save_library(out, names, frames, energies, origin):
    representatives, families, comparisons = classify(names, frames, energies)
    with (out / "minima.xyz").open("w") as stream:
        for i in representatives:
            write(
                stream,
                frames[i],
                format="xyz",
                comment=names[i] + "; rigid TIP3P relaxed seed",
            )
    with (out / "relaxed-seeds.xyz").open("w") as stream:
        for name, frame in zip(names, frames, strict=True):
            write(stream, frame, format="xyz", comment=name + "; relaxed tour hold")
    report = dict(
        origin=origin,
        families=families,
        energies_eV=dict(zip(names, energies, strict=True)),
        equivalence_criteria=dict(
            proper_all_atom_rmsd_A=0.01, energy_difference_eV=1e-5
        ),
        comparisons=comparisons,
        interpretation=(
            "Seed-derived relaxed structures, not an exhaustive inventory "
            "of TIP3P minima. Proper water/H permutations merge duplicates; improper "
            "fits are reported separately. A matching operation does not establish "
            "a dynamically accessible pathway or an effective tunneling group."
        ),
    )
    (out / "minima.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tour", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.tour / "summary.json").read_text())
    trajectory = read(args.tour / "trajectory.traj", ":")
    names, frames, energies = [], [], []
    for visit in summary["visits"]:
        a = min(
            trajectory, key=lambda a: abs(a.info["time_fs"] - visit["time_fs"])
        ).copy()
        # XYZ does not preserve FixBondLengths: always install constraints explicitly.
        a.set_constraint([FixBondLengths(PAIRS, tolerance=1e-12), FixCom()])
        a.calc = TIP3P(rc=20, width=1)
        lengths = np.array([a.get_distance(i, j) for i, j in PAIRS])
        for stage, tolerance in [("coarse", 0.005), ("tight", 1e-4)]:
            ok = BFGS(
                a,
                logfile=str(args.output / f"{visit['name']}-{stage}.log"),
                maxstep=0.03,
            ).run(fmax=tolerance, steps=1600)
            if not ok:
                raise RuntimeError(f"{visit['name']} did not converge")
        assert (
            max(abs(np.array([a.get_distance(i, j) for i, j in PAIRS]) - lengths))
            < 1e-8
        )
        names.append(visit["name"])
        frames.append(a)
        energies.append(float(a.get_potential_energy()))
        write(args.output / f"{visit['name']}.extxyz", a)
    report = save_library(
        args.output,
        names,
        frames,
        energies,
        dict(tour_parameters=summary["parameters"], force_tolerance_eV_A=1e-4),
    )
    report["preparation_seconds"] = time.monotonic() - started
    (args.output / "minima.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["families"])


if __name__ == "__main__":
    main()
