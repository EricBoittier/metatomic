"""Independently audit saved frames: geometry, topology, constraints and bias."""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_scan_endpoints import hb_graph
from ase.io import read
from conformations import load_conformations
from run_md import descriptors
from scan_cv import PAIRS, oxygen_rmsd
from tour_cv import ORDER, full_rmsd


def verify(directory):
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text())
    args = summary["parameters"]
    frames = read(out / "trajectory.traj", ":")
    raw = load_conformations()
    names = [v["name"] for v in summary["visits"]]
    assert names == ORDER, "The report must include all nine templates in tour order"
    refs = [read(out / f"{n}-target.xyz", format="xyz") for n in names]
    q = np.array(
        [[np.sqrt(full_rmsd(a, ref) ** 2 + 1e-16) for ref in refs] for a in frames]
    )
    colvar = np.loadtxt(out / "COLVAR")
    observables = np.loadtxt(out / "observables.csv", delimiter=",", skiprows=1)
    assert np.isfinite(observables).all()
    assert len(frames) == len(colvar) == len(observables)
    np.testing.assert_allclose(observables[:, 0], colvar[:, 0], atol=1e-10)
    assert np.max(abs(q - colvar[:, 3:12])) < 1e-6
    assert (
        np.max(abs(np.array([descriptors(a) for a in frames]) - colvar[:, 1:3])) < 1e-8
    )
    steps = colvar[:, 0] / args["dt"]
    stages = [0, args["hold"]]
    weights = [np.eye(9)[0], np.eye(9)[0]]
    for i in range(1, 9):
        stages.extend(
            [stages[-1] + args["ramp"], stages[-1] + args["ramp"] + args["hold"]]
        )
        weights.extend([np.eye(9)[i], np.eye(9)[i]])
    assert colvar[0, 0] == 0
    assert colvar[-1, 0] == stages[-1] * args["dt"]
    w = np.array(
        [np.interp(steps, stages, np.array(weights)[:, i]) for i in range(9)]
    ).T
    bias = 0.5 * args["kappa"] * np.sum(w * q * q, axis=1)
    assert np.max(abs(bias - colvar[:, 12])) < 1e-3
    bonds = np.array([[a.get_distance(i, j) for i, j in PAIRS] for a in frames])
    bond_error = float(np.max(abs(bonds - bonds[0])))
    assert bond_error < 1e-8
    max_diameter = max(
        np.linalg.norm(a.positions[::3, None] - a.positions[None, ::3], axis=-1).max()
        for a in frames
    )
    assert max_diameter < 19
    results = []
    for name, target, visit in zip(names, refs, summary["visits"], strict=False):
        end = visit["time_fs"]
        # Require at least three consecutive saved frames in the final 50 fs.
        indices = np.flatnonzero((colvar[:, 0] >= end - 50) & (colvar[:, 0] <= end))
        hits, longest, consecutive = [], 0, 0
        for idx in indices:
            a = frames[idx]
            hit = (
                full_rmsd(a, target) < 0.15
                and oxygen_rmsd(a, raw[name]) < 0.15
                and np.array_equal(hb_graph(a), hb_graph(target))
            )
            hits.append(bool(hit))
            consecutive = consecutive + 1 if hit else 0
            longest = max(longest, consecutive)
        results.append(
            dict(
                name=name,
                matching_frames=sum(hits),
                checked_frames=len(hits),
                longest_consecutive_frames=longest,
                passed=longest >= 3,
            )
        )
    report = dict(
        visits=results,
        all_visited=all(r["passed"] for r in results),
        max_bond_error_A=bond_error,
        max_oxygen_diameter_A=float(max_diameter),
        max_bias_energy_error_eV=float(np.max(abs(bias - colvar[:, 12]))),
    )
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert report["all_visited"], (
        "Not all nine templates were visited for three consecutive saved frames"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
