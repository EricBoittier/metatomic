"""Audit loop closure and show the net molecular permutation of a driven loop."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.utils import rotate
from plot_scan import draw_atoms
from symmetry import match
from tour_cv import full_rmsd


def label_waters(ax, atoms):
    for i, p in enumerate(atoms.positions[::3]):
        ax.text(
            p[0] + 4,
            p[1] + 4,
            "ABCDEF"[i],
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=9,
            zorder=10,
        )


def analyze(directory):
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    assert summary["parameters"]["minima"] and verification["all_visited"]
    frames = read(out / "trajectory.traj", ":")
    first = min(
        frames, key=lambda a: abs(a.info["time_fs"] - summary["visits"][0]["time_fs"])
    )
    last = frames[-1]
    fits = match(last, first)
    assert fits["proper"]["rmsd_A"] < 0.15
    report = dict(
        connected_relaxed_references=[v["name"] for v in summary["visits"][:-1]],
        edges=[
            dict(
                start=a["name"],
                end=b["name"],
                start_fs=a["time_fs"],
                end_fs=b["time_fs"],
            )
            for a, b in zip(summary["visits"][:-1], summary["visits"][1:], strict=True)
        ],
        closed=True,
        fixed_label_closure_rmsd_A=full_rmsd(last, first),
        symmetry_matches=fits,
        interpretation=(
            "A prescribed biased closed loop with a best-fit net atom "
            "permutation. Edges are driven trajectory segments, not minimum-energy "
            "paths or transition-state connections. This one loop does not determine "
            "the effective molecular symmetry group; symmetry-equivalent fits may "
            "not be unique."
        ),
    )
    (out / "connection-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    a, b = first.copy(), last.copy()
    center = a.positions.mean(0)
    b.positions = (b.positions - b.positions.mean(0)) @ np.array(
        fits["proper"]["rotation"]
    ) + center
    camera = rotate("35x,25y,-15z")
    origin = a.positions[::3].mean(0)
    for frame in (a, b):
        frame.positions = (frame.positions - origin) @ camera
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    for ax, frame, title in zip(
        axes,
        (a, b),
        ("First book hold", "Return to book · symmetry-aligned"),
        strict=True,
    ):
        draw_atoms(ax, frame)
        label_waters(ax, frame)
        ax.set_title(title)
    op = fits["proper"]
    fig.suptitle(
        f"Water identities through the closed loop: {op['water_cycles']}\n"
        f"Permutation-aligned RMSD {op['rmsd_A']:.3f} Å · "
        f"fixed-label RMSD {report['fixed_label_closure_rmsd_A']:.3f} Å"
    )
    fig.savefig(out / "closure-symmetry.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)
