"""Render actual driven MD frames with a synchronized PCA and visitation map."""

import argparse
import json
import time
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from analyze_loop import label_waters
from analyze_sampling import analyze, histogram_axes, pca_axes, plot_summary
from matplotlib.animation import FuncAnimation, PillowWriter
from plot_scan import aligned_frames, draw_atoms


def main(directory):
    started = time.monotonic()
    out = Path(directory)
    summary = json.loads((out / "summary.json").read_text())
    sampling = analyze(out)
    plot_summary(out, sampling)
    frames = sampling["frames"]
    aligned = aligned_frames(frames)
    visits = summary["visits"]
    times = sampling["time"]
    chosen = [int(np.argmin(abs(times - v["time_fs"] / 1000))) for v in visits]
    loop = summary["parameters"].get("minima", False)
    label = (
        "Closed loop through relaxed minima" if loop else "Nine-template driven MD tour"
    )
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout="constrained")
    for ax, idx, visit in zip(axes.flat, chosen, visits, strict=False):
        draw_atoms(ax, aligned[idx])
        if loop:
            label_waters(ax, aligned[idx])
        ax.set_title(
            f"{visit['name']} · {times[idx]:.2f} ps\n"
            f"All-atom RMSD {visit['all_atom_rmsd_A']:.3f} Å",
            fontsize=11,
        )
    for unused in list(axes.flat)[len(visits) :]:
        unused.axis("off")
    fig.suptitle(
        label + " · actual hold-end frames\n"
        "ASE plot_atoms · dashed lines: geometric hydrogen bonds"
    )
    fig.savefig(out / "ase-storyboard.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), layout="constrained")
    ax, traceax, pcaax, histax = axes.flat
    pca_axes(pcaax, sampling)
    mesh = histogram_axes(histax, sampling)
    fig.colorbar(mesh, ax=histax, label="Saved-frame fraction")
    (pc_marker,) = pcaax.plot([], [], "o", color="#00bfff", mec="black", ms=8, zorder=5)
    (cv_marker,) = histax.plot(
        [], [], "o", color="#00bfff", mec="black", ms=8, zorder=5
    )
    # Aligned all-atom RMSDs to each fixed, labeled template retain proton ordering.
    colvar = np.loadtxt(out / "COLVAR")
    for i, visit in enumerate(visits):
        traceax.plot(colvar[:, 0] / 1000, colvar[:, 3 + i], label=visit["name"], lw=1)
    traceax.set(
        xlabel="Time (ps)",
        ylabel="Target RMSD CV (Å)",
        title="Aligned all-atom RMSD",
    )
    traceax.legend(ncol=3, fontsize=8, frameon=False)
    marker = traceax.axvline(0, color="k", lw=1)
    title = fig.suptitle("")

    def draw(idx):
        draw_atoms(ax, aligned[idx])
        if loop:
            label_waters(ax, aligned[idx])
        t = times[idx]
        marker.set_xdata([t, t])
        pc_marker.set_data([sampling["points"][idx, 0]], [sampling["points"][idx, 1]])
        cv_marker.set_data([sampling["cv"][idx, 0]], [sampling["cv"][idx, 1]])
        target = min(
            np.searchsorted([v["time_fs"] / 1000 for v in visits], t), len(visits) - 1
        )
        title.set_text(
            f"{label} · {t:.2f} ps · steering toward {visits[target]['name']}"
        )

    indices = sorted(
        set(np.linspace(0, len(frames) - 1, 65).astype(int).tolist() + chosen)
    )
    animation = FuncAnimation(fig, draw, frames=indices, interval=120)
    animation.save(out / "ase-trajectory.gif", writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)
    runtime = dict(
        md_seconds=summary["elapsed_seconds"],
        analysis_and_render_seconds=time.monotonic() - started,
    )
    runtime["total_seconds"] = (
        runtime["md_seconds"] + runtime["analysis_and_render_seconds"]
    )
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
