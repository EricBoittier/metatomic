"""Descriptive sampling maps, not equilibrium probabilities or free energies."""

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from conformations import load_conformations
from run_md import descriptors
from scan_cv import oxygen_rmsd


def distance_features(frames):
    """Sorted 15 O–O distances: rotation/translation/permutation invariant.

    Sorting discards connectivity and all proton ordering. This is a geometric
    projection, not an injective structure descriptor or basin classifier.
    """
    result = []
    for frame in frames:
        p = frame.positions[frame.numbers == 8]
        if len(p) != 6:
            raise ValueError("Expected six oxygens")
        d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        result.append(np.sort(d[np.triu_indices(6, 1)]))
    return np.array(result)


def project_pca(features, references):
    """Fit centered, unscaled PCA to trajectory only, then project references."""
    if len(features) < 3 or not np.isfinite(features).all():
        raise ValueError("PCA needs at least three finite trajectory frames")
    mean = features.mean(axis=0)
    _, s, vt = np.linalg.svd(features - mean, full_matrices=False)
    if np.sum(s**2) == 0:
        raise ValueError("Trajectory has no geometric variance")
    # Stable sign convention for reproducible exports.
    for row in vt:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    components = vt[:2]
    return (
        (features - mean) @ components.T,
        (references - mean) @ components.T,
        s[:2] ** 2 / np.sum(s**2),
        mean,
        components,
    )


def analyze(directory):
    out = Path(directory)
    frames = read(out / "trajectory.traj", ":")
    library = out / "reference-library.xyz"
    refs = load_conformations(library if library.exists() else None)
    names = list(refs)
    time = np.array([a.info["time_fs"] for a in frames]) / 1000
    cv = np.array([descriptors(a) for a in frames])
    ref_cv = np.array([descriptors(a) for a in refs.values()])
    points, ref_points, variance, mean, components = project_pca(
        distance_features(frames), distance_features(refs.values())
    )
    rmsd = np.array([[oxygen_rmsd(a, b) for b in refs.values()] for a in frames])
    threshold = 0.35
    nearest = rmsd.argmin(axis=1)
    classified = np.where(rmsd.min(axis=1) <= threshold, nearest, len(names))
    fractions = np.bincount(classified, minlength=len(names) + 1) / len(frames)
    np.savetxt(
        out / "sampling-pca.csv",
        np.column_stack([time, points, cv, rmsd]),
        delimiter=",",
        comments="",
        header="time_ps,PC1_A,PC2_A,rg_A,cn,"
        + ",".join("rmsd_" + n + "_A" for n in names),
    )
    hist, xedges, yedges = np.histogram2d(cv[:, 0], cv[:, 1], bins=24)
    np.savez(
        out / "sampling-map.npz",
        mean=mean,
        components=components,
        explained_variance_ratio=variance,
        scores=points,
        reference_scores=ref_points,
        cv_counts=hist,
        rg_edges=xedges,
        cn_edges=yedges,
    )
    report = dict(
        frames=len(frames),
        time_range_ps=[float(time[0]), float(time[-1])],
        reference_names=names,
        reference_kind="TIP3P-relaxed minima"
        if library.exists()
        else "unrelaxed source geometries",
        pca_features=(
            "15 sorted oxygen pair distances in angstrom; "
            "centered, unscaled; trajectory-only fit"
        ),
        explained_variance_ratio=variance.tolist(),
        rmsd_threshold_A=threshold,
        nearest_reference_frame_fraction=dict(
            zip(names + ["unassigned"], fractions.tolist(), strict=True)
        ),
        interpretation=(
            "Biased nonequilibrium saved-frame frequencies, not equilibrium "
            "populations. Oxygen geometry ignores proton order; "
            "sorted distances are not unique."
        ),
    )
    (out / "sampling-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return dict(
        frames=frames,
        refs=refs,
        names=names,
        time=time,
        cv=cv,
        ref_cv=ref_cv,
        points=points,
        ref_points=ref_points,
        variance=variance,
        rmsd=rmsd,
        hist=hist,
        xedges=xedges,
        yedges=yedges,
        fractions=fractions,
    )


def pca_axes(ax, result):
    p, r = result["points"], result["ref_points"]
    ax.plot(*p.T, color="0.75", lw=0.7, zorder=0)
    scatter = ax.scatter(*p.T, c=result["time"], cmap="viridis", s=12)
    ax.scatter(*r.T, marker="X", c="#cf503d", edgecolors="white", s=65, zorder=3)
    # Labels use staggered offsets for the similar cyclic references.
    for i, (name, point) in enumerate(zip(result["names"], r, strict=True)):
        ax.annotate(
            name,
            point,
            xytext=(5, 7 + (i % 3) * 11),
            textcoords="offset points",
            fontsize=8,
            arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5),
        )
    v = result["variance"] * 100
    ax.set(
        xlabel=f"PC1 (Å; {v[0]:.1f}% variance)",
        ylabel=f"PC2 (Å; {v[1]:.1f}% variance)",
        title="Oxygen geometry PCA · × reference geometries",
    )
    ax.margins(0.2)
    return scatter


def histogram_axes(ax, result):
    h = result["hist"] / len(result["frames"])
    mesh = ax.pcolormesh(
        result["xedges"],
        result["yedges"],
        np.ma.masked_equal(h.T, 0),
        cmap="magma",
        shading="auto",
    )
    ax.set(
        xlabel="Oxygen Rg (Å)",
        ylabel="Mean O coordination",
        title="Visited CV bins · saved-frame fraction",
    )
    return mesh


def plot_summary(out, result):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    sc = pca_axes(axes[0, 0], result)
    fig.colorbar(sc, ax=axes[0, 0], label="Time (ps)")
    mesh = histogram_axes(axes[0, 1], result)
    fig.colorbar(mesh, ax=axes[0, 1], label="Fraction of saved frames per bin")
    labels = result["names"] + ["unassigned"]
    axes[1, 0].barh(labels, result["fractions"], color="#4d8796")
    axes[1, 0].set(
        xlabel="Fraction of saved frames",
        xlim=(0, 1),
        title="Nearest reference O geometry within 0.35 Å RMSD",
    )
    axes[1, 0].invert_yaxis()
    for i, name in enumerate(result["names"]):
        axes[1, 1].plot(result["time"], result["rmsd"][:, i], label=name, lw=1)
    axes[1, 1].axhline(0.35, color="k", ls=":", lw=0.8)
    axes[1, 1].set(
        xlabel="Time (ps)",
        ylabel="Permutation-minimized O RMSD (Å)",
        title="Distances to all reference conformations",
    )
    axes[1, 1].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "Visited regions of the biased trajectory\n"
        "Descriptive frame counts — not equilibrium populations "
        "or proton-order assignments"
    )
    fig.savefig(Path(out) / "sampling-regions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=Path("cv-scan"))
    directory = parser.parse_args().directory
    plot_summary(directory, analyze(directory))
