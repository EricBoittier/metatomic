"""
.. _c-tutorial-symd-engine:

Case study: driving symd through the C API
===========================================

The other C tutorials show how to build a system, attach a pair list, and
(most recently) define and run a model in isolation. This one closes the loop
with a real, independent
simulation engine: `symd <https://github.com/whitead/symd>`_, a small C
molecular-dynamics code (its specialty is symmetry-constrained crystal MD,
though nothing here uses that -- see below).

symd has its own force plugin ABI (a ``force_t`` vtable: a ``gather``
function that fills a ``forces`` array and returns the energy, plus a
``free``). Wiring a new ``force_type: "metatomic"`` into it means writing the
engine side of the C API for real. The walkthrough below follows the same
12-step data flow as the :ref:`torch engine/model diagram <model-dataflow>`
on the overview page, point by point, so it should read as "the same
contract, a different language" rather than a separate story.

The full integration lives in symd's own tree, on the ``feat/metatomic-c-api``
branch (not part of this repository). Every result plotted below comes from
actually running the compiled ``symd`` binary, not from replaying saved
numbers. To isolate *just* the C API bridge -- not symd's neighbor-list
bookkeeping, not its symmetry machinery -- it runs the simplest possible
scenario: a small free (non-periodic) 3D cluster of particles under NVE,
where whether energy is conserved is a direct, unambiguous check on whether
the physics coming back through the C API is correct.
"""

# sphinx_gallery_thumbnail_number = 3

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import ase
import chemiscope
import matplotlib.pyplot as plt
import numpy as np

# %%
#
# The bridge, 12 points at a time
# ---------------------------------
#
# Every snippet below is quoted directly out of symd's real
# ``src/metatomic_force.c`` (via ``literalinclude``, not retyped), so it
# can't drift out of sync with what actually runs. Larger ones are
# collapsed by default -- click to expand.
#
# .. literalinclude:: /_include/metatomic_force.c
#    :language: c
#    :lines: 30-39
#    :caption: metatomic_force.c, top of file
#
# ``metatomic_force.h`` itself pulls in symd's own ``force.h`` (the
# ``force_t`` vtable) and ``nlist.h`` (symd's neighbor list). Before the
# 12 points: the physics the model actually computes, independently
# derived rather than copied from symd's own ``lj()`` (so the two can be
# cross-checked, see the results below):
#
# .. literalinclude:: /_include/metatomic_force.c
#    :language: c
#    :lines: 66-93
#    :caption: mtm_lj_pair() -- shifted Lennard-Jones, energy and force
#
# #. **The engine loads an exported model.** symd doesn't load a file --
#    it registers an in-process plugin and loads it by name, once, in
#    ``build_metatomic()``:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 783-800
#
# #. **The engine requests and gets the model's capabilities.**
#    :c:func:`mta_execute_model` does this itself (calling ``mtm_lj_capabilities``)
#    as part of consistency checking -- symd's own code never calls it directly.
# #. **The engine creates evaluation options.** The C API has no separate
#    options object; the equivalent is just the ``requested_outputs_json``
#    string handed straight to :c:func:`mta_execute_model` (point 9).
# #. **The engine creates a list of System.** ``mta_system_create`` wraps
#    symd's own position/cell/pbc arrays as DLPack views -- no copies:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 613-615
#
# #. **The engine asks the model for the neighbor lists it needs.**
#    Handled internally by :c:func:`mta_execute_model` (``mtm_lj_requested_pair_lists``
#    reports the cutoff as a ``PairListOptions`` JSON string).
# #. **The engine computes those neighbor lists and registers them.**
#    symd's own cutoff filtering, then attaching the result:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 670-673
#
#    .. details:: Show how symd's own neighbor list becomes a metatomic pair list
#
#       Collecting the pairs symd's cell/Verlet list already found, within the
#       model's *true* cutoff (the list itself uses a slightly larger skin
#       radius):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 549-582
#
#       Then packaging them as a metatomic pair-list block (displacement
#       vectors, plus the ``first_atom``/``second_atom`` sample labels):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 633-668
#
# #. **The engine asks for any extra required input data.** ``mtm_lj_requested_inputs``
#    returns ``[]`` -- this model needs nothing beyond positions.
# #. **The engine registers that extra data.** Nothing to do, since point 7
#    asked for nothing.
# #. **The engine calls the model.** Not ``forward()`` -- the C API's
#    equivalent is :c:func:`mta_execute_model` itself, which also does unit
#    conversion and consistency checking on top of whatever the model computes:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 681-695
#
#    This only became available partway through this exercise
#    (``metatomic-core@8b0d8975``); earlier revisions of this code called
#    ``execute_inner`` directly, since ``mta_execute_model`` was not ready yet.
# #. **The model runs.** ``mtm_lj_execute_inner`` -- called internally by
#    :c:func:`mta_execute_model` through the ``execute_inner`` function pointer.
#
#    .. details:: Show the full mtm_lj_execute_inner()
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 358-486
#
# #. **The model returns its outputs.** A :py:class:`TensorMap`-shaped
#    ``"energy"`` output, with a ``"positions"`` gradient block attached --
#    that gradient *is* the force, and driving a real engine needs it filled
#    in. See :ref:`the previous tutorial <c-tutorial-add-model>` for how,
#    including a finite-difference check that it's actually correct.
#
#    .. details:: Show how the gradient block is built, and how the engine reads it back
#
#       Building it (one row per atom, values = ``-force``):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 240-298
#
#       Reading it back out, on the engine side:
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 704-756
#
# #. **The engine runs backward() for gradients, if needed.** Not used here:
#    the C API has no autodiff pass. Forces come back already computed, as
#    the ``"positions"`` gradient block from point 11 -- exactly the
#    *explicit* gradients path the torch docs' own tip on this step mentions
#    as the alternative to backward-mode differentiation.

# %%
#
# Setting up a run
# ----------------
#
# Eight particles in a small 3D cluster, in a box much larger than the
# interaction cutoff -- so there is nothing for symd's own neighbor list to
# do beyond simple distance checks, and nothing for the C API bridge to get
# wrong. ``p1`` is symd's trivial symmetry group (one member, the identity):
# it still goes through symd's normal group machinery, just without imposing
# any actual constraint.

P1_GROUP = {
    "name": "p1", "size": 1, "dof": 3,
    # a single 4x4 identity (homogeneous 3D transform: rotation + translation)
    "members": [np.eye(4).flatten().tolist()],
    # symd projects box updates through this at startup regardless of
    # box_update_period -- a 9x9 identity imposes no shape constraint
    "projector": np.eye(9).flatten().tolist(),
}
# a compact 2x2x2 grid, spacing 1.15 sigma, small jitter -- fractional
# coordinates in a 20x20x20 box (sigma = 1, cutoff = 3 sigma: nothing here is
# within reach of a periodic image of itself or of another particle)
CLUSTER_XYZ = """0.471774 0.472290 0.471487
0.527936 0.471160 0.470411
0.471862 0.531352 0.471160
0.528342 0.530077 0.472434
0.471930 0.470446 0.529355
0.530315 0.469826 0.528712
0.468920 0.527408 0.526636
0.528920 0.527441 0.529805
"""


def run_symd(force_type, workdir, symd_binary, steps=4000, print_period=10):
    """Write inputs for one force_type, run symd, and return (trace, frames)."""
    workdir = Path(workdir)
    (workdir / "cluster.xyz").write_text(CLUSTER_XYZ)
    (workdir / "p1.json").write_text(json.dumps(P1_GROUP))

    run_params = {
        # NVE: no thermostat, no pressure coupling -- total energy should be
        # conserved, full stop, and any drift is a bug in the force.
        "steps": steps, "time_step": 0.005,
        "cell": [20, 20, 20], "n_images": 0, "n_particles": 8,
        "start_positions": "cluster.xyz",
        "print_period": print_period, "positions_log_file": "positions.xyz",
        "position_log_period": print_period, "force_type": force_type,
        "lj_epsilon": 1.0, "lj_sigma": 1.0,
        "group": "p1.json",
        "start_temperature": 0.3, "seed": 42,
        "final_positions": "final_positions.dat",
    }
    (workdir / "run_params.json").write_text(json.dumps(run_params))

    result = subprocess.run(
        [str(symd_binary), "run_params.json"],
        cwd=workdir, capture_output=True, text=True, check=True,
    )

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if re.match(r"^\d", line):
            step, t, T, PE, KE, E, Htherm, V = line.split()
            rows.append(
                {"t": float(t), "T": float(T), "PE": float(PE),
                 "KE": float(KE), "E": float(E)}
            )

    # parse the logged xyz trajectory: "n\nFrame: k\nH x y z\n..." repeated
    frames = []
    xyz_lines = (workdir / "positions.xyz").read_text().splitlines()
    i = 0
    while i < len(xyz_lines):
        n_atoms = int(xyz_lines[i])
        i += 2  # atom count, then the "Frame: k" comment line
        coords = []
        for _ in range(n_atoms):
            parts = xyz_lines[i].split()
            coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
            i += 1
        frames.append(np.array(coords))

    return rows, frames


# %%
#
# Running it
# ----------
#
# ``lj`` is symd's own hand-written, independently-implemented Lennard-Jones;
# ``metatomic`` is the C API bridge above. Same starting configuration, same
# seed, both NVE.

SYMD_BUILD = Path(os.environ.get("SYMD_BUILD_DIR", "/home/ericb/metawork/etc/symd/build"))
symd_binary = SYMD_BUILD / "symd3"  # N_DIMS=3, matches this 3D cluster

traces = {}
frames_by_type = {}
with tempfile.TemporaryDirectory() as tmp:
    for force_type in ["lj", "metatomic"]:
        run_dir = Path(tmp) / force_type
        run_dir.mkdir()
        traces[force_type], frames_by_type[force_type] = run_symd(force_type, run_dir, symd_binary)

for name, rows in traces.items():
    Es = [r["E"] for r in rows]
    drift = (max(Es) - min(Es)) / abs(sum(Es) / len(Es))
    print(f"{name:>10}: {len(rows)} samples, E(0) = {rows[0]['E']:.5f}, "
          f"E(end) = {rows[-1]['E']:.5f}, (max-min)/|mean| = {drift:.4%}")

# %%
#
# The cluster, in motion
# -----------------------
#
# The actual 3D trajectories, straight from symd's own log, one
# `chemiscope <https://chemiscope.org>`_ viewer per force type -- drag to
# rotate, use the play button to step through time. Both start from the
# identical 2x2x2 grid.

def to_ase_trajectory(frames, stride=5):
    return [
        ase.Atoms("H8", positions=xyz, cell=np.eye(3) * 20.0, pbc=False)
        for xyz in frames[::stride]
    ]


viewer_settings = {"bonds": False, "playbackDelay": 90}

chemiscope.show(
    to_ase_trajectory(frames_by_type["lj"]),
    mode="structure",
    settings={"structure": [viewer_settings]},
)

# %%
#
# ``metatomic``, same starting configuration:

chemiscope.show(
    to_ase_trajectory(frames_by_type["metatomic"]),
    mode="structure",
    settings={"structure": [viewer_settings]},
)

# %%
#
# Results
# -------
#
# The metatomic run's own energy bookkeeping: potential and kinetic energy
# trade off, total energy stays flat. This is the actual point of running
# NVE -- an MD code that gets forces wrong from a model almost always shows
# it here first, as drift.

mtm = traces["metatomic"]
t = [r["t"] for r in mtm]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(t, [r["PE"] for r in mtm], label="potential energy")
ax.plot(t, [r["KE"] for r in mtm], label="kinetic energy")
ax.plot(t, [r["E"] for r in mtm], label="total energy", color="k", linewidth=1.5)
ax.legend()
ax.set_xlabel("t (reduced units)")
ax.set_ylabel("energy")
ax.set_title("metatomic force_type, NVE")
fig.tight_layout()
fig.show()

# %%
#
# And the cross-check this tutorial exists to make: does the C API path
# agree with symd's own independently-implemented force, for the *entire*
# trajectory? Both total-energy traces are shown as their deviation from
# their own mean, ``E(t) - <E>`` -- this removes the (well-understood, see
# below) constant offset between them and puts both conservation qualities
# on the same scale.

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, color in [("lj", "C0"), ("metatomic", "C2")]:
    rows = traces[name]
    Es = np.array([r["E"] for r in rows])
    ax.plot([r["t"] for r in rows], Es - Es.mean(), label=name, color=color)
ax.axhline(0, color="0.7", linewidth=1, zorder=0)
ax.legend()
ax.set_xlabel("t (reduced units)")
ax.set_ylabel("total energy - <total energy>")
ax.set_title("same trajectory, two independent force implementations")
fig.tight_layout()
fig.show()

# %%
#
# Both fluctuate around zero -- energy is conserved on both sides, which is
# the whole check. The offset removed by centering each trace on its own
# mean is not drift and not a bug: symd's own ``lj()`` shifts the *force* to
# zero smoothly at the cutoff, while this model only shifts the *energy* and
# truncates the force there. Both are standard, legitimate LJ cutoff
# conventions; they just are not bit-identical.
