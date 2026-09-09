"""
.. _c-tutorial-symd-engine:

Case study: driving symd through the C API
===========================================

The other C tutorials show how to build a system, attach a pair list, and run
a model in isolation. This one closes the loop with a real, independent
simulation engine: `symd <https://github.com/whitead/symd>`_, a small C
molecular-dynamics code with its own neighbor lists, thermostats, and (for
this tutorial) 2D wallpaper-group symmetry constraints.

symd has its own force plugin ABI (a ``force_t`` vtable: a ``gather``
function that fills a ``forces`` array and returns the energy, plus a
``free``). Wiring a new ``force_type: "metatomic"`` into it means writing the
engine side of the C API for real: wrap symd's own arrays into an
:c:type:`mta_system_t`, reuse symd's own neighbor list as the pair list, and
call :c:func:`mta_execute_model` on a toy shifted Lennard-Jones
:c:type:`mta_model_t` -- then read the result back into symd's own force
buffer. Every result plotted below comes from actually running the compiled
``symd`` binary, not from replaying saved numbers.

The full integration lives in symd's own tree, on the ``feat/metatomic-c-api``
branch (not part of this repository) -- this tutorial reproduces its smoke
test and its validation against symd's own hand-written potentials.
"""

# sphinx_gallery_thumbnail_number = 1

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

# %%
#
# The bridge, in two snippets
# ----------------------------
#
# The model is a shifted Lennard-Jones potential, independently derived (not
# copied from symd's own ``lj()`` helper) so the two can be cross-checked:
#
# .. code-block:: c
#
#     static void mtm_lj_pair(
#         double dx, double dy, double dz,
#         const MtmLennardJones *lj,
#         double *energy,
#         double force_on_i[3]
#     ) {
#         double r2 = dx * dx + dy * dy + dz * dz;
#         if (r2 <= 0.0 || r2 >= lj->cutoff * lj->cutoff) {
#             *energy = 0.0;
#             force_on_i[0] = force_on_i[1] = force_on_i[2] = 0.0;
#             return;
#         }
#         double inv2 = (lj->sigma * lj->sigma) / r2;
#         double inv6 = inv2 * inv2 * inv2;
#         double inv12 = inv6 * inv6;
#         *energy = 4.0 * lj->epsilon * (inv12 - inv6) - lj->shift;
#
#         /* dE/d(r^2) = (12 * epsilon / r^2) * (inv6 - 2 * inv12) */
#         double dedr2 = (12.0 * lj->epsilon / r2) * (inv6 - 2.0 * inv12);
#         force_on_i[0] = 2.0 * dedr2 * dx;
#         force_on_i[1] = 2.0 * dedr2 * dy;
#         force_on_i[2] = 2.0 * dedr2 * dz;
#     }
#
# ``metatomic_gather_forces`` (symd's ``force_t.gather``) wraps symd's
# positions into an :c:type:`mta_system_t`, attaches a pair list built from
# symd's own neighbor list, and calls :c:func:`mta_execute_model`:
#
# .. code-block:: c
#
#     static const char *requested_outputs = "[{"
#         "\"type\": \"metatomic_quantity\","
#         "\"name\": \"energy\","
#         "\"unit\": \"eV\","
#         "\"gradients\": [\"positions\"],"
#         "\"sample_kind\": \"system\""
#         "}]";
#     status = mta_execute_model(
#         mp->model, systems, 1, NULL, requested_outputs, true, &output, 1
#     );
#
# :c:func:`mta_execute_model` (not ``execute_inner`` directly) is deliberate:
# it is the entry point that actually does unit conversion and consistency
# checking -- ``metatomic-core@8b0d8975`` finished it partway through this
# exercise, replacing an earlier workaround that called ``execute_inner``
# directly while it was still ``todo!()``.
#
# The result comes back as a :py:class:`TensorMap`-shaped ``"energy"``
# output with a ``"positions"`` gradient block -- the C API tutorials stop at
# energy-only ("position gradients are still TODO"); driving a real engine
# needs the gradient filled in, since that *is* the force.

# %%
#
# Setting up a run
# ----------------
#
# The scenario is a 6-particle, symmetry-constrained 2D packing (wallpaper
# group p6, symd's own ``wp-15`` example) -- embedded here so this tutorial
# is self-contained.

WP15_GROUP = {
    "name": "wp-15", "size": 6, "dof": 2,
    "members": [
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ],
    "projector": [
        1.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.5,
        0.0, 0.0, 0.0, 0.0, 0.8660254, 0.8660254, 0.8660254, 0.8660254,
    ],
}
WP15_WYCKOFFS = {
    "wp-15-00": {"name": "wp-15-0", "size": 3, "dof": 2,
                 "members": [[1, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1],
                             [-1, 0, 0, -1, 0, 0, 0, 0, 1]],
                 "projector": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
    "wp-15-01": {"name": "wp-15-1", "size": 2, "dof": 0,
                 "members": [[0, 0, 0.66666667, 0, 0, 0.33333333, 0, 0, 1],
                             [0, 0, 0.33333333, 0, 0, 0.66666667, 0, 0, 1]],
                 "projector": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
    "wp-15-02": {"name": "wp-15-2", "size": 1, "dof": 0,
                 "members": [[0, 0, 0, 0, 0, 0, 0, 0, 1]],
                 "projector": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
}
WP15_XYZ = """    0.716213    0.0897165
    0.869673     0.112382
    0.259615     0.123343
 9.71127e-24  1.94225e-23
    0.406883  7.76902e-23
    0.666667     0.333333
"""


def run_symd(force_type, workdir, symd_binary, steps=4000, print_period=10):
    """Write inputs for one force_type, run symd, and return the parsed trace."""
    workdir = Path(workdir)
    (workdir / "wp-15.xyz").write_text(WP15_XYZ)
    (workdir / "wp-15.json").write_text(json.dumps(WP15_GROUP))
    for name, group in WP15_WYCKOFFS.items():
        (workdir / f"{name}.json").write_text(json.dumps(group))

    run_params = {
        "steps": steps, "temperature": 0.2, "start_temperature": 0.5,
        "langevin_gamma": 0.005, "cell": [25, 25], "n_images": 2,
        "n_particles": 6, "start_positions": "wp-15.xyz",
        "print_period": print_period, "positions_log_file": "positions.xyz",
        "position_log_period": print_period, "force_type": force_type,
        "group": "wp-15.json",
        "wyckoffs": [
            {"group": "wp-15-02.json", "n_particles": 1},
            {"group": "wp-15-00.json", "n_particles": 1},
            {"group": "wp-15-01.json", "n_particles": 1},
        ],
        "thermostat": "baoab", "pressure": 1, "box_update_period": 25,
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
    return rows


# %%
#
# Running the three force types
# ------------------------------
#
# ``lj`` and ``nlj`` are symd's own hand-written Lennard-Jones (O(n^2), and a
# cell/Verlet-list version); ``metatomic`` is the C API bridge above,
# reusing ``nlj``'s neighbor list. Same starting configuration, same seed.

SYMD_BUILD = Path(os.environ.get("SYMD_BUILD_DIR", "/home/ericb/metawork/etc/symd/build"))
symd_binary = SYMD_BUILD / "symd2"  # N_DIMS=2, matches this 2D packing

traces = {}
with tempfile.TemporaryDirectory() as tmp:
    for force_type in ["lj", "nlj", "metatomic"]:
        run_dir = Path(tmp) / force_type
        run_dir.mkdir()
        traces[force_type] = run_symd(force_type, run_dir, symd_binary)

for name, rows in traces.items():
    print(f"{name:>10}: {len(rows)} samples, E(0) = {rows[0]['E']:.5f}, "
          f"E(end) = {rows[-1]['E']:.5f}")

# %%
#
# Results
# -------
#
# The metatomic run's own energy bookkeeping: potential and kinetic energy
# trade off as the baoab thermostat pulls the system from T = 0.5 toward its
# T = 0.2 target -- total energy is *not* expected to be flat, since this is
# NVT, not NVE.

mtm = traces["metatomic"]
t = [r["t"] for r in mtm]

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

ax[0].plot(t, [r["PE"] for r in mtm], label="potential energy")
ax[0].plot(t, [r["KE"] for r in mtm], label="kinetic energy")
ax[0].plot(t, [r["E"] for r in mtm], label="total energy")
ax[0].legend()
ax[0].set_xlabel("t (reduced units)")
ax[0].set_ylabel("energy")
ax[0].set_title("metatomic force_type")

ax[1].plot(t, [r["T"] for r in mtm], color="C3")
ax[1].axhline(0.2, color="0.6", linestyle="--", linewidth=1, label="target T")
ax[1].legend()
ax[1].set_xlabel("t (reduced units)")
ax[1].set_ylabel("temperature")
ax[1].set_title("thermostat")

fig.tight_layout()
fig.show()

# %%
#
# And the cross-check this tutorial exists to make: does the C API path
# agree with symd's own force code, for the *entire* trajectory, not just
# the first step?

fig, ax = plt.subplots(figsize=(6, 4.2))
for name, color in [("lj", "C0"), ("nlj", "C1"), ("metatomic", "C2")]:
    rows = traces[name]
    ax.plot([r["t"] for r in rows], [r["E"] for r in rows], label=name, color=color)
ax.legend()
ax.set_xlabel("t (reduced units)")
ax.set_ylabel("total energy")
ax.set_title("same trajectory, three force implementations")
fig.tight_layout()
fig.show()

# %%
#
# ``nlj`` and ``metatomic`` track each other almost exactly for all 4000
# steps -- expected, since the C API path reuses ``nlj``'s own neighbor
# list. The constant offset from ``lj`` is a pre-existing double-count in
# symd's own ghost-pair energy accounting on this symmetry-heavy example
# (not introduced by, and not fixed by, this integration); forces are
# unaffected; see symd's own ``NOTES-metatomic.md`` for the full writeup.
