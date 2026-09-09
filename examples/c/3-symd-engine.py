"""
.. _c-tutorial-symd-engine:

Case study: driving symd through the C API
===========================================

The other C tutorials show how to build a system, attach a pair list, and run
a model in isolation. This one closes the loop with a real, independent
simulation engine: `symd <https://github.com/whitead/symd>`_, a small C
molecular-dynamics code (its specialty is symmetry-constrained crystal MD,
though nothing here uses that -- see below).

symd has its own force plugin ABI (a ``force_t`` vtable: a ``gather``
function that fills a ``forces`` array and returns the energy, plus a
``free``). Wiring a new ``force_type: "metatomic"`` into it means writing the
engine side of the C API for real: wrap symd's own arrays into an
:c:type:`mta_system_t`, attach a pair list built from symd's own neighbor
list, and call :c:func:`mta_execute_model` on a toy shifted Lennard-Jones
:c:type:`mta_model_t` -- then read the result back into symd's own force
buffer. Every result plotted below comes from actually running the compiled
``symd`` binary, not from replaying saved numbers.

The full integration lives in symd's own tree, on the ``feat/metatomic-c-api``
branch (not part of this repository). To isolate *just* the C API bridge --
not symd's neighbor-list bookkeeping, not its symmetry machinery -- this
tutorial runs the simplest possible scenario: a small free (non-periodic)
cluster of particles under NVE, where whether energy is conserved is a
direct, unambiguous check on whether the physics coming back through the C
API is correct.
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
# Calling :c:func:`mta_execute_model` -- not ``execute_inner`` directly -- is
# deliberate: it is the real entry point, handling unit conversion and
# consistency checking on top of whatever the model itself computes. It only
# became available partway through this exercise (``metatomic-core@8b0d8975``);
# earlier revisions called ``execute_inner`` directly to work around it still
# being ``todo!()``.
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
# Eight particles in a small 2D cluster, in a box much larger than the
# interaction cutoff -- so there is nothing for symd's own neighbor list to
# do beyond simple distance checks, and nothing for the C API bridge to get
# wrong. ``group.p1`` is symd's trivial symmetry group (one member, the
# identity): it still goes through symd's normal group machinery, just
# without imposing any actual constraint.

P1_GROUP = {
    "name": "p1", "size": 1, "dof": 2,
    "members": [[1, 0, 0, 0, 1, 0, 0, 0, 1]],
    "projector": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
}
# a compact 2x4 grid, spacing 1.15 sigma, small jitter -- fractional
# coordinates in a 20x20 box (sigma = 1, cutoff = 3 sigma: nothing here is
# within reach of a periodic image of itself or of another particle)
CLUSTER_XYZ = """0.413752 0.471698
0.470839 0.469914
0.528068 0.469763
0.586340 0.473260
0.413012 0.527819
0.471985 0.529285
0.528908 0.527354
0.586206 0.529793
"""


def run_symd(force_type, workdir, symd_binary, steps=4000, print_period=10):
    """Write inputs for one force_type, run symd, and return the parsed trace."""
    workdir = Path(workdir)
    (workdir / "cluster.xyz").write_text(CLUSTER_XYZ)
    (workdir / "p1.json").write_text(json.dumps(P1_GROUP))

    run_params = {
        # NVE: no thermostat, no pressure coupling -- total energy should be
        # conserved, full stop, and any drift is a bug in the force.
        "steps": steps, "time_step": 0.005,
        "cell": [20, 20], "n_images": 0, "n_particles": 8,
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
    return rows


# %%
#
# Running it
# ----------
#
# ``lj`` is symd's own hand-written, independently-implemented Lennard-Jones;
# ``metatomic`` is the C API bridge above. Same starting configuration, same
# seed, both NVE.

SYMD_BUILD = Path(os.environ.get("SYMD_BUILD_DIR", "/home/ericb/metawork/etc/symd/build"))
symd_binary = SYMD_BUILD / "symd2"  # N_DIMS=2, matches this 2D cluster

traces = {}
with tempfile.TemporaryDirectory() as tmp:
    for force_type in ["lj", "metatomic"]:
        run_dir = Path(tmp) / force_type
        run_dir.mkdir()
        traces[force_type] = run_symd(force_type, run_dir, symd_binary)

for name, rows in traces.items():
    Es = [r["E"] for r in rows]
    drift = (max(Es) - min(Es)) / abs(sum(Es) / len(Es))
    print(f"{name:>10}: {len(rows)} samples, E(0) = {rows[0]['E']:.5f}, "
          f"E(end) = {rows[-1]['E']:.5f}, (max-min)/|mean| = {drift:.4%}")

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
# trajectory?

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name, color in [("lj", "C0"), ("metatomic", "C2")]:
    rows = traces[name]
    ax.plot([r["t"] for r in rows], [r["E"] for r in rows], label=name, color=color)
ax.legend()
ax.set_xlabel("t (reduced units)")
ax.set_ylabel("total energy")
ax.set_title("same trajectory, two independent force implementations")
fig.tight_layout()
fig.show()

# %%
#
# Both are flat -- energy is conserved on both sides, which is the whole
# check. The small constant offset between them is not drift and not a bug:
# symd's own ``lj()`` shifts the *force* to zero smoothly at the cutoff,
# while this model only shifts the *energy* and truncates the force there.
# Both are standard, legitimate LJ cutoff conventions; they just are not
# bit-identical. See symd's own ``NOTES-metatomic.md`` for the full writeup.
