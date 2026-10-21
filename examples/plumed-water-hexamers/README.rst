Water hexamers through the metatomic core API
============================================

This example evaluates two analytic collective variables on prism, cage, book,
ring, bag, chair, a second book, and two boat water hexamers (18 atoms each). It loads a C++ plugin through the
metatomic core API. It does not use TorchScript or a trained potential.

This is an experimental documentation draft for the new core API. The build
instructions below target macOS and exact development revisions, not the
released TorchScript interface. Generated trajectories are not committed; the scripts reproduce them.
Selected rendered figures are included in the PLUMED documentation.

After completing the source build below, run from this directory::

    sh run.sh

The source-built PLUMED writes ``COLVAR`` and ``driver.log``. ``verify.py``
compares all eighteen CV values against independent Python calculations, with
an absolute tolerance of 1e-8. The first COLVAR column is a frame time/index,
not a physical MD trajectory. Frames are prism, cage, book, ring, bag, chair, book-2, boat-1 and boat-2,
in that order.

Descriptors
-----------

``rg`` is the unweighted radius of gyration of the six oxygen positions, in
angstrom. ``cn`` is their mean smooth coordination number:

    cn = (2/6) sum over i<j of 1 / (1 + (r_ij / 3.2 angstrom)^6)

These are geometric descriptors, not energies or a free-energy ranking. They
use nonperiodic coordinates and do not account for hydrogen-bond orientation.
The supported oxygen-cluster diameter is at most 100 angstrom; larger clusters
are rejected, matching the finite interaction range declared to this core API.
Both CVs provide analytic position and homogeneous-strain gradients. Hydrogen
position gradients are zero. For row-vector coordinates transformed as r' = r F,
the strain gradient is dCV/dF_ab = sum_i r_ia dCV/dr_ib. The position gradient of
Rg is (r_i - r_center)/(6 Rg); coordination derivatives follow by differentiating
the switching function. Rg rejects coincident oxygen positions because its
gradient is undefined there.

Gradient validation and biased MD
---------------------------------

From this directory, with the existing source build::

    sh build-plugin.sh
    sh run-md.sh --output md-repeat

``run-md.sh`` checks gradients before starting MD. Output directories must not
already exist, to protect earlier trajectories. The default output directory is
``md-run/``. ``build.sh`` now also enables PLUMED's ``bias`` module. For a fresh
build, run ``sh build.sh`` first. The additional Python packages used here are::

    python -m pip install -r python-requirements.txt

The Python plumed package is only a runtime wrapper: ``bias_utils.py`` explicitly
selects the source-built kernel, which links to the pinned metatomic core.
``HEXAMER_PYTHON`` can select another Python executable with these packages.

``check_gradients.py`` compares PLUMED bias forces and virial against central
finite differences of its bias energy, E = CV^2/2, with displacement 1e-5 A and
strain increment 1e-5. It checks both CVs on all nine hexamers, all 54 Cartesian
coordinates and all nine strain components. This exercises the plugin gradient
layout, core execution, PLUMED force contraction, unit conversion, and virial
sign. A repeat evaluation with PLUMED length units set to nanometers verifies
force and virial equivalence to the angstrom setup. It also checks zero hydrogen forces, net force and torque, and translation
invariance. Detailed errors are in ``gradient-checks/results.json``.

Important pinned-version limitation: this revision's ``feature`` consistency
validator only permits outputs without gradients
(``metatomic-core/src/quantity/feature.rs``). Consequently ``CHECK_CONSISTENCY``
is omitted; enabling it fails even for correctly formed gradient blocks.
The metatomic source remains unchanged. Numerical validation through PLUMED is
performed explicitly instead; this does not claim the pinned schema validator
supports feature gradients.

The MD example starts from the prism, restores exact TIP3P intramolecular
geometry, and relaxes with rigid bonds to a projected force threshold of
0.02 eV/A. Two trajectories share identical initial positions, momenta and
thermostat random seed: an unbiased control and a harmonic Rg restraint with
KAPPA=10 eV/A^2, centered 0.15 A above the relaxed initial radius. Each runs
1000 steps at 0.5 fs (0.5 ps), using ASE rigid TIP3P and Langevin dynamics at
100 K with friction 0.01/fs. ``FixCom`` constrains center-of-mass motion and
Langevin's deprecated ``fixcm`` option is disabled. All three intramolecular
pair distances are constrained in each water molecule.

These are nonperiodic clusters. TIP3P's switching region starts at 19 A and
cutoff is 20 A; the script verifies all O-O distances remain below 19 A.
It checks finite energies/temperatures, rigid-bond errors below 1e-8 A, and
agreement of PLUMED CVs and bias energies with independent calculations at
every step. These short trajectories demonstrate force coupling and bias
response, not equilibrium sampling or a water-hexamer free-energy result.

In the completed 0.5 ps test, the mean Rg over the last 0.25 ps was
2.18016 A without bias and 2.28390 A with bias (target 2.28368 A). The maximum
rigid-bond deviation was below 1.9e-12 A. The largest finite-difference force
error was 1.4e-10 eV/A and strain-derivative error 5.8e-10 eV.

A default run creates ``md-run/`` containing ``summary.json`` and ``comparison.png``, plus, for each run,
``plumed.dat``, ``plumed.log``, ``COLVAR``, ``observables.csv``, and trajectories
in ASE and extended XYZ formats (saved every ten steps).

ASE implementations: https://docs.ase-lib.org/_modules/ase/calculators/tip3p.html
and https://docs.ase-lib.org/_modules/ase/calculators/plumed.html .

Pinned source and build
-----------------------

* Metatomic: https://github.com/EricBoittier/metatomic/tree/codex/plumed-new-api
  at ``b02b9ff3df6458cca21fa8690d229bbda97e6f78``.
* PLUMED: https://github.com/EricBoittier/plumed2/tree/EricBoittier/metatomic-capi-update
  at ``8226d8f682372750f46b3ed859c37e3dc0657803``.

The metatomic source is unchanged. ``plumed-core-api.patch`` adapts two calls
in the PLUMED fork to this core revision's argument-free ``requested_inputs``
and ``requested_pair_lists`` methods. ``Cargo.lock.pinned`` preserves the
existing local checkout's dependency resolution, including zip 8.6.0 (temporarily yanked when the initial CI ran),
which Cargo can still build from a lockfile.

``PLUGIN_LIBRARY`` is specified only on the first METATOMIC action: loading
the same plugin twice fails with this core revision. The second action reuses
the registered plugin by name. ``run.sh`` supplies a zero box explicitly to
PLUMED's XYZ reader, so comment lines can retain the isomer names.

Place the PLUMED checkout beside this repository. From this example directory::

    git clone https://github.com/EricBoittier/plumed2.git ../../../plumed2
    git -C ../../../plumed2 checkout 8226d8f682372750f46b3ed859c37e3dc0657803

Activate the dependency environment described below and set::

    export HEXAMER_DEPS="$CONDA_PREFIX"
    export HEXAMER_PYTHON="$CONDA_PREFIX/bin/python"
    sh build.sh

For micromamba, use the activated environment's prefix if CONDA_PREFIX is unset.
The build creates a separate ``metatomic-pinned`` worktree beside this repository
at the exact core commit, with the installed core in sibling ``install/``.
It builds the core, patched PLUMED and plugin on macOS and checks both source
commit IDs. CMake, make, Apple Clang and a recent Rust toolchain are required. Set ``HEXAMER_DEPS`` and
``HEXAMER_CARGO`` to override the dependency prefix and Cargo executable.
The dependency environment supplies metatensor and numerical libraries; its
prebuilt PLUMED and Torch libraries are not used by this example.

The initial environment was created with::

    micromamba create -n water-hexamer-plumed -c metatensor -c conda-forge \
        'plumed-metatomic=*=*nompi*' python=3.12 pytorch numpy pip

The source build uses Rust 1.98 through rustup and Apple's Clang. The older
Homebrew Cargo on PATH is insufficient for this branch.

Structure provenance
--------------------

Coordinates were copied from the local i-PI checkout at commit
``e1460fef7cb889472324c47b644a6f8340a474e2``:

* ``examples/temp/pes/geop/waterhexamers/{prism,cage,book}/init.xyz``
* ``examples/init_files/waterhexamer_ring.xyz``

Upstream: https://github.com/i-pi/i-pi . Coordinates are in angstrom, as
specified by the accompanying i-PI inputs. Only XYZ comment lines were
replaced; original coordinates and atom order were retained.


Book/prism CV scan and ASE atom plots
-----------------------------------

Run a fresh forward/reverse scan (output directory must not exist)::

    sh run-scan.sh cv-scan-repeat

The default scan output directory is ``cv-scan/``. The same pinned core plugin drives a
PLUMED ``MOVINGRESTRAINT`` on both Rg and mean oxygen coordination. The system
starts from rigid-TIP3P-relaxed book. Targets move to relaxed prism over 0.8 ps,
hold 0.2 ps, return to the book targets over 0.8 ps, then hold 0.2 ps. The timestep
is 0.5 fs, temperature 150 K, and spring constants are 30 eV/A^2 for Rg and
5 eV for coordination. The 2 ps scan is nonequilibrium steering, not a free
energy calculation. Neither CV uniquely identifies a water-hexamer isomer.

``scan_cv.py`` checks the recorded targets, bias energies, and rigid-water
constraints. It compares oxygen geometries to independently relaxed prism,
cage, book and ring references, minimizing RMSD over all 720 oxygen permutations
and proper rotations. ``analyze_scan_endpoints.py`` relaxes the end of each hold
without bias to test basin membership and compares geometric directed
hydrogen-bond graphs (H...O < 2.4 A and O-H...O angle > 150 degrees). Oxygen shape
and hydrogen-bond topology are reported separately.

``plot_scan.py`` uses ``ase.visualize.plot.plot_atoms`` to produce
``ase-storyboard.png`` and ``ase-trajectory.gif`` from the actual MD trajectory.
Frames are aligned by labeled oxygens to remove overall rotation and translation
for display only; a fixed camera and bounding box are used. Red/white atoms are
O/H, solid gray lines are covalent O-H bonds, and dashed blue lines show the
geometric hydrogen bonds. ``scan-diagnostics.png`` shows both moving targets
and the oxygen-reference RMSDs. Re-render existing outputs with::

    micromamba run -n water-hexamer-plumed python plot_scan.py cv-scan

Observed scan result: at 0.8 ps the oxygen geometry was 0.193 A RMSD from
relaxed prism. The 1 ps forward endpoint stayed prism-like after removing the
bias and minimizing (0.192 A oxygen RMSD), but its directed geometric H-bond
graph differs from the prism reference by two edges. Thus this is a prism-like
oxygen rearrangement, not a verified conversion to the identical proton-order
isomer. The reverse scan did not restore book: its final oxygen RMSD to book
was 0.977 A. The two compactness CVs alone do not select a unique structure or
ensure a reversible book/prism transformation in this short scan.

NumPy, PyTorch and JAX integration tests
--------------------------------------

Run ``sh test-backends.sh -q`` for 35 tests through the source-built PLUMED
API and the same pinned core. They cover values, biased forces, position and
strain derivatives, units, model lifetime and errors. The pinned revision
requires a test-only Python C ABI adapter; this does not test a production
Python model loader. See ``tests/README.rst`` for setup and coverage details.


Expanded reference library and visited regions
---------------------------------------------

``conformations.py`` reads labels directly from XYZ comments, so the driver,
gradient checks, scan and analysis share the same nine-reference library.
The five additions (bag, chair, book-2, boat-1 and boat-2) are taken from the
ForceBalance Interactions_CCSDT dataset at commit
``446533fc6b1f45b3d498f0fc5477283efce43ff6``:
https://github.com/leucinw/forcebalance-1.3.2/tree/446533fc6b1f45b3d498f0fc5477283efce43ff6/studies/015_amoeba_tinker/targets/Interactions_CCSDT/convert

The upstream XYZ comments attribute these geometries to Dahlke and Truhlar,
JPCA 2008. Original files, comments, SHA-256 digests and the atom-order mapping
are preserved in ``references/``. Only atom order changes when appending to
``hexamers.xyz``: each hydrogen is assigned to its nearest oxygen to produce
six OHH groups. Coordinates remain in angstrom without rounding or optimization.
The original four i-PI references remain unchanged. Ring and chair are retained
as separately sourced structures, not asserted to represent distinct basins.

``plot_scan.py`` now renders a nine-conformation ASE gallery and a four-panel
animation: atoms, Rg/time, PCA and a CV visitation histogram with synchronized
current-frame markers. ``analyze_sampling.py`` can independently regenerate
``sampling-regions.png``, ``sampling-pca.csv``, ``sampling-map.npz`` and
``sampling-summary.json`` from an existing trajectory::

    python analyze_sampling.py cv-scan
    python plot_scan.py cv-scan

PCA uses the 15 sorted O–O distances, in angstrom, centered without scaling.
It is fitted only to the trajectory; source references are projected afterward.
The distance descriptor is invariant to translation, rotation and oxygen
permutation, but loses connectivity and proton ordering. Explained variance,
centering vector and components are exported. The first two components explain
68.7% and 20.9% of the original 201-frame scan's variance.

The 24-by-24 Rg/coordination histogram reports fractions of saved frames per bin.
The reference bar chart counts the nearest source oxygen geometry only if its
permutation-minimized RMSD is at most 0.35 A; other frames are unassigned. This
illustrative threshold is not a validated basin boundary. These are descriptive
counts from a short driven trajectory, not equilibrium populations, a free-energy
surface, or evidence of visiting every reference conformation.

The gallery, PCA and sampling charts compare against **unrelaxed source
references**, preserving the published labels even if TIP3P relaxation changes
a structure. The scan's ``oxygen-rmsd.csv`` and endpoint analysis instead compare
against its **TIP3P-relaxed references**. These reference sets are intentionally
reported separately; the updated plots do not overwrite the original scan report.

A kernel-independent CI job checks provenance, OHH grouping, rigid-motion and
permutation invariance, trajectory-only PCA fitting and histogram normalization::

    python -m pytest analysis_tests -q

The native NumPy/PyTorch/JAX suite remains a separate source-build test, run with
``sh test-backends.sh -q``.

Nine-template targeted MD tour
-----------------------------

For a trajectory that deliberately visits all nine templates, rebuild the plugin
and run the bounded pipeline::

    sh build-plugin.sh
    sh run-tour.sh cv-tour

The existing pinned core and PLUMED source builds are required. The eight-minute
budget applies to gradient checks, MD, saved-frame verification and rendering;
installing dependencies or compiling the source stack is outside that budget.
``run-tour.sh`` enforces a 480-second wall-clock timeout, exits unsuccessfully
if verification fails, and writes measured timings to ``runtime.json``.
Performance depends on the host. Output directories must not already exist.

This replaces compactness steering with nine structure-specific CVs. For target
``k``, the plugin calculates the all-atom RMSD after optimizing translation and proper rotation::

    q_k = sqrt(min_{R in SO(3)} sum_i |(r_i - center) R
               - (r_i,target_k - center_target_k)|^2 / 18 + 1e-16)
    V_bias = K/2 * sum_k w_k(t) q_k^2

Lengths are in angstrom. A quaternion eigensolver determines the optimal
proper rotation; reflections are excluded. Gradients use the aligned displacement
and the envelope theorem, including the centering correction. The regularizer
makes the CV derivative defined at its minimum for nondegenerate alignments. The PLUMED feature output is numerically expressed in angstrom but,
like the original Rg feature, declares an empty quantity unit at this pinned
API revision. Position and homogeneous-strain gradients are analytic; hydrogens
now contribute. ``check_rmsd_gradients.py`` checks every Cartesian and strain
component for all nine targets through PLUMED, verifies the independently
calculated bias, and repeats in nanometer units.

The target XYZ path is supplied as ``MODEL=rmsd-target:/absolute/target.xyz``.
Only the first METATOMIC action loads ``hexamer.dylib``; subsequent actions reuse
the registered plugin. The core API, its pin, and the force propagation through
PLUMED are unchanged.

The sequence is book, book-2, bag, cage, prism, boat-1, boat-2, chair, ring.
Each source template retains its oxygen positions; O-H distances and H-O-H angles
are restored to rigid TIP3P geometry without minimizing the structure. Before MD,
water labels and equivalent hydrogen labels are matched to the preceding target
using proper rotations and whole-water geometry. These are fixed target label
choices, not coordinate replacements during dynamics. The trajectory is one
continuous integration with no resets, interpolated structures or restarts.

Default settings are 200 K, 0.5 fs, Langevin friction 0.05/fs, K=2000 eV/A^2,
400-step ramps and 400-step holds (6800 steps, 3.4 ps). The weights transfer
linearly from the preceding target to the next. All intramolecular TIP3P distances
and the center of mass are constrained. The strong, time-dependent restraint is
intended to enforce a tour; it is not a physical transition pathway or evidence
that the system spontaneously samples these structures.

A visit requires proper-rotation all-atom RMSD below 0.15 A to the fixed labeled
rigid template, oxygen RMSD below 0.15 A to the original source (minimized over
oxygen permutations), and an identical directed geometric hydrogen-bond graph.
``verify_tour.py`` requires at least three consecutive saved frames to meet all
three conditions within the final 50 fs of each hold. It independently checks
CVs, bias energies, rigid-bond errors and the TIP3P cutoff. The visit audit uses an independent NumPy SVD alignment, separate from the
plugin quaternion eigensolver; both exclude reflections.
Ring and chair remain separately sourced templates, not two claimed distinct
free-energy basins. No claim is made that the driven structures remain stable
after removing the restraint.

``plot_tour.py`` produces the ASE atom animation, a nine-panel storyboard of
actual hold-end frames, trajectory-fitted oxygen-distance PCA, and a visitation
histogram. The PCA and histogram have the same limitations described above:
they summarize the driven trajectory, not equilibrium populations or proton
order. The stricter visit audit is saved separately in ``verification.json``.

Measured result on an Apple M1 Pro (macOS): 308.7 seconds for MD and
369.0 seconds for the complete checks/MD/verification/rendering pipeline. All
nine targets passed the sustained-visit audit; each had 7–11 consecutive matching
saved frames in its final 50 fs. Hold-end all-atom RMSDs were 0.013–0.017 A,
with matching directed hydrogen-bond graphs. The maximum rigid-bond drift was
1.6e-12 A. These results and timings are included alongside the documentation
figures as ``tour-summary.json``, ``tour-verification.json`` and
``tour-runtime.json``.

Closed loop through relaxed representatives and permutation symmetry
-------------------------------------------------------------------

To connect the relaxed structures and return to the starting basin::

    sh run-minima.sh cv-minima-loop

This uses the same pinned core, RMSD plugin, thermostat and timestep. The route
is book -> bag -> cage -> prism -> boat-1 -> chair -> book-return: 5200 steps,
2.6 ps, with seven holds and six connecting ramps. No coordinates or atom
identities are reset during MD. ``run-minima.sh`` includes gradient checks,
trajectory verification, symmetry analysis and plots in its 480-second budget.
It uses the checked-in ``minima.xyz`` library.

The library was obtained by removing the bias from each hold-end frame of the
nine-template tour, retaining all rigid TIP3P bond constraints, and minimizing
in two stages to projected force thresholds 0.005 and 0.0001 eV/A. Under this
potential, book/book-2, boat-1/boat-2 and chair/ring converge to equivalent relaxed
structures. The nine seeds therefore give six distinct representatives. These
are seed-derived labels; relaxation can change the published geometry. This is
not an exhaustive inventory of TIP3P minima, and no Hessian or transition-state
classification is claimed.

``relaxed-seeds.xyz`` retains all nine quenched geometries. ``minima.json`` records
energies, the originating tour commit and hold times, family assignments and
matching operations. ``prepare_minima.py`` can regenerate the library from an
existing nine-template trajectory::

    python prepare_minima.py cv-tour-rmsd minima-preparation

Preparation is separate from the timed connection run. Constraints are explicitly
installed after reading coordinates: XYZ does not preserve ``FixBondLengths``.
The analysis tests recompute the stored energies and projected forces, verify
all duplicate-family assignments, and test known rotations, permutations and
reflections independently.

``symmetry.py`` enumerates all 720 permutations of the six intact waters and
all 64 choices of equivalent hydrogen exchange within them. For each of these
46,080 assignments it computes separate optimal proper and improper fits.
Two quenched seeds merge only if a **proper** all-atom fit has RMSD below 0.01 A
and their energies differ by less than 1e-5 eV. Reflected fits are recorded
separately; they are not silently merged with rotational equivalence.

The geometric matching set is the intact-water permutation group
``(S2)^6 semidirect S6``, optionally extended by inversion. This is not an
assignment of the effective molecular symmetry group: which rearrangements are
physically accessible depends on the potential and barriers. The distinction
between equivalent minima, permutation-inversion operations and feasible
rearrangements is discussed in `Wales's review
<https://arxiv.org/abs/physics/9810032>`_ and in the `water-hexamer tunneling study
<https://pubs.rsc.org/en/content/articlehtml/2025/cp/d5cp00155b>`_.

The loop tests a specific operation: after returning to book,
``analyze_loop.py`` aligns the final hold to the first hold while allowing water
and hydrogen permutations. It reports the best proper and improper RMSDs, the
fixed-label RMSD, and the net atom and water permutations in cycle notation.
The map sends final molecule/atom ``i`` to the initial reference site it occupies;
atom cycles use one-based OHH indices. ``closure-symmetry.png`` shows the two
actual frames with fixed A–F molecule identities. Those labels also appear in
the loop animation.

The route and target labeling are deliberately chosen before dynamics. A net
permutation demonstrates a driven connection between equivalent labeled copies;
it is not a spontaneously discovered tunneling mechanism. Connections between
different relaxed representatives are driven trajectory segments, not validated
minimum-energy paths. One closed loop does not establish a full symmetry group,
and best-fit operations need not be unique for a symmetric structure.

Observed closed-loop result: the complete checks/MD/analysis/rendering pipeline
took 285.2 seconds (4 min 45 s), including 235.5 seconds of MD. Every target,
including book-return, passed all eleven saved-frame checks in the final 50 fs
of its hold. The first/last book frames differ by 2.035 A with fixed labels,
but only 0.020 A after the proper fit and the net water permutation
``(A B F E)(C D)``. The best improper fit is 0.752 A. The full atom permutation
is recorded in ``connection-analysis.json``; its algebraic order is four.
This is one measured representative operation for the prescribed loop, not
a determination of the full effective symmetry group.
