Water hexamers through the metatomic core API
============================================

This example evaluates two analytic collective variables on prism, cage, book,
and ring water hexamers (18 atoms each). It loads a C++ plugin through the
metatomic core API. It does not use TorchScript or a trained potential.

This is an experimental documentation draft for the new core API. The build
instructions below target macOS and exact development revisions, not the
released TorchScript interface. Generated trajectories and figures are not
committed; the scripts reproduce them.

After completing the source build below, run from this directory::

    sh run.sh

The source-built PLUMED writes ``COLVAR`` and ``driver.log``. ``verify.py``
compares all eight CV values against independent Python calculations, with
an absolute tolerance of 1e-8. The first COLVAR column is a frame time/index,
not a physical MD trajectory. Frames are prism, cage, book, ring, in that order.

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
strain increment 1e-5. It checks both CVs on all four hexamers, all 54 Cartesian
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
existing local checkout's dependency resolution, including yanked zip 8.6.0,
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
