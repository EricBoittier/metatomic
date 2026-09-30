# metatomic-torch plugin for metatomic-core

A plugin for the metatomic C API that runs TorchScript `AtomisticModel` files (`*.pt`,
exported with metatomic-torch, e.g. PET-MAD and PET-OMol), so engines built on
metatomic-core can use existing models. It is the "plugin to run existing (legacy)
TorchScript models from C++" item of the metatomic-core roadmap.

What it does (`src/torch_plugin.cpp`):

* loads the model with `metatomic_torch::load_atomistic_model` and declares its
  capabilities, pair lists and requested inputs (`charge`, `spin_multiplicity`, ...) in
  metatomic-core terms, in the model's own units and dtype; deprecated output names of
  legacy models (`features`, `non_conservative_forces`) are not declared;
* converts each system (DLPack tensors), its pair lists and its custom data to
  metatomic-torch, runs the model on `device`, and returns the outputs as metatensor-core
  tensor maps;
* for outputs requested with gradients, multiplies positions and cell by a per-system strain
  and returns `positions` and `strain` gradients of the total from autograd, attached to the
  first sample of each system;
* loads PyTorch's CUDA linear algebra once, so engines evaluating from several threads do
  not race on first use;
* reads pair vectors and custom data in the model dtype (float32 for PET-OMol), and converts
  labels without re-checking their uniqueness, which metatensor-core guarantees.

metatomic-core converts the systems to the model's length unit before calling the plugin,
and the outputs from the model's units to the requested ones afterwards.

## Building

The plugin, metatomic-core and metatensor-torch must share one metatensor-core, as they
all end up in the same process. With released packages:

```bash
uv venv .venv-mtaplugin --python 3.12
uv pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install metatensor-core==0.2.5 metatensor-torch==0.10.6 metatomic-torch==0.1.18

V=.venv-mtaplugin/lib/python3.12/site-packages
# metatomic-core against that metatensor-core
cmake -S metatomic-core -B build-core -DCMAKE_PREFIX_PATH="$V/metatensor/lib/cmake" \
    -DCMAKE_INSTALL_PREFIX=$PWD/core-prefix
cmake --build build-core --target install
# the plugin
cmake -S metatomic-torch-plugin -B build-plugin -DCMAKE_PREFIX_PATH="$V/torch/share/cmake;\
$V/metatensor/lib/cmake;$V/metatensor_torch/torch-2.13/lib/cmake;\
$V/metatomic/torch/torch-2.13/lib/cmake;$PWD/core-prefix"
cmake --build build-plugin     # -> libmetatomic_torch_plugin.so
```

nlohmann_json (>= 3.11) must be findable, and CUDA builds of torch need
`-DCMAKE_CUDA_COMPILER` and `-DCMAKE_CUDA_ARCHITECTURES`.

## Using it

Load the plugin, then load the `.pt` file: `mta_load_plugin("libmetatomic_torch_plugin.so")`
and `mta_load_model("model.pt", options, NULL, &model)`. Load options (JSON object of
strings):

* `device`: `cuda`, `cpu`, ...; default CUDA when available, or the `METATOMIC_TORCH_DEVICE`
  environment variable;
* `threads`: torch's CPU threads, or `METATOMIC_TORCH_THREADS` (default: torch's choice);
* `extensions_directory`.

The plugin also uses torch's dynamic-shape JIT fusion (as LAMMPS and GROMACS do for these
models). With `METATOMIC_TORCH_TIMER` set, it prints the mean time of its phases (inputs,
forward, backward and outputs) every 1000 calls; that synchronizes the device at each phase.

In GROMACS (C-API force provider): `metatomic-model = model.pt` and
`metatomic-extensions = /path/to/libmetatomic_torch_plugin.so`.

## Checked

PET-MAD xs v1.5.0 on a cyclic peptide (87 ML atoms, ONIOM, in 6,486 atoms of water),
GROMACS C API + this plugin against GROMACS's own libtorch path, same inputs:

| | C API + plugin | libtorch path |
|---|---|---|
| ML energy, step 0 | -54934.011719 kJ/mol | -54934.011719 kJ/mol |
| ML energy, step 20 | -54966.808594 kJ/mol | -54966.808594 kJ/mol |
| total energy, steps 0 -> 20 | -123989.66 -> -123989.95 | -123990.13 -> -123990.26 |
| ms/step, 1000 steps, 8 pinned threads, RTX 4070 Ti SUPER (4 runs) | 15.40-15.47 | 15.00-15.11 |

Unpinned runs on the same machine vary by +-1.5 ms/step, so pin threads (`-pin on`) when
comparing. Torch's thread count (1 or 24) changes nothing here.

PET-OMol s v1.0.0 on hemoglobin, the four hemes as four ML sites (per-system `charge` -2 and
`spin_multiplicity` 5, float32 model, 51,402 atoms):

| | C API + plugin | libtorch path |
|---|---|---|
| ML energy | -162503.281250 kJ/mol | -162503.281250 kJ/mol |
| forces | max difference 0.010 kJ/mol/nm, for forces up to 5,370 | |
| NVE, 1 ps (dt 0.5 fs), total energy range / drift | 12.50 kJ/mol / 0.81 kJ/mol/ps | 12.62 / 1.76 |
