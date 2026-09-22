# Metatomic ecosystem regression tests

These tests run models from the metatomic ecosystem that were **exported by
previous versions of metatomic**, and check that they still load and still
produce the same outputs.

`tests/regressions.py` discovers every `references/*/input.json`, runs that
model on the structure in the file, and compares each listed output to the
`.mts` file named there. In-repo examples (Lennard-Jones, the ASE calculator)
are not in this suite. They stay in `python/metatomic_torch/tests`, run with
`tox -e torch-tests`.

### Coverage

| Case | Structure | Outputs checked |
| --- | --- | --- |
| `references/pet-mad-s-v1.5.0` | 8 Si atoms, periodic | `energy` (positions and strain gradients), `non_conservative_force`, `non_conservative_stress`, `feature` |
| `references/pet-mad-dos-v1.0` | same Si cell | `mtt::dos`, `feature` |
| `references/pet-mad-featurizer-v1.0.0` | same Si cell | `feature` |
| `references/flashmd-pet-omatpes-v2-16fs` | same Si cell, plus `momenta.mts` | `positions`, `momenta`, `features` |
| `references/shiftml3-model-1` | one water molecule, non-periodic | `mtt::cs_iso`, `feature` |

There is no separately published "PET 1.6" metatomic archive. The PET export in
this suite is PET-MAD 1.5.0.

Not checked here:

- auxiliary outputs (`mtt::aux::*`)
- a second structure, `selected_atoms`, or a GPU device
- gradients, other than energy positions and strain on `pet-mad-s-v1.5.0`

`references/pet-mad-featurizer-v1.0.0/input.json` sets `"check_consistency": false`
because that export declares `float64` and returns `float32`. FlashMD reads
`momenta` off the system (`"inputs"` in its `input.json`) even though
`requested_inputs()` does not list it.

### Running

From `regtests/`, with this checkout's `metatomic.torch` installed and the
model files either published or already in `cache/<sha256>.pt`:

```bash
python -m pytest tests/regressions.py                 # every case
python -m pytest tests/regressions.py -k shiftml3     # one case
```

`tox -e regtests` installs the checkout and runs the same file, but it first
runs `python export-models.py --check`. That check fails while a model's `url`
in `models.lock` is still null. Publish with `--upload` (below) before relying
on tox. A local `cache/` hit is enough for pytest itself.

### Adding a model

`models.json` says *what* to build: the packages a model needs, and the code
that writes it out. Replace `<model-name>` with the name of the model.

```json
"<model-name>": {
    "dependencies": ["upet"],
    "source": "import upet; upet.save_upet(model='pet-mad', size='s', version='1.5.0')"
}
```

The `source` runs in an empty directory and must write exactly one `.pt` file;
its name does not matter.

```bash
python export-models.py --list
python export-models.py <model-name>
python export-models.py <model-name> --relock     # re-resolve the dependencies
```

This creates a virtual environment under `cache/venvs/`, installs the
dependencies in it, runs the ``source``, stores the model in
`build/<sha256>.pt`, and records in `models.lock` the exact version of *every*
package that was installed.

Once a model is in `models.lock`, building it again reuses those pinned versions
rather than resolving them afresh, so the same model can be rebuilt later even
as the upstream projects move on. Pass `--relock` to deliberately move to a
newer set of dependencies.

The next step is to upload the model so it can be used for tests. This requires
write access to the https://huggingface.co/metatensor/metatomic-regtests
repository. The address of the uploaded model is recorded in `models.lock`.

```bash
python export-models.py <model-name> --upload
```

### Adding a test case

Create `references/<case>/input.json`:

```json
{
  "model": "pet-mad-s-v1.5.0",
  "length_unit": "angstrom",
  "systems": [
    {
      "types": [14, ...],
      "positions": [
        [1.0, 2.0, 3.0],
        [...]
      ],
      "cell": [
        [9.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 9.0]
      ],
      "pbc": [true, false, true]
    }
  ],
  "selected_atoms": null,
  "outputs": {
    "energy": {
      "unit": "eV",
      "sample_kind": "system",
      "gradients": ["positions", "strain"],
      "reference": "energy.mts",
      "rtol": 1e-6,
      "atol": 1e-6
    }
  }
}

```

Then generate the reference outputs:

```bash
python update-references.py --list
python update-references.py <name-of-case>
```
