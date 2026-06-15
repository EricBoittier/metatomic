#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${METATOMIC_CONDA_ENV:-metatomic-torch}"
ENV_FILE="${ROOT}/environment.yml"
PIP_REQUIREMENTS="${ROOT}/conda/requirements-pip.txt"
ACTIVATE_HOOK="${ROOT}/conda/activate-metatomic-cuda-host.sh"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Updating existing conda environment '${ENV_NAME}'"
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune "$@"
else
    echo "Creating conda environment '${ENV_NAME}'"
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}" "$@"
fi

CONDA_PREFIX="$(conda run -n "${ENV_NAME}" python -c "import os; print(os.environ['CONDA_PREFIX'])")"
ENV_PYTHON="${CONDA_PREFIX}/bin/python"

echo "Installing pip packages into ${ENV_NAME}"
"${ENV_PYTHON}" -m pip install -r "${PIP_REQUIREMENTS}"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
install -m 0755 "${ACTIVATE_HOOK}" "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh"

CONDA_BASE="$(conda info --base)"

cat <<EOF

Environment '${ENV_NAME}' is ready at ${CONDA_PREFIX}

Activate it (micromamba must not override conda on PATH):

  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate ${ENV_NAME}
  export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
  which python   # should be ${CONDA_PREFIX}/bin/python

Or use the helper script from the repo root:

  source scripts/activate-metatomic-env.sh

On GPU nodes, load your CUDA module first if needed, then activate the env.
METATOMIC_CUDA_HOST_COMPILER is set automatically to the conda GCC 13 g++.

Install metatomic-torch from a checkout (editable installs are not supported):

  pip install --no-deps --no-build-isolation --check-build-dependencies python/metatomic_torch

Run tests (same as CI/tox):

  tox -e torch-tests-cxx,torch-tests

EOF
