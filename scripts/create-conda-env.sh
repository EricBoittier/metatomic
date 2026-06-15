#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${METATOMIC_CONDA_ENV:-metatomic-torch}"
ENV_FILE="${ROOT}/environment.yml"
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
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
install -m 0755 "${ACTIVATE_HOOK}" "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh"

cat <<EOF

Environment '${ENV_NAME}' is ready.

  conda activate ${ENV_NAME}

On GPU nodes, load your CUDA module first if needed, then activate the env.
METATOMIC_CUDA_HOST_COMPILER is set automatically to the conda GCC 13 g++.

Install metatomic-torch from a checkout (editable installs are not supported):

  pip install --no-deps --no-build-isolation --check-build-dependencies python/metatomic_torch

Run tests (same as CI/tox):

  tox -e torch-tests-cxx,torch-tests

EOF
