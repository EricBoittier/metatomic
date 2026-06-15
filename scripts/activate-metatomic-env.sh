#!/usr/bin/env bash
# Usage: source scripts/activate-metatomic-env.sh
#
# Activates the metatomic-torch conda env and puts it ahead of micromamba on PATH.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source this script instead of executing it:" >&2
    echo "  source scripts/activate-metatomic-env.sh" >&2
    exit 1
fi

ENV_NAME="${METATOMIC_CONDA_ENV:-metatomic-torch}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found in PATH" >&2
    return 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if command -v micromamba >/dev/null 2>&1; then
    while [[ "${CONDA_DEFAULT_ENV:-}" == "base" || "${MAMBA_ROOT_PREFIX:-}" != "" ]]; do
        if ! micromamba deactivate 2>/dev/null; then
            break
        fi
    done
fi

conda activate "${ENV_NAME}"

# micromamba shell hooks can leave their bin directory ahead of the active env
export PATH="${CONDA_PREFIX}/bin:${PATH}"

if [[ -f "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh"
fi

echo "Using ${CONDA_PREFIX}/bin/python ($(python --version))"
echo "METATOMIC_CUDA_HOST_COMPILER=${METATOMIC_CUDA_HOST_COMPILER:-unset}"
