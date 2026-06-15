#!/usr/bin/env bash
# Usage: source scripts/activate-metatomic-env.sh
#
# Puts the metatomic-torch conda env ahead of micromamba on PATH without
# calling conda activate (which is slow when micromamba is also initialized).

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source this script instead of executing it:" >&2
    echo "  source scripts/activate-metatomic-env.sh" >&2
    exit 1
fi

ENV_NAME="${METATOMIC_CONDA_ENV:-metatomic-torch}"
CONDA_PREFIX="${METATOMIC_CONDA_PREFIX:-${HOME}/.conda/envs/${ENV_NAME}}"

if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "conda env not found at ${CONDA_PREFIX}" >&2
    echo "run ./scripts/create-conda-env.sh first" >&2
    return 1
fi

export CONDA_PREFIX
export CONDA_DEFAULT_ENV="${ENV_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTHONHOME PYTHONPATH

if [[ -f "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_PREFIX}/etc/conda/activate.d/metatomic-cuda-host.sh"
fi

echo "Using ${CONDA_PREFIX}/bin/python ($("${CONDA_PREFIX}/bin/python" --version))"
echo "METATOMIC_CUDA_HOST_COMPILER=${METATOMIC_CUDA_HOST_COMPILER:-unset}"
