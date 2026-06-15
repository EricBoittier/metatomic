#!/usr/bin/env bash
# Set nvcc host compiler for CUDA 12.x when the system CXX is GCC 14+.
# Installed into $CONDA_PREFIX/etc/conda/activate.d/ by scripts/create-conda-env.sh
export METATOMIC_CUDA_HOST_COMPILER="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
# PyTorch CMake (e.g. metatomic-lj-test) reads CUDAHOSTCXX, not METATOMIC_CUDA_HOST_COMPILER
export CUDAHOSTCXX="${METATOMIC_CUDA_HOST_COMPILER}"
