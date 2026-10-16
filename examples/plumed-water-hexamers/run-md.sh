#!/bin/sh
set -eu
cd "$(dirname "$0")"
python_exe="${HEXAMER_PYTHON:-python3}"
"$python_exe" check_gradients.py
"$python_exe" run_md.py "$@"
