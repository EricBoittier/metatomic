#!/bin/sh
set -eu
cd "$(dirname "$0")"
python_exe="${HEXAMER_PYTHON:-python3}"
output="${1:-cv-scan-repeat}"
"$python_exe" check_gradients.py
"$python_exe" scan_cv.py --output "$output"
"$python_exe" analyze_scan_endpoints.py "$output"
"$python_exe" plot_scan.py "$output"
