#!/bin/sh
set -eu
cd "$(dirname "$0")"
# Source-built PLUMED in the sibling worktree, never the conda executable.
PLUMED_EXE="${PLUMED_EXE:-../../../plumed2/src/lib/plumed}"
"$PLUMED_EXE" driver --plumed plumed.dat --ixyz hexamers.xyz --length-units A --box 0,0,0 --timestep 0.001 > driver.log 2>&1
python3 verify.py
