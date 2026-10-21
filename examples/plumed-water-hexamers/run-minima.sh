#!/bin/sh
# Requires the pinned source build and a rebuilt hexamer plugin.
set -eu
cd "$(dirname "$0")"
exec "${HEXAMER_PYTHON:-python}" - "${1:-cv-minima-loop}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
import time

started = time.monotonic()
out = Path(sys.argv[1]).resolve()
for command in [
    ['check_rmsd_gradients.py'],
    ['tour_cv.py', '--minima', '--output', str(out)],
    ['verify_tour.py', str(out)],
    ['analyze_loop.py', str(out)],
    ['plot_tour.py', str(out)],
]:
    remaining = 480 - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError('Eight-minute pipeline budget exhausted')
    subprocess.run([sys.executable, *command], check=True, timeout=remaining)
runtime = json.loads((out / 'runtime.json').read_text())
runtime['complete_pipeline_seconds'] = time.monotonic() - started
(out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
print(json.dumps(runtime, indent=2))
PY
