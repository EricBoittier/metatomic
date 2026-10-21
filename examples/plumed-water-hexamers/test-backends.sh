#!/bin/sh
set -eu
cd "$(dirname "$0")"
python_exe="${HEXAMER_PYTHON:-python3}"
# Avoid unrelated globally installed pytest plugins in this native integration test.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$python_exe" -m pytest -c tests/pytest.ini tests "$@"
