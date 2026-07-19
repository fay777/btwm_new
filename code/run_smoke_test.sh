#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
FULL_TRAIN=${FULL_TRAIN:-0}

cd "$PROJ"

echo "[1/4] Checking syntax for key modules"
"$PYTHON_BIN" -m py_compile \
  code/btwm_head.py \
  dreamerv3/dreamerv3/agent.py \
  dreamerv3/dreamerv3/rssm.py

echo "[2/4] Running BTWM unit tests if dependencies are available"
if "$PYTHON_BIN" -c "import elements" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m unittest code.test_btwm_head
else
  echo "Skipping unit tests: Python module 'elements' is not installed in the current environment."
fi

echo "[3/4] Verifying experiment command generation"
PROJ="$PROJ" DRY_RUN=1 MODEL=btwm DOMAIN=dmc TASK=walker_walk RUN_TAG=smoke \
  bash "$PROJ/code/run_experiment.sh"
PROJ="$PROJ" DRY_RUN=1 MODEL=dreamerv3 DOMAIN=crafter TASK=reward RUN_TAG=smoke \
  bash "$PROJ/code/run_experiment.sh"

if [[ "$FULL_TRAIN" == "1" ]]; then
  echo "[4/4] Running optional CPU debug training smoke"
  "$PYTHON_BIN" "$PROJ/dreamerv3/dreamerv3/main.py" \
    --logdir "$PROJ/runs/smoke_debug" \
    --configs debug \
    --task dummy_disc \
    --run.steps 200
else
  echo "[4/4] Skipping optional CPU debug training smoke (set FULL_TRAIN=1 to enable)"
fi

echo "Smoke test completed successfully."
