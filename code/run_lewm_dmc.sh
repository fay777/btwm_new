#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=lewm DOMAIN=dmc TASK=${TASK:-reacher} GPU=${GPU:-0}
exec "$SCRIPT_DIR/run_lewm_experiment.sh" "$@"
