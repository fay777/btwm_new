#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=btwm DOMAIN=crafter TASK=reward GPU=${GPU:-7}
exec "$SCRIPT_DIR/run_experiment.sh" "$@"
