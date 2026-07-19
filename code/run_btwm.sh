#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=btwm DOMAIN=atari TASK=pong GPU=${GPU:-3}
exec "$SCRIPT_DIR/run_experiment.sh" "$@"
