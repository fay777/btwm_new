#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=dreamerv3 DOMAIN=atari TASK=pong GPU=${GPU:-0}
exec "$SCRIPT_DIR/run_experiment.sh" "$@"
