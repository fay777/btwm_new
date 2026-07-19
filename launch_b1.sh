#!/usr/bin/env bash
set -euo pipefail

PROJ=/data2/liguoqi/habmm/hab_z/btwm
SEED=${SEED:-0}
GPU=${GPU:-0}
mkdir -p "$PROJ/runs/pids" "$PROJ/logs"

nohup env MODEL=btwm DOMAIN=atari TASK=pong SEED="$SEED" GPU="$GPU" \
  RUN_STEPS="${RUN_STEPS:-100000}" \
  "$PROJ/code/run_experiment.sh" \
  > "$PROJ/logs/launcher_btwm_atari_pong_seed${SEED}.log" 2>&1 &

PID=$!
echo "$PID" > "$PROJ/runs/pids/btwm_atari_pong_seed${SEED}.pid"
echo "Launched BTWM Atari 100k Pong seed=$SEED GPU=$GPU PID=$PID"
