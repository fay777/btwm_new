#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJ=${PROJ:-$(cd -- "$SCRIPT_DIR/.." && pwd)}

MODEL=${MODEL:-lewm}
DOMAIN=${DOMAIN:?Set DOMAIN to dmc}
TASK=${TASK:?Set TASK, for example reacher}
GPU=${GPU:-0}
SEED=${SEED:-0}
RUN_TAG=${RUN_TAG:-}
MIN_FREE_MB=${MIN_FREE_MB:-16000}

LEWM_ROOT=${LEWM_ROOT:-$PROJ/lewm_base}
LEWM_PYTHON=${LEWM_PYTHON:-python}
LEWM_VENV=${LEWM_VENV:-}
LEWM_STABLEWM_HOME=${LEWM_STABLEWM_HOME:-$PROJ/runs/lewm_cache}
LEWM_DATASET_DIR=${LEWM_DATASET_DIR:-$LEWM_STABLEWM_HOME}
LEWM_MAX_EPOCHS=${LEWM_MAX_EPOCHS:-100}
LEWM_BATCH_SIZE=${LEWM_BATCH_SIZE:-128}
LEWM_DEVICES=${LEWM_DEVICES:-1}
LEWM_IMG_SIZE=${LEWM_IMG_SIZE:-224}

case "$MODEL" in
  lewm) ;;
  *) echo "Unsupported MODEL: $MODEL" >&2; exit 2 ;;
esac

case "$DOMAIN" in
  dmc) ;;
  *) echo "Unsupported DOMAIN: $DOMAIN (LeWM baseline currently wired for dmc only)" >&2; exit 2 ;;
esac

case "$TASK" in
  reacher|reacher_easy)
    DATA_CFG=dmc
    DATASET_NAME=${LEWM_DATASET_NAME:-$LEWM_DATASET_DIR/reacher.h5}
    ;;
  *)
    echo "Unsupported DMC task for LeWM baseline: $TASK" >&2
    echo "Expected one of: reacher, reacher_easy" >&2
    exit 2
    ;;
esac

RUN_NAME=${MODEL}_${DOMAIN}_${TASK}_seed${SEED}
if [[ -n "$RUN_TAG" ]]; then
  RUN_NAME=${RUN_NAME}_${RUN_TAG}
fi

RUN_DIR=$PROJ/runs/$RUN_NAME
LOG_DIR=$PROJ/logs
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG=$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log
mkdir -p "$RUN_DIR" "$LOG_DIR" "$LEWM_STABLEWM_HOME"

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -n 1 | tr -d ' ' || true)
  if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB < MIN_FREE_MB )); then
    echo "GPU $GPU has ${FREE_MB} MiB free; require ${MIN_FREE_MB} MiB." >&2
    exit 3
  fi
fi

if [[ -n "$LEWM_VENV" ]]; then
  # Optional isolated environment for LeWM and stable-worldmodel dependencies.
  source "$LEWM_VENV"
fi

export CUDA_VISIBLE_DEVICES=$GPU
export STABLEWM_HOME=$LEWM_STABLEWM_HOME
export LOCAL_DATASET_DIR=$LEWM_DATASET_DIR
export PYTHONPATH=$LEWM_ROOT${PYTHONPATH:+:$PYTHONPATH}

ARGS=(
  "$LEWM_PYTHON" "$LEWM_ROOT/train.py"
  "data=$DATA_CFG"
  "data.dataset.name=$DATASET_NAME"
  "trainer.max_epochs=$LEWM_MAX_EPOCHS"
  "trainer.devices=$LEWM_DEVICES"
  "loader.batch_size=$LEWM_BATCH_SIZE"
  "img_size=$LEWM_IMG_SIZE"
  "seed=$SEED"
  "subdir=$RUN_NAME"
  "output_model_name=$RUN_NAME"
)

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] model=$MODEL domain=$DOMAIN task=$TASK"
  echo "seed=$SEED gpu=$GPU logdir=$RUN_DIR"
  echo "stablewm_home=$STABLEWM_HOME dataset_dir=$LOCAL_DATASET_DIR dataset_name=$DATASET_NAME"
  printf 'command:'
  printf ' %q' "${ARGS[@]}" "$@"
  printf '\n'
  echo "note: LeWM is an offline world-model baseline. Report results separately from DreamerV3/BTWM online RL tables."
} | tee -a "$LOG"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

(
  cd "$LEWM_ROOT"
  "${ARGS[@]}" "$@"
) 2>&1 | tee -a "$LOG"
