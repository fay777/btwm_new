#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJ=${PROJ:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
MODEL=${MODEL:?Set MODEL to dreamerv3 or btwm}
DOMAIN=${DOMAIN:?Set DOMAIN to atari, dmc, or crafter}
TASK=${TASK:?Set TASK, for example pong, walker_walk, or reward}
GPU=${GPU:-0}
SEED=${SEED:-0}
RUN_STEPS=${RUN_STEPS:-1000000}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
RUN_TAG=${RUN_TAG:-}
PARAM_MATCHED=${PARAM_MATCHED:-1}
BTWM_VARIANT=${BTWM_VARIANT:-v2}
INV_LOSS_WEIGHT=${INV_LOSS_WEIGHT:-0.05}
INV_NUM_BINS=${INV_NUM_BINS:-21}

case "$BTWM_VARIANT" in
  v2)
    INV_HEAD_LOSS_WEIGHT=${INV_HEAD_LOSS_WEIGHT:-0.0}
    INV_POLICY_WEIGHT=${INV_POLICY_WEIGHT:-0.0}
    INV_CONFIDENCE_GATING=${INV_CONFIDENCE_GATING:-False}
    ;;
  v3)
    INV_HEAD_LOSS_WEIGHT=${INV_HEAD_LOSS_WEIGHT:-0.02}
    INV_POLICY_WEIGHT=${INV_POLICY_WEIGHT:-1.0}
    INV_CONFIDENCE_GATING=${INV_CONFIDENCE_GATING:-True}
    ;;
  *)
    echo "Unsupported BTWM_VARIANT: $BTWM_VARIANT (expected v2 or v3)" >&2
    exit 2
    ;;
esac

case "$MODEL" in
  dreamerv3|btwm) ;;
  *) echo "Unsupported MODEL: $MODEL" >&2; exit 2 ;;
esac

case "$DOMAIN" in
  atari)
    CONFIG=atari100k
    TASK_NAME=atari100k_${TASK}
    REPLAY_SIZE=${REPLAY_SIZE:-500000}
    ;;
  dmc)
    CONFIG=dmc_vision
    TASK_NAME=dmc_${TASK}
    REPLAY_SIZE=${REPLAY_SIZE:-300000}
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    ;;
  crafter)
    CONFIG=crafter
    TASK_NAME=crafter_${TASK}
    REPLAY_SIZE=${REPLAY_SIZE:-300000}
    ;;
  *)
    echo "Unsupported DOMAIN: $DOMAIN" >&2
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
mkdir -p "$RUN_DIR" "$LOG_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    -i "$GPU" 2>/dev/null | head -n 1 | tr -d ' ' || true)
  if [[ "$FREE_MB" =~ ^[0-9]+$ ]] && (( FREE_MB < MIN_FREE_MB )); then
    echo "GPU $GPU has ${FREE_MB} MiB free; require ${MIN_FREE_MB} MiB." >&2
    exit 3
  fi
fi

source "$PROJ/venv_dv3/bin/activate"
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$PROJ${PYTHONPATH:+:$PYTHONPATH}

ARGS=(
  python "$PROJ/dreamerv3/dreamerv3/main.py"
  --logdir "$RUN_DIR"
  --configs "$CONFIG"
  --task "$TASK_NAME"
  --seed "$SEED"
  --run.steps "$RUN_STEPS"
  --replay.size "$REPLAY_SIZE"
)

if [[ "$MODEL" == btwm ]]; then
  ARGS+=(
    --agent.use_btwm True
    --agent.inv_loss_weight "$INV_LOSS_WEIGHT"
    --agent.inv_head_loss_weight "$INV_HEAD_LOSS_WEIGHT"
    --agent.inv_num_bins "$INV_NUM_BINS"
    --agent.inv_confidence_gating "$INV_CONFIDENCE_GATING"
    --agent.inv_policy_weight "$INV_POLICY_WEIGHT"
  )
elif [[ "$PARAM_MATCHED" == 1 ]]; then
  # Keep architecture, parameter count, and forward compute identical to
  # BTWM. A zero auxiliary weight leaves the DreamerV3 objective unchanged.
  ARGS+=(
    --agent.use_btwm True
    --agent.inv_loss_weight 0.0
    --agent.inv_head_loss_weight 0.0
    --agent.inv_num_bins "$INV_NUM_BINS"
    --agent.inv_confidence_gating "$INV_CONFIDENCE_GATING"
    --agent.inv_policy_weight 0.0
  )
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] model=$MODEL domain=$DOMAIN task=$TASK_NAME"
  echo "seed=$SEED gpu=$GPU steps=$RUN_STEPS logdir=$RUN_DIR"
  echo "variant=$BTWM_VARIANT param_matched=$PARAM_MATCHED inv_weight=$INV_LOSS_WEIGHT head_weight=$INV_HEAD_LOSS_WEIGHT inv_bins=$INV_NUM_BINS confidence_gating=$INV_CONFIDENCE_GATING policy_weight=$INV_POLICY_WEIGHT"
  printf 'command:'
  printf ' %q' "${ARGS[@]}" "$@"
  printf '\n'
} | tee -a "$LOG"

if [[ ${DRY_RUN:-0} == 1 ]]; then
  exit 0
fi

"${ARGS[@]}" "$@" 2>&1 | tee -a "$LOG"
