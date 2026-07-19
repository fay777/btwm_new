# LeWM DMC Baseline

This repository includes a dedicated launch path for running LeWorldModel
(LeWM) as a DMC baseline without mixing it into the DreamerV3/BTWM online RL
runner.

## Scope

- Supported family: `dmc`
- Currently wired task: `reacher` / `reacher_easy`
- Entry points:
  - `code/run_lewm_experiment.sh`
  - `code/run_lewm_dmc.sh`

LeWM must still be reported as a `cross-framework comparison`, not as a
same-framework matched baseline, because it uses an offline dataset plus a
planner rather than online environment interaction.

## Required environment

LeWM depends on packages that are not part of the DreamerV3 stack:

- `stable-worldmodel[train,env]==0.1.0`
- `stable-pretraining`
- `datasets<3`
- `pyarrow==20.0.0`
- `hdf5plugin`
- `einops`
- `lightning`

The recommended Linux setup uses a separate `lewm` environment. See:

- [experiments/lewm-venv-linux-zh.md](E:/课题组2025-2026/具身智能/btwm-7.7/experiments/lewm-venv-linux-zh.md:1)

## Required data

LeWM training expects an HDF5 dataset.

For the current DMC hook:

- task: `reacher`
- file on disk: `reacher.h5`
- recommended directory:
  `/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/`

The launcher now passes the dataset as a local HDF5 path:

- `"$LEWM_DATASET_DIR/reacher.h5"`

So `LEWM_DATASET_DIR` should point directly to the folder that contains
`reacher.h5`.

## Example

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7

export LEWM_PYTHON=python
export LEWM_STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export LEWM_DATASET_DIR=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets
export GPU=5
export TASK=reacher
export SEED=0
export LEWM_BATCH_SIZE=32
export LEWM_MAX_EPOCHS=5

bash code/run_lewm_dmc.sh
```

Outputs follow the same repository convention as other launchers:

- logs -> `logs/`
- run names -> `runs/lewm_dmc_<task>_seed<seed>`

The cache/checkpoint root is still controlled by `STABLEWM_HOME`, but the
training input should come from the explicit dataset directory above.

## Official Checkpoint Eval

For the main BTWM vs LeWM comparison, prefer the official `reacher`
checkpoint and evaluate it locally from an absolute checkpoint path.

Why:

- the original `lewm_base/eval.py` path uses
  `swm.wm.utils.load_pretrained(cfg.policy)`
- in practice, that may interpret `policy=...` as a Hugging Face repo id
- this can trigger an unnecessary remote download even when a local
  `_object.ckpt` already exists

To avoid that, this repository provides two helper scripts:

- `code/convert_lewm_hf_ckpt.py`
- `code/run_lewm_official_eval.py`

Recommended server flow:

1. Download the Hugging Face checkpoint to a local cache directory.
2. Convert `weights.pt + config.json` into a local `_object.ckpt`.
3. Prepare the expected eval dataset path
   `datasets/dmc/reacher_random.h5`.
4. Run `code/run_lewm_official_eval.py` from the local absolute checkpoint.

Example:

```bash
conda activate lewm
cd /home/zhangpeiying01/lfy/btwm-7.7

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export HF_ENDPOINT=https://hf-mirror.com

hf download quentinll/lewm-reacher --local-dir "$STABLEWM_HOME/hf_reacher"

python code/convert_lewm_hf_ckpt.py \
  --src "$STABLEWM_HOME/hf_reacher" \
  --out "$STABLEWM_HOME/reacher/lewm_object.ckpt" \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base

mkdir -p "$STABLEWM_HOME/datasets/dmc"
ln -sf \
  "$STABLEWM_HOME/datasets/reacher.h5" \
  "$STABLEWM_HOME/datasets/dmc/reacher_random.h5"

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5

python code/run_lewm_official_eval.py \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base \
  --config-name reacher \
  --policy-path "$STABLEWM_HOME/reacher/lewm_object.ckpt"
```

Confirmed result on July 18, 2026:

- task: `reacher`
- eval count: `50`
- success rate: `80.0`

For the full Chinese migration guide, see:

- [experiments/lewm-official-eval-linux-zh.md](E:/课题组2025-2026/具身智能/btwm-7.7/experiments/lewm-official-eval-linux-zh.md:1)
