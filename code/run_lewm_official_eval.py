#!/usr/bin/env python
"""Run LeWM evaluation from a local converted official checkpoint.

This bypasses `stable_worldmodel.wm.utils.load_pretrained()`, which otherwise
may try to resolve the policy name via Hugging Face even when a local object
checkpoint already exists.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate LeWM from a local `_object.ckpt` file."
    )
    parser.add_argument(
        "--lewm-root",
        type=Path,
        default=root / "lewm_base",
        help="Path to the local `lewm_base` source tree.",
    )
    parser.add_argument(
        "--config-name",
        default="reacher",
        help="Eval config name under `lewm_base/config/eval/`.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Absolute path to the local `_object.ckpt` file.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for the loaded policy.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Additional Hydra-style overrides, e.g. eval.num_eval=10.",
    )
    return parser.parse_args()


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def main() -> None:
    args = parse_args()
    lewm_root = args.lewm_root.expanduser().resolve()
    policy_path = args.policy_path.expanduser().resolve()

    if not lewm_root.is_dir():
        raise FileNotFoundError(f"LeWM source tree not found: {lewm_root}")
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_path}")

    sys.path.insert(0, str(lewm_root))

    config_dir = lewm_root / "config" / "eval"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    model = torch.load(policy_path, map_location="cpu", weights_only=False)
    model = model.to(args.device)
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )

    results_path = policy_path.parent

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    print(metrics)

    output_path = results_path / cfg.output.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")
        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    main()
