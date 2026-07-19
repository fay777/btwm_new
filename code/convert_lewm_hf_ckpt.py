#!/usr/bin/env python
"""Convert a Hugging Face LeWM checkpoint into a local `_object.ckpt` file.

Example:
    python code/convert_lewm_hf_ckpt.py \
      --src /path/to/hf_reacher \
      --out /path/to/lewm_cache/reacher/lewm_object.ckpt
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import stable_pretraining as spt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face LeWM checkpoint directory containing "
            "`config.json` and `weights.pt` into a local `_object.ckpt` file."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Directory containing `config.json` and `weights.pt`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Target path for the converted `_object.ckpt` file.",
    )
    parser.add_argument(
        "--lewm-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "lewm_base",
        help="Path to the local `lewm_base` source tree.",
    )
    return parser.parse_args()


def import_lewm_modules(lewm_root: Path):
    sys.path.insert(0, str(lewm_root))
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    return JEPA, ARPredictor, Embedder, MLP


def make_mlp(cfg: dict, key: str, mlp_cls):
    mlp_cfg = cfg[key]
    return mlp_cls(
        input_dim=mlp_cfg["input_dim"],
        output_dim=mlp_cfg["output_dim"],
        hidden_dim=mlp_cfg["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )


def filter_kwargs(cls, raw_cfg: dict) -> dict:
    """Keep only kwargs supported by the local constructor."""

    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in raw_cfg.items() if k in valid}


def main() -> None:
    args = parse_args()
    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    lewm_root = args.lewm_root.expanduser().resolve()

    config_path = src / "config.json"
    weights_path = src / "weights.pt"

    if not lewm_root.is_dir():
        raise FileNotFoundError(f"LeWM source tree not found: {lewm_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing weight file: {weights_path}")

    JEPA, ARPredictor, Embedder, MLP = import_lewm_modules(lewm_root)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    predictor_cfg = filter_kwargs(ARPredictor, cfg["predictor"])
    action_encoder_cfg = filter_kwargs(Embedder, cfg["action_encoder"])

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_cfg),
        action_encoder=Embedder(**action_encoder_cfg),
        projector=make_mlp(cfg, "projector", MLP),
        pred_proj=make_mlp(cfg, "pred_proj", MLP),
    )

    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
