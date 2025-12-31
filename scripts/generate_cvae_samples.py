"""Generate tactic samples from a trained CVAE model.

This script:
- Loads a CVAE checkpoint (default: best.ckpt under checkpoint_dir in config)
- Loads one sample from a CVAE dataset (pickle)
- Generates multiple samples via prior sampling (CVAEGenerator.generate)
- Saves outputs to a pickle (and optional denormalized arrays)

Notes on normalization in this repo:
- x,y are expected to be normalized to roughly [0,1]
- vx,vy are expected to be normalized by /70.0 (see ReceiverSchema + CVAESchema)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

from tacticai.dataio import CVAEDataset
from tacticai.modules import get_device, set_seed
from tacticai.train.train_cvae import create_model


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
MAX_VEL = 70.0


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def denormalize_states(x: np.ndarray) -> np.ndarray:
    """Denormalize [*,N,4] where (x,y,vx,vy) normalized in this repo."""
    y = np.array(x, dtype=np.float32, copy=True)
    y[..., 0] = y[..., 0] * FIELD_LENGTH
    y[..., 1] = y[..., 1] * FIELD_WIDTH
    y[..., 2] = y[..., 2] * MAX_VEL
    y[..., 3] = y[..., 3] * MAX_VEL
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="CVAE config yaml")
    ap.add_argument("--checkpoint", default="", type=str, help="Path to CVAE ckpt (default: <checkpoint_dir>/best.ckpt)")
    ap.add_argument(
        "--data-path",
        default="",
        type=str,
        help="Path to CVAE dataset pickle (default: config.data.test_path)",
    )
    ap.add_argument("--index", type=int, default=0, help="Sample index in dataset")
    ap.add_argument("--num-samples", type=int, default=10, help="Number of generated samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    ap.add_argument("--out", type=str, default="", help="Output pickle path")
    ap.add_argument("--denormalize", action="store_true", help="Also save denormalized outputs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(args.seed))
    device = get_device(args.device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.get("checkpoint_dir", "checkpoints")) / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data_path = args.data_path or cfg["data"]["test_path"]
    ds = CVAEDataset(data_path, file_format=cfg["data"].get("format", "pickle"))
    if len(ds) == 0:
        raise ValueError(f"Dataset is empty: {data_path}")
    if not (0 <= int(args.index) < len(ds)):
        raise IndexError(f"index out of range: {args.index} (len={len(ds)})")

    # Load sample
    input_data, targets = ds[int(args.index)]
    x = input_data["x"].to(device)
    edge_index = input_data["edge_index"].to(device)
    batch = input_data["batch"].to(device)
    cond = input_data["conditions"].to(device).unsqueeze(0)  # [1,C]
    edge_attr = input_data.get("edge_attr", None)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # Load model
    model = create_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        samples = model.generate(
            x=x,
            edge_index=edge_index,
            batch=batch,
            conditions=cond,
            num_samples=int(args.num_samples),
            edge_attr=edge_attr,
        )  # [B=1,S,N,4]

    out_path = Path(args.out) if args.out else (Path(cfg.get("log_dir", "runs")) / "generated" / f"cvae_gen_idx{args.index}_S{args.num_samples}.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_obj: Dict[str, Any] = {
        "config": cfg,
        "config_path": args.config,
        "checkpoint_path": str(ckpt_path),
        "data_path": str(data_path),
        "sample_index": int(args.index),
        "conditions": cond.detach().cpu().numpy().astype(np.float32),
        "x_input": x.detach().cpu().numpy().astype(np.float32),
        "edge_index": edge_index.detach().cpu().numpy().astype(np.int64),
        "edge_attr": edge_attr.detach().cpu().numpy().astype(np.float32) if edge_attr is not None else None,
        "targets": targets.detach().cpu().numpy().astype(np.float32),
        "generated": samples.detach().cpu().numpy().astype(np.float32),
    }
    if args.denormalize:
        out_obj["targets_denorm"] = denormalize_states(out_obj["targets"].reshape(1, -1, 4)).reshape(-1, 4)
        out_obj["generated_denorm"] = denormalize_states(out_obj["generated"])

    with open(out_path, "wb") as f:
        pickle.dump(out_obj, f)

    print(f"Saved: {out_path}")
    print(f"generated shape: {out_obj['generated'].shape}  (B,S,N,4)")


if __name__ == "__main__":
    main()


