"""Build CVAE dataset for Corner Kicks from processed_ck receiver data.

Why:
- SoccerData (tracking.csv) may not be available inside the training container.
- processed_ck contains CK-only samples with fields like kicker_idx and shot_occurred.

We generate new pickle files that contain:
  - x,y,vx,vy,height,weight,team,ball,mask,match_id,frame
  - shot_occurred (condition)
  - receiver_node_index (condition; derived from target_idx/receiver_node_index/receiver_id)
  - swing (condition; 0=in, 1=out, 2=short) inferred heuristically without tracking

Swing heuristic (no tracking/foot available):
  - If kicker is farther than short_threshold_m from nearest corner -> short (2)
  - Else decide in/out by receiver y relative to centerline:
      in  if |y_receiver| < |y_corner| - margin_m
      out otherwise
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def _to_m(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    looks_norm = (
        x.size > 0
        and y.size > 0
        and (x.min() >= -0.5) and (x.max() <= 1.5)
        and (y.min() >= -0.5) and (y.max() <= 1.5)
    )
    if looks_norm:
        return (x - 0.5) * FIELD_LENGTH, (y - 0.5) * FIELD_WIDTH
    return x, y


def _nearest_corner(xm: float, ym: float) -> Tuple[float, float]:
    hl = FIELD_LENGTH / 2
    hw = FIELD_WIDTH / 2
    corners = [(-hl, -hw), (-hl, hw), (hl, -hw), (hl, hw)]
    cx, cy = min(corners, key=lambda c: (xm - c[0]) ** 2 + (ym - c[1]) ** 2)
    return float(cx), float(cy)


def _get_receiver_node_index(sample: Dict[str, Any]) -> Optional[int]:
    for k in ("receiver_node_index", "target_idx", "receiver_id"):
        if k in sample and sample[k] is not None:
            try:
                v = int(sample[k])
                if 0 <= v <= 21:
                    return v
            except Exception:
                pass
    return None


def _get_kicker_idx(sample: Dict[str, Any]) -> Optional[int]:
    if "kicker_idx" in sample and sample["kicker_idx"] is not None:
        try:
            k = int(sample["kicker_idx"])
            if 0 <= k <= 21:
                return k
        except Exception:
            pass
    # fallback: ball flag
    ball = np.asarray(sample.get("ball", []), dtype=float)
    if ball.size == 22 and float(ball.sum()) > 0:
        return int(ball.argmax())
    return None


def infer_swing_from_sample(
    sample: Dict[str, Any],
    short_threshold_m: float = 8.0,
    in_margin_m: float = 2.0,
) -> int:
    """Infer swing label without tracking.

    Returns: 0=in, 1=out, 2=short
    """
    kicker_idx = _get_kicker_idx(sample)
    if kicker_idx is None:
        return 0

    x = np.asarray(sample.get("x", []), dtype=float)
    y = np.asarray(sample.get("y", []), dtype=float)
    if x.size < kicker_idx + 1 or y.size < kicker_idx + 1:
        return 0

    x_m, y_m = _to_m(x, y)
    kx, ky = float(x_m[kicker_idx]), float(y_m[kicker_idx])
    cx, cy = _nearest_corner(kx, ky)
    dist = float(((kx - cx) ** 2 + (ky - cy) ** 2) ** 0.5)
    if dist > float(short_threshold_m):
        return 2

    ridx = _get_receiver_node_index(sample)
    if ridx is None or ridx >= len(y_m):
        return 0
    ry = float(y_m[ridx])

    # in: toward centerline (|y| decreases), out: toward sideline
    if abs(ry) < abs(cy) - float(in_margin_m):
        return 0
    return 1


def _load_samples(p: Path) -> List[Dict[str, Any]]:
    obj = pickle.load(open(p, "rb"))
    if isinstance(obj, dict) and "samples" in obj:
        return obj["samples"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported pickle format: {p}")


def build_split(
    split: str,
    in_dir: Path,
    out_dir: Path,
    short_threshold_m: float,
    in_margin_m: float,
) -> None:
    src = in_dir / f"receiver_{split}" / "data.pickle"
    meta = in_dir / f"receiver_{split}" / "metadata.json"
    samples = _load_samples(src)

    out_samples: List[Dict[str, Any]] = []
    for s in samples:
        s2 = dict(s)
        # Ensure conditions exist
        if "shot_occurred" not in s2:
            s2["shot_occurred"] = 0
        ridx = _get_receiver_node_index(s2)
        if ridx is not None:
            s2["receiver_node_index"] = int(ridx)
        s2["swing"] = int(infer_swing_from_sample(s2, short_threshold_m=short_threshold_m, in_margin_m=in_margin_m))
        out_samples.append(s2)

    dst_dir = out_dir / f"cvae_ck_{split}"
    dst_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_dir / "data.pickle", "wb") as f:
        pickle.dump(out_samples, f)

    if meta.exists():
        m = json.load(open(meta, "r"))
    else:
        m = {}
    m["source"] = str(src)
    m["generated_by"] = "scripts/build_cvae_ck_dataset.py"
    m["swing_encoding"] = {"in": 0, "out": 1, "short": 2}
    m["swing_heuristic"] = {"short_threshold_m": short_threshold_m, "in_margin_m": in_margin_m}
    json.dump(m, open(dst_dir / "metadata.json", "w"), ensure_ascii=False, indent=2)

    print(f"[{split}] wrote {len(out_samples)} -> {dst_dir}/data.pickle")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, default="data/processed_ck")
    ap.add_argument("--out-dir", type=str, default="data/processed_ck")
    ap.add_argument("--short-threshold-m", type=float, default=8.0)
    ap.add_argument("--in-margin-m", type=float, default=2.0)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        build_split(
            split=split,
            in_dir=in_dir,
            out_dir=out_dir,
            short_threshold_m=float(args.short_threshold_m),
            in_margin_m=float(args.in_margin_m),
        )


if __name__ == "__main__":
    main()


