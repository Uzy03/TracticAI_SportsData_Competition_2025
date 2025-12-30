"""Enrich CVAE processed dataset with conditions: shot_occurred + swing(in/out/short).

This script writes new pickle files so training does NOT need to read SoccerData tracking.csv
every epoch.

Input:
  - data/processed/cvae_{train,val,test}/data.pickle (list[dict])
  - data/processed/shot_{train,val,test}/data.pickle (list[dict]) for shot_occurred join
  - SoccerData/{year}_data/{match_id}/tracking.csv for swing inference

Output:
  - data/processed/cvae_{split}_enriched/data.pickle
  - metadata.json copied from original split if present

Swing encoding:
  0 = in-swing
  1 = out-swing
  2 = short corner
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def _tracking_path(soccerdata_dir: Path, match_id: str) -> Path:
    year = str(match_id)[:4]
    return soccerdata_dir / f"{year}_data" / str(match_id) / "tracking.csv"


def _load_ball_segment(
    tracking_csv: Path,
    frame: int,
    lookback_frames: int,
    window_frames: int,
) -> Optional[pd.DataFrame]:
    if not tracking_csv.exists():
        return None
    usecols = ["Frame", "HA", "SysTarget", "No", "X", "Y"]
    df = pd.read_csv(tracking_csv, usecols=usecols)
    ball = df[(df["HA"] == 0) | (df["No"] == 0) | (df["SysTarget"] == 0)].copy()
    lb = max(0, int(frame) - int(lookback_frames))
    end = int(frame) + int(window_frames)
    seg = ball[(ball["Frame"] >= lb) & (ball["Frame"] <= end)].copy()
    if len(seg) < 3:
        return None

    # to meters if looks like cm
    x = seg["X"].to_numpy(dtype=float)
    y = seg["Y"].to_numpy(dtype=float)
    if max(np.max(np.abs(x)), np.max(np.abs(y))) > 200.0:
        seg["X"] = x / 100.0
        seg["Y"] = y / 100.0
    return seg


def _nearest_corner(x: float, y: float) -> Tuple[float, float]:
    hl = FIELD_LENGTH / 2
    hw = FIELD_WIDTH / 2
    corners = [(-hl, -hw), (-hl, hw), (hl, -hw), (hl, hw)]
    best = min(corners, key=lambda c: (x - c[0]) ** 2 + (y - c[1]) ** 2)
    return float(best[0]), float(best[1])


def infer_swing_from_tracking(
    tracking_csv: Path,
    match_id: str,
    frame: int,
    lookback_frames: int = 300,
    window_frames: int = 120,
    corner_radius_m: float = 2.5,
    short_corner_threshold_m: float = 8.0,
    initial_frames_for_curve: int = 10,
) -> int:
    """Infer swing label from tracking.

    Returns: 0=in, 1=out, 2=short
    """
    seg = _load_ball_segment(tracking_csv, frame=frame, lookback_frames=lookback_frames, window_frames=window_frames)
    if seg is None:
        return 0  # default to in

    frames = seg["Frame"].to_numpy(dtype=int)
    x = seg["X"].to_numpy(dtype=float)
    y = seg["Y"].to_numpy(dtype=float)

    # find latest <= frame where ball is near ANY corner
    hl = FIELD_LENGTH / 2
    hw = FIELD_WIDTH / 2
    corners = np.array([[-hl, -hw], [-hl, hw], [hl, -hw], [hl, hw]], dtype=float)
    coords = np.stack([x, y], axis=1)
    d2 = ((coords[:, None, :] - corners[None, :, :]) ** 2).sum(axis=2)
    near_corner = np.any(d2 <= float(corner_radius_m) ** 2, axis=1)
    candidates = np.where(near_corner & (frames <= int(frame)))[0]
    if candidates.size > 0:
        k0 = int(candidates[-1])
    else:
        # no corner moment found -> treat as short-corner (ball already away)
        k0 = int(np.where(frames <= int(frame))[0][-1]) if np.any(frames <= int(frame)) else 0

    bx0 = float(x[k0])
    by0 = float(y[k0])
    cx, cy = _nearest_corner(bx0, by0)
    dist = float(((bx0 - cx) ** 2 + (by0 - cy) ** 2) ** 0.5)
    if dist > float(short_corner_threshold_m):
        return 2  # short

    # in/out by whether |y| moves toward centerline shortly after kick
    k1 = min(k0 + int(initial_frames_for_curve), len(y) - 1)
    y0 = float(y[k0])
    y1 = float(y[k1])
    if abs(y1) < abs(y0):
        return 0  # in
    if abs(y1) > abs(y0):
        return 1  # out
    # tie-breaker by direction relative to corner side
    if cy >= 0:
        return 0 if y1 < y0 else 1
    return 0 if y1 > y0 else 1


def build_shot_map(shot_samples: List[dict]) -> Dict[Tuple[str, int], int]:
    m: Dict[Tuple[str, int], int] = {}
    for s in shot_samples:
        mid = str(s.get("match_id"))
        fr = int(s.get("frame"))
        so = int(s.get("shot_occurred", 0))
        m[(mid, fr)] = so
    return m


def enrich_split(
    split: str,
    processed_dir: Path,
    soccerdata_dir: Path,
    out_dir: Path,
    lookback_frames: int,
    window_frames: int,
    corner_radius_m: float,
    short_corner_threshold_m: float,
) -> None:
    cvae_path = processed_dir / f"cvae_{split}" / "data.pickle"
    shot_path = processed_dir / f"shot_{split}" / "data.pickle"
    if not cvae_path.exists():
        raise FileNotFoundError(cvae_path)
    if not shot_path.exists():
        raise FileNotFoundError(shot_path)

    cvae_samples = pickle.load(open(cvae_path, "rb"))
    shot_samples = pickle.load(open(shot_path, "rb"))
    if not isinstance(cvae_samples, list) or not isinstance(shot_samples, list):
        raise ValueError("Expected list[dict] pickle format for both cvae_* and shot_*")

    shot_map = build_shot_map(shot_samples)
    missing_shot = 0
    missing_tracking = 0

    out_samples: List[dict] = []
    for s in cvae_samples:
        s2 = dict(s)
        mid = str(s2.get("match_id"))
        fr = int(s2.get("frame"))

        so = shot_map.get((mid, fr), None)
        if so is None:
            missing_shot += 1
            so = 0
        s2["shot_occurred"] = int(so)

        tracking_csv = _tracking_path(soccerdata_dir, mid)
        if not tracking_csv.exists():
            missing_tracking += 1
        swing = infer_swing_from_tracking(
            tracking_csv=tracking_csv,
            match_id=mid,
            frame=fr,
            lookback_frames=lookback_frames,
            window_frames=window_frames,
            corner_radius_m=corner_radius_m,
            short_corner_threshold_m=short_corner_threshold_m,
        )
        s2["swing"] = int(swing)

        out_samples.append(s2)

    split_out = out_dir / f"cvae_{split}_enriched"
    split_out.mkdir(parents=True, exist_ok=True)
    with open(split_out / "data.pickle", "wb") as f:
        pickle.dump(out_samples, f)

    # copy metadata if present
    meta_in = processed_dir / f"cvae_{split}" / "metadata.json"
    if meta_in.exists():
        meta = json.load(open(meta_in, "r"))
        meta["enriched"] = True
        meta["condition_fields"] = ["shot_occurred", "swing", "receiver_id"]
        meta["swing_encoding"] = {"in": 0, "out": 1, "short": 2}
        json.dump(meta, open(split_out / "metadata.json", "w"), ensure_ascii=False, indent=2)

    print(
        f"[{split}] wrote {len(out_samples)} samples -> {split_out}/data.pickle | "
        f"missing_shot={missing_shot}, missing_tracking={missing_tracking}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=str, default="data/processed")
    ap.add_argument("--soccerdata-dir", type=str, default="SoccerData")
    ap.add_argument("--out-dir", type=str, default="data/processed")
    ap.add_argument("--lookback-frames", type=int, default=300)
    ap.add_argument("--window-frames", type=int, default=120)
    ap.add_argument("--corner-radius-m", type=float, default=2.5)
    ap.add_argument("--short-threshold-m", type=float, default=8.0)
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    soccerdata_dir = Path(args.soccerdata_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "val", "test"]:
        enrich_split(
            split=split,
            processed_dir=processed_dir,
            soccerdata_dir=soccerdata_dir,
            out_dir=out_dir,
            lookback_frames=args.lookback_frames,
            window_frames=args.window_frames,
            corner_radius_m=args.corner_radius_m,
            short_corner_threshold_m=args.short_threshold_m,
        )


if __name__ == "__main__":
    main()


