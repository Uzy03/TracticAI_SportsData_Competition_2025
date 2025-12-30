"""Utility functions for soccer field visualization.

This module provides functions for drawing soccer field and loading data
for the retrieval visualization system.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.graph_objects import Figure


# Field dimensions (standard FIFA field)
FIELD_LENGTH = 105.0  # meters
FIELD_WIDTH = 68.0    # meters
PENALTY_AREA_LENGTH = 16.5  # meters
PENALTY_AREA_WIDTH = 40.32  # meters
GOAL_AREA_LENGTH = 5.5      # meters
GOAL_AREA_WIDTH = 18.32     # meters
CENTER_CIRCLE_RADIUS = 9.15  # meters
CORNER_ARC_RADIUS = 1.0      # meters
GOAL_WIDTH = 7.32     # meters
GOAL_DEPTH = 2.0      # meters


def draw_soccer_field(fig: Figure, field_length: float = FIELD_LENGTH, field_width: float = FIELD_WIDTH) -> None:
    """Draw soccer field shapes on a Plotly figure.
    
    Args:
        fig: Plotly figure object to add shapes to
        field_length: Length of the field in meters (default: 105.0)
        field_width: Width of the field in meters (default: 68.0)
    """
    half_length = field_length / 2
    half_width = field_width / 2
    
    # Field background (green)
    fig.add_shape(
        type="rect",
        x0=-half_length, y0=-half_width,
        x1=half_length, y1=half_width,
        fillcolor="rgb(34, 139, 34)",  # Forest green
        line=dict(color="white", width=2),
        layer="below",
    )
    
    # Touch lines and goal lines (outer boundary)
    # Already drawn by the rectangle above, but we can add additional lines if needed
    
    # Center line
    fig.add_shape(
        type="line",
        x0=0, y0=-half_width,
        x1=0, y1=half_width,
        line=dict(color="white", width=2),
    )
    
    # Center circle
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-CENTER_CIRCLE_RADIUS, y0=-CENTER_CIRCLE_RADIUS,
        x1=CENTER_CIRCLE_RADIUS, y1=CENTER_CIRCLE_RADIUS,
        line=dict(color="white", width=2),
    )
    
    # Left penalty area
    fig.add_shape(
        type="rect",
        x0=-half_length, y0=-PENALTY_AREA_WIDTH / 2,
        x1=-half_length + PENALTY_AREA_LENGTH, y1=PENALTY_AREA_WIDTH / 2,
        line=dict(color="white", width=2),
    )
    
    # Left goal area
    fig.add_shape(
        type="rect",
        x0=-half_length, y0=-GOAL_AREA_WIDTH / 2,
        x1=-half_length + GOAL_AREA_LENGTH, y1=GOAL_AREA_WIDTH / 2,
        line=dict(color="white", width=2),
    )
    
    # Left penalty spot
    penalty_spot_x = -half_length + 11.0  # 11 meters from goal line
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=penalty_spot_x - 0.3, y0=-0.3,
        x1=penalty_spot_x + 0.3, y1=0.3,
        fillcolor="white",
        line=dict(color="white", width=1),
    )
    
    # Right penalty area
    fig.add_shape(
        type="rect",
        x0=half_length - PENALTY_AREA_LENGTH, y0=-PENALTY_AREA_WIDTH / 2,
        x1=half_length, y1=PENALTY_AREA_WIDTH / 2,
        line=dict(color="white", width=2),
    )
    
    # Right goal area
    fig.add_shape(
        type="rect",
        x0=half_length - GOAL_AREA_LENGTH, y0=-GOAL_AREA_WIDTH / 2,
        x1=half_length, y1=GOAL_AREA_WIDTH / 2,
        line=dict(color="white", width=2),
    )
    
    # Right penalty spot
    penalty_spot_x = half_length - 11.0
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=penalty_spot_x - 0.3, y0=-0.3,
        x1=penalty_spot_x + 0.3, y1=0.3,
        fillcolor="white",
        line=dict(color="white", width=1),
    )
    
    # Left goal
    goal_y_start = -GOAL_WIDTH / 2
    goal_y_end = GOAL_WIDTH / 2
    fig.add_shape(
        type="rect",
        x0=-half_length - GOAL_DEPTH, y0=goal_y_start,
        x1=-half_length, y1=goal_y_end,
        line=dict(color="white", width=3),
    )
    
    # Right goal
    fig.add_shape(
        type="rect",
        x0=half_length, y0=goal_y_start,
        x1=half_length + GOAL_DEPTH, y1=goal_y_end,
        line=dict(color="white", width=3),
    )
    
    # Corner arcs (important for CK analysis)
    corner_arc_points = 20
    
    # Top-left corner
    theta = np.linspace(np.pi, np.pi / 2, corner_arc_points)
    arc_x = -half_length + CORNER_ARC_RADIUS * np.cos(theta)
    arc_y = half_width - CORNER_ARC_RADIUS * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(color="white", width=2),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Top-right corner
    theta = np.linspace(np.pi / 2, 0, corner_arc_points)
    arc_x = half_length - CORNER_ARC_RADIUS * np.cos(theta)
    arc_y = half_width - CORNER_ARC_RADIUS * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(color="white", width=2),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Bottom-left corner
    theta = np.linspace(-np.pi, -np.pi / 2, corner_arc_points)
    arc_x = -half_length + CORNER_ARC_RADIUS * np.cos(theta)
    arc_y = -half_width - CORNER_ARC_RADIUS * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(color="white", width=2),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Bottom-right corner
    theta = np.linspace(-np.pi / 2, 0, corner_arc_points)
    arc_x = half_length - CORNER_ARC_RADIUS * np.cos(theta)
    arc_y = -half_width - CORNER_ARC_RADIUS * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y,
        mode='lines',
        line=dict(color="white", width=2),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Update layout
    fig.update_layout(
        xaxis=dict(
            range=[-half_length - 5, half_length + 5],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[-half_width - 5, half_width + 5],
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="rgb(34, 139, 34)",
        paper_bgcolor="white",
    )


def load_raw_sample_data(
    sample: Dict[str, Any],
    schema: Optional[Any] = None,
) -> pd.DataFrame:
    """Load raw sample data from ReceiverDataset sample into DataFrame format.
    
    Args:
        sample: Raw data sample dictionary from ReceiverDataset
        schema: Optional schema object (for field dimensions)
    
    Returns:
        DataFrame with columns: player_id, team_id, x, y, vx, vy, ball
    """
    # Extract data from sample
    x = np.array(sample.get("x", []))
    y = np.array(sample.get("y", []))
    x_raw = x.copy()
    y_raw = y.copy()
    team = np.array(sample.get("team", []))
    ball = np.array(sample.get("ball", []))
    mask = np.array(sample.get("mask", []))
    # Some samples may not have an explicit ball flag. Use kicker position as proxy.
    if (ball.size > 0) and (ball.sum() == 0) and ("kicker_idx" in sample) and (sample["kicker_idx"] is not None):
        try:
            k = int(sample["kicker_idx"])
            if 0 <= k < ball.size:
                ball = ball.astype(float)
                ball[k] = 1.0
        except Exception:
            pass
    
    # Get velocities if available
    vx = np.array(sample.get("vx", [0.0] * len(x)))
    vy = np.array(sample.get("vy", [0.0] * len(y)))
    
    # Ensure all arrays have the same length
    lengths = [len(x), len(y), len(team), len(ball)]
    if mask.size > 0:
        lengths.append(len(mask))
    min_len = min(lengths) if lengths else 0
    if len(vx) < min_len:
        vx = np.pad(vx, (0, min_len - len(vx)), mode='constant')
    if len(vy) < min_len:
        vy = np.pad(vy, (0, min_len - len(vy)), mode='constant')
    
    x = x[:min_len]
    y = y[:min_len]
    x_raw = x_raw[:min_len]
    y_raw = y_raw[:min_len]
    team = team[:min_len]
    ball = ball[:min_len]
    vx = vx[:min_len]
    vy = vy[:min_len]
    if mask.size > 0:
        mask = mask[:min_len]
    else:
        mask = np.ones(min_len, dtype=float)
    
    # Heuristic: many preprocessors store positions normalized to [0, 1] (or slightly beyond).
    # Plotly field is drawn in meters with origin at center, so convert if values look normalized.
    def _looks_normalized(arr: np.ndarray) -> bool:
        if arr.size == 0:
            return False
        a = np.asarray(arr).reshape(-1)
        # Typical ranges we observed in this project: ~0.3..1.0 (sometimes slightly >1)
        return (a.min() >= -0.5) and (a.max() <= 1.5)

    normalized = False
    if _looks_normalized(x) and _looks_normalized(y):
        normalized = True
        # Convert normalized [0,1] -> meters centered at (0,0)
        x = (x - 0.5) * FIELD_LENGTH
        y = (y - 0.5) * FIELD_WIDTH
        # NOTE: Do NOT auto-scale velocities here. In this project vx/vy may already be in
        # real-world units (and can be large). Vector visualization is handled by clipping
        # in the plotting function.

    # Some preprocess pipelines pad/mask a missing player by placing them exactly at a corner.
    # In this dataset, we observed a frequent dummy at (-FIELD_LENGTH/2, -FIELD_WIDTH/2).
    # Also, we observed a center placeholder at normalized (0.5,0.5) -> meters (0,0).
    # Keep the row (to preserve indices) but mark it so visualization can ignore it.
    is_corner_dummy = (
        np.isclose(x, -FIELD_LENGTH / 2, atol=1e-9)
        & np.isclose(y, -FIELD_WIDTH / 2, atol=1e-9)
        & np.isclose(vx, 0.0, atol=1e-6)
        & np.isclose(vy, 0.0, atol=1e-6)
        & (ball.astype(float) < 0.5)
    )
    is_center_dummy = np.zeros(min_len, dtype=bool)
    if normalized:
        is_center_dummy = (
            np.isclose(x_raw, 0.5, atol=1e-9)
            & np.isclose(y_raw, 0.5, atol=1e-9)
            & np.isclose(vx, 0.0, atol=1e-6)
            & np.isclose(vy, 0.0, atol=1e-6)
            & (ball.astype(float) < 0.5)
        )

    is_masked_out = (mask.astype(float) < 0.5)
    is_dummy = is_corner_dummy | is_center_dummy | is_masked_out

    # Create DataFrame
    df = pd.DataFrame({
        'orig_idx': np.arange(min_len, dtype=int),
        'player_id': [f'player_{i}' for i in range(min_len)],
        'team_id': team,
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'ball': ball.astype(float),
        'mask': mask.astype(float),
        'is_dummy': is_dummy.astype(bool),
    })

    # Attach metadata (useful for loading tracking-based ball trajectory)
    if "match_id" in sample:
        df["match_id"] = sample["match_id"]
    if "frame" in sample:
        df["frame"] = sample["frame"]
    
    # If ball possession is indicated, mark that player
    if ball.sum() > 0:
        ball_idx = int(np.argmax(ball))
        df.loc[ball_idx, 'player_id'] = 'ball'
    
    return df


def get_ball_trajectory(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ball trajectory from DataFrame.
    
    Note: For single-frame data (CK snapshot), this returns a single point.
    For multi-frame data, this would return the full trajectory.
    
    Args:
        df: DataFrame with ball data
    
    Returns:
        Tuple of (x_coords, y_coords) arrays
    """
    ball_data = df[df['ball'] > 0.5]
    if len(ball_data) > 0:
        return ball_data['x'].values, ball_data['y'].values
    else:
        # If no ball data, return empty arrays
        return np.array([]), np.array([])


def _nearest_corner(x: float, y: float) -> Tuple[float, float]:
    """Return nearest pitch corner (in meters) to a point."""
    half_length = FIELD_LENGTH / 2
    half_width = FIELD_WIDTH / 2
    corners = [
        (-half_length, -half_width),
        (-half_length, +half_width),
        (+half_length, -half_width),
        (+half_length, +half_width),
    ]
    best = min(corners, key=lambda c: (x - c[0]) ** 2 + (y - c[1]) ** 2)
    return float(best[0]), float(best[1])


def infer_swing_mode(
    df: pd.DataFrame,
    mode: str = "auto",
) -> str:
    """Infer in/out-swing mode.

    Data does not contain true ball trajectory, so this is a heuristic.
    """
    m = (mode or "auto").lower().replace("_", "-")
    if m in {"in", "in-swing", "inswing"}:
        return "in"
    if m in {"out", "out-swing", "outswing"}:
        return "out"
    if m in {"none", "off", "disable", "disabled"}:
        return "none"

    # Auto heuristic:
    # - Use the kicker position (ball marker) to determine the corner (top/bottom).
    # - Compare attacking team center-of-mass Y to corner Y:
    #   if attackers are more "inside" (closer to 0), assume in-swing; else out-swing.
    ball_rows = df[df["ball"] > 0.5]
    if len(ball_rows) == 0:
        return "in"
    bx = float(ball_rows.iloc[0]["x"])
    by = float(ball_rows.iloc[0]["y"])
    _, cy = _nearest_corner(bx, by)

    # Determine attacking team label from kicker's team_id
    kicker_team = int(ball_rows.iloc[0]["team_id"]) if "team_id" in df.columns else 0
    attacking = df[df["team_id"] == kicker_team] if "team_id" in df.columns else df[df["team_id"] == 0]
    if len(attacking) == 0:
        return "in"
    ay = float(attacking["y"].mean())

    # Top corner: inside means smaller y than corner y.
    # Bottom corner: inside means larger y than corner y (since corner y is negative).
    if cy >= 0:
        return "in" if ay < cy * 0.7 else "out"
    return "in" if ay > cy * 0.7 else "out"


def get_ball_swing_arc(
    df: pd.DataFrame,
    mode: str = "auto",
    num_points: int = 40,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Create a stylized ball flight arc (in/out-swing) from nearest corner into the box.

    Returns (x_points, y_points, label). Empty arrays if not available/disabled.
    """
    swing = infer_swing_mode(df, mode=mode)
    if swing == "none":
        return np.array([]), np.array([]), "none"

    ball_rows = df[df["ball"] > 0.5]
    if len(ball_rows) == 0:
        return np.array([]), np.array([]), swing

    bx = float(ball_rows.iloc[0]["x"])
    by = float(ball_rows.iloc[0]["y"])
    # Start from the ball position, but use nearest corner only to decide "side"
    cx, cy = _nearest_corner(bx, by)
    # If ball is far from the corner, it's likely a short corner / restart, and a curved corner arc is misleading.
    dist_to_corner = float(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5)
    if dist_to_corner > 8.0:
        return np.array([]), np.array([]), swing

    # End point: into the penalty area near the goal side.
    half_length = FIELD_LENGTH / 2
    goal_side = 1.0 if cx > 0 else -1.0
    x2 = goal_side * (half_length - 12.0)  # ~12m inside from goal line
    y2 = 0.0

    # Control point: determines curvature (in-swing vs out-swing)
    # Make curvature direction very explicit:
    # - In-swing: bend toward the pitch centerline (y -> 0)
    # - Out-swing: bend toward the sideline (|y| increases)
    x1 = goal_side * (half_length - 6.0)
    sgn = 1.0 if cy >= 0 else -1.0
    delta = 12.0  # meters of lateral curvature for visual clarity
    if swing == "in":
        y1 = cy - sgn * delta  # toward center
    else:
        y1 = cy + sgn * delta  # toward sideline

    # Quadratic Bezier curve: P0=(ball), P1=(control), P2=(target)
    t = np.linspace(0.0, 1.0, num_points)
    x = (1 - t) ** 2 * bx + 2 * (1 - t) * t * x1 + t**2 * x2
    y = (1 - t) ** 2 * by + 2 * (1 - t) * t * y1 + t**2 * y2
    return x, y, swing


def get_short_pass_arrow(
    df: pd.DataFrame,
    receiver_idx: Optional[int] = None,
    short_corner_threshold_m: float = 8.0,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """If this looks like a short corner, return a short pass arrow (start,end).

    - Start: ball/kicker position
    - End: receiver_idx if provided; otherwise nearest same-team player (excluding kicker)
    """
    ball_rows = df[df["ball"] > 0.5]
    if len(ball_rows) == 0:
        return None, None

    # Kicker/ball position
    bx = float(ball_rows.iloc[0]["x"])
    by = float(ball_rows.iloc[0]["y"])
    cx, cy = _nearest_corner(bx, by)
    dist_to_corner = float(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5)
    if dist_to_corner <= float(short_corner_threshold_m):
        return None, None

    start = (bx, by)

    # Determine receiver
    if receiver_idx is not None:
        try:
            ridx = int(receiver_idx)
            if "orig_idx" in df.columns:
                hit = df[df["orig_idx"] == ridx]
                if len(hit) > 0:
                    end_row = hit.iloc[0]
                    return start, (float(end_row["x"]), float(end_row["y"]))
            else:
                if 0 <= ridx < len(df):
                    end_row = df.iloc[ridx]
                    return start, (float(end_row["x"]), float(end_row["y"]))
        except Exception:
            pass

    # Fallback: nearest same-team player (exclude kicker)
    kicker_team = int(ball_rows.iloc[0]["team_id"]) if "team_id" in df.columns else 0
    kicker_pos = np.array([bx, by], dtype=float)

    candidates = df[df["team_id"] == kicker_team].copy() if "team_id" in df.columns else df.copy()
    if "is_dummy" in candidates.columns:
        candidates = candidates[~candidates["is_dummy"]]
    if len(candidates) <= 1:
        return None, None

    # Compute distances, exclude the ball row itself by distance ~0
    coords = candidates[["x", "y"]].to_numpy(dtype=float)
    d = np.sqrt(((coords - kicker_pos) ** 2).sum(axis=1))
    d = np.where(d < 1e-6, np.inf, d)
    j = int(np.argmin(d))
    if not np.isfinite(d[j]):
        return None, None
    end = (float(coords[j, 0]), float(coords[j, 1]))
    return start, end


def load_ball_trajectory_from_tracking(
    match_id: str,
    frame: int,
    soccerdata_dir: str = "SoccerData",
    window_frames: int = 120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ball trajectory from SoccerData tracking.csv around (match_id, frame).

    SoccerData format (per README):
    - tracking.csv columns include: Frame, HA, SysTarget, No, X, Y
    - Ball rows are typically HA==0 and No==0 (SysTarget==0)
    - X,Y are in centimeters with origin at pitch center in many files.
    """
    try:
        mid = str(match_id)
        year = mid[:4]
        tracking_path = Path(soccerdata_dir) / f"{year}_data" / mid / "tracking.csv"
        if not tracking_path.exists():
            return np.array([]), np.array([])

        usecols = ["Frame", "HA", "SysTarget", "No", "X", "Y"]
        df = pd.read_csv(tracking_path, usecols=usecols)
        # Ball candidate rows (empirically: HA==0, No==0, SysTarget==0)
        ball = df[(df["HA"] == 0) | (df["No"] == 0) | (df["SysTarget"] == 0)]
        seg = ball[(ball["Frame"] >= int(frame)) & (ball["Frame"] <= int(frame) + int(window_frames))]
        if len(seg) < 2:
            return np.array([]), np.array([])

        x = seg["X"].to_numpy(dtype=float)
        y = seg["Y"].to_numpy(dtype=float)

        # Units heuristic: if values look like centimeters, convert to meters.
        if max(np.max(np.abs(x)), np.max(np.abs(y))) > 200.0:
            x = x / 100.0
            y = y / 100.0
        return x, y
    except Exception:
        return np.array([]), np.array([])


def infer_swing_from_ball_trajectory(x: np.ndarray, y: np.ndarray) -> str:
    """Infer in/out-swing from real ball trajectory.

    Heuristic:
    - Determine nearest corner at start.
    - If |y| moves toward 0 (centerline) shortly after kick -> in
      else -> out
    """
    if x is None or y is None or len(x) < 3 or len(y) < 3:
        return "none"
    x0 = float(x[0])
    y0 = float(y[0])
    _, cy = _nearest_corner(x0, y0)
    # Use first ~10 frames for "initial curve"
    n = min(10, len(y) - 1)
    y_start = float(y[0])
    y_end = float(y[n])

    # Compare absolute distance from centerline (y=0)
    if abs(y_end) < abs(y_start):
        return "in"
    if abs(y_end) > abs(y_start):
        return "out"
    # Tie-breaker: compare sign relative to corner
    if cy >= 0:
        return "in" if y_end < y_start else "out"
    return "in" if y_end > y_start else "out"


def get_emphasized_swing_arc_from_start(
    start_x: float,
    start_y: float,
    swing: str,
    num_points: int = 40,
    short_corner_threshold_m: float = 8.0,
    delta_m: float = 18.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a visually emphasized swing arc from a start position.

    This is meant for *visual clarity*, not physical accuracy.
    - in-swing: bends more toward the goal and toward the pitch centerline
    - out-swing: bends away from the goal and toward the sideline
    """
    s = (swing or "in").lower()
    if s not in {"in", "out"}:
        return np.array([]), np.array([])

    bx = float(start_x)
    by = float(start_y)
    cx, cy = _nearest_corner(bx, by)
    dist_to_corner = float(((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5)
    if dist_to_corner > float(short_corner_threshold_m):
        return np.array([]), np.array([])

    half_length = FIELD_LENGTH / 2
    goal_side = 1.0 if cx > 0 else -1.0

    # End point (toward/away from goal for clarity)
    if s == "in":
        x2 = goal_side * (half_length - 6.0)   # closer to goal
    else:
        x2 = goal_side * (half_length - 16.0)  # farther from goal
    y2 = 0.0

    # Control point: strong lateral curvature
    x1 = goal_side * (half_length - 8.0)
    sgn = 1.0 if cy >= 0 else -1.0
    if s == "in":
        y1 = cy - sgn * float(delta_m)  # toward centerline
    else:
        y1 = cy + sgn * float(delta_m)  # toward sideline

    t = np.linspace(0.0, 1.0, int(num_points))
    x = (1 - t) ** 2 * bx + 2 * (1 - t) * t * x1 + t**2 * x2
    y = (1 - t) ** 2 * by + 2 * (1 - t) * t * y1 + t**2 * y2
    return x, y

