"""Streamlit application for visualizing similar CK retrieval results.

This application provides an interactive interface to visualize search results
from the similar CK retrieval system.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
import yaml
import torch
from torch.utils.data import ConcatDataset
import math
import copy

# Proposed structural similarity (optional dependency)
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover
    linear_sum_assignment = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from my_method.retrieval import SimilarCKSearch, SimilarCKIndex
from my_method.dataio import ReceiverDataset
from my_method.modules import get_device

# Import utils (handle both relative and absolute imports)
try:
    from .utils import draw_soccer_field, load_raw_sample_data, get_ball_swing_arc
except ImportError:
    from my_method.retrieval.visualization.utils import draw_soccer_field, load_raw_sample_data, get_ball_swing_arc
try:
    from .utils import get_short_pass_arrow
except ImportError:
    from my_method.retrieval.visualization.utils import get_short_pass_arrow
try:
    from .utils import load_ball_trajectory_from_tracking, infer_swing_from_ball_trajectory
except ImportError:
    from my_method.retrieval.visualization.utils import load_ball_trajectory_from_tracking, infer_swing_from_ball_trajectory
try:
    from .utils import get_emphasized_swing_arc_from_start
except ImportError:
    from my_method.retrieval.visualization.utils import get_emphasized_swing_arc_from_start


def _get_raw_sample(dataset: Any, idx: int) -> Dict[str, Any]:
    """Get underlying raw dict sample for both ReceiverDataset and ConcatDataset[ReceiverDataset]."""
    if isinstance(dataset, ReceiverDataset):
        return dataset.data[idx]
    if isinstance(dataset, ConcatDataset):
        i = idx
        for ds in dataset.datasets:
            n = len(ds)
            if i < n:
                # ds is ReceiverDataset
                return ds.data[i]
            i -= n
        raise IndexError(idx)
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


# Page config
st.set_page_config(
    page_title="Similar CK Retrieval Visualization",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_search_system(config_path: str, checkpoint_path: Optional[str] = None):
    """Load search system (cached for performance)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device(config.get("device", "auto"))
    
    if checkpoint_path is None:
        d2_enabled = config.get("d2", {}).get("enabled", False)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        model_save_dir = config.get("model_save_dir", f"{checkpoint_dir}/receiver_shot")
        run_name = config.get("run_name", None)
        if run_name:
            model_save_dir = f"{model_save_dir}/{run_name}"
        if d2_enabled:
            checkpoint_path = f"{model_save_dir}/best_d2.ckpt"
        else:
            checkpoint_path = f"{model_save_dir}/best_no_d2.ckpt"

    # Helpful error message when checkpoint is missing
    ckpt_p = Path(str(checkpoint_path))
    if not ckpt_p.exists():
        parent = ckpt_p.parent.parent  # checkpoints/.../receiver_shot/<run_name>
        try:
            # list available checkpoints under checkpoints/my_method/receiver_shot/*
            base_dir = Path("checkpoints") / "my_method" / "receiver_shot"
            cand = sorted([str(p.as_posix()) for p in base_dir.glob("*/best_*.ckpt")])[:20]
        except Exception:
            cand = []
        msg = (
            f"Backbone checkpoint not found: {checkpoint_path}\n\n"
            f"ヒント: 選択中のconfigの run_name と checkpoints 配下のディレクトリ名が一致している必要があります。\n"
            f"例: baseline_stable を比較したい場合は、configも "
            f"`configs_my_method/multitask_receiver_shot_d2_baseline_stable.yaml` を選んでください。\n\n"
            f"見つかった候補（最大20件）:\n- " + "\n- ".join(cand) if cand else
            f"Backbone checkpoint not found: {checkpoint_path}"
        )
        raise FileNotFoundError(msg)
    
    search_system = SimilarCKSearch(
        backbone_checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    return search_system, config


@st.cache_resource
def load_index(index_path: str, embedding_dim: int):
    """Load search index (cached for performance)."""
    index = SimilarCKIndex(
        embedding_dim=embedding_dim,
        index_path=index_path,
    )
    index.load(index_path)
    return index


def plot_ck_snapshot(
    df: pd.DataFrame,
    title: str = "CK Snapshot",
    show_vectors: bool = True,
    vector_scale: float = 0.5,
    max_vector_len: float = 6.0,
    vector_offset_m: float = 1.0,
    min_vector_len: float = 0.0,
    show_ids: bool = False,
    show_ball_arc: bool = True,
    swing_mode: str = "auto",
    trajectory_source: str = "stylized",
    soccerdata_dir: str = "SoccerData",
    traj_window_frames: int = 120,
    emphasize_swing: bool = True,
    show_raw_tracking: bool = False,
    swing_curvature_m: float = 30.0,
    receiver_idx: Optional[int] = None,
) -> go.Figure:
    """Plot a single CK snapshot on the soccer field.
    
    Args:
        df: DataFrame with player data (columns: player_id, team_id, x, y, vx, vy, ball)
        title: Plot title
        show_vectors: Whether to show velocity vectors
        vector_scale: Scale factor for velocity vectors
        show_ids: Whether to show player IDs
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Draw soccer field
    draw_soccer_field(fig)
    
    # Determine attacking/defending team IDs.
    # Do NOT assume team_id==0 is always attacking: some samples flip.
    # Also guard against older dataframes that don't have is_dummy by filtering the known (0,0)->corner placeholder.
    vis_mask = (df["team_id"] == df["team_id"])
    if "is_dummy" in df.columns:
        vis_mask = vis_mask & (~df["is_dummy"])
    # Fallback dummy detection (corner placeholder)
    try:
        is_corner_dummy = (
            (df["ball"] < 0.5)
            & (df["vx"].abs() < 1e-6)
            & (df["vy"].abs() < 1e-6)
            & ((df["x"] + 52.5).abs() < 1e-6)
            & ((df["y"] + 34.0).abs() < 1e-6)
        )
        vis_mask = vis_mask & (~is_corner_dummy)
    except Exception:
        pass
    df_vis = df[vis_mask]

    ball_data = df_vis[df_vis['ball'] > 0.5]
    if len(ball_data) > 0:
        attacking_team_id = int(ball_data.iloc[0]['team_id'])
    else:
        attacking_team_id = 0

    unique_team_ids = [int(x) for x in df_vis['team_id'].dropna().unique().tolist()]
    defending_team_id = None
    for tid in unique_team_ids:
        if tid != attacking_team_id:
            defending_team_id = tid
            break
    if defending_team_id is None and attacking_team_id in (0, 1):
        defending_team_id = 1 - attacking_team_id
    if defending_team_id is None:
        defending_team_id = 1

    # Separate attacking and defending teams
    attacking = df_vis[df_vis['team_id'] == attacking_team_id]
    defending = df_vis[df_vis['team_id'] == defending_team_id]
    
    # Plot attacking team (red)
    if len(attacking) > 0:
        fig.add_trace(go.Scatter(
            x=attacking['x'],
            y=attacking['y'],
            mode='markers+text' if show_ids else 'markers',
            text=attacking['player_id'] if show_ids else None,
            textposition="top center",
            marker=dict(
                size=10,
                color='red',
                line=dict(width=2, color='darkred'),
            ),
            name='Attacking Team',
            hovertemplate='<b>%{text}</b><br>' +
                          'x: %{x:.2f}<br>' +
                          'y: %{y:.2f}<extra></extra>',
            texttemplate='%{text}' if show_ids else None,
        ))
    
    # Plot defending team (blue)
    if len(defending) > 0:
        fig.add_trace(go.Scatter(
            x=defending['x'],
            y=defending['y'],
            mode='markers+text' if show_ids else 'markers',
            text=defending['player_id'] if show_ids else None,
            textposition="top center",
            marker=dict(
                size=10,
                color='blue',
                line=dict(width=2, color='darkblue'),
            ),
            name='Defending Team',
            hovertemplate='<b>%{text}</b><br>' +
                          'x: %{x:.2f}<br>' +
                          'y: %{y:.2f}<extra></extra>',
            texttemplate='%{text}' if show_ids else None,
        ))
    
    # Compute ball marker position.
    # NOTE: In processed data, ball==1 is often "ball holder player", not true ball location.
    # When using tracking, prefer the tracking start position as the ball marker.
    ball_marker = None
    # Keep the actual ball-holder position (for short-corner decision), even if we snap the display marker.
    ball_holder_pos = None
    traj_x = traj_y = np.array([])
    swing = None
    if show_ball_arc and (trajectory_source or "stylized").lower() == "tracking":
        try:
            match_id = str(df["match_id"].iloc[0]) if "match_id" in df.columns else None
            frame = int(df["frame"].iloc[0]) if "frame" in df.columns else None
        except Exception:
            match_id, frame = None, None
        if match_id is not None and frame is not None:
            traj_x, traj_y = load_ball_trajectory_from_tracking(
                match_id=match_id,
                frame=frame,
                soccerdata_dir=soccerdata_dir,
                lookback_frames=300,
                window_frames=traj_window_frames,
            )
        if len(traj_x) > 0:
            ball_marker = (float(traj_x[0]), float(traj_y[0]))
            swing = infer_swing_from_ball_trajectory(traj_x, traj_y) if len(traj_x) > 1 else None

    if ball_marker is None and len(ball_data) > 0:
        ball_marker = (float(ball_data.iloc[0]["x"]), float(ball_data.iloc[0]["y"]))
    if len(ball_data) > 0:
        ball_holder_pos = (float(ball_data.iloc[0]["x"]), float(ball_data.iloc[0]["y"]))

    # If tracking is enabled but the detected start is far from a corner (common in short corners),
    # snap the displayed marker to the nearest corner for clarity.
    if (show_ball_arc and (trajectory_source or "stylized").lower() == "tracking") and (ball_marker is not None):
        bx, by = float(ball_marker[0]), float(ball_marker[1])
        hl = 105.0 / 2.0
        hw = 68.0 / 2.0
        corners = [(-hl, -hw), (-hl, hw), (hl, -hw), (hl, hw)]
        cx, cy = min(corners, key=lambda c: (bx - c[0]) ** 2 + (by - c[1]) ** 2)
        dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
        if dist > 8.0 and (ball_holder_pos is not None):
            # use ball-holder pos to choose corner side
            rx, ry = ball_holder_pos
            cx2, cy2 = min(corners, key=lambda c: (rx - c[0]) ** 2 + (ry - c[1]) ** 2)
            ball_marker = (float(cx2), float(cy2))

    # Plot ball marker
    if ball_marker is not None:
        fig.add_trace(go.Scatter(
            x=[ball_marker[0]],
            y=[ball_marker[1]],
            mode='markers',
            marker=dict(
                size=15,
                color='yellow',
                symbol='circle',
                line=dict(width=2, color='black'),
            ),
            name='Ball',
            hovertemplate='<b>Ball</b><br>' +
                          'x: %{x:.2f}<br>' +
                          'y: %{y:.2f}<extra></extra>',
        ))

        # Ball trajectory visualization
        if show_ball_arc:
            if (trajectory_source or "stylized").lower() == "tracking":
                # Use real ball trajectory from SoccerData tracking.csv
                if len(traj_x) > 1:
                    is_in = (swing == "in")
                    arc_color = 'deepskyblue' if is_in else 'orange'
                    # For clarity, optionally show a stylized emphasized arc regardless of raw trajectory shape.
                    if emphasize_swing:
                        arc_x, arc_y = get_emphasized_swing_arc_from_start(
                            start_x=float(ball_marker[0]),
                            start_y=float(ball_marker[1]),
                            swing=swing,
                            delta_m=float(swing_curvature_m),
                        )
                        if len(arc_x) > 1:
                            fig.add_trace(go.Scatter(
                                x=arc_x,
                                y=arc_y,
                                mode='lines',
                                line=dict(color=arc_color, width=4, dash='solid' if is_in else 'dash'),
                                name="Swing",
                            ))
                            fig.add_annotation(
                                x=float(arc_x[-1]),
                                y=float(arc_y[-1]),
                                ax=float(arc_x[-2]),
                                ay=float(arc_y[-2]),
                                xref="x",
                                yref="y",
                                axref="x",
                                ayref="y",
                                showarrow=True,
                                arrowhead=3,
                                arrowsize=2.0,
                                arrowwidth=3,
                                arrowcolor=arc_color,
                            )
                        else:
                            # likely short corner -> fall back to short pass arrow
                            start, end = get_short_pass_arrow(
                                df,
                                receiver_idx=receiver_idx,
                                start_override=ball_marker,
                                short_corner_ref_point=ball_holder_pos,
                            )
                            if start is not None and end is not None:
                                fig.add_trace(go.Scatter(
                                    x=[start[0], end[0]],
                                    y=[start[1], end[1]],
                                    mode='lines',
                                    line=dict(color=arc_color, width=3, dash=('solid' if is_in else 'dash')),
                                    name='Swing',
                                ))
                                fig.add_annotation(
                                    x=end[0],
                                    y=end[1],
                                    ax=start[0],
                                    ay=start[1],
                                    xref="x",
                                    yref="y",
                                    axref="x",
                                    ayref="y",
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=1.6,
                                    arrowwidth=2,
                                    arrowcolor=arc_color,
                                )
                    if show_raw_tracking:
                        fig.add_trace(go.Scatter(
                            x=traj_x,
                            y=traj_y,
                            mode='lines',
                            line=dict(color='rgba(255,255,255,0.45)', width=2, dash='dot'),
                            name="Ball trajectory (raw)",
                            showlegend=False,
                        ))
                else:
                    # Fallback to stylized arc + short-pass arrow if tracking not available
                    arc_x, arc_y, swing = get_ball_swing_arc(df, mode=swing_mode)
                    if len(arc_x) > 1:
                        is_in = (swing == "in")
                        arc_color = 'deepskyblue' if is_in else 'orange'
                        fig.add_trace(go.Scatter(
                            x=arc_x,
                            y=arc_y,
                            mode='lines',
                            line=dict(
                                color=arc_color,
                                width=3,
                                dash='solid' if is_in else 'dash',
                            ),
                            name="Swing",
                        ))
                        fig.add_annotation(
                            x=float(arc_x[-1]),
                            y=float(arc_y[-1]),
                            ax=float(arc_x[-2]),
                            ay=float(arc_y[-2]),
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1.8,
                            arrowwidth=2,
                            arrowcolor=arc_color,
                        )
                    else:
                        start, end = get_short_pass_arrow(
                            df,
                            receiver_idx=receiver_idx,
                            start_override=ball_marker,
                            short_corner_ref_point=ball_holder_pos,
                        )
                        if start is not None and end is not None:
                            is_in = (swing == "in")
                            arc_color = 'deepskyblue' if is_in else 'orange'
                            fig.add_trace(go.Scatter(
                                x=[start[0], end[0]],
                                y=[start[1], end[1]],
                                mode='lines',
                                line=dict(color=arc_color, width=3, dash=('solid' if is_in else 'dash')),
                                name="Swing",
                            ))
                            fig.add_annotation(
                                x=end[0],
                                y=end[1],
                                ax=start[0],
                                ay=start[1],
                                xref="x",
                                yref="y",
                                axref="x",
                                ayref="y",
                                showarrow=True,
                                arrowhead=3,
                                arrowsize=1.6,
                                arrowwidth=2,
                                arrowcolor=arc_color,
                            )
            else:
                # Stylized swing arc (in/out) from nearest corner into the box
                arc_x, arc_y, swing = get_ball_swing_arc(df, mode=swing_mode)
                if len(arc_x) > 1:
                    is_in = (swing == "in")
                    arc_color = 'deepskyblue' if is_in else 'orange'
                    fig.add_trace(go.Scatter(
                        x=arc_x,
                        y=arc_y,
                        mode='lines',
                        line=dict(
                            color=arc_color,
                            width=3,
                            dash='solid' if is_in else 'dash',
                        ),
                        name="Swing",
                    ))
                    fig.add_annotation(
                        x=float(arc_x[-1]),
                        y=float(arc_y[-1]),
                        ax=float(arc_x[-2]),
                        ay=float(arc_y[-2]),
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.8,
                        arrowwidth=2,
                        arrowcolor=arc_color,
                    )
                else:
                    start, end = get_short_pass_arrow(
                        df,
                        receiver_idx=receiver_idx,
                        start_override=ball_marker,
                        short_corner_ref_point=ball_holder_pos,
                    )
                    if start is not None and end is not None:
                        is_in = (swing == "in")
                        arc_color = 'deepskyblue' if is_in else 'orange'
                        fig.add_trace(go.Scatter(
                            x=[start[0], end[0]],
                            y=[start[1], end[1]],
                            mode='lines',
                            line=dict(color=arc_color, width=3, dash=('solid' if is_in else 'dash')),
                            name="Swing",
                        ))
                        fig.add_annotation(
                            x=end[0],
                            y=end[1],
                            ax=start[0],
                            ay=start[1],
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1.6,
                            arrowwidth=2,
                            arrowcolor=arc_color,
                        )
    
    # Plot velocity vectors
    if show_vectors:
        # Attacking team vectors
        if len(attacking) > 0:
            for idx, row in attacking.iterrows():
                if abs(row['vx']) > 1e-9 or abs(row['vy']) > 1e-9:
                    dx = float(row['vx']) * float(vector_scale)
                    dy = float(row['vy']) * float(vector_scale)
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm < float(min_vector_len) and norm > 1e-9:
                        up = float(min_vector_len) / norm
                        dx *= up
                        dy *= up
                        norm = float(min_vector_len)
                    if norm > float(max_vector_len) and norm > 1e-9:
                        scale = float(max_vector_len) / norm
                        dx *= scale
                        dy *= scale
                    # Offset arrow base so it doesn't overlap the player marker
                    if norm > 1e-9:
                        ux = dx / norm
                        uy = dy / norm
                        off = min(float(vector_offset_m), 0.5 * norm)
                    else:
                        ux = uy = 0.0
                        off = 0.0
                    fig.add_annotation(
                        x=float(row['x']) + dx,
                        y=float(row['y']) + dy,
                        ax=float(row['x']) + ux * off,
                        ay=float(row['y']) + uy * off,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.4,
                        arrowwidth=2,
                        arrowcolor="red",
                    )
        
        # Defending team vectors
        if len(defending) > 0:
            for idx, row in defending.iterrows():
                if abs(row['vx']) > 1e-9 or abs(row['vy']) > 1e-9:
                    dx = float(row['vx']) * float(vector_scale)
                    dy = float(row['vy']) * float(vector_scale)
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm < float(min_vector_len) and norm > 1e-9:
                        up = float(min_vector_len) / norm
                        dx *= up
                        dy *= up
                        norm = float(min_vector_len)
                    if norm > float(max_vector_len) and norm > 1e-9:
                        scale = float(max_vector_len) / norm
                        dx *= scale
                        dy *= scale
                    if norm > 1e-9:
                        ux = dx / norm
                        uy = dy / norm
                        off = min(float(vector_offset_m), 0.5 * norm)
                    else:
                        ux = uy = 0.0
                        off = 0.0
                    fig.add_annotation(
                        x=float(row['x']) + dx,
                        y=float(row['y']) + dy,
                        ax=float(row['x']) + ux * off,
                        ay=float(row['y']) + uy * off,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.4,
                        arrowwidth=2,
                        arrowcolor="blue",
                    )
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig


def compute_all_similarities(
    search_system: SimilarCKSearch,
    query_data: Dict[str, Any],
    index: SimilarCKIndex,
) -> np.ndarray:
    """Compute cosine similarity between query and all index embeddings."""
    if index.embeddings is None:
        raise ValueError("Index embeddings are not loaded.")

    x = query_data["x"].to(search_system.device)
    edge_index = query_data["edge_index"].to(search_system.device)
    edge_attr = query_data.get("edge_attr")
    if edge_attr is not None:
        edge_attr = edge_attr.to(search_system.device)
    batch = query_data.get("batch")
    if batch is not None:
        batch = batch.to(search_system.device)

    with torch.no_grad():
        q = search_system._forward_batch(x, edge_index, edge_attr, batch).detach().cpu().numpy()
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    qn = np.where(qn == 0, 1.0, qn)
    q = q / qn

    # index.embeddings are L2-normalized when built
    sims = np.dot(q, index.embeddings.T).reshape(-1)  # [N]
    return sims


def _get_team_split_positions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (attacking_positions, defending_positions) as arrays [M,2], filtered for visualization dummies."""
    df_vis = df.copy()
    if "is_dummy" in df_vis.columns:
        df_vis = df_vis[~df_vis["is_dummy"]]

    # Determine attacking team by ball-holder (kicker proxy)
    ball_rows = df_vis[df_vis.get("ball", 0.0) > 0.5] if "ball" in df_vis.columns else df_vis.iloc[0:0]
    if len(ball_rows) > 0 and "team_id" in df_vis.columns:
        attacking_team_id = int(ball_rows.iloc[0]["team_id"])
    else:
        attacking_team_id = int(df_vis["team_id"].iloc[0]) if "team_id" in df_vis.columns and len(df_vis) > 0 else 0

    if "team_id" in df_vis.columns:
        attacking = df_vis[df_vis["team_id"] == attacking_team_id]
        defending = df_vis[df_vis["team_id"] != attacking_team_id]
    else:
        attacking = df_vis
        defending = df_vis.iloc[0:0]

    att = attacking[["x", "y"]].to_numpy(dtype=np.float32) if len(attacking) > 0 else np.zeros((0, 2), dtype=np.float32)
    dfd = defending[["x", "y"]].to_numpy(dtype=np.float32) if len(defending) > 0 else np.zeros((0, 2), dtype=np.float32)

    # Keep at most 11 players per side for stability (CK snapshot)
    att = att[:11]
    dfd = dfd[:11]
    return att, dfd


def _hungarian_mean_distance(a: np.ndarray, b: np.ndarray, pad_cost: float = 200.0) -> float:
    """Mean assignment distance between two point sets using Hungarian algorithm.

    If scipy is unavailable, falls back to greedy matching (slightly worse but works).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = int(a.shape[0]), int(b.shape[0])
    if na == 0 and nb == 0:
        return 0.0
    if na == 0 or nb == 0:
        # All unmatched -> penalize
        return float(pad_cost)

    n = max(na, nb)
    # Build padded arrays
    A = np.zeros((n, 2), dtype=np.float32)
    B = np.zeros((n, 2), dtype=np.float32)
    A[:na] = a
    B[:nb] = b

    # Cost matrix: Euclidean distances
    diff = A[:, None, :] - B[None, :, :]
    cost = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

    # Penalize matches involving padded rows/cols
    if na < n:
        cost[na:, :] = pad_cost
    if nb < n:
        cost[:, nb:] = pad_cost

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(cost[row_ind, col_ind].mean())

    # Fallback greedy matching
    used_cols = set()
    dists = []
    for i in range(n):
        j = int(np.argmin([cost[i, jj] if jj not in used_cols else 1e9 for jj in range(n)]))
        used_cols.add(j)
        dists.append(float(cost[i, j]))
    return float(np.mean(dists)) if dists else float(pad_cost)


def compute_structural_similarities(
    query_sample: Dict[str, Any],
    dataset: Any,
    max_n: Optional[int] = None,
    w_att: float = 1.0,
    w_def: float = 1.0,
) -> np.ndarray:
    """Compute proposed structural similarity for all samples (higher is better).

    Similarity is defined as 1 / (1 + distance), where distance is a weighted
    Hungarian matching distance between attacking/defending point sets.
    """
    q_df = load_raw_sample_data(query_sample)
    q_att, q_def = _get_team_split_positions(q_df)

    n_total = len(dataset)
    if max_n is not None:
        n_total = min(n_total, int(max_n))

    sims = np.zeros((n_total,), dtype=np.float32)
    for i in range(n_total):
        cand_raw = _get_raw_sample(dataset, i)
        c_df = load_raw_sample_data(cand_raw)
        c_att, c_def = _get_team_split_positions(c_df)
        d_att = _hungarian_mean_distance(q_att, c_att)
        d_def = _hungarian_mean_distance(q_def, c_def)
        dist = float(w_att) * float(d_att) + float(w_def) * float(d_def)
        sims[i] = 1.0 / (1.0 + np.float32(dist))
    return sims


def main():
    """Main Streamlit application."""
    st.title("⚽ Similar CK Retrieval Visualization")
    st.markdown("Visualize search results from the similar corner kick retrieval system.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Quick presets (recommended): avoid mismatched config/index/checkpoint.
    preset = st.sidebar.radio(
        "Preset（おすすめ）",
        options=["consistency_stable", "baseline_stable", "custom"],
        index=0,
        help="普段は consistency_stable と baseline_stable の2つを比較するだけなので、まずはプリセットを推奨。",
    )

    # Cache clear helper (often needed when switching configs)
    if st.sidebar.button("♻️ キャッシュクリア（読み直し）"):
        try:
            load_search_system.clear()
            load_index.clear()
            st.sidebar.success("Cache cleared.")
        except Exception:
            # Best-effort: if Streamlit changes API, ignore.
            st.sidebar.info("Cache clear requested (restart browser if not reflected).")

    # ---- Easy selectors (recommended) ----
    # Config selection (dropdown)
    default_config = "configs_my_method/multitask_receiver_shot_d2_consistency_stable.yaml"
    try:
        cfg_candidates = sorted([str(p.as_posix()) for p in Path("configs_my_method").glob("*.yaml")])
    except Exception:
        cfg_candidates = []
    if default_config not in cfg_candidates:
        cfg_candidates = [default_config] + cfg_candidates

    # Apply preset defaults
    if preset == "consistency_stable":
        config_path = "configs_my_method/multitask_receiver_shot_d2_consistency_stable.yaml"
    elif preset == "baseline_stable":
        config_path = "configs_my_method/multitask_receiver_shot_d2_baseline_stable.yaml"
    else:
        config_path = st.sidebar.selectbox(
            "Config（選択）",
            options=cfg_candidates,
            index=0,
            help="customのときのみ選択（普段はプリセット推奨）",
        )

    # Infer default index path from selected config (run_name + d2)
    inferred_index_path = None
    try:
        with open(config_path, "r") as f:
            _cfg_tmp = yaml.safe_load(f)
        rn = _cfg_tmp.get("run_name", "default_run")
        d2_enabled = _cfg_tmp.get("d2", {}).get("enabled", False)
        inferred_index_path = f"runs/my_method/{rn}/indices/index_{'d2' if d2_enabled else 'no_d2'}.pkl"
    except Exception:
        inferred_index_path = "runs/my_method/consistency_stable/indices/index_d2.pkl"

    # Index selection (dropdown)
    try:
        idx_candidates = sorted([str(p.as_posix()) for p in Path("runs/my_method").glob("*/indices/index_*.pkl")])
    except Exception:
        idx_candidates = []
    if inferred_index_path not in idx_candidates:
        idx_candidates = [inferred_index_path] + idx_candidates
    if preset in ("consistency_stable", "baseline_stable"):
        index_path = inferred_index_path
        st.sidebar.caption(f"Index（自動）: `{index_path}`")
    else:
        index_path = st.sidebar.selectbox(
            "Index（選択）",
            options=idx_candidates,
            index=0,
            help="consistency_stable / baseline_stable など run_name ごとの index を選択",
        )

    # Advanced: allow manual override (optional)
    with st.sidebar.expander("詳細設定（手入力したい場合）", expanded=False):
        custom_config_path = st.text_input("Config file path（手入力）", value=config_path)
        custom_index_path = st.text_input("Index file path（手入力）", value=index_path)
        custom_checkpoint_path = st.text_input(
            "Backbone checkpoint path（手入力）",
            value="",
            help="空ならconfigから自動選択。run_name不一致などで見つからない場合に指定してください。",
        )
        use_custom_paths = st.checkbox("手入力のパスを使う", value=False)

    if "use_custom_paths" in locals() and use_custom_paths:
        config_path = custom_config_path
        index_path = custom_index_path
        checkpoint_path_override = custom_checkpoint_path.strip() or None
    else:
        checkpoint_path_override = None

    # Data path is rarely needed; keep it in advanced section
    data_path = "data/processed_ck/receiver_train/data.pickle"
    with st.sidebar.expander("データセット設定（通常は触らない）", expanded=False):
        data_path = st.text_input(
            "Data path（手入力）",
            value=data_path,
            help="通常は下の「configのreceiver_train/val/testを使う」をONにしておけばOK",
    )

    use_config_splits = st.sidebar.checkbox(
        "Use config receiver_train/val/test paths (recommended for index=373)",
        value=True,
        help="If enabled, load and concatenate train+val+test based on the selected config file.",
    )
    
    # Query index
    query_index = st.sidebar.number_input(
        "Query sample index",
        min_value=0,
        value=0,
        help="Index of the query sample in the dataset",
    )
    
    # Top/Bottom-k selection
    top_k = st.sidebar.number_input(
        "Top/Bottom-k results",
        min_value=1,
        max_value=20,
        value=5,
        help="Show Top-k most similar and Bottom-k least similar CKs.",
    )

    st.sidebar.header("Comparison Mode")
    compare_mode = st.sidebar.selectbox(
        "Compare",
        options=["Side-by-side (Cosine vs Proposed)", "Cosine only", "Proposed only"],
        index=0,
        help="左=cos類似度、右=提案手法（構造的類似度）",
    )
    enable_horizontal_scroll = False
    if compare_mode == "Side-by-side (Cosine vs Proposed)":
        enable_horizontal_scroll = st.sidebar.checkbox(
            "横スクロールで左右比較（潰れ防止）",
            value=True,
            help="画面が狭い場合に左右の結果が潰れないよう、横スクロールで表示します。",
        )
    # Results rendering: avoid squished plots by forcing horizontal scrolling rows with a minimum card width.
    results_horizontal_scroll = st.sidebar.checkbox(
        "検索結果を横スクロール（Queryと同じくらい大きく表示）",
        value=True,
        help="Top-k / Bottom-k を1行横並びにして、各カードを潰さず横スクロールで見られるようにします。",
    )
    result_card_min_width_px = st.sidebar.slider(
        "結果カード最小幅（px）",
        min_value=500,
        max_value=1400,
        value=900,
        step=50,
        help="Query CKと同じくらいの見た目にしたい場合は大きめ（例: 900〜1200）にしてください。",
    )
    with st.sidebar.expander("Proposed similarity settings", expanded=False):
        w_att = st.slider("Weight: attacking", 0.0, 3.0, 1.0, 0.1)
        w_def = st.slider("Weight: defending", 0.0, 3.0, 1.0, 0.1)
        st.caption("提案手法は、攻撃/守備それぞれの配置をHungarianマッチングで比較します。")
    
    # Display options
    st.sidebar.header("Display Options")
    show_vectors = st.sidebar.checkbox("Show velocity vectors", value=True)
    vector_scale = st.sidebar.slider(
        "Vector scale",
        min_value=0.05,
        max_value=3.0,
        value=0.72,
        step=0.01,
        help="Scale factor for velocity vectors",
    )
    max_vector_len = st.sidebar.slider(
        "Max vector length (m)",
        min_value=1.0,
        max_value=15.0,
        value=12.0,
        step=0.5,
        help="Clip velocity arrows to this maximum length for readability.",
    )
    min_vector_len = st.sidebar.slider(
        "Min vector length (m)",
        min_value=0.0,
        max_value=6.0,
        value=2.0,
        step=0.5,
        help="If non-zero, scales up small vectors so arrows are visible (for visualization clarity).",
    )
    vector_offset_m = st.sidebar.slider(
        "Vector base offset (m)",
        min_value=0.0,
        max_value=3.0,
        value=1.2,
        step=0.1,
        help="Shift arrow start away from the player marker to reduce overlap.",
    )
    show_ids = st.sidebar.checkbox("Show player IDs", value=False)

    st.sidebar.header("Ball Trajectory")
    show_ball_arc = st.sidebar.checkbox("Show in/out-swing arc", value=True)
    trajectory_source = st.sidebar.selectbox(
        "Trajectory source",
        options=["tracking", "stylized"],
        index=0,
        help="tracking: draw real ball trajectory from SoccerData tracking.csv (if available). stylized: draw a heuristic arc.",
    )
    emphasize_swing = st.sidebar.checkbox(
        "Emphasize swing (easy to see)",
        value=True,
        help="When enabled, draws a clearly curved in/out arc for quick recognition (even if the real trajectory is subtle).",
    )
    swing_curvature_m = st.sidebar.slider(
        "Swing curvature (m)",
        min_value=10.0,
        max_value=45.0,
        value=30.0,
        step=1.0,
        help="Bigger value = more 'curvy' in/out swing arc (for easy recognition).",
    )
    show_raw_tracking = st.sidebar.checkbox(
        "Show raw tracking trajectory (thin)",
        value=False,
        help="Overlay the real tracking trajectory as a thin dotted line (for reference).",
    )
    soccerdata_dir = st.sidebar.text_input(
        "SoccerData directory",
        value="SoccerData",
        help="Root directory containing 2023_data/ and 2024_data/.",
    )
    traj_window_frames = st.sidebar.slider(
        "Trajectory window (frames)",
        min_value=20,
        max_value=300,
        value=120,
        step=10,
        help="How many frames after the kick to draw for tracking trajectory.",
    )
    swing_mode = st.sidebar.selectbox(
        "Swing mode",
        options=["auto", "in", "out", "none"],
        index=0,
        help="Used only for stylized trajectory. Auto is a heuristic.",
    )
    
    # Grid layout option
    num_cols = st.sidebar.selectbox(
        "Number of columns",
        options=[1, 2, 3, 4],
        index=1,
        help="Number of columns for displaying results",
    )
    
    # Load search system and index
    try:
        search_system, config = load_search_system(config_path, checkpoint_path=checkpoint_path_override)
        embedding_dim = config["model"]["hidden_dim"]
        index = load_index(index_path, embedding_dim)
        
        st.sidebar.success(f"Loaded index with {len(index)} embeddings")
    except Exception as e:
        st.sidebar.error(f"Error loading search system: {str(e)}")
        st.stop()
    
    # Load query dataset
    try:
        if use_config_splits and isinstance(config, dict) and "data" in config:
            dcfg = config["data"]
            if all(k in dcfg for k in ["receiver_train_path", "receiver_val_path", "receiver_test_path"]):
                ds_train = ReceiverDataset(dcfg["receiver_train_path"], file_format="pickle", phase="train")
                ds_val = ReceiverDataset(dcfg["receiver_val_path"], file_format="pickle", phase="val")
                ds_test = ReceiverDataset(dcfg["receiver_test_path"], file_format="pickle", phase="test")
                dataset = ConcatDataset([ds_train, ds_val, ds_test])
            else:
                dataset = ReceiverDataset(data_path=data_path, file_format="pickle", phase="train")
        else:
            dataset = ReceiverDataset(data_path=data_path, file_format="pickle", phase="train")
        
        if query_index >= len(dataset):
            st.error(f"Query index {query_index} is out of range (dataset size: {len(dataset)})")
            st.stop()
        
        # Get query sample
        query_data_dict, query_target = dataset[query_index]
        query_raw_sample = _get_raw_sample(dataset, query_index)
        
        st.sidebar.info(f"Query target receiver: {query_target.item()}")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()
    
    # Perform search
    if st.sidebar.button("🔍 Search", type="primary"):
        try:
            with st.spinner("Searching for similar CKs..."):
                # Cosine similarities (embedding-based)
                sims_cos = compute_all_similarities(search_system, query_data_dict, index)

                # Proposed structural similarities (geometry-based)
                sims_prop = None
                if compare_mode != "Cosine only":
                    sims_prop = compute_structural_similarities(
                        query_raw_sample,
                        dataset,
                        max_n=len(sims_cos),
                        w_att=float(w_att),
                        w_def=float(w_def),
                    )

                n = int(len(sims_cos))
                k = int(min(int(top_k), n))

                def _pack_results(sims_arr: np.ndarray) -> tuple[list[dict], list[dict]]:
                    top_indices = np.argsort(sims_arr)[::-1][:k]
                    bottom_indices = np.argsort(sims_arr)[:k]
                    top_r = [{
                        "similarity": float(sims_arr[i]),
                    "metadata": index.metadata[int(i)].copy() if (index.metadata and int(i) < len(index.metadata)) else {},
                    "index": int(i),
                } for i in top_indices]
                    bot_r = [{
                        "similarity": float(sims_arr[i]),
                    "metadata": index.metadata[int(i)].copy() if (index.metadata and int(i) < len(index.metadata)) else {},
                    "index": int(i),
                } for i in bottom_indices]
                    return top_r, bot_r

                top_cos, bottom_cos = _pack_results(sims_cos)
                if sims_prop is not None:
                    top_prop, bottom_prop = _pack_results(sims_prop)
                else:
                    top_prop, bottom_prop = [], []
            
            st.session_state['top_results_cos'] = top_cos
            st.session_state['bottom_results_cos'] = bottom_cos
            st.session_state['top_results_prop'] = top_prop
            st.session_state['bottom_results_prop'] = bottom_prop
            st.session_state['query_sample'] = query_raw_sample
            st.session_state['query_index'] = query_index
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
    
    # Display results
    if 'top_results_cos' in st.session_state and 'bottom_results_cos' in st.session_state:
        top_cos = st.session_state['top_results_cos']
        bottom_cos = st.session_state['bottom_results_cos']
        top_prop = st.session_state.get('top_results_prop', [])
        bottom_prop = st.session_state.get('bottom_results_prop', [])
        query_sample = st.session_state['query_sample']
        query_idx = st.session_state['query_index']
        
        st.header("Search Results")
        
        # Display query CK
        st.subheader(f"Query CK (Index {query_idx})")
        query_df = load_raw_sample_data(query_sample)
        query_fig = plot_ck_snapshot(
            query_df,
            title=f"Query CK (Index {query_idx}, Receiver: {query_target.item()})",
            show_vectors=show_vectors,
            vector_scale=vector_scale,
            max_vector_len=max_vector_len,
            vector_offset_m=vector_offset_m,
            min_vector_len=min_vector_len,
            show_ids=show_ids,
            show_ball_arc=show_ball_arc,
            swing_mode=swing_mode,
            trajectory_source=trajectory_source,
            soccerdata_dir=soccerdata_dir,
            traj_window_frames=int(traj_window_frames),
            emphasize_swing=emphasize_swing,
            show_raw_tracking=show_raw_tracking,
            swing_curvature_m=float(swing_curvature_m),
            receiver_idx=int(query_target.item()),
        )
        st.plotly_chart(query_fig, use_container_width=True, key=f"query_fig_{int(query_idx)}")

        def _build_tiled_row_figure(
            results: list[dict],
            panel_key: str,
            *,
            card_min_width_px: int,
        ) -> Optional[go.Figure]:
            """Build a single wide Plotly figure by tiling multiple CK snapshot figures horizontally.

            This avoids Streamlit's `st.columns` shrink behavior (which squashes plots on narrow screens).
            """
            if not results:
                return None

            # Build per-sample figures first (re-using the same snapshot function).
            per_figs: list[go.Figure] = []
            per_meta: list[dict] = []
            for r_i, result in enumerate(results):
                idx = int(result["index"])
                if idx >= len(dataset):
                    continue
                _d, similar_target = dataset[idx]
                                similar_raw_sample = _get_raw_sample(dataset, idx)
                                similar_df = load_raw_sample_data(similar_raw_sample)
                fig_i = plot_ck_snapshot(
                                    similar_df,
                    title=f"Idx {idx}",
                                    show_vectors=show_vectors,
                                    vector_scale=vector_scale,
                                    max_vector_len=max_vector_len,
                                    vector_offset_m=vector_offset_m,
                                    min_vector_len=min_vector_len,
                                    show_ids=show_ids,
                                    show_ball_arc=show_ball_arc,
                                    swing_mode=swing_mode,
                                    trajectory_source=trajectory_source,
                                    soccerdata_dir=soccerdata_dir,
                                    traj_window_frames=int(traj_window_frames),
                                    emphasize_swing=emphasize_swing,
                                    show_raw_tracking=show_raw_tracking,
                                    swing_curvature_m=float(swing_curvature_m),
                                    receiver_idx=int(similar_target.item()),
                                )
                per_figs.append(fig_i)
                per_meta.append(
                    {
                        "rank": r_i + 1,
                        "idx": idx,
                        "receiver": int(similar_target.item()),
                        "similarity": result.get("similarity", None),
                    }
                )

            if not per_figs:
                return None

            # Determine base axis ranges from the first figure.
            base_xrange = [-60.0, 60.0]
            base_yrange = [-39.0, 39.0]
            try:
                xr = per_figs[0].layout.xaxis.range
                yr = per_figs[0].layout.yaxis.range
                if xr is not None and len(xr) == 2:
                    base_xrange = [float(xr[0]), float(xr[1])]
                if yr is not None and len(yr) == 2:
                    base_yrange = [float(yr[0]), float(yr[1])]
            except Exception:
                pass

            pitch_span = (base_xrange[1] - base_xrange[0]) + 15.0  # add gap between pitches

            big = go.Figure()

            # Helper to shift a trace's x coordinates
            def _shift_trace_x(tr: go.BaseTraceType, dx: float, showlegend: bool) -> go.BaseTraceType:
                tr2 = copy.deepcopy(tr)
                try:
                    if hasattr(tr2, "x") and tr2.x is not None:
                        x_arr = np.asarray(tr2.x, dtype=float)
                        tr2.x = (x_arr + float(dx)).tolist()
                except Exception:
                    pass
                tr2.showlegend = bool(showlegend and getattr(tr2, "showlegend", True))
                return tr2

            # Helper to shift a shape dict's x coordinates
            def _shift_shape_x(shape_obj: Any, dx: float) -> dict:
                s = shape_obj.to_plotly_json() if hasattr(shape_obj, "to_plotly_json") else dict(shape_obj)
                if "x0" in s:
                    s["x0"] = float(s["x0"]) + float(dx)
                if "x1" in s:
                    s["x1"] = float(s["x1"]) + float(dx)
                # Keep xref/yref default ("x","y") since we tile on a single axis.
                s.pop("xref", None)
                s.pop("yref", None)
                return s

            for i, (fig_i, meta) in enumerate(zip(per_figs, per_meta)):
                dx = float(i) * float(pitch_span)
                showlegend = (i == 0)  # show legend only once to avoid clutter

                # Add shapes (field markings etc.)
                try:
                    for sh in (fig_i.layout.shapes or []):
                        big.add_shape(**_shift_shape_x(sh, dx))
                except Exception:
                    pass

                # Add traces (players, ball, swing, corner arcs etc.)
                for tr in fig_i.data:
                    big.add_trace(_shift_trace_x(tr, dx, showlegend=showlegend))

                # Add a clear title annotation above each pitch
                sim_val = meta.get("similarity", None)
                sim_str = "N/A"
                try:
                    sim_f = float(sim_val)
                    if math.isfinite(sim_f):
                        sim_str = f"{sim_f:.4f}"
                except Exception:
                    pass
                big.add_annotation(
                    x=dx + 0.5 * (base_xrange[0] + base_xrange[1]),
                    y=base_yrange[1] + 2.5,
                    xref="x",
                    yref="y",
                    text=f"Rank {meta['rank']} | idx {meta['idx']} | sim {sim_str} | recv {meta['receiver']}",
                    showarrow=False,
                    font=dict(size=12),
                )

            # Layout: wide fixed width so the browser can scroll horizontally
            n = len(per_figs)
            big.update_layout(
                width=int(max(900, n * int(card_min_width_px))),
                height=520,
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            )
            big.update_xaxes(
                range=[base_xrange[0], base_xrange[0] + pitch_span * (n - 1) + (base_xrange[1] - base_xrange[0])],
                showgrid=False,
                zeroline=False,
            )
            big.update_yaxes(range=base_yrange, showgrid=False, zeroline=False, scaleanchor=None)
            return big

        def _render_result_grid(
            results: list[dict],
            header: str,
            panel_key: str,
            *,
            horizontal_scroll: bool,
            card_min_width_px: int,
        ):
            st.subheader(header)
            num_results = len(results)
            if horizontal_scroll and num_results > 0:
                # Render as a single wide figure (tiled pitches). This avoids "squished/overlapped" plots.
                marker_id = f"{panel_key}_wideplot_marker"
                st.markdown(
                    f"""
<style>
#{marker_id} + div div[data-testid="stPlotlyChart"] {{
  overflow-x: auto;
}}
</style>
<div id="{marker_id}"></div>
""",
                    unsafe_allow_html=True,
                )
                big_fig = _build_tiled_row_figure(results, panel_key, card_min_width_px=int(card_min_width_px))
                if big_fig is not None:
                    st.plotly_chart(
                        big_fig,
                        use_container_width=False,
                        key=f"{panel_key}_wideplot",
                    )
                else:
                    st.info("No valid results to display.")
                return

            # Default grid layout
        num_rows = (num_results + num_cols - 1) // num_cols
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                    r_i = row * num_cols + col_idx
                    if r_i >= num_results:
                        continue
                    result = results[r_i]
                    idx = int(result["index"])
                    with cols[col_idx]:
                        sim_val = result.get("similarity", None)
                        sim_str = "N/A"
                        try:
                            sim_f = float(sim_val)
                            if math.isfinite(sim_f):
                                sim_str = f"{sim_f:.4f}"
                        except Exception:
                            pass
                        st.markdown(f"**Rank {r_i + 1}** (Similarity: {sim_str})")
                        st.caption(f"Index: {idx}")
                        try:
                            if idx < len(dataset):
                                _d, similar_target = dataset[idx]
                                similar_raw_sample = _get_raw_sample(dataset, idx)
                                similar_df = load_raw_sample_data(similar_raw_sample)
                                fig = plot_ck_snapshot(
                                    similar_df,
                                    title=f"Idx {idx}",
                                    show_vectors=show_vectors,
                                    vector_scale=vector_scale,
                                    max_vector_len=max_vector_len,
                                    vector_offset_m=vector_offset_m,
                                    min_vector_len=min_vector_len,
                                    show_ids=show_ids,
                                    show_ball_arc=show_ball_arc,
                                    swing_mode=swing_mode,
                                    trajectory_source=trajectory_source,
                                    soccerdata_dir=soccerdata_dir,
                                    traj_window_frames=int(traj_window_frames),
                                    emphasize_swing=emphasize_swing,
                                    show_raw_tracking=show_raw_tracking,
                                    swing_curvature_m=float(swing_curvature_m),
                                    receiver_idx=int(similar_target.item()),
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    key=f"{panel_key}_rank{r_i+1}_idx{idx}",
                                )
                                st.caption(f"Receiver: {int(similar_target.item())}")
                            else:
                                st.warning(f"Index {idx} out of dataset range")
                        except Exception as e:
                            st.error(f"Error loading sample {idx}: {str(e)}")

        if compare_mode == "Side-by-side (Cosine vs Proposed)":
            if enable_horizontal_scroll:
                # Insert a marker element and CSS to target the next horizontal block (the columns)
                st.markdown(
                    """
<style>
/* Make the compare columns horizontally scrollable instead of shrinking. */
#compare_panels_marker + div[data-testid="stHorizontalBlock"] {
  overflow-x: auto;
  flex-wrap: nowrap;
  gap: 1rem;
}
#compare_panels_marker + div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
  min-width: 560px; /* prevent squishing on narrow screens */
}
</style>
<div id="compare_panels_marker"></div>
""",
                    unsafe_allow_html=True,
                )
            col_l, col_r = st.columns(2)
            with col_l:
                _render_result_grid(
                    top_cos,
                    f"[Cosine] Top-{len(top_cos)} Similar CKs",
                    panel_key="cos_top",
                    horizontal_scroll=results_horizontal_scroll,
                    card_min_width_px=int(result_card_min_width_px),
                )
                _render_result_grid(
                    bottom_cos,
                    f"[Cosine] Bottom-{len(bottom_cos)} Dissimilar CKs",
                    panel_key="cos_bottom",
                    horizontal_scroll=results_horizontal_scroll,
                    card_min_width_px=int(result_card_min_width_px),
                )
            with col_r:
                _render_result_grid(
                    top_prop,
                    f"[Proposed] Top-{len(top_prop)} Similar CKs",
                    panel_key="prop_top",
                    horizontal_scroll=results_horizontal_scroll,
                    card_min_width_px=int(result_card_min_width_px),
                )
                _render_result_grid(
                    bottom_prop,
                    f"[Proposed] Bottom-{len(bottom_prop)} Dissimilar CKs",
                    panel_key="prop_bottom",
                    horizontal_scroll=results_horizontal_scroll,
                    card_min_width_px=int(result_card_min_width_px),
                )
        elif compare_mode == "Cosine only":
            _render_result_grid(
                top_cos,
                f"Top-{len(top_cos)} Similar CKs (Cosine)",
                panel_key="cos_top",
                horizontal_scroll=results_horizontal_scroll,
                card_min_width_px=int(result_card_min_width_px),
            )
            _render_result_grid(
                bottom_cos,
                f"Bottom-{len(bottom_cos)} Dissimilar CKs (Cosine)",
                panel_key="cos_bottom",
                horizontal_scroll=results_horizontal_scroll,
                card_min_width_px=int(result_card_min_width_px),
            )
        else:
            _render_result_grid(
                top_prop,
                f"Top-{len(top_prop)} Similar CKs (Proposed)",
                panel_key="prop_top",
                horizontal_scroll=results_horizontal_scroll,
                card_min_width_px=int(result_card_min_width_px),
            )
            _render_result_grid(
                bottom_prop,
                f"Bottom-{len(bottom_prop)} Dissimilar CKs (Proposed)",
                panel_key="prop_bottom",
                horizontal_scroll=results_horizontal_scroll,
                card_min_width_px=int(result_card_min_width_px),
            )
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Search' to view results.")


if __name__ == "__main__":
    main()

