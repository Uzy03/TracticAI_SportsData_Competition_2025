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
    team = np.array(sample.get("team", []))
    ball = np.array(sample.get("ball", []))
    
    # Get velocities if available
    vx = np.array(sample.get("vx", [0.0] * len(x)))
    vy = np.array(sample.get("vy", [0.0] * len(y)))
    
    # Ensure all arrays have the same length
    min_len = min(len(x), len(y), len(team), len(ball))
    if len(vx) < min_len:
        vx = np.pad(vx, (0, min_len - len(vx)), mode='constant')
    if len(vy) < min_len:
        vy = np.pad(vy, (0, min_len - len(vy)), mode='constant')
    
    x = x[:min_len]
    y = y[:min_len]
    team = team[:min_len]
    ball = ball[:min_len]
    vx = vx[:min_len]
    vy = vy[:min_len]
    
    # Heuristic: many preprocessors store positions normalized to [0, 1] (or slightly beyond).
    # Plotly field is drawn in meters with origin at center, so convert if values look normalized.
    def _looks_normalized(arr: np.ndarray) -> bool:
        if arr.size == 0:
            return False
        a = np.asarray(arr).reshape(-1)
        # Typical ranges we observed in this project: ~0.3..1.0 (sometimes slightly >1)
        return (a.min() >= -0.5) and (a.max() <= 1.5)

    if _looks_normalized(x) and _looks_normalized(y):
        # Convert normalized [0,1] -> meters centered at (0,0)
        x = (x - 0.5) * FIELD_LENGTH
        y = (y - 0.5) * FIELD_WIDTH

        # Velocities often follow same normalization scale; convert if they look small.
        if _looks_normalized(vx):
            vx = vx * FIELD_LENGTH
        if _looks_normalized(vy):
            vy = vy * FIELD_WIDTH

    # Create DataFrame
    df = pd.DataFrame({
        'player_id': [f'player_{i}' for i in range(min_len)],
        'team_id': team,
        'x': x,
        'y': y,
        'vx': vx,
        'vy': vy,
        'ball': ball.astype(float),
    })
    
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

