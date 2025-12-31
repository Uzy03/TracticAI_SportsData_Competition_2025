"""Streamlit app to visualize CVAE generated tactics on a soccer pitch.

Reads output pickle produced by scripts/generate_cvae_samples.py:
  - targets: [N,4]  (x,y,vx,vy) normalized
  - generated: [1,S,N,4] normalized
  - x_input: [N,16] node features (includes ball/team)

Displays:
  - Target snapshot
  - Generated snapshots (S samples) in a grid
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add repo root to import path (for running in Docker without install)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tacticai.retrieval.visualization.utils import FIELD_LENGTH, FIELD_WIDTH, draw_soccer_field


def _to_meter_xy(xy01: np.ndarray) -> np.ndarray:
    """[N,2] in [0,1] -> meters centered at (0,0)."""
    out = np.array(xy01, dtype=np.float32, copy=True)
    out[:, 0] = (out[:, 0] - 0.5) * float(FIELD_LENGTH)
    out[:, 1] = (out[:, 1] - 0.5) * float(FIELD_WIDTH)
    return out


def _make_df(states_n4: np.ndarray, x_input_n16: np.ndarray) -> pd.DataFrame:
    """Create plotting DataFrame from normalized states + input features."""
    s = np.asarray(states_n4, dtype=np.float32).reshape(-1, 4)
    x_in = np.asarray(x_input_n16, dtype=np.float32).reshape(-1, 16)
    assert s.shape[0] == 22 and x_in.shape[0] == 22, f"Expected 22 players, got {s.shape[0]} and {x_in.shape[0]}"

    xy_m = _to_meter_xy(s[:, :2])
    # velocity is normalized by /70 in this repo; scale to meters/sec for arrow display
    vxvy_ms = s[:, 2:4] * 70.0

    df = pd.DataFrame(
        {
            "player_id": np.arange(22, dtype=int),
            "x": xy_m[:, 0],
            "y": xy_m[:, 1],
            "vx": vxvy_ms[:, 0],
            "vy": vxvy_ms[:, 1],
            "ball_flag": x_in[:, 6],  # ball possession (0/1)
            "team_id": x_in[:, 15],   # team_id (0/1)
        }
    )
    return df


def _infer_attacking_team_id(df: pd.DataFrame) -> int:
    if df["ball_flag"].sum() > 0:
        kicker = int(df["ball_flag"].values.argmax())
        return int(round(float(df.loc[kicker, "team_id"])))
    # fallback
    return 0


def plot_snapshot(
    df: pd.DataFrame,
    title: str,
    show_vectors: bool,
    vector_scale: float,
    max_vector_len: float,
) -> go.Figure:
    fig = go.Figure()
    draw_soccer_field(fig)

    attacking_team_id = _infer_attacking_team_id(df)
    is_att = df["team_id"].round().astype(int) == attacking_team_id

    # players
    fig.add_trace(
        go.Scatter(
            x=df.loc[is_att, "x"],
            y=df.loc[is_att, "y"],
            mode="markers",
            marker=dict(size=9, color="red", line=dict(color="white", width=1)),
            name="Attack",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.loc[~is_att, "x"],
            y=df.loc[~is_att, "y"],
            mode="markers",
            marker=dict(size=9, color="blue", line=dict(color="white", width=1)),
            name="Defense",
        )
    )

    # ball marker at ball-holder if present
    if df["ball_flag"].sum() > 0:
        kicker = int(df["ball_flag"].values.argmax())
        fig.add_trace(
            go.Scatter(
                x=[float(df.loc[kicker, "x"])],
                y=[float(df.loc[kicker, "y"])],
                mode="markers",
                marker=dict(size=10, color="yellow", line=dict(color="black", width=1)),
                name="Ball",
            )
        )

    # velocity vectors
    if show_vectors:
        xs, ys = df["x"].to_numpy(), df["y"].to_numpy()
        vxs, vys = df["vx"].to_numpy(), df["vy"].to_numpy()
        # scale and clip
        dx = vxs * float(vector_scale)
        dy = vys * float(vector_scale)
        lens = np.sqrt(dx * dx + dy * dy) + 1e-9
        clip = np.minimum(lens, float(max_vector_len)) / lens
        dx *= clip
        dy *= clip

        # plot as small line segments
        x_lines = []
        y_lines = []
        for i in range(len(xs)):
            x_lines += [xs[i], xs[i] + dx[i], None]
            y_lines += [ys[i], ys[i] + dy[i], None]
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color="white", width=2),
                name="Velocity",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=[-FIELD_LENGTH / 2 - 2, FIELD_LENGTH / 2 + 2]),
        yaxis=dict(visible=False, range=[-FIELD_WIDTH / 2 - 2, FIELD_WIDTH / 2 + 2], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
    )
    return fig


def _load_pickle(path: str) -> Dict[str, Any]:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


st.set_page_config(page_title="CVAE Generation Visualization", layout="wide")
st.title("CVAE 生成結果の可視化")

st.sidebar.header("入力")
default_path = "runs/cvae_d2/generated/cvae_gen_idx0_S20.pkl"
pickle_path = st.sidebar.text_input("Generated pickle path", value=default_path)

st.sidebar.header("表示")
cols = st.sidebar.slider("Columns", min_value=2, max_value=6, value=4, step=1)
show_vectors = st.sidebar.checkbox("Show velocity vectors", value=True)
vector_scale = st.sidebar.slider("Vector scale", min_value=0.01, max_value=1.5, value=0.15, step=0.01)
max_vec = st.sidebar.slider("Max vector length (m)", min_value=1.0, max_value=15.0, value=6.0, step=0.5)

obj: Optional[Dict[str, Any]] = None
err: Optional[str] = None
try:
    obj = _load_pickle(pickle_path)
except Exception as e:
    err = str(e)

if obj is None:
    st.error(f"pickleを読めません: {err}")
    st.stop()

generated = np.asarray(obj["generated"], dtype=np.float32)  # [1,S,N,4]
targets = np.asarray(obj["targets"], dtype=np.float32)
x_input = np.asarray(obj["x_input"], dtype=np.float32)

S = int(generated.shape[1])

st.caption(f"Loaded: {pickle_path} | generated: {tuple(generated.shape)}")

# Target / Posterior recon (debug)
try:
    target_df = _make_df(targets.reshape(22, 4), x_input)
    fig_t = plot_snapshot(target_df, title="Target (GT)", show_vectors=show_vectors, vector_scale=vector_scale, max_vector_len=max_vec)
    st.plotly_chart(fig_t, use_container_width=True)
except Exception as e:
    st.error(f"Targetの描画に失敗: {e}")

if "recon_posterior" in obj:
    try:
        recon = np.asarray(obj["recon_posterior"], dtype=np.float32)  # [1,N,4]
        recon_df = _make_df(recon.reshape(22, 4), x_input)
        fig_r = plot_snapshot(recon_df, title="Posterior recon (x_gtあり / デバッグ)", show_vectors=show_vectors, vector_scale=vector_scale, max_vector_len=max_vec)
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception as e:
        st.error(f"Posterior reconの描画に失敗: {e}")

st.subheader("Generated samples")

grid = st.columns(cols)
for i in range(S):
    df_i = _make_df(generated[0, i], x_input)
    fig_i = plot_snapshot(df_i, title=f"Gen #{i}", show_vectors=show_vectors, vector_scale=vector_scale, max_vector_len=max_vec)
    grid[i % cols].plotly_chart(fig_i, use_container_width=True)


