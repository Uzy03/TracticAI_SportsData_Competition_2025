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
from torch.utils.data import ConcatDataset
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset
from tacticai.modules import get_device

# Import utils (handle both relative and absolute imports)
try:
    from .utils import draw_soccer_field, load_raw_sample_data, get_ball_swing_arc
except ImportError:
    from tacticai.retrieval.visualization.utils import draw_soccer_field, load_raw_sample_data, get_ball_swing_arc


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
        if d2_enabled:
            checkpoint_path = f"{model_save_dir}/best_d2.ckpt"
        else:
            checkpoint_path = f"{model_save_dir}/best_no_d2.ckpt"
    
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
    show_ids: bool = False,
    show_ball_arc: bool = True,
    swing_mode: str = "auto",
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
    
    # Separate attacking and defending teams
    attacking = df[df['team_id'] == 0]  # Attacking team (usually team 0)
    defending = df[df['team_id'] == 1]  # Defending team (usually team 1)
    
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
    
    # Plot ball
    ball_data = df[df['ball'] > 0.5]
    if len(ball_data) > 0:
        fig.add_trace(go.Scatter(
            x=ball_data['x'],
            y=ball_data['y'],
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

        # Stylized swing arc (in/out) from nearest corner into the box
        if show_ball_arc:
            arc_x, arc_y, swing = get_ball_swing_arc(df, mode=swing_mode)
            if len(arc_x) > 1:
                is_in = (swing == "in")
                fig.add_trace(go.Scatter(
                    x=arc_x,
                    y=arc_y,
                    mode='lines',
                    line=dict(
                        color='deepskyblue' if is_in else 'orange',
                        width=3,
                        dash='solid' if is_in else 'dash',
                    ),
                    name=f"{'In' if is_in else 'Out'}-swing",
                ))
                # Arrow head at the end (last segment)
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
                    arrowcolor=('deepskyblue' if is_in else 'orange'),
                )
                # Label near the curve so the direction is visually obvious
                mid = len(arc_x) // 2
                fig.add_annotation(
                    x=float(arc_x[mid]),
                    y=float(arc_y[mid]),
                    text=("In-swing (内側)" if is_in else "Out-swing (外側)"),
                    showarrow=False,
                    font=dict(color=('deepskyblue' if is_in else 'orange'), size=12),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor=('deepskyblue' if is_in else 'orange'),
                    borderwidth=1,
                )
    
    # Plot velocity vectors
    if show_vectors:
        # Attacking team vectors
        if len(attacking) > 0:
            for idx, row in attacking.iterrows():
                if abs(row['vx']) > 0.01 or abs(row['vy']) > 0.01:  # Only show if significant
                    dx = float(row['vx']) * float(vector_scale)
                    dy = float(row['vy']) * float(vector_scale)
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm > float(max_vector_len) and norm > 1e-9:
                        scale = float(max_vector_len) / norm
                        dx *= scale
                        dy *= scale
                    fig.add_annotation(
                        x=float(row['x']) + dx,
                        y=float(row['y']) + dy,
                        ax=row['x'],
                        ay=row['y'],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1,
                        arrowcolor="red",
                    )
        
        # Defending team vectors
        if len(defending) > 0:
            for idx, row in defending.iterrows():
                if abs(row['vx']) > 0.01 or abs(row['vy']) > 0.01:
                    dx = float(row['vx']) * float(vector_scale)
                    dy = float(row['vy']) * float(vector_scale)
                    norm = (dx * dx + dy * dy) ** 0.5
                    if norm > float(max_vector_len) and norm > 1e-9:
                        scale = float(max_vector_len) / norm
                        dx *= scale
                        dy *= scale
                    fig.add_annotation(
                        x=float(row['x']) + dx,
                        y=float(row['y']) + dy,
                        ax=row['x'],
                        ay=row['y'],
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.2,
                        arrowwidth=1,
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


def main():
    """Main Streamlit application."""
    st.title("⚽ Similar CK Retrieval Visualization")
    st.markdown("Visualize search results from the similar corner kick retrieval system.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Config file selection
    config_path = st.sidebar.text_input(
        "Config file path",
        value="configs/multitask_receiver_shot_d2.yaml",
        help="Path to YAML config file",
    )
    
    # Index file selection
    index_path = st.sidebar.text_input(
        "Index file path",
        value="runs/retrieval/index_d2.pkl",
        help="Path to search index file",
    )
    
    # Data path selection
    data_path = st.sidebar.text_input(
        "Data path",
        value="data/processed_ck/receiver_train/data.pickle",
        help="Path to receiver dataset",
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
    
    # Top-k selection
    top_k = st.sidebar.number_input(
        "Top-k results",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of similar CKs to retrieve",
    )
    
    # Display options
    st.sidebar.header("Display Options")
    show_vectors = st.sidebar.checkbox("Show velocity vectors", value=True)
    vector_scale = st.sidebar.slider(
        "Vector scale",
        min_value=0.02,
        max_value=1.5,
        value=0.25,
        step=0.01,
        help="Scale factor for velocity vectors",
    )
    max_vector_len = st.sidebar.slider(
        "Max vector length (m)",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.5,
        help="Clip velocity arrows to this maximum length for readability.",
    )
    show_ids = st.sidebar.checkbox("Show player IDs", value=False)

    st.sidebar.header("Ball Trajectory")
    show_ball_arc = st.sidebar.checkbox("Show in/out-swing arc", value=True)
    swing_mode = st.sidebar.selectbox(
        "Swing mode",
        options=["auto", "in", "out", "none"],
        index=0,
        help="Data has no true ball trajectory; this draws a stylized arc from the corner. Auto is a heuristic.",
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
        search_system, config = load_search_system(config_path)
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
                results = search_system.search_similar(
                    query_data_dict,
                    index=index,
                    top_k=top_k,
                )
            
            st.session_state['search_results'] = results
            st.session_state['query_sample'] = query_raw_sample
            st.session_state['query_index'] = query_index
        except Exception as e:
            st.error(f"Error performing search: {str(e)}")
    
    # Display results
    if 'search_results' in st.session_state:
        results = st.session_state['search_results']
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
            show_ids=show_ids,
            show_ball_arc=show_ball_arc,
            swing_mode=swing_mode,
        )
        st.plotly_chart(query_fig, use_container_width=True)
        
        # Display similar CKs
        st.subheader(f"Top-{len(results)} Similar CKs")
        
        # Create grid layout
        num_results = len(results)
        num_rows = (num_results + num_cols - 1) // num_cols
        
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col_idx in range(num_cols):
                result_idx = row * num_cols + col_idx
                if result_idx < num_results:
                    result = results[result_idx]
                    similarity = result['similarity']
                    idx = result['index']
                    metadata = result['metadata']
                    
                    with cols[col_idx]:
                        sim_val = result.get("similarity", None)
                        sim_str = "N/A"
                        try:
                            sim_f = float(sim_val)
                            if math.isfinite(sim_f):
                                sim_str = f"{sim_f:.4f}"
                        except Exception:
                            pass

                        st.markdown(f"**Rank {result_idx + 1}** (Similarity: {sim_str})")
                        st.caption(f"Index: {idx}")
                        
                        # Load and plot similar CK
                        # Note: We need to load the actual sample from the dataset
                        # For now, we'll use the index to get the sample
                        # This assumes the index metadata contains enough info to load the sample
                        try:
                            # Try to get sample from dataset using index
                            if idx < len(dataset):
                                similar_data_dict, similar_target = dataset[idx]
                                similar_raw_sample = _get_raw_sample(dataset, idx)
                                
                                similar_df = load_raw_sample_data(similar_raw_sample)
                                similar_fig = plot_ck_snapshot(
                                    similar_df,
                                    title=f"Rank {result_idx + 1} (Idx {idx})",
                                    show_vectors=show_vectors,
                                    vector_scale=vector_scale,
                                    max_vector_len=max_vector_len,
                                    show_ids=show_ids,
                                    show_ball_arc=show_ball_arc,
                                    swing_mode=swing_mode,
                                )
                                st.plotly_chart(similar_fig, use_container_width=True)
                                
                                st.caption(f"Receiver: {similar_target.item()}")
                            else:
                                st.warning(f"Index {idx} out of dataset range")
                        except Exception as e:
                            st.error(f"Error loading sample {idx}: {str(e)}")
    
    else:
        st.info("👈 Configure settings in the sidebar and click 'Search' to view results.")


if __name__ == "__main__":
    main()

