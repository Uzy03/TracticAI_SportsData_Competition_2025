"""Edge and graph feature computation for TacticAI.

This module implements edge and graph feature computation including:
- Distance and bearing between players
- Team membership information
- Set piece metadata (corner kicks, free kicks, etc.)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def compute_distance_bearing(
    pos1: torch.Tensor, 
    pos2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute distance and bearing between two positions.
    
    Args:
        pos1: First position [..., 2] (x, y)
        pos2: Second position [..., 2] (x, y)
        
    Returns:
        Distance [..., 1] and bearing [..., 1] (in radians)
    """
    # Compute distance
    diff = pos2 - pos1
    distance = torch.norm(diff, dim=-1, keepdim=True)
    
    # Compute bearing (angle from pos1 to pos2)
    bearing = torch.atan2(diff[..., 1:2], diff[..., 0:1])
    
    return distance, bearing


def compute_edge_features(
    node_positions: torch.Tensor,
    edge_index: torch.Tensor,
    team_ids: Optional[torch.Tensor] = None,
    ball_positions: Optional[torch.Tensor] = None,
    set_piece_info: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute comprehensive edge features.
    
    Args:
        node_positions: Node positions [N, 2] (x, y)
        edge_index: Edge connectivity [2, E]
        team_ids: Team IDs [N] (0 for team A, 1 for team B)
        ball_positions: Ball positions [N] (optional)
        set_piece_info: Set piece information [N] (optional)
        
    Returns:
        Edge features [E, edge_dim]
    """
    src, dst = edge_index[0], edge_index[1]
    
    # Get source and destination positions
    src_pos = node_positions[src]  # [E, 2]
    dst_pos = node_positions[dst]  # [E, 2]
    
    # Compute distance and bearing
    distance, bearing = compute_distance_bearing(src_pos, dst_pos)
    
    # Initialize edge features
    edge_features = [distance, bearing]
    
    # Team membership features
    if team_ids is not None:
        src_team = team_ids[src].float().unsqueeze(-1)  # [E, 1]
        dst_team = team_ids[dst].float().unsqueeze(-1)  # [E, 1]
        same_team = (src_team == dst_team).float()  # [E, 1]
        
        edge_features.extend([src_team, dst_team, same_team])
    
    # Ball proximity features
    if ball_positions is not None:
        # Distance from source to ball
        src_to_ball_dist, _ = compute_distance_bearing(src_pos, ball_positions[src])
        # Distance from destination to ball
        dst_to_ball_dist, _ = compute_distance_bearing(dst_pos, ball_positions[dst])
        
        edge_features.extend([src_to_ball_dist, dst_to_ball_dist])
    
    # Set piece features
    if set_piece_info is not None:
        src_set_piece = set_piece_info[src].float().unsqueeze(-1)  # [E, 1]
        dst_set_piece = set_piece_info[dst].float().unsqueeze(-1)  # [E, 1]
        
        edge_features.extend([src_set_piece, dst_set_piece])
    
    # Concatenate all features
    edge_features = torch.cat(edge_features, dim=-1)  # [E, edge_dim]
    
    return edge_features


def compute_graph_features(
    node_positions: torch.Tensor,
    team_ids: Optional[torch.Tensor] = None,
    ball_positions: Optional[torch.Tensor] = None,
    set_piece_info: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute graph-level features.
    
    Args:
        node_positions: Node positions [N, 2] (x, y)
        team_ids: Team IDs [N] (optional)
        ball_positions: Ball positions [N] (optional)
        set_piece_info: Set piece information [N] (optional)
        
    Returns:
        Graph features [graph_dim]
    """
    graph_features = []
    
    # Field coverage features
    field_bounds = torch.tensor([[-52.5, -34.0], [52.5, 34.0]], device=node_positions.device)
    
    # Normalize positions to [0, 1]
    normalized_pos = (node_positions - field_bounds[0]) / (field_bounds[1] - field_bounds[0])
    
    # Field coverage statistics
    coverage_x = normalized_pos[:, 0].mean().unsqueeze(0)  # [1]
    coverage_y = normalized_pos[:, 1].mean().unsqueeze(0)  # [1]
    spread_x = normalized_pos[:, 0].std().unsqueeze(0)  # [1]
    spread_y = normalized_pos[:, 1].std().unsqueeze(0)  # [1]
    
    graph_features.extend([coverage_x, coverage_y, spread_x, spread_y])
    
    # Team distribution features
    if team_ids is not None:
        team_a_pos = node_positions[team_ids == 0]
        team_b_pos = node_positions[team_ids == 1]
        
        if len(team_a_pos) > 0:
            team_a_center = team_a_pos.mean(dim=0)  # [2]
            team_a_spread = team_a_pos.std(dim=0)  # [2]
        else:
            team_a_center = torch.zeros(2, device=node_positions.device)
            team_a_spread = torch.zeros(2, device=node_positions.device)
        
        if len(team_b_pos) > 0:
            team_b_center = team_b_pos.mean(dim=0)  # [2]
            team_b_spread = team_b_pos.std(dim=0)  # [2]
        else:
            team_b_center = torch.zeros(2, device=node_positions.device)
            team_b_spread = torch.zeros(2, device=node_positions.device)
        
        # Distance between team centers
        team_distance = torch.norm(team_a_center - team_b_center).unsqueeze(0)  # [1]
        
        graph_features.extend([
            team_a_center, team_a_spread,
            team_b_center, team_b_spread,
            team_distance
        ])
    
    # Ball position features
    if ball_positions is not None:
        ball_center = ball_positions.mean(dim=0)  # [2]
        ball_spread = ball_positions.std(dim=0)  # [2]
        
        graph_features.extend([ball_center, ball_spread])
    
    # Set piece features
    if set_piece_info is not None:
        set_piece_count = set_piece_info.sum().float().unsqueeze(0)  # [1]
        set_piece_ratio = set_piece_count / len(set_piece_info)
        
        graph_features.extend([set_piece_count, set_piece_ratio])
    
    # Concatenate all features
    graph_features = torch.cat(graph_features, dim=-1)  # [graph_dim]
    
    return graph_features


class EdgeFeatureEncoder(nn.Module):
    """Neural network encoder for edge features.
    
    Args:
        edge_dim: Input edge feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output edge feature dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            edge_features: Raw edge features [E, edge_dim]
            
        Returns:
            Encoded edge features [E, output_dim]
        """
        return self.encoder(edge_features)


class GraphFeatureEncoder(nn.Module):
    """Neural network encoder for graph features.
    
    Args:
        graph_dim: Input graph feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output graph feature dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        graph_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.graph_dim = graph_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            graph_features: Raw graph features [graph_dim]
            
        Returns:
            Encoded graph features [output_dim]
        """
        return self.encoder(graph_features)


def get_edge_feature_dim(
    include_team: bool = True,
    include_ball: bool = True,
    include_set_piece: bool = True,
) -> int:
    """Get edge feature dimension based on included features.
    
    Args:
        include_team: Whether to include team features
        include_ball: Whether to include ball features
        include_set_piece: Whether to include set piece features
        
    Returns:
        Edge feature dimension
    """
    dim = 2  # distance + bearing
    
    if include_team:
        dim += 3  # src_team + dst_team + same_team
    
    if include_ball:
        dim += 2  # src_to_ball_dist + dst_to_ball_dist
    
    if include_set_piece:
        dim += 2  # src_set_piece + dst_set_piece
    
    return dim


def get_graph_feature_dim(
    include_team: bool = True,
    include_ball: bool = True,
    include_set_piece: bool = True,
) -> int:
    """Get graph feature dimension based on included features.
    
    Args:
        include_team: Whether to include team features
        include_ball: Whether to include ball features
        include_set_piece: Whether to include set piece features
        
    Returns:
        Graph feature dimension
    """
    dim = 4  # coverage_x + coverage_y + spread_x + spread_y
    
    if include_team:
        dim += 9  # team_a_center(2) + team_a_spread(2) + team_b_center(2) + team_b_spread(2) + team_distance(1)
    
    if include_ball:
        dim += 2  # ball_center + ball_spread
    
    if include_set_piece:
        dim += 2  # set_piece_count + set_piece_ratio
    
    return dim
