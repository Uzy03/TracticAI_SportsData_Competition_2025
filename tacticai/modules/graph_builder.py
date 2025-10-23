"""Graph construction utilities for TacticAI.

This module provides utilities for building graphs from tactical data,
including complete graphs and k-nearest neighbor graphs.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import numpy as np


def build_complete_graph(num_nodes: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Build a complete graph connectivity matrix.
    
    Args:
        num_nodes: Number of nodes in the graph
        device: Device to place the tensor on
        
    Returns:
        Edge index tensor [2, num_edges] where num_edges = num_nodes * (num_nodes - 1)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Create all possible edges (excluding self-loops)
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def build_knn_graph(
    positions: torch.Tensor, 
    k: int = 5,
    include_self: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build k-nearest neighbor graph from positions.
    
    Args:
        positions: Node positions [N, 2] (x, y coordinates)
        k: Number of nearest neighbors
        include_self: Whether to include self-loops
        device: Device to place the tensor on
        
    Returns:
        Edge index tensor [2, num_edges]
    """
    if device is None:
        device = positions.device
    
    num_nodes = positions.size(0)
    
    # Compute pairwise distances
    dist = torch.cdist(positions, positions, p=2)  # [N, N]
    
    # Get k nearest neighbors (excluding self if include_self=False)
    if include_self:
        _, indices = torch.topk(dist, k, dim=1)
    else:
        # Set diagonal to infinity to exclude self
        dist = dist + torch.eye(num_nodes, device=device) * float('inf')
        _, indices = torch.topk(dist, k, dim=1)
    
    # Build edge index
    edges = []
    for i in range(num_nodes):
        for j in indices[i]:
            edges.append([i, j.item()])
    
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def compute_edge_attributes(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    include_distance: bool = True,
    include_bearing: bool = True,
    include_team_info: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Compute edge attributes from node positions and optional team information.
    
    Args:
        positions: Node positions [N, 2]
        edge_index: Edge connectivity [2, E]
        include_distance: Whether to include distance as edge attribute
        include_bearing: Whether to include bearing as edge attribute
        include_team_info: Team information [N] (optional)
        
    Returns:
        Edge attributes [E, edge_dim] or None if no attributes requested
    """
    num_edges = edge_index.size(1)
    edge_attrs = []
    
    src, dst = edge_index[0], edge_index[1]
    
    # Distance
    if include_distance:
        src_pos = positions[src]
        dst_pos = positions[dst]
        distance = torch.norm(dst_pos - src_pos, dim=1, keepdim=True)
        edge_attrs.append(distance)
    
    # Bearing (angle from source to destination)
    if include_bearing:
        src_pos = positions[src]
        dst_pos = positions[dst]
        diff = dst_pos - src_pos
        bearing = torch.atan2(diff[:, 1], diff[:, 0]).unsqueeze(1)
        edge_attrs.append(bearing)
    
    # Team information
    if include_team_info is not None:
        src_team = include_team_info[src].float()
        dst_team = include_team_info[dst].float()
        same_team = (src_team == dst_team).float().unsqueeze(1)
        edge_attrs.append(same_team)
    
    if not edge_attrs:
        return None
    
    return torch.cat(edge_attrs, dim=1)


class GraphBuilder:
    """Graph builder for tactical data.
    
    This class provides utilities for building graphs from tactical frames,
    including node feature extraction and edge construction.
    """
    
    def __init__(
        self,
        graph_type: str = "complete",
        k: int = 5,
        include_self_loops: bool = True,
        include_edge_attrs: bool = True,
        normalize_positions: bool = True,
    ):
        """Initialize graph builder.
        
        Args:
            graph_type: Type of graph ('complete' or 'knn')
            k: Number of neighbors for knn graph
            include_self_loops: Whether to include self-loops
            include_edge_attrs: Whether to compute edge attributes
            normalize_positions: Whether to normalize positions to [0, 1]
        """
        self.graph_type = graph_type
        self.k = k
        self.include_self_loops = include_self_loops
        self.include_edge_attrs = include_edge_attrs
        self.normalize_positions = normalize_positions
        
        # Field dimensions (standard football field)
        self.field_length = 105.0  # meters
        self.field_width = 68.0   # meters
    
    def extract_node_features(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        player_attributes: Optional[torch.Tensor] = None,
        ball_info: Optional[torch.Tensor] = None,
        team_info: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract node features from tactical data.
        
        Args:
            positions: Player positions [N, 2] (x, y)
            velocities: Player velocities [N, 2] (vx, vy) (optional)
            player_attributes: Player attributes [N, A] (optional)
            ball_info: Ball information [N] (0=no ball, 1=has ball) (optional)
            team_info: Team information [N] (0=team1, 1=team2) (optional)
            
        Returns:
            Node features [N, feature_dim]
        """
        num_nodes = positions.size(0)
        features = []
        
        # Normalize positions to [0, 1]
        if self.normalize_positions:
            normalized_pos = positions.clone()
            normalized_pos[:, 0] = positions[:, 0] / self.field_length
            normalized_pos[:, 1] = positions[:, 1] / self.field_width
            features.append(normalized_pos)
        else:
            features.append(positions)
        
        # Add velocities if provided
        if velocities is not None:
            features.append(velocities)
        else:
            # Add zero velocities as placeholder
            zero_vel = torch.zeros(num_nodes, 2, device=positions.device)
            features.append(zero_vel)
        
        # Add player attributes if provided
        if player_attributes is not None:
            features.append(player_attributes)
        else:
            # Add default attributes (height, weight, etc.)
            default_attrs = torch.zeros(num_nodes, 2, device=positions.device)
            features.append(default_attrs)
        
        # Add ball information if provided
        if ball_info is not None:
            features.append(ball_info.unsqueeze(1))
        else:
            # Add zero ball info as placeholder
            zero_ball = torch.zeros(num_nodes, 1, device=positions.device)
            features.append(zero_ball)
        
        # Add team information if provided
        if team_info is not None:
            features.append(team_info.unsqueeze(1))
        else:
            # Add zero team info as placeholder
            zero_team = torch.zeros(num_nodes, 1, device=positions.device)
            features.append(zero_team)
        
        return torch.cat(features, dim=1)
    
    def build_graph(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        player_attributes: Optional[torch.Tensor] = None,
        ball_info: Optional[torch.Tensor] = None,
        team_info: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Build complete graph from tactical data.
        
        Args:
            positions: Player positions [N, 2]
            velocities: Player velocities [N, 2] (optional)
            player_attributes: Player attributes [N, A] (optional)
            ball_info: Ball information [N] (optional)
            team_info: Team information [N] (optional)
            
        Returns:
            Tuple of (node_features, edge_index, edge_attributes)
        """
        # Extract node features
        node_features = self.extract_node_features(
            positions, velocities, player_attributes, ball_info, team_info
        )
        
        # Build edge index
        if self.graph_type == "complete":
            edge_index = build_complete_graph(
                positions.size(0), device=positions.device
            )
        elif self.graph_type == "knn":
            edge_index = build_knn_graph(
                positions, k=self.k, include_self=self.include_self_loops, 
                device=positions.device
            )
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        # Compute edge attributes
        edge_attrs = None
        if self.include_edge_attrs:
            edge_attrs = compute_edge_attributes(
                positions, edge_index, 
                include_distance=True, include_bearing=True,
                include_team_info=team_info
            )
        
        return node_features, edge_index, edge_attrs
    
    def create_dummy_data(
        self, 
        num_players: int = 22,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Create dummy tactical data for testing.
        
        Args:
            num_players: Number of players
            device: Device to place tensors on
            
        Returns:
            Tuple of (node_features, edge_index, edge_attributes)
        """
        if device is None:
            device = torch.device('cpu')
        
        # Random positions on field
        positions = torch.rand(num_players, 2, device=device)
        positions[:, 0] *= self.field_length  # x: 0 to 105
        positions[:, 1] *= self.field_width   # y: 0 to 68
        
        # Random velocities
        velocities = torch.randn(num_players, 2, device=device) * 2.0
        
        # Random player attributes (height, weight)
        player_attrs = torch.rand(num_players, 2, device=device)
        player_attrs[:, 0] = 1.7 + player_attrs[:, 0] * 0.3  # height: 1.7-2.0m
        player_attrs[:, 1] = 60 + player_attrs[:, 1] * 30   # weight: 60-90kg
        
        # Team information (first half = team 0, second half = team 1)
        team_info = torch.zeros(num_players, device=device, dtype=torch.long)
        team_info[num_players//2:] = 1
        
        # Ball information (randomly assign to one player)
        ball_info = torch.zeros(num_players, device=device)
        ball_owner = torch.randint(0, num_players, (1,), device=device)
        ball_info[ball_owner] = 1
        
        return self.build_graph(
            positions, velocities, player_attrs, ball_info, team_info
        )
