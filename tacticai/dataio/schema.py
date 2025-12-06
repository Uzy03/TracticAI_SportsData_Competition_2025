"""Data schema definitions for TacticAI.

This module defines data schemas for different TacticAI tasks and provides
utilities for mapping between different data formats.

Enhanced with specific edge and graph attributes:
- edge_attr: [distance, bearing, same_team]
- graph_attr: [side(L/R), swing(in/out), kicker_foot, defense_scheme]
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
import numpy as np
import pandas as pd
import math


class EdgeAttributeSchema:
    """Schema for edge attributes in tactical graphs.
    
    Defines standardized edge attributes:
    - distance: Euclidean distance between players (meters)
    - bearing: Angle from source to destination player (radians)
    - same_team: Binary indicator for same team (0/1)
    """
    
    def __init__(self, normalize_distance: bool = True, max_distance: float = 100.0):
        """Initialize edge attribute schema.
        
        Args:
            normalize_distance: Whether to normalize distance to [0, 1]
            max_distance: Maximum distance for normalization (meters)
        """
        self.normalize_distance = normalize_distance
        self.max_distance = max_distance
    
    def compute_edge_attributes(
        self, 
        positions: torch.Tensor, 
        edge_index: torch.Tensor,
        team_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute edge attributes from positions and connectivity (same_team only).
        
        Args:
            positions: Player positions [N, 2] (x, y in meters)
            edge_index: Edge connectivity [2, E] (including self-loops)
            team_ids: Team IDs [N] (optional)
            
        Returns:
            Edge attributes [E, 1] (same_team only)
            - same_team = 1 for same team edges and self-loops
            - same_team = 0 for opponent edges
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Compute same team indicator
        if team_ids is not None:
            same_team = (team_ids[src] == team_ids[dst]).float().unsqueeze(-1)  # [E, 1]
        else:
            # Default: assume alternating teams
            same_team = ((src // 11) == (dst // 11)).float().unsqueeze(-1)  # [E, 1]
        
        # Self-loops (i==j) are always same_team=1 (TacticAI spec)
        self_loop_mask = (src == dst)
        same_team[self_loop_mask] = 1.0
        
        return same_team


class GraphAttributeSchema:
    """Schema for graph-level attributes in tactical graphs.
    
    Defines standardized graph attributes:
    - side: Field side (L/R) - 0 for left, 1 for right
    - swing: Ball swing direction (in/out) - 0 for in, 1 for out
    - kicker_foot: Kicker's preferred foot (left/right) - 0 for left, 1 for right
    - defense_scheme: Defensive formation (4-4-2, 3-5-2, etc.) - encoded as integer
    """
    
    def __init__(self):
        """Initialize graph attribute schema."""
        self.defense_schemes = {
            "4-4-2": 0,
            "3-5-2": 1,
            "4-3-3": 2,
            "3-4-3": 3,
            "5-3-2": 4,
            "4-5-1": 5,
            "3-4-1-2": 6,
            "4-2-3-1": 7,
            "unknown": 8,
        }
    
    def compute_graph_attributes(
        self,
        positions: torch.Tensor,
        ball_position: Optional[torch.Tensor] = None,
        kicker_id: Optional[int] = None,
        defense_scheme: Optional[str] = None,
        field_length: float = 105.0,
        field_width: float = 68.0,
    ) -> torch.Tensor:
        """Compute graph-level attributes.
        
        Args:
            positions: Player positions [N, 2] (x, y in meters)
            ball_position: Ball position [2] (optional)
            kicker_id: ID of the kicker (optional)
            defense_scheme: Defensive formation name (optional)
            field_length: Field length in meters
            field_width: Field width in meters
            
        Returns:
            Graph attributes [4] (side, swing, kicker_foot, defense_scheme)
        """
        # Determine field side based on ball position or center of mass
        if ball_position is not None:
            side = 1 if ball_position[0] > field_length / 2 else 0
        else:
            # Use center of mass of attacking team (first 11 players)
            attacking_center = positions[:11].mean(dim=0)
            side = 1 if attacking_center[0] > field_length / 2 else 0
        
        # Determine swing direction based on ball movement or field position
        if ball_position is not None:
            # Simple heuristic: closer to sidelines = out swing
            swing = 1 if abs(ball_position[1] - field_width / 2) > field_width / 4 else 0
        else:
            # Default to in swing
            swing = 0
        
        # Kicker foot (default to right foot)
        kicker_foot = 1  # Right foot
        if kicker_id is not None:
            # This would need actual player data - using default for now
            pass
        
        # Defense scheme
        defense_scheme_id = self.defense_schemes.get(defense_scheme, 8)  # Default to unknown
        
        # Stack attributes
        graph_attrs = torch.tensor([side, swing, kicker_foot, defense_scheme_id], dtype=torch.float32)
        
        return graph_attrs


class DataSchema(ABC):
    """Abstract base class for data schemas.
    
    Defines the interface for data schemas used in TacticAI tasks.
    """
    
    @abstractmethod
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features from raw data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Node features tensor [N, F]
        """
        pass
    
    @abstractmethod
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity from raw data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E]
        """
        pass
    
    @abstractmethod
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract target labels from raw data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Target labels tensor
        """
        pass
    
    def get_edge_attributes(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract edge attributes from raw data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge attributes tensor [E, edge_dim] or None
        """
        return None
    
    def get_graph_attributes(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract graph-level attributes from raw data.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Graph attributes tensor [graph_dim] or None
        """
        return None


class ReceiverSchema(DataSchema):
    """Schema for receiver prediction task.
    
    Maps raw data to receiver prediction format.
    Enhanced with edge and graph attributes.
    """
    
    def __init__(
        self,
        position_columns: List[str] = ["x", "y"],
        velocity_columns: Optional[List[str]] = None,
        player_attr_columns: Optional[List[str]] = None,
        team_column: Optional[str] = None,
        ball_column: Optional[str] = None,
        receiver_column: str = "receiver_id",
        field_length: float = 105.0,
        field_width: float = 68.0,
        use_edge_attributes: bool = True,
        use_graph_attributes: bool = True,
    ):
        """Initialize receiver schema.
        
        Args:
            position_columns: Column names for player positions
            velocity_columns: Column names for player velocities (optional)
            player_attr_columns: Column names for player attributes (optional)
            team_column: Column name for team information (optional)
            ball_column: Column name for ball possession (optional)
            receiver_column: Column name for receiver ID
            field_length: Field length for normalization
            field_width: Field width for normalization
            use_edge_attributes: Whether to compute edge attributes
            use_graph_attributes: Whether to compute graph attributes
        """
        self.position_columns = position_columns
        self.velocity_columns = velocity_columns or []
        self.player_attr_columns = player_attr_columns or []
        self.team_column = team_column
        self.ball_column = ball_column
        self.receiver_column = receiver_column
        self.field_length = field_length
        self.field_width = field_width
        self.use_edge_attributes = use_edge_attributes
        self.use_graph_attributes = use_graph_attributes
        
        # Initialize attribute schemas
        if use_edge_attributes:
            self.edge_schema = EdgeAttributeSchema(normalize_distance=True, max_distance=100.0)
        else:
            self.edge_schema = None
            
        if use_graph_attributes:
            self.graph_schema = GraphAttributeSchema()
        else:
            self.graph_schema = None
    
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features for receiver prediction (enhanced with relative features: 16 dimensions).
        
        Args:
            data: Raw data dictionary containing player information
            
            Returns:
            Node features [N, 16] where:
                - dim 0-1: x, y (normalized positions to [0, 1])
                - dim 2-3: vx, vy (normalized velocities to [-1, 1] range)
                - dim 4: height (normalized to [0, 1] range)
                - dim 5: weight (normalized to [0, 1] range)
                - dim 6: ball_possession (1.0 if player has ball, 0.0 otherwise)
                - dim 7-10: dx_to_kicker, dy_to_kicker, dist_to_kicker, angle_to_kicker (relative to kicker)
                - dim 11-14: dx_to_goal, dy_to_goal, dist_to_goal, angle_to_goal (relative to goal)
                - dim 15: team_id (0=attacking team, 1=defending team)
        """
        features = []
        
        # Extract positions (x, y) - TacticAI paper baseline
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        # Normalize positions to [0, 1]
        normalized_positions = positions.copy()
        normalized_positions[:, 0] = positions[:, 0] / self.field_length
        normalized_positions[:, 1] = positions[:, 1] / self.field_width
        features.append(torch.tensor(normalized_positions, dtype=torch.float32))
        
        # Extract velocities (vx, vy) - TacticAI paper baseline
        if self.velocity_columns:
            if isinstance(data, pd.DataFrame):
                velocities = data[self.velocity_columns].values
            else:
                velocities = np.array([data[col] for col in self.velocity_columns]).T
            # Normalize velocities to similar scale as positions
            # Maximum speed: typically 0-70 m/s (soccer players), normalize to [-1, 1] range (divide by 70.0)
            # This allows both positive and negative velocities (directions)
            max_velocity = 70.0  # m/s (typical maximum speed for soccer players)
            normalized_velocities = velocities / max_velocity
            features.append(torch.tensor(normalized_velocities, dtype=torch.float32))
        else:
            # Add zero velocities as placeholder
            features.append(torch.zeros(positions.shape[0], 2, dtype=torch.float32))
        
        # Extract player attributes (height, weight) - TacticAI paper baseline
        if self.player_attr_columns:
            if isinstance(data, pd.DataFrame):
                attrs = data[self.player_attr_columns].values
            else:
                attrs = np.array([data[col] for col in self.player_attr_columns]).T
            # Normalize attributes to similar scale as positions
            # Height: typically 1.7-2.0m, normalize to [0, 1] range (divide by 2.0)
            # Weight: typically 60-90kg, normalize to [0, 1] range (divide by 100.0)
            if attrs.shape[1] >= 2:
                normalized_attrs = attrs.copy()
                normalized_attrs[:, 0] = attrs[:, 0] / 2.0  # height: divide by 2.0m
                normalized_attrs[:, 1] = attrs[:, 1] / 100.0  # weight: divide by 100.0kg
                features.append(torch.tensor(normalized_attrs, dtype=torch.float32))
            else:
                features.append(torch.tensor(attrs, dtype=torch.float32))
        else:
            # Add default attributes (height, weight)
            features.append(torch.zeros(positions.shape[0], 2, dtype=torch.float32))
        
        # Extract ball possession (ball_possession) - TacticAI paper baseline
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_info = data[self.ball_column].values
            else:
                ball_info = np.array(data[self.ball_column])
            features.append(torch.tensor(ball_info, dtype=torch.float32).unsqueeze(1))
        else:
            # Add zero ball info as placeholder
            features.append(torch.zeros(positions.shape[0], 1, dtype=torch.float32))
        
        # Extract relative features to kicker (dx_to_kicker, dy_to_kicker, dist_to_kicker, angle_to_kicker)
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        num_nodes = positions_tensor.shape[0]
        
        # Get kicker index (same logic as get_global_features)
        kicker_idx = None
        ball_idx = None
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_info = data[self.ball_column].values
            else:
                ball_info = np.array(data[self.ball_column])
            ball_info = np.array(ball_info)
            if ball_info.sum() > 0:
                ball_idx = int(np.argmax(ball_info))
        
        if "kicker_idx" in data and data["kicker_idx"] is not None:
            kicker_idx = int(data["kicker_idx"])
        elif ball_idx is not None:
            kicker_idx = ball_idx
        
        if kicker_idx is not None and kicker_idx < num_nodes:
            kicker_pos = positions_tensor[kicker_idx]  # [2]
            # Compute relative position to kicker for each node
            dx_to_kicker = positions_tensor[:, 0] - kicker_pos[0]  # [N]
            dy_to_kicker = positions_tensor[:, 1] - kicker_pos[1]  # [N]
            dist_to_kicker = torch.sqrt(dx_to_kicker ** 2 + dy_to_kicker ** 2 + 1e-6)  # [N] (add small epsilon to avoid zero)
            angle_to_kicker = torch.atan2(dy_to_kicker, dx_to_kicker)  # [N] (in radians, -π to π)
            
            # Normalize relative features
            # Distance: normalize by field diagonal (max distance on field)
            max_dist = math.sqrt(self.field_length ** 2 + self.field_width ** 2)
            dist_to_kicker = dist_to_kicker / max_dist  # Normalize to [0, 1]
            # Angle: normalize to [-1, 1] (divide by π)
            angle_to_kicker = angle_to_kicker / math.pi  # Normalize to [-1, 1]
            # dx, dy: normalize by field dimensions
            dx_to_kicker = dx_to_kicker / self.field_length  # Normalize to [-1, 1]
            dy_to_kicker = dy_to_kicker / self.field_width   # Normalize to [-1, 1]
            
            relative_to_kicker = torch.stack([dx_to_kicker, dy_to_kicker, dist_to_kicker, angle_to_kicker], dim=1)  # [N, 4]
        else:
            # If no kicker info, use zeros
            relative_to_kicker = torch.zeros(num_nodes, 4, dtype=torch.float32)
        
        features.append(relative_to_kicker)
        
        # Extract relative features to goal (dx_to_goal, dy_to_goal, dist_to_goal, angle_to_goal)
        # Goal position: typically at the end of the field (x = field_length for attacking team)
        # For corner kicks, attacking team is usually on one side
        # Use goal at x = field_length (opponent's goal)
        goal_x = self.field_length
        goal_y = self.field_width / 2.0  # Center of goal (goal width is field width)
        goal_pos = torch.tensor([goal_x, goal_y], dtype=torch.float32)
        
        # Compute relative position to goal for each node
        dx_to_goal = positions_tensor[:, 0] - goal_pos[0]  # [N]
        dy_to_goal = positions_tensor[:, 1] - goal_pos[1]  # [N]
        dist_to_goal = torch.sqrt(dx_to_goal ** 2 + dy_to_goal ** 2 + 1e-6)  # [N]
        angle_to_goal = torch.atan2(dy_to_goal, dx_to_goal)  # [N]
        
        # Normalize relative features (same as kicker)
        max_dist = math.sqrt(self.field_length ** 2 + self.field_width ** 2)
        dist_to_goal = dist_to_goal / max_dist  # Normalize to [0, 1]
        angle_to_goal = angle_to_goal / math.pi  # Normalize to [-1, 1]
        dx_to_goal = dx_to_goal / self.field_length  # Normalize to [-1, 1]
        dy_to_goal = dy_to_goal / self.field_width   # Normalize to [-1, 1]
        
        relative_to_goal = torch.stack([dx_to_goal, dy_to_goal, dist_to_goal, angle_to_goal], dim=1)  # [N, 4]
        features.append(relative_to_goal)
        
        # Extract team ID (0=attacking team, 1=defending team)
        if self.team_column:
            if isinstance(data, pd.DataFrame):
                team_info = data[self.team_column].values
            else:
                team_info = np.array(data[self.team_column])
            team_tensor = torch.tensor(team_info, dtype=torch.float32)
            if team_tensor.dim() == 1:
                team_tensor = team_tensor.unsqueeze(1)  # [N, 1]
            features.append(team_tensor)
        else:
            # Default: assume alternating teams (first 11 = attacking team, last 11 = defending team)
            team_id_tensor = torch.zeros(num_nodes, 1, dtype=torch.float32)
            team_id_tensor[11:] = 1.0  # Last 11 players are defending team
            features.append(team_id_tensor)
        
        # Total: 7 (baseline) + 4 (kicker) + 4 (goal) + 1 (team) = 16 dimensions
        return torch.cat(features, dim=1)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity (complete graph with self-loops).
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E] where E = num_nodes * num_nodes (22×22 = 484)
        """
        # For receiver prediction, use complete graph with self-loops (TacticAI spec)
        num_nodes = self._get_num_nodes(data)
        
        # Create complete graph with self-loops (22×22 = 484 edges)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                edges.append([i, j])  # Include self-loops
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract receiver targets.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Receiver node index tensor (0-21)
        """
        # Prefer receiver_node_index if available (0-21), otherwise fall back to receiver_id
        receiver_idx = None
        
        if isinstance(data, pd.DataFrame):
            if "receiver_node_index" in data.columns:
                receiver_idx = data["receiver_node_index"].iloc[0]
                if pd.isna(receiver_idx):
                    receiver_idx = None
            if receiver_idx is None:
                # Fallback: use receiver_id, but this should be node index (0-21)
                receiver_idx = data[self.receiver_column].iloc[0]
        else:
            if "receiver_node_index" in data and data["receiver_node_index"] is not None:
                receiver_idx = data["receiver_node_index"]
            if receiver_idx is None:
                # Fallback: use receiver_id, but this should be node index (0-21)
                receiver_id = data[self.receiver_column]
                # Check if it's a valid node index (0-21) or a player ID
                if receiver_id <= 21:
                    receiver_idx = receiver_id
                else:
                    # This is a player ID, not a node index
                    # Fallback: use ball owner (closest to ball) or first node
                    if "ball" in data:
                        ball = np.array(data["ball"])
                        ball_owner = int(np.argmax(ball)) if ball.sum() > 0 else 0
                        receiver_idx = ball_owner
                    else:
                        # Ultimate fallback: use node 0
                        receiver_idx = 0
        
        return torch.tensor(int(receiver_idx), dtype=torch.long)
    
    def get_edge_attributes(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract edge attributes for receiver prediction (enhanced with distance and angle).
        
        Returns 10-dimensional edge features:
        - dx: x方向の位置差（正規化）
        - dy: y方向の位置差（正規化）
        - dist_ij: ノードiとjの距離（正規化）
        - angle_ij: ノードiからjへの角度（正規化）
        - same_team: binary indicator (1.0 if same team or self-loop, 0.0 otherwise)
        - dvx: x方向の速度差（正規化）
        - dvy: y方向の速度差（正規化）
        - rel_speed: 相対速度の大きさ（正規化）
        - from_kicker: キッカーから出るエッジかどうか (1.0/0.0)
        - to_kicker: キッカーへ向かうエッジかどうか (1.0/0.0)
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge attributes tensor [E, 10] or None
        """
        if not self.use_edge_attributes:
            return None
        
        # Get positions (x, y) - already normalized to [0, 1] in get_node_features
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        # Convert to tensor and denormalize to meters for distance calculations
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        positions_meters = positions_tensor.clone()
        positions_meters[:, 0] *= self.field_length
        positions_meters[:, 1] *= self.field_width
        
        # Get velocities (vx, vy)
        velocities = None
        if self.velocity_columns:
            if isinstance(data, pd.DataFrame):
                velocities = data[self.velocity_columns].values
            else:
                velocities = np.array([data[col] for col in self.velocity_columns]).T
            velocities = torch.tensor(velocities, dtype=torch.float32)
        else:
            velocities = torch.zeros(positions_tensor.shape[0], 2, dtype=torch.float32)
        
        # Get team IDs
        team_ids = None
        if self.team_column:
            if isinstance(data, pd.DataFrame):
                team_ids = data[self.team_column].values
            else:
                team_ids = np.array(data[self.team_column])
            team_ids = torch.tensor(team_ids, dtype=torch.long)
        else:
            # Fallback: assume alternating teams (0-10: team 0, 11-21: team 1)
            num_nodes = positions_tensor.shape[0]
            team_ids = torch.arange(num_nodes, dtype=torch.long) // 11
        
        # Get edge index
        edge_index = self.get_edge_index(data)
        src, dst = edge_index[0], edge_index[1]
        
        # Compute edge features: dx, dy, dist_ij, angle_ij, same_team
        pos_src = positions_meters[src]  # [E, 2]
        pos_dst = positions_meters[dst]  # [E, 2]
        
        # Position differences (dx, dy) in meters
        dx = pos_dst[:, 0] - pos_src[:, 0]  # [E]
        dy = pos_dst[:, 1] - pos_src[:, 1]  # [E]
        
        # Distance (dist_ij) in meters
        dist_ij = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)  # [E] (add small epsilon to avoid zero)
        
        # Angle (angle_ij) in radians [-π, π]
        angle_ij = torch.atan2(dy, dx)  # [E]
        
        # Normalize edge features
        # dx, dy: normalize by field dimensions
        max_dist = math.sqrt(self.field_length ** 2 + self.field_width ** 2)  # Field diagonal
        dx_norm = dx / self.field_length  # Normalize to [-1, 1] range
        dy_norm = dy / self.field_width   # Normalize to [-1, 1] range
        dist_ij_norm = dist_ij / max_dist  # Normalize to [0, 1] range
        angle_ij_norm = angle_ij / math.pi  # Normalize to [-1, 1] range
        
        # Same team indicator
        same_team = (team_ids[src] == team_ids[dst]).float()  # [E]
        # Explicitly set self-loops to 1.0
        self_loop_mask = (src == dst)
        same_team[self_loop_mask] = 1.0
        
        # Velocity differences (dvx, dvy)
        vel_src = velocities[src]  # [E, 2]
        vel_dst = velocities[dst]  # [E, 2]
        dvx = vel_dst[:, 0] - vel_src[:, 0]
        dvy = vel_dst[:, 1] - vel_src[:, 1]
        rel_speed = torch.sqrt(dvx ** 2 + dvy ** 2 + 1e-6)
        
        # Normalize velocity features (relative speed ~20m/s max)
        max_rel_speed = 20.0
        dvx_norm = dvx / max_rel_speed
        dvy_norm = dvy / max_rel_speed
        rel_speed_norm = rel_speed / max_rel_speed
        
        # Kicker indicators
        # Identify kicker
        ball_idx = None
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_info = data[self.ball_column].values
            else:
                ball_info = np.array(data[self.ball_column])
            ball_info = np.array(ball_info)
            if ball_info.sum() > 0:
                ball_idx = int(np.argmax(ball_info))
        
        kicker_idx = None
        if "kicker_idx" in data and data["kicker_idx"] is not None:
            kicker_idx = int(data["kicker_idx"])
        elif ball_idx is not None:
            kicker_idx = ball_idx
            
        from_kicker = torch.zeros_like(same_team)
        to_kicker = torch.zeros_like(same_team)
        
        if kicker_idx is not None and kicker_idx < positions_tensor.shape[0]:
            from_kicker[src == kicker_idx] = 1.0
            to_kicker[dst == kicker_idx] = 1.0
        
        # Stack all edge features: [E, 10]
        edge_attrs = torch.stack([
            dx_norm,        # [E]
            dy_norm,        # [E]
            dist_ij_norm,   # [E]
            angle_ij_norm,  # [E]
            same_team,      # [E]
            dvx_norm,       # [E]
            dvy_norm,       # [E]
            rel_speed_norm, # [E]
            from_kicker,    # [E]
            to_kicker       # [E]
        ], dim=1)  # [E, 10]
        
        return edge_attrs
    
    def get_graph_attributes(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract graph attributes for receiver prediction.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Graph attributes tensor [4] or None
        """
        if not self.use_graph_attributes or self.graph_schema is None:
            return None
        
        # Get positions
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        # Convert to meters
        positions = torch.tensor(positions, dtype=torch.float32)
        positions[:, 0] *= self.field_length
        positions[:, 1] *= self.field_width
        
        # Get ball position if available
        ball_position = None
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_pos = data[self.ball_column].iloc[0]  # Assuming same for all players
            else:
                ball_pos = data[self.ball_column]
            if isinstance(ball_pos, (list, tuple)) and len(ball_pos) >= 2:
                ball_position = torch.tensor([ball_pos[0] * self.field_length, ball_pos[1] * self.field_width], dtype=torch.float32)
        
        # Compute graph attributes
        graph_attrs = self.graph_schema.compute_graph_attributes(
            positions, ball_position, field_length=self.field_length, field_width=self.field_width
        )
        
        return graph_attrs
    
    def get_global_features(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract global features for receiver prediction (TacticAI spec).
        
        Returns 8-dimensional global features:
        - ball_x, ball_y, ball_vx, ball_vy: ball position and velocity
        - kicker_x, kicker_y, kicker_vx, kicker_vy: kicker position and velocity
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Global features tensor [8] or None
        """
        # Get positions (x, y)
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        num_nodes = positions_tensor.shape[0]
        
        # Get velocities (vx, vy)
        velocities = None
        if self.velocity_columns:
            if isinstance(data, pd.DataFrame):
                velocities = data[self.velocity_columns].values
            else:
                velocities = np.array([data[col] for col in self.velocity_columns]).T
            velocities = torch.tensor(velocities, dtype=torch.float32)
        else:
            velocities = torch.zeros(num_nodes, 2, dtype=torch.float32)
        
        # Get ball information
        ball_idx = None
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_info = data[self.ball_column].values
            else:
                ball_info = np.array(data[self.ball_column])
            ball_info = np.array(ball_info)
            if ball_info.sum() > 0:
                ball_idx = int(np.argmax(ball_info))
        
        # Get kicker index
        # Priority: 1) data["kicker_idx"] if present, 2) ball_idx (ball owner = kicker)
        kicker_idx = None
        if "kicker_idx" in data and data["kicker_idx"] is not None:
            # Use explicit kicker index from sample dict
            kicker_idx = int(data["kicker_idx"])
        elif ball_idx is not None:
            # Fallback: ball owner node index = kicker index
            kicker_idx = ball_idx
        
        # Extract ball features (use kicker's position/velocity if ball info not available)
        if ball_idx is not None and ball_idx < num_nodes:
            ball_x = positions_tensor[ball_idx, 0].item()
            ball_y = positions_tensor[ball_idx, 1].item()
            ball_vx = velocities[ball_idx, 0].item()
            ball_vy = velocities[ball_idx, 1].item()
        elif kicker_idx is not None and kicker_idx < num_nodes:
            # Fallback to kicker's position if ball info not available
            ball_x = positions_tensor[kicker_idx, 0].item()
            ball_y = positions_tensor[kicker_idx, 1].item()
            ball_vx = velocities[kicker_idx, 0].item()
            ball_vy = velocities[kicker_idx, 1].item()
        else:
            # Default to zero if no ball/kicker info
            ball_x = ball_y = ball_vx = ball_vy = 0.0
        
        # Extract kicker features
        if kicker_idx is not None and kicker_idx < num_nodes:
            kicker_x = positions_tensor[kicker_idx, 0].item()
            kicker_y = positions_tensor[kicker_idx, 1].item()
            kicker_vx = velocities[kicker_idx, 0].item()
            kicker_vy = velocities[kicker_idx, 1].item()
        else:
            # Default to zero if no kicker info
            kicker_x = kicker_y = kicker_vx = kicker_vy = 0.0
        
        # Stack global features: [8]
        global_features = torch.tensor([
            ball_x, ball_y, ball_vx, ball_vy,
            kicker_x, kicker_y, kicker_vx, kicker_vy
        ], dtype=torch.float32)
        
        return global_features
    
    def _get_num_nodes(self, data: Dict[str, Any]) -> int:
        """Get number of nodes from data."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        else:
            return len(data[self.position_columns[0]])


class ShotSchema(DataSchema):
    """Schema for shot prediction task.
    
    Maps raw data to shot prediction format.
    """
    
    def __init__(
        self,
        position_columns: List[str] = ["x", "y"],
        velocity_columns: Optional[List[str]] = ["vx", "vy"],
        player_attr_columns: Optional[List[str]] = None,
        team_column: Optional[str] = "team",
        ball_column: Optional[str] = "ball",
        shot_column: str = "shot_occurred",
        receiver_column: Optional[str] = "receiver_id",
        field_length: float = 105.0,
        field_width: float = 68.0,
    ):
        """Initialize shot schema.
        
        Args:
            position_columns: Column names for player positions
            velocity_columns: Column names for player velocities (optional)
            player_attr_columns: Column names for player attributes (optional)
            team_column: Column name for team information (optional)
            ball_column: Column name for ball possession (optional)
            shot_column: Column name for shot occurrence
            receiver_column: Column name for receiver identifier (optional)
            field_length: Field length for normalization
            field_width: Field width for normalization
        """
        self.position_columns = position_columns
        self.velocity_columns = velocity_columns or []
        self.player_attr_columns = player_attr_columns or []
        self.team_column = team_column
        self.ball_column = ball_column
        self.shot_column = shot_column
        self.receiver_column = receiver_column
        self.field_length = field_length
        self.field_width = field_width
    
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features for shot prediction.
        
        Args:
            data: Raw data dictionary containing player information
            
        Returns:
            Node features [N, F]
        """
        # Extract positions
        try:
            positions = np.array([data[col] for col in self.position_columns]).T
        except Exception as e:
            print(f"Error in get_node_features: {e}")
            print(f"data type: {type(data)}")
            print(f"data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            print(f"position_columns: {self.position_columns}")
            raise
        
        # Extract velocities
        velocities = np.array([data[col] for col in self.velocity_columns]).T
        
        # Extract player attributes
        if self.player_attr_columns:
            player_attrs = np.array([data[col] for col in self.player_attr_columns]).T
        else:
            # Default attributes (height, weight)
            player_attrs = np.zeros((positions.shape[0], 2))
        
        # Extract team information
        team = np.array(data[self.team_column])
        
        # Extract ball information
        ball = np.array(data[self.ball_column])
        
        # Normalize positions to [-1, 1]
        positions[:, 0] = 2 * positions[:, 0] / self.field_length - 1
        positions[:, 1] = 2 * positions[:, 1] / self.field_width - 1
        
        # Combine all features
        features = np.column_stack([
            positions,      # [N, 2] - x, y positions
            velocities,     # [N, 2] - vx, vy velocities
            player_attrs,   # [N, 2] - height, weight
            team,           # [N, 1] - team_id
            ball,           # [N, 1] - is_ball
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity (complete graph).
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E]
        """
        # Use same edge extraction as receiver schema
        receiver_schema = ReceiverSchema(
            position_columns=self.position_columns,
            velocity_columns=self.velocity_columns,
            player_attr_columns=self.player_attr_columns,
            team_column=self.team_column,
            ball_column=self.ball_column,
            field_length=self.field_length,
            field_width=self.field_width,
        )
        return receiver_schema.get_edge_index(data)
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract shot targets.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Shot occurrence tensor [1]
        """
        if isinstance(data, pd.DataFrame):
            shot_occurred = data[self.shot_column].iloc[0]
        else:
            shot_occurred = data[self.shot_column]
        
        return torch.tensor(shot_occurred, dtype=torch.float32).unsqueeze(0)

    def get_receiver_target(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract receiver target if available."""
        if not self.receiver_column:
            raise ValueError("ShotSchema receiver_column is not set")

        if isinstance(data, pd.DataFrame):
            receiver_id = int(data[self.receiver_column].iloc[0])
        else:
            receiver_id = int(data[self.receiver_column])

        return torch.tensor(receiver_id, dtype=torch.long)


class CVAESchema(DataSchema):
    """Schema for CVAE tactic generation task.
    
    Maps raw data to CVAE format.
    """
    
    def __init__(
        self,
        position_columns: List[str] = ["x", "y"],
        velocity_columns: Optional[List[str]] = None,
        player_attr_columns: Optional[List[str]] = None,
        team_column: Optional[str] = None,
        ball_column: Optional[str] = None,
        condition_columns: Optional[List[str]] = None,
        target_position_columns: Optional[List[str]] = None,
        field_length: float = 105.0,
        field_width: float = 68.0,
    ):
        """Initialize CVAE schema.
        
        Args:
            position_columns: Column names for current player positions
            velocity_columns: Column names for player velocities (optional)
            player_attr_columns: Column names for player attributes (optional)
            team_column: Column name for team information (optional)
            ball_column: Column name for ball possession (optional)
            condition_columns: Column names for conditions (optional)
            target_position_columns: Column names for target positions (optional)
            field_length: Field length for normalization
            field_width: Field width for normalization
        """
        self.position_columns = position_columns
        self.velocity_columns = velocity_columns or []
        self.player_attr_columns = player_attr_columns or []
        self.team_column = team_column
        self.ball_column = ball_column
        self.condition_columns = condition_columns or []
        self.target_position_columns = target_position_columns or position_columns
        self.field_length = field_length
        self.field_width = field_width
    
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features for CVAE.
        
        Args:
            data: Raw data dictionary containing player information
            
        Returns:
            Node features [N, F]
        """
        # Use same feature extraction as receiver schema
        receiver_schema = ReceiverSchema(
            position_columns=self.position_columns,
            velocity_columns=self.velocity_columns,
            player_attr_columns=self.player_attr_columns,
            team_column=self.team_column,
            ball_column=self.ball_column,
            field_length=self.field_length,
            field_width=self.field_width,
        )
        return receiver_schema.get_node_features(data)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity (complete graph).
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E]
        """
        # Use same edge extraction as receiver schema
        receiver_schema = ReceiverSchema(
            position_columns=self.position_columns,
            velocity_columns=self.velocity_columns,
            player_attr_columns=self.player_attr_columns,
            team_column=self.team_column,
            ball_column=self.ball_column,
            field_length=self.field_length,
            field_width=self.field_width,
        )
        return receiver_schema.get_edge_index(data)
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract target positions for reconstruction.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Target positions tensor [N, 2]
        """
        if isinstance(data, pd.DataFrame):
            target_positions = data[self.target_position_columns].values
        else:
            target_positions = np.array([data[col] for col in self.target_position_columns]).T
        
        # Normalize positions to [0, 1]
        normalized_positions = target_positions.copy()
        normalized_positions[:, 0] = target_positions[:, 0] / self.field_length
        normalized_positions[:, 1] = target_positions[:, 1] / self.field_width
        
        return torch.tensor(normalized_positions, dtype=torch.float32)
    
    def get_conditions(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract conditions for CVAE.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Conditions tensor [condition_dim]
        """
        if not self.condition_columns:
            # Return default conditions if none specified
            return torch.zeros(8, dtype=torch.float32)
        
        if isinstance(data, pd.DataFrame):
            conditions = data[self.condition_columns].iloc[0].values
        else:
            conditions = np.array([data[col] for col in self.condition_columns])
        
        return torch.tensor(conditions, dtype=torch.float32)


def create_schema_mapping(
    task: str,
    custom_columns: Optional[Dict[str, List[str]]] = None
) -> DataSchema:
    """Create schema mapping for a specific task.
    
    Args:
        task: Task type ('receiver', 'shot', 'cvae')
        custom_columns: Custom column mappings (optional)
        
    Returns:
        Appropriate schema instance
    """
    if custom_columns is None:
        custom_columns = {}
    
    if task == "receiver":
        return ReceiverSchema(**custom_columns.get("receiver", {}))
    elif task == "shot":
        return ShotSchema(**custom_columns.get("shot", {}))
    elif task == "cvae":
        return CVAESchema(**custom_columns.get("cvae", {}))
    else:
        raise ValueError(f"Unknown task: {task}")


class FlexibleSchema(DataSchema):
    """Flexible schema that can adapt to different data formats.
    
    This schema automatically detects column names and creates appropriate mappings.
    """
    
    def __init__(self, task: str, data_sample: Union[pd.DataFrame, Dict[str, Any]]):
        """Initialize flexible schema.
        
        Args:
            task: Task type ('receiver', 'shot', 'cvae')
            data_sample: Sample of data to analyze column structure
        """
        self.task = task
        self.columns = self._analyze_columns(data_sample)
        self.base_schema = create_schema_mapping(task, self._create_column_mapping())
    
    def _analyze_columns(self, data_sample: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze data structure to identify relevant columns.
        
        Args:
            data_sample: Sample of data
            
        Returns:
            Dictionary mapping column types to column names
        """
        if isinstance(data_sample, pd.DataFrame):
            columns = data_sample.columns.tolist()
        else:
            columns = list(data_sample.keys())
        
        # Identify position columns
        position_columns = [col for col in columns if col.lower() in ['x', 'y', 'pos_x', 'pos_y', 'position_x', 'position_y']]
        
        # Identify velocity columns
        velocity_columns = [col for col in columns if col.lower() in ['vx', 'vy', 'vel_x', 'vel_y', 'velocity_x', 'velocity_y']]
        
        # Identify team column
        team_column = next((col for col in columns if col.lower() in ['team', 'team_id', 'side']), None)
        
        # Identify ball column
        ball_column = next((col for col in columns if col.lower() in ['ball', 'has_ball', 'ball_owner']), None)
        
        # Identify target columns based on task
        target_columns = []
        if self.task == "receiver":
            target_columns = [col for col in columns if col.lower() in ['receiver', 'receiver_id', 'target_player']]
        elif self.task == "shot":
            target_columns = [col for col in columns if col.lower() in ['shot', 'shot_occurred', 'is_shot']]
        
        return {
            "position": position_columns[:2],  # Take first 2 position columns
            "velocity": velocity_columns[:2],  # Take first 2 velocity columns
            "team": team_column,
            "ball": ball_column,
            "target": target_columns[0] if target_columns else None,
        }
    
    def _create_column_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Create column mapping for base schema.
        
        Returns:
            Column mapping dictionary
        """
        mapping = {
            "position_columns": self.columns["position"],
            "velocity_columns": self.columns["velocity"] if self.columns["velocity"] else None,
            "team_column": self.columns["team"],
            "ball_column": self.columns["ball"],
        }
        
        if self.task == "receiver":
            mapping["receiver_column"] = self.columns["target"]
        elif self.task == "shot":
            mapping["shot_column"] = self.columns["target"]
        
        return {self.task: mapping}
    
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features using base schema."""
        return self.base_schema.get_node_features(data)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge index using base schema."""
        return self.base_schema.get_edge_index(data)
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract targets using base schema."""
        return self.base_schema.get_targets(data)
