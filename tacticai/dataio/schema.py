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
        """Compute edge attributes from positions and connectivity.
        
        Args:
            positions: Player positions [N, 2] (x, y in meters)
            edge_index: Edge connectivity [2, E]
            team_ids: Team IDs [N] (optional)
            
        Returns:
            Edge attributes [E, 3] (distance, bearing, same_team)
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Compute distances
        src_pos = positions[src]  # [E, 2]
        dst_pos = positions[dst]  # [E, 2]
        distances = torch.norm(dst_pos - src_pos, dim=1)  # [E]
        
        # Normalize distances if requested
        if self.normalize_distance:
            distances = distances / self.max_distance
        
        # Compute bearings (angle from source to destination)
        dx = dst_pos[:, 0] - src_pos[:, 0]  # [E]
        dy = dst_pos[:, 1] - src_pos[:, 1]  # [E]
        bearings = torch.atan2(dy, dx)  # [E] in radians
        
        # Normalize bearings to [0, 2π]
        bearings = torch.fmod(bearings + 2 * math.pi, 2 * math.pi)
        
        # Compute same team indicator
        if team_ids is not None:
            same_team = (team_ids[src] == team_ids[dst]).float()  # [E]
        else:
            # Default: assume alternating teams
            same_team = ((src // 11) == (dst // 11)).float()  # Assuming 11 players per team
        
        # Stack attributes
        edge_attrs = torch.stack([distances, bearings, same_team], dim=1)  # [E, 3]
        
        return edge_attrs


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
        """Extract node features for receiver prediction.
        
        Args:
            data: Raw data dictionary containing player information
            
        Returns:
            Node features [N, F] where F includes positions, velocities, attributes
        """
        features = []
        
        # Extract positions
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        # Normalize positions to [0, 1]
        normalized_positions = positions.copy()
        normalized_positions[:, 0] = positions[:, 0] / self.field_length
        normalized_positions[:, 1] = positions[:, 1] / self.field_width
        features.append(torch.tensor(normalized_positions, dtype=torch.float32))
        
        # Extract velocities if available
        if self.velocity_columns:
            if isinstance(data, pd.DataFrame):
                velocities = data[self.velocity_columns].values
            else:
                velocities = np.array([data[col] for col in self.velocity_columns]).T
            features.append(torch.tensor(velocities, dtype=torch.float32))
        else:
            # Add zero velocities as placeholder
            features.append(torch.zeros(positions.shape[0], 2, dtype=torch.float32))
        
        # Extract player attributes if available
        if self.player_attr_columns:
            if isinstance(data, pd.DataFrame):
                attrs = data[self.player_attr_columns].values
            else:
                attrs = np.array([data[col] for col in self.player_attr_columns]).T
            features.append(torch.tensor(attrs, dtype=torch.float32))
        else:
            # Add default attributes (height, weight)
            features.append(torch.zeros(positions.shape[0], 2, dtype=torch.float32))
        
        # Extract ball information if available
        if self.ball_column:
            if isinstance(data, pd.DataFrame):
                ball_info = data[self.ball_column].values
            else:
                ball_info = np.array(data[self.ball_column])
            features.append(torch.tensor(ball_info, dtype=torch.float32).unsqueeze(1))
        else:
            # Add zero ball info as placeholder
            features.append(torch.zeros(positions.shape[0], 1, dtype=torch.float32))
        
        # Extract team information if available
        if self.team_column:
            if isinstance(data, pd.DataFrame):
                team_info = data[self.team_column].values
            else:
                team_info = np.array(data[self.team_column])
            features.append(torch.tensor(team_info, dtype=torch.float32).unsqueeze(1))
        else:
            # Add zero team info as placeholder
            features.append(torch.zeros(positions.shape[0], 1, dtype=torch.float32))
        
        return torch.cat(features, dim=1)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity (complete graph).
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E]
        """
        # For receiver prediction, we typically use complete graphs
        num_nodes = self._get_num_nodes(data)
        
        # Create complete graph
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract receiver targets.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Receiver ID tensor [N]
        """
        if isinstance(data, pd.DataFrame):
            receiver_id = data[self.receiver_column].iloc[0]  # Assuming same for all players
        else:
            receiver_id = data[self.receiver_column]
        
        return torch.tensor(receiver_id, dtype=torch.long)
    
    def get_edge_attributes(self, data: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract edge attributes for receiver prediction.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge attributes tensor [E, 3] or None
        """
        if not self.use_edge_attributes or self.edge_schema is None:
            return None
        
        # Get positions
        if isinstance(data, pd.DataFrame):
            positions = data[self.position_columns].values
        else:
            positions = np.array([data[col] for col in self.position_columns]).T
        
        # Convert to meters (assuming positions are in normalized coordinates)
        positions = torch.tensor(positions, dtype=torch.float32)
        positions[:, 0] *= self.field_length
        positions[:, 1] *= self.field_width
        
        # Get edge index
        edge_index = self.get_edge_index(data)
        
        # Get team IDs if available
        team_ids = None
        if self.team_column:
            if isinstance(data, pd.DataFrame):
                team_ids = data[self.team_column].values
            else:
                team_ids = np.array(data[self.team_column])
            team_ids = torch.tensor(team_ids, dtype=torch.long)
        
        # Compute edge attributes
        edge_attrs = self.edge_schema.compute_edge_attributes(positions, edge_index, team_ids)
        
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
            field_length: Field length for normalization
            field_width: Field width for normalization
        """
        self.position_columns = position_columns
        self.velocity_columns = velocity_columns or []
        self.player_attr_columns = player_attr_columns or []
        self.team_column = team_column
        self.ball_column = ball_column
        self.shot_column = shot_column
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
