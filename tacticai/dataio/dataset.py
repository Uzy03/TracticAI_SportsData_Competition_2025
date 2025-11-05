"""Dataset classes for TacticAI tasks.

This module provides PyTorch dataset classes for different TacticAI tasks
including receiver prediction, shot prediction, and tactic generation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle

from .schema import DataSchema, create_schema_mapping


class TacticAIDataset(Dataset, ABC):
    """Abstract base class for TacticAI datasets.
    
    Provides common functionality for all TacticAI dataset implementations.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        schema: DataSchema,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
    ):
        """Initialize TacticAI dataset.
        
        Args:
            data_path: Path to data files
            schema: Data schema for parsing
            transform: Optional transform for input data
            target_transform: Optional transform for target data
        """
        self.data_path = Path(data_path)
        self.schema = schema
        self.transform = transform
        self.target_transform = target_transform
        
        # Load data
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from files.
        
        Returns:
            List of data samples
        """
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, target)
        """
        pass
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def _apply_transforms(
        self, 
        input_data: Dict[str, torch.Tensor], 
        target: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Apply transforms to data.
        
        Args:
            input_data: Input data dictionary
            target: Target tensor
            
        Returns:
            Transformed (input_data, target)
        """
        if self.transform is not None:
            input_data = self.transform(input_data)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return input_data, target


class ReceiverDataset(TacticAIDataset):
    """Dataset for receiver prediction task.
    
    Loads and processes data for predicting pass receivers.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        schema: Optional[DataSchema] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        file_format: str = "parquet",
    ):
        """Initialize receiver dataset.
        
        Args:
            data_path: Path to data files
            schema: Data schema (optional, will create default if None)
            transform: Optional transform for input data
            target_transform: Optional transform for target data
            file_format: Data file format ('parquet', 'csv', 'pickle')
        """
        if schema is None:
            schema = create_schema_mapping("receiver")
        
        self.file_format = file_format
        super().__init__(data_path, schema, transform, target_transform)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load receiver prediction data.
        
        Returns:
            List of data samples
        """
        if self.data_path.is_file():
            # Single file
            return self._load_single_file(self.data_path)
        else:
            # Directory of files
            data = []
            pattern = f"*.{self.file_format}"
            for file_path in self.data_path.glob(pattern):
                data.extend(self._load_single_file(file_path))
            return data
    
    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of data samples
        """
        if self.file_format == "parquet":
            df = pd.read_parquet(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "csv":
            df = pd.read_csv(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "pickle":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # If it's already a list, return it; otherwise convert to list
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single receiver prediction sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, receiver_target)
        """
        sample = self.data[idx]
        
        # Extract features using schema
        node_features = self.schema.get_node_features(sample)
        edge_index = self.schema.get_edge_index(sample)
        edge_attr = self.schema.get_edge_attributes(sample)  # TacticAI spec: include edge_attr
        target = self.schema.get_targets(sample)
        
        # Create batch tensor (all nodes belong to same graph)
        batch = torch.zeros(node_features.size(0), dtype=torch.long)
        
        input_data = {
            "x": node_features,
            "edge_index": edge_index,
            "batch": batch,
        }
        
        # Add edge_attr if available (TacticAI spec: same_team feature)
        if edge_attr is not None:
            input_data["edge_attr"] = edge_attr
        
        # Add mask, team, ball if available in sample
        if isinstance(sample, dict):
            if "mask" in sample:
                mask = np.array(sample["mask"])
                input_data["mask"] = torch.tensor(mask, dtype=torch.float32)
            if "team" in sample:
                team = np.array(sample["team"])
                input_data["team"] = torch.tensor(team, dtype=torch.long)
            if "ball" in sample:
                ball = np.array(sample["ball"])
                input_data["ball"] = torch.tensor(ball, dtype=torch.float32)
        
        # Apply transforms
        input_data, target = self._apply_transforms(input_data, target)
        
        return input_data, target


class ShotDataset(TacticAIDataset):
    """Dataset for shot prediction task.
    
    Loads and processes data for predicting shot occurrence.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        schema: Optional[DataSchema] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        file_format: str = "parquet",
    ):
        """Initialize shot dataset.
        
        Args:
            data_path: Path to data files
            schema: Data schema (optional, will create default if None)
            transform: Optional transform for input data
            target_transform: Optional transform for target data
            file_format: Data file format ('parquet', 'csv', 'pickle')
        """
        if schema is None:
            schema = create_schema_mapping("shot")
        
        self.file_format = file_format
        super().__init__(data_path, schema, transform, target_transform)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load shot prediction data.
        
        Returns:
            List of data samples
        """
        if self.data_path.is_file():
            # Single file
            return self._load_single_file(self.data_path)
        else:
            # Directory of files
            data = []
            pattern = f"*.{self.file_format}"
            for file_path in self.data_path.glob(pattern):
                data.extend(self._load_single_file(file_path))
            return data
    
    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of data samples
        """
        if self.file_format == "parquet":
            df = pd.read_parquet(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "csv":
            df = pd.read_csv(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "pickle":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # If it's already a list, return it
                if isinstance(data, list):
                    return data
                # If it's a dict, it should be a single sample, wrap it in a list
                elif isinstance(data, dict):
                    return [data]
                else:
                    return [data]
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single shot prediction sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, shot_target)
        """
        sample = self.data[idx]
        
        # Debug information (commented out for production)
        # print(f"Dataset __getitem__ called with idx={idx}")
        # print(f"Sample type: {type(sample)}")
        # print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
        # print(f"Sample content: {sample}")
        
        # Extract features using schema
        try:
            node_features = self.schema.get_node_features(sample)
        except Exception as e:
            print(f"Error in get_node_features: {e}")
            print(f"Sample type: {type(sample)}")
            print(f"Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
            print(f"Sample content: {sample}")
            print(f"Schema position_columns: {self.schema.position_columns}")
            print(f"Schema velocity_columns: {self.schema.velocity_columns}")
            print(f"Schema type: {type(self.schema)}")
            print(f"Schema dir: {dir(self.schema)}")
            print(f"Schema vars: {vars(self.schema)}")
            print(f"Schema repr: {repr(self.schema)}")
            print(f"Schema str: {str(self.schema)}")
            print(f"Schema getattr: {getattr(self.schema, 'position_columns', 'NOT_FOUND')}")
            print(f"Schema hasattr: {hasattr(self.schema, 'position_columns')}")
            print(f"Schema __dict__: {self.schema.__dict__}")
            print(f"Schema __class__: {self.schema.__class__}")
            print(f"Schema __bases__: {self.schema.__class__.__bases__}")
            print(f"Schema __mro__: {self.schema.__class__.__mro__}")
            print(f"Schema __module__: {self.schema.__class__.__module__}")
            print(f"Schema __qualname__: {self.schema.__class__.__qualname__}")
            print(f"Schema __name__: {self.schema.__class__.__name__}")
            print(f"Schema __doc__: {self.schema.__class__.__doc__}")
            print(f"Schema __init__: {self.schema.__class__.__init__}")
            print(f"Schema __new__: {self.schema.__class__.__new__}")
            print(f"Schema __call__: {self.schema.__class__.__call__}")
            print(f"Schema __getattribute__: {self.schema.__class__.__getattribute__}")
            print(f"Schema __setattr__: {self.schema.__class__.__setattr__}")
            print(f"Schema __delattr__: {self.schema.__class__.__delattr__}")
            print(f"Schema __hash__: {self.schema.__class__.__hash__}")
            print(f"Schema __eq__: {self.schema.__class__.__eq__}")
            raise
        edge_index = self.schema.get_edge_index(sample)
        target = self.schema.get_targets(sample)
        
        # Create batch tensor (all nodes belong to same graph)
        batch = torch.zeros(node_features.size(0), dtype=torch.long)
        
        input_data = {
            "x": node_features,
            "edge_index": edge_index,
            "batch": batch,
        }
        
        # Apply transforms
        input_data, target = self._apply_transforms(input_data, target)
        
        return input_data, target


class CVAEDataset(TacticAIDataset):
    """Dataset for CVAE tactic generation task.
    
    Loads and processes data for generating tactical formations.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        schema: Optional[DataSchema] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        file_format: str = "parquet",
    ):
        """Initialize CVAE dataset.
        
        Args:
            data_path: Path to data files
            schema: Data schema (optional, will create default if None)
            transform: Optional transform for input data
            target_transform: Optional transform for target data
            file_format: Data file format ('parquet', 'csv', 'pickle')
        """
        if schema is None:
            schema = create_schema_mapping("cvae")
        
        self.file_format = file_format
        super().__init__(data_path, schema, transform, target_transform)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load CVAE data.
        
        Returns:
            List of data samples
        """
        if self.data_path.is_file():
            # Single file
            return self._load_single_file(self.data_path)
        else:
            # Directory of files
            data = []
            pattern = f"*.{self.file_format}"
            for file_path in self.data_path.glob(pattern):
                data.extend(self._load_single_file(file_path))
            return data
    
    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load a single data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of data samples
        """
        if self.file_format == "parquet":
            df = pd.read_parquet(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "csv":
            df = pd.read_csv(file_path)
            return [df.iloc[[i]].to_dict('list') for i in range(len(df))]
        elif self.file_format == "pickle":
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # If it's already a list, return it; otherwise convert to list
                if isinstance(data, list):
                    return data
                else:
                    return [data]
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single CVAE sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, target_positions)
        """
        sample = self.data[idx]
        
        # Extract features using schema
        node_features = self.schema.get_node_features(sample)
        edge_index = self.schema.get_edge_index(sample)
        target = self.schema.get_targets(sample)
        
        # Extract conditions if schema supports it
        conditions = torch.zeros(8, dtype=torch.float32)  # Default condition size
        if hasattr(self.schema, 'get_conditions'):
            conditions = self.schema.get_conditions(sample)
        
        # Create batch tensor (all nodes belong to same graph)
        batch = torch.zeros(node_features.size(0), dtype=torch.long)
        
        input_data = {
            "x": node_features,
            "edge_index": edge_index,
            "batch": batch,
            "conditions": conditions,
        }
        
        # Apply transforms
        input_data, target = self._apply_transforms(input_data, target)
        
        return input_data, target


def collate_fn(batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Collate function for batching graph data.
    
    Args:
        batch: List of (input_data, target) tuples
        
    Returns:
        Batched (input_data, targets)
    """
    input_data_list, targets = zip(*batch)
    
    # Concatenate node features
    node_features = torch.cat([data["x"] for data in input_data_list], dim=0)
    
    # Concatenate edge indices with offset
    edge_indices = []
    node_offset = 0
    
    for data in input_data_list:
        edge_index = data["edge_index"] + node_offset
        edge_indices.append(edge_index)
        node_offset += data["x"].size(0)
    
    edge_index = torch.cat(edge_indices, dim=1)
    
    # Create batch assignment
    batch_tensor = torch.cat([
        torch.full((data["x"].size(0),), i, dtype=torch.long)
        for i, data in enumerate(input_data_list)
    ])
    
    # Handle conditions if present
    conditions = None
    if "conditions" in input_data_list[0]:
        conditions = torch.stack([data["conditions"] for data in input_data_list])
    
    batched_input = {
        "x": node_features,
        "edge_index": edge_index,
        "batch": batch_tensor,
    }
    
    # Concatenate edge_attr if present (TacticAI spec: same_team feature)
    if all("edge_attr" in data and data["edge_attr"] is not None for data in input_data_list):
        edge_attrs = []
        for data in input_data_list:
            edge_attrs.append(data["edge_attr"])
        batched_input["edge_attr"] = torch.cat(edge_attrs, dim=0)
    
    # Optional per-node fields: mask, team, ball
    # Concatenate if present in all items
    opt_keys = ["mask", "team", "ball"]
    for k in opt_keys:
        if all(k in data for data in input_data_list):
            batched_input[k] = torch.cat([data[k] for data in input_data_list], dim=0)
    
    if conditions is not None:
        batched_input["conditions"] = conditions
    
    # Handle different target shapes
    if all(t.dim() == 0 for t in targets):
        batched_targets = torch.stack(targets)
    elif all(t.dim() == 1 for t in targets):
        batched_targets = torch.stack(targets)
    else:
        # For multi-dimensional targets (like CVAE), flatten and stack
        batched_targets = torch.stack([t.flatten() for t in targets])
    
    return batched_input, batched_targets


def create_dataloader(
    dataset: TacticAIDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
) -> DataLoader:
    """Create a DataLoader for TacticAI dataset.
    
    Args:
        dataset: TacticAI dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (for GPU acceleration)
        prefetch_factor: Number of batches prefetched per worker
        persistent_workers: Whether to keep workers alive between epochs
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn,
    )


def create_dummy_dataset(
    task: str,
    num_samples: int = 100,
    num_players: int = 22,
    device: Optional[torch.device] = None,
) -> TacticAIDataset:
    """Create dummy dataset for testing.
    
    Args:
        task: Task type ('receiver', 'shot', 'cvae')
        num_samples: Number of samples to generate
        num_players: Number of players per sample
        device: Device to place tensors on
        
    Returns:
        Dummy dataset
    """
    if device is None:
        device = torch.device('cpu')
    
    # Generate dummy data
    dummy_data = []
    for _ in range(num_samples):
        sample = {}
        
        # Generate random positions
        positions = torch.rand(num_players, 2) * 100  # Rough field size
        sample["x"] = positions[:, 0].numpy()
        sample["y"] = positions[:, 1].numpy()
        
        # Generate random velocities
        velocities = torch.randn(num_players, 2) * 2
        sample["vx"] = velocities[:, 0].numpy()
        sample["vy"] = velocities[:, 1].numpy()
        
        # Generate team information
        sample["team"] = np.random.randint(0, 2, num_players)
        
        # Generate ball information
        ball_owner = np.random.randint(0, num_players)
        sample["ball"] = np.zeros(num_players)
        sample["ball"][ball_owner] = 1
        
        # Generate task-specific targets
        if task == "receiver":
            sample["receiver_id"] = np.random.randint(0, num_players)
        elif task == "shot":
            sample["shot_occurred"] = np.random.randint(0, 2)
        elif task == "cvae":
            # Target positions (slightly different from input)
            target_pos = positions + torch.randn_like(positions) * 5
            sample["target_x"] = target_pos[:, 0].numpy()
            sample["target_y"] = target_pos[:, 1].numpy()
            
            # Conditions
            conditions = np.random.randn(8)
            sample["condition"] = conditions
        
        dummy_data.append(sample)
    
    # Create temporary directory and save dummy data
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "dummy_data.pickle")
    
    with open(temp_file, 'wb') as f:
        pickle.dump(dummy_data, f)
    
    # Create dataset
    if task == "receiver":
        dataset = ReceiverDataset(temp_file, file_format="pickle")
    elif task == "shot":
        dataset = ShotDataset(temp_file, file_format="pickle")
    elif task == "cvae":
        dataset = CVAEDataset(temp_file, file_format="pickle")
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Clean up temp file
    import os
    os.remove(temp_file)
    os.rmdir(temp_dir)
    
    return dataset
