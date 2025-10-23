"""Script to scaffold additional TacticAI tasks.

This script helps create boilerplate code for new tasks.
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import yaml


def create_task_config(task_name: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create configuration for a new task.
    
    Args:
        task_name: Name of the new task
        base_config: Base configuration to modify
        
    Returns:
        New task configuration
    """
    config = base_config.copy()
    
    # Update task-specific settings
    if "model" in config:
        config["model"]["task"] = task_name
    
    # Update data paths
    if "data" in config:
        config["data"]["train_path"] = f"data/processed/{task_name}_train"
        config["data"]["val_path"] = f"data/processed/{task_name}_val"
        config["data"]["test_path"] = f"data/processed/{task_name}_test"
    
    return config


def create_task_schema(task_name: str, schema_template: str) -> str:
    """Create schema class for a new task.
    
    Args:
        task_name: Name of the new task
        schema_template: Template for schema class
        
    Returns:
        Schema class code
    """
    class_name = f"{task_name.title()}Schema"
    
    schema_code = f'''"""Schema for {task_name} task.

This module defines the data schema for {task_name} task.
"""

from typing import Optional, List, Tuple, Dict, Any
import torch
import numpy as np
import pandas as pd

from .schema import DataSchema


class {class_name}(DataSchema):
    """Schema for {task_name} task.
    
    Maps raw data to {task_name} format.
    """
    
    def __init__(
        self,
        position_columns: List[str] = ["x", "y"],
        velocity_columns: Optional[List[str]] = None,
        player_attr_columns: Optional[List[str]] = None,
        team_column: Optional[str] = None,
        ball_column: Optional[str] = None,
        # Add task-specific parameters here
        field_length: float = 105.0,
        field_width: float = 68.0,
    ):
        """Initialize {task_name} schema.
        
        Args:
            position_columns: Column names for player positions
            velocity_columns: Column names for player velocities (optional)
            player_attr_columns: Column names for player attributes (optional)
            team_column: Column name for team information (optional)
            ball_column: Column name for ball possession (optional)
            # Add task-specific parameters here
            field_length: Field length for normalization
            field_width: Field width for normalization
        """
        self.position_columns = position_columns
        self.velocity_columns = velocity_columns or []
        self.player_attr_columns = player_attr_columns or []
        self.team_column = team_column
        self.ball_column = ball_column
        # Add task-specific attributes here
        self.field_length = field_length
        self.field_width = field_width
    
    def get_node_features(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract node features for {task_name}.
        
        Args:
            data: Raw data dictionary containing player information
            
        Returns:
            Node features [N, F]
        """
        # Implement feature extraction here
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
        
        # Add other features as needed
        # ...
        
        return torch.cat(features, dim=1)
    
    def get_edge_index(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract edge connectivity.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Edge index tensor [2, E]
        """
        # Implement edge extraction here
        num_nodes = self._get_num_nodes(data)
        
        # Create complete graph (or implement task-specific logic)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def get_targets(self, data: Dict[str, Any]) -> torch.Tensor:
        """Extract targets.
        
        Args:
            data: Raw data dictionary
            
        Returns:
            Target tensor
        """
        # Implement target extraction here
        # This depends on the specific task
        pass
    
    def _get_num_nodes(self, data: Dict[str, Any]) -> int:
        """Get number of nodes from data."""
        if isinstance(data, pd.DataFrame):
            return len(data)
        else:
            return len(data[self.position_columns[0]])
'''
    
    return schema_code


def create_task_dataset(task_name: str) -> str:
    """Create dataset class for a new task.
    
    Args:
        task_name: Name of the new task
        
    Returns:
        Dataset class code
    """
    class_name = f"{task_name.title()}Dataset"
    schema_class = f"{task_name.title()}Schema"
    
    dataset_code = f'''"""Dataset for {task_name} task.

This module provides PyTorch dataset class for {task_name} task.
"""

from typing import Optional, Union, Tuple, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle

from .dataset import TacticAIDataset
from .schema import {schema_class}


class {class_name}(TacticAIDataset):
    """Dataset for {task_name} task.
    
    Loads and processes data for {task_name}.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        schema: Optional[{schema_class}] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        file_format: str = "parquet",
    ):
        """Initialize {task_name} dataset.
        
        Args:
            data_path: Path to data files
            schema: Data schema (optional, will create default if None)
            transform: Optional transform for input data
            target_transform: Optional transform for target data
            file_format: Data file format ('parquet', 'csv', 'pickle')
        """
        if schema is None:
            schema = {schema_class}()
        
        self.file_format = file_format
        super().__init__(data_path, schema, transform, target_transform)
    
    def _load_data(self) -> list[Dict[str, Any]]:
        """Load {task_name} data.
        
        Returns:
            List of data samples
        """
        data = []
        
        if self.data_path.is_file():
            # Single file
            data.append(self._load_single_file(self.data_path))
        else:
            # Directory of files
            pattern = f"*.{{self.file_format}}"
            for file_path in self.data_path.glob(pattern):
                data.append(self._load_single_file(file_path))
        
        return data
    
    def _load_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single data file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Data dictionary
        """
        if self.file_format == "parquet":
            return pd.read_parquet(file_path)
        elif self.file_format == "csv":
            return pd.read_csv(file_path)
        elif self.file_format == "pickle":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {{self.file_format}}")
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single {task_name} sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, target)
        """
        sample = self.data[idx]
        
        # Extract features using schema
        node_features = self.schema.get_node_features(sample)
        edge_index = self.schema.get_edge_index(sample)
        target = self.schema.get_targets(sample)
        
        # Create batch tensor (all nodes belong to same graph)
        batch = torch.zeros(node_features.size(0), dtype=torch.long)
        
        input_data = {{
            "x": node_features,
            "edge_index": edge_index,
            "batch": batch,
        }}
        
        # Apply transforms
        input_data, target = self._apply_transforms(input_data, target)
        
        return input_data, target
'''
    
    return dataset_code


def create_task_head(task_name: str) -> str:
    """Create head class for a new task.
    
    Args:
        task_name: Name of the new task
        
    Returns:
        Head class code
    """
    class_name = f"{task_name.title()}Head"
    
    head_code = f'''"""Head for {task_name} task.

This module implements task-specific head for {task_name}.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class {class_name}(nn.Module):
    """Head for {task_name} task.
    
    Implements the final layer for {task_name}.
    """
    
    def __init__(
        self,
        input_dim: int,
        # Add task-specific parameters here
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        """Initialize {task_name} head.
        
        Args:
            input_dim: Input feature dimension
            # Add task-specific parameters here
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Implement head architecture here
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Add final layer based on task requirements
            nn.Linear(hidden_dim, 1),  # Adjust output dimension as needed
        )
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input_features: Input features
            
        Returns:
            Task-specific outputs
        """
        return self.mlp(input_features)
'''
    
    return head_code


def create_task_training_script(task_name: str) -> str:
    """Create training script for a new task.
    
    Args:
        task_name: Name of the new task
        
    Returns:
        Training script code
    """
    script_code = f'''"""Training script for {task_name} task.

This script trains a model for {task_name}.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network, {task_name.title()}Head
from tacticai.dataio import {task_name.title()}Dataset, create_dataloader
from tacticai.modules import (
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping,
)
# Import appropriate loss and metrics for the task


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create {task_name} model."""
    model_config = config["model"]
    
    # Create backbone
    backbone = GATv2Network(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
        readout="mean",
        residual=True,
    )
    
    # Create head
    head = {task_name.title()}Head(
        input_dim=model_config["hidden_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
    )
    
    # Combine into full model
    model = nn.Sequential(backbone, head)
    
    return model.to(device)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train {task_name} model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug_overfit", action="store_true", help="Debug overfit test")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO")
    )
    
    logger.info(f"Training {task_name} model on {{device}}")
    
    # TODO: Implement training loop
    logger.info("Training loop not yet implemented")


if __name__ == "__main__":
    main()
'''
    
    return script_code


def scaffold_task(task_name: str, output_dir: Path):
    """Scaffold a new TacticAI task.
    
    Args:
        task_name: Name of the new task
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base configuration
    config_path = Path("configs/receiver.yaml")  # Use receiver as base
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create task configuration
    task_config = create_task_config(task_name, base_config)
    
    # Save configuration
    config_output_path = output_dir / "configs" / f"{task_name}.yaml"
    config_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_output_path, 'w') as f:
        yaml.dump(task_config, f, default_flow_style=False, indent=2)
    
    # Create schema
    schema_code = create_task_schema(task_name, "base")
    schema_output_path = output_dir / "tacticai" / "dataio" / f"{task_name}_schema.py"
    schema_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(schema_output_path, 'w') as f:
        f.write(schema_code)
    
    # Create dataset
    dataset_code = create_task_dataset(task_name)
    dataset_output_path = output_dir / "tacticai" / "dataio" / f"{task_name}_dataset.py"
    
    with open(dataset_output_path, 'w') as f:
        f.write(dataset_code)
    
    # Create head
    head_code = create_task_head(task_name)
    head_output_path = output_dir / "tacticai" / "models" / f"{task_name}_head.py"
    
    with open(head_output_path, 'w') as f:
        f.write(head_code)
    
    # Create training script
    training_script = create_task_training_script(task_name)
    training_output_path = output_dir / "tacticai" / "train" / f"train_{task_name}.py"
    
    with open(training_output_path, 'w') as f:
        f.write(training_script)
    
    print(f"Scaffolded {task_name} task:")
    print(f"  Config: {config_output_path}")
    print(f"  Schema: {schema_output_path}")
    print(f"  Dataset: {dataset_output_path}")
    print(f"  Head: {head_output_path}")
    print(f"  Training script: {training_output_path}")
    print("\nNext steps:")
    print("1. Implement the schema's get_targets() method")
    print("2. Implement the head's forward() method")
    print("3. Add appropriate loss function and metrics")
    print("4. Complete the training script")
    print("5. Update __init__.py files to include new classes")


def main():
    """Main scaffolding function."""
    parser = argparse.ArgumentParser(description="Scaffold new TacticAI task")
    parser.add_argument("task_name", type=str, help="Name of the new task")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    scaffold_task(args.task_name, output_dir)


if __name__ == "__main__":
    main()
