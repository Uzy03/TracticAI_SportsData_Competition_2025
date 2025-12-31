"""Multi-task dataset for receiver and shot prediction."""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch

from .dataset import ReceiverDataset, ShotDataset


class MultiTaskDataset(torch.utils.data.Dataset):
    """Multi-task dataset combining receiver and shot prediction data.
    
    This dataset loads both receiver and shot data, assuming they are aligned
    by sample index (i.e., sample i in receiver data corresponds to sample i in shot data).
    """
    
    def __init__(
        self,
        receiver_data_path: Union[str, Path],
        shot_data_path: Union[str, Path],
        receiver_file_format: str = "pickle",
        shot_file_format: str = "pickle",
        phase: str = "train",
    ):
        """Initialize multi-task dataset.
        
        Args:
            receiver_data_path: Path to receiver data
            shot_data_path: Path to shot data
            receiver_file_format: File format for receiver data
            shot_file_format: File format for shot data
            phase: Dataset phase (train/val/test)
        """
        # Load receiver dataset
        self.receiver_dataset = ReceiverDataset(
            data_path=receiver_data_path,
            file_format=receiver_file_format,
            phase=phase,
        )
        
        # Load shot dataset
        self.shot_dataset = ShotDataset(
            data_path=shot_data_path,
            file_format=shot_file_format,
        )
        
        # Verify datasets have same length
        if len(self.receiver_dataset) != len(self.shot_dataset):
            raise ValueError(
                f"Receiver dataset ({len(self.receiver_dataset)} samples) and "
                f"shot dataset ({len(self.shot_dataset)} samples) must have the same length"
            )
        
        self.phase = phase
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.receiver_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Get a multi-task sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_data, targets) where:
                - input_data: Dictionary containing graph data (x, edge_index, edge_attr, batch, etc.)
                - targets: Dictionary containing 'receiver_target' and 'shot_target'
        """
        # Get receiver sample
        receiver_input, receiver_target = self.receiver_dataset[idx]
        
        # Get shot sample (should have same graph data)
        shot_input, shot_target = self.shot_dataset[idx]
        
        # Use receiver input data (they should be the same)
        # Both datasets use the same underlying data, so input_data should be identical
        input_data = receiver_input
        
        # Combine targets
        targets = {
            'receiver_target': receiver_target,
            'shot_target': shot_target,
        }
        
        return input_data, targets

