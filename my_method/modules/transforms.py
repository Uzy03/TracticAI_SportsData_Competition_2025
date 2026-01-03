"""Data transformation utilities for TacticAI.

This module provides data augmentation and transformation utilities,
including D2 (horizontal/vertical flip) transformations and group pooling.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


class RandomFlipTransform:
    """Random horizontal and vertical flip transformation for D2 equivariance.
    
    This transform applies random horizontal and/or vertical flips to
    maintain D2 (dihedral group) equivariance in football tactics.
    """
    
    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        field_length: float = 105.0,
        field_width: float = 68.0,
    ):
        """Initialize random flip transform.
        
        Args:
            hflip_prob: Probability of horizontal flip
            vflip_prob: Probability of vertical flip
            field_length: Field length for coordinate transformation
            field_width: Field width for coordinate transformation
        """
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.field_length = field_length
        self.field_width = field_width
    
    def __call__(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Apply random flip transformation.
        
        Args:
            positions: Player positions [N, 2]
            velocities: Player velocities [N, 2] (optional)
            **kwargs: Additional data to transform
            
        Returns:
            Dictionary with transformed data
        """
        result = {"positions": positions.clone()}
        
        # Apply horizontal flip
        if torch.rand(1) < self.hflip_prob:
            result["positions"][:, 0] = self.field_length - result["positions"][:, 0]
            if velocities is not None:
                if "velocities" not in result:
                    result["velocities"] = velocities.clone()
                result["velocities"][:, 0] = -result["velocities"][:, 0]
        
        # Apply vertical flip
        if torch.rand(1) < self.vflip_prob:
            result["positions"][:, 1] = self.field_width - result["positions"][:, 1]
            if velocities is not None:
                if "velocities" not in result:
                    result["velocities"] = velocities.clone()
                result["velocities"][:, 1] = -result["velocities"][:, 1]
        
        # Copy other data unchanged
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.clone()
            else:
                result[key] = value
        
        return result
    
    def inverse(
        self,
        positions: torch.Tensor,
        hflip: bool = False,
        vflip: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Apply inverse transformation.
        
        Args:
            positions: Transformed positions [N, 2]
            hflip: Whether horizontal flip was applied
            vflip: Whether vertical flip was applied
            **kwargs: Additional data (ignored)
            
        Returns:
            Original positions [N, 2]
        """
        result = positions.clone()
        
        # Apply inverse vertical flip
        if vflip:
            result[:, 1] = self.field_width - result[:, 1]
        
        # Apply inverse horizontal flip
        if hflip:
            result[:, 0] = self.field_length - result[:, 0]
        
        return result


class GroupPoolingWrapper(nn.Module):
    """Wrapper for applying group pooling over D2 transformations.
    
    This wrapper applies a model to all D2 transformations of the input
    and pools the results (average logits or features).
    """
    
    def __init__(
        self,
        model: nn.Module,
        average_logits: bool = True,
        field_length: float = 105.0,
        field_width: float = 68.0,
    ):
        """Initialize group pooling wrapper.
        
        Args:
            model: Base model to wrap
            average_logits: Whether to average logits (True) or features (False)
            field_length: Field length for coordinate transformation
            field_width: Field width for coordinate transformation
        """
        super().__init__()
        self.model = model
        self.average_logits = average_logits
        self.field_length = field_length
        self.field_width = field_width
        
        # D2 group elements (identity, horizontal flip, vertical flip, both flips)
        self.transforms = [
            (False, False),  # identity
            (True, False),   # horizontal flip
            (False, True),   # vertical flip
            (True, True),    # both flips
        ]
    
    def _apply_transform(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
        hflip: bool = False,
        vflip: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply specific transformation.
        
        Args:
            positions: Player positions [N, 2]
            velocities: Player velocities [N, 2] (optional)
            hflip: Whether to apply horizontal flip
            vflip: Whether to apply vertical flip
            
        Returns:
            Tuple of (transformed_positions, transformed_velocities)
        """
        transformed_pos = positions.clone()
        transformed_vel = velocities.clone() if velocities is not None else None
        
        # Apply horizontal flip
        if hflip:
            transformed_pos[:, 0] = self.field_length - transformed_pos[:, 0]
            if transformed_vel is not None:
                transformed_vel[:, 0] = -transformed_vel[:, 0]
        
        # Apply vertical flip
        if vflip:
            transformed_pos[:, 1] = self.field_width - transformed_pos[:, 1]
            if transformed_vel is not None:
                transformed_vel[:, 1] = -transformed_vel[:, 1]
        
        return transformed_pos, transformed_vel
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        velocities: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with group pooling.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, E_attr] (optional)
            batch: Batch assignment [N] (optional)
            positions: Player positions [N, 2] (optional)
            velocities: Player velocities [N, 2] (optional)
            **kwargs: Additional arguments for model
            
        Returns:
            Averaged model output
        """
        outputs = []
        
        for hflip, vflip in self.transforms:
            # Apply transformation if positions are provided
            if positions is not None:
                transformed_pos, transformed_vel = self._apply_transform(
                    positions, velocities, hflip, vflip
                )
                
                # Update node features with transformed positions
                if x.size(1) >= 2:  # Assume first 2 features are x, y coordinates
                    transformed_x = x.clone()
                    transformed_x[:, 0] = transformed_pos[:, 0] / self.field_length
                    transformed_x[:, 1] = transformed_pos[:, 1] / self.field_width
                else:
                    transformed_x = x
            else:
                transformed_x = x
                transformed_vel = velocities
            
            # Forward pass through model
            output = self.model(
                transformed_x, edge_index, edge_attr, batch, **kwargs
            )
            outputs.append(output)
        
        # Average outputs
        if self.average_logits:
            # Average logits (for classification tasks)
            return torch.stack(outputs).mean(dim=0)
        else:
            # Average features (for representation learning)
            return torch.stack(outputs).mean(dim=0)


class NormalizeTransform:
    """Normalization transform for features.
    
    Normalizes features to have zero mean and unit variance.
    """
    
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        """Initialize normalization transform.
        
        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.mean = mean
        self.std = std
        self._fitted = mean is not None and std is not None
    
    def fit(self, data: torch.Tensor, dim: int = 0) -> "NormalizeTransform":
        """Fit normalization parameters.
        
        Args:
            data: Data to fit on [N, F]
            dim: Dimension to compute statistics over
            
        Returns:
            Self for chaining
        """
        self.mean = torch.mean(data, dim=dim, keepdim=True)
        self.std = torch.std(data, dim=dim, keepdim=True)
        self.std = torch.clamp(self.std, min=1e-8)  # Avoid division by zero
        self._fitted = True
        return self
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply normalization.
        
        Args:
            data: Input data [N, F]
            
        Returns:
            Normalized data [N, F]
        """
        if not self._fitted:
            raise ValueError("Transform not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse normalization.
        
        Args:
            data: Normalized data [N, F]
            
        Returns:
            Original scale data [N, F]
        """
        if not self._fitted:
            raise ValueError("Transform not fitted. Call fit() first.")
        
        return data * self.std + self.mean


class StandardizeTransform:
    """Standardization transform for features.
    
    Standardizes features to have zero mean and unit variance.
    Similar to NormalizeTransform but with different naming convention.
    """
    
    def __init__(self):
        """Initialize standardization transform."""
        self.mean = None
        self.std = None
        self._fitted = False
    
    def fit(self, data: torch.Tensor, dim: int = 0) -> "StandardizeTransform":
        """Fit standardization parameters.
        
        Args:
            data: Data to fit on [N, F]
            dim: Dimension to compute statistics over
            
        Returns:
            Self for chaining
        """
        self.mean = torch.mean(data, dim=dim, keepdim=True)
        self.std = torch.std(data, dim=dim, keepdim=True)
        self.std = torch.clamp(self.std, min=1e-8)  # Avoid division by zero
        self._fitted = True
        return self
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply standardization.
        
        Args:
            data: Input data [N, F]
            
        Returns:
            Standardized data [N, F]
        """
        if not self._fitted:
            raise ValueError("Transform not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization.
        
        Args:
            data: Standardized data [N, F]
            
        Returns:
            Original scale data [N, F]
        """
        if not self._fitted:
            raise ValueError("Transform not fitted. Call fit() first.")
        
        return data * self.std + self.mean


class Compose:
    """Compose multiple transforms together.
    
    Applies transforms in sequence.
    """
    
    def __init__(self, transforms: List):
        """Initialize compose transform.
        
        Args:
            transforms: List of transforms to apply
        """
        self.transforms = transforms
    
    def __call__(self, *args, **kwargs):
        """Apply all transforms in sequence.
        
        Args:
            *args: Arguments to pass to transforms
            **kwargs: Keyword arguments to pass to transforms
            
        Returns:
            Result of applying all transforms
        """
        result = args[0] if args else kwargs
        for transform in self.transforms:
            if isinstance(result, dict):
                result = transform(**result)
            else:
                result = transform(result)
        return result
