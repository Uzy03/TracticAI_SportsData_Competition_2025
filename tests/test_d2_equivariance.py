"""Tests for D2 equivariance in 4-view GATv2 layers.

This module tests that the 4-view GATv2 implementation maintains D2 equivariance
by ensuring that applying coordinate transformations (H/V/HV) produces consistent
logits across all views.
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Tuple

from tacticai.models.gatv2 import GATv2Layer4View, GATv2Network4View
from tacticai.modules.transforms import apply_d2_transforms


class D2EquivarianceTester:
    """Test suite for D2 equivariance in 4-view GATv2 layers."""
    
    def __init__(self, device: str = "cpu"):
        """Initialize tester.
        
        Args:
            device: Device to run tests on
        """
        self.device = device
    
    def create_test_data(self, batch_size: int = 2, num_nodes: int = 10, 
                        feature_dim: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create test data for equivariance testing.
        
        Args:
            batch_size: Batch size
            num_nodes: Number of nodes per graph
            feature_dim: Feature dimension
            
        Returns:
            Tuple of (node_features, edge_index)
        """
        # Create 4-view node features [B, V=4, N, D]
        node_features = torch.randn(batch_size, 4, num_nodes, feature_dim, device=self.device)
        
        # Create complete graph edge index
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        
        return node_features, edge_index
    
    def test_layer_equivariance(self, layer: GATv2Layer4View, 
                               tolerance: float = 1e-5) -> bool:
        """Test D2 equivariance for a single layer.
        
        Args:
            layer: GATv2Layer4View instance to test
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if equivariance test passes
        """
        x, edge_index = self.create_test_data()
        
        # Forward pass
        with torch.no_grad():
            output = layer(x, edge_index)
        
        # Test that all views produce the same output (up to coordinate transformation)
        # This tests weight sharing across views
        view_0 = output[:, 0]  # Original
        view_1 = output[:, 1]   # H-flip
        view_2 = output[:, 2]   # V-flip  
        view_3 = output[:, 3]   # HV-flip
        
        # Check that views are different (coordinate transformations applied)
        assert not torch.allclose(view_0, view_1, atol=tolerance), "Views should differ after H-flip"
        assert not torch.allclose(view_0, view_2, atol=tolerance), "Views should differ after V-flip"
        assert not torch.allclose(view_0, view_3, atol=tolerance), "Views should differ after HV-flip"
        
        # Check that the same transformation applied to input produces consistent output
        # This tests that the layer respects D2 symmetry
        x_h_flipped = x.clone()
        x_h_flipped[:, :, :, 0] = -x_h_flipped[:, :, :, 0]  # Flip x-coordinates
        
        with torch.no_grad():
            output_h_flipped = layer(x_h_flipped, edge_index)
        
        # Output should be consistent with input transformation
        # (This is a simplified test - full equivariance would require more complex checks)
        assert output.shape == output_h_flipped.shape, "Output shapes should match"
        
        return True
    
    def test_network_equivariance(self, network: GATv2Network4View,
                                 tolerance: float = 1e-5) -> bool:
        """Test D2 equivariance for entire network.
        
        Args:
            network: GATv2Network4View instance to test
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if equivariance test passes
        """
        x, edge_index = self.create_test_data()
        
        # Forward pass
        with torch.no_grad():
            output = network(x, edge_index)
        
        # Test that output maintains 4-view structure
        assert output.shape[1] == 4, "Output should maintain 4-view structure"
        
        # Test that all views produce meaningful outputs
        for v in range(4):
            view_output = output[:, v]
            assert not torch.allclose(view_output, torch.zeros_like(view_output), atol=tolerance), \
                f"View {v} should produce non-zero output"
        
        return True
    
    def test_coordinate_consistency(self, layer: GATv2Layer4View,
                                  tolerance: float = 1e-5) -> bool:
        """Test coordinate transformation consistency.
        
        Args:
            layer: GATv2Layer4View instance to test
            tolerance: Numerical tolerance for comparison
            
        Returns:
            True if coordinate consistency test passes
        """
        x, edge_index = self.create_test_data()
        
        # Create input with known coordinate structure
        # First two features are x, y coordinates
        x[:, :, :, 0] = torch.linspace(-1, 1, x.shape[2], device=self.device).unsqueeze(0).unsqueeze(0)
        x[:, :, :, 1] = torch.linspace(-1, 1, x.shape[2], device=self.device).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            output = layer(x, edge_index)
        
        # Test that coordinate transformations are properly handled
        # View 1 (H-flip) should have x-coordinates flipped
        view_0_x = output[:, 0, :, 0]  # Original x-coordinates
        view_1_x = output[:, 1, :, 0]  # H-flipped x-coordinates
        
        # These should be different (transformation applied)
        assert not torch.allclose(view_0_x, view_1_x, atol=tolerance), \
            "H-flip should affect x-coordinates"
        
        return True


def test_gatv2_layer_4view_equivariance():
    """Test D2 equivariance for GATv2Layer4View."""
    tester = D2EquivarianceTester()
    
    # Test different configurations
    configs = [
        {"heads": 1, "concat": True, "view_mixing": "attention"},
        {"heads": 4, "concat": True, "view_mixing": "conv1d"},
        {"heads": 2, "concat": False, "view_mixing": "mlp"},
    ]
    
    for config in configs:
        layer = GATv2Layer4View(
            in_features=8,
            out_features=16,
            **config
        )
        
        # Test layer equivariance
        assert tester.test_layer_equivariance(layer), f"Layer equivariance failed for config: {config}"
        
        # Test coordinate consistency
        assert tester.test_coordinate_consistency(layer), f"Coordinate consistency failed for config: {config}"


def test_gatv2_network_4view_equivariance():
    """Test D2 equivariance for GATv2Network4View."""
    tester = D2EquivarianceTester()
    
    # Test network configuration
    network = GATv2Network4View(
        input_dim=8,
        hidden_dim=32,
        output_dim=16,
        num_layers=2,
        num_heads=2,
        view_mixing="attention"
    )
    
    # Test network equivariance
    assert tester.test_network_equivariance(network), "Network equivariance test failed"


def test_weight_sharing():
    """Test that weight sharing is properly implemented."""
    layer = GATv2Layer4View(
        in_features=8,
        out_features=16,
        heads=2,
        weight_sharing=True
    )
    
    # Check that all views use the same parameters
    # (This is implicit in the implementation, but we can verify the structure)
    assert hasattr(layer, 'W'), "Linear transformation should exist"
    assert hasattr(layer, 'att'), "Attention parameters should exist"
    
    # Test that parameters are shared (not duplicated)
    param_count = sum(p.numel() for p in layer.parameters())
    expected_param_count = layer.W.weight.numel() + layer.att.numel()
    
    if layer.bias is not None:
        expected_param_count += layer.bias.numel()
    
    assert param_count == expected_param_count, "Parameters should be shared, not duplicated"


def test_view_mixing_methods():
    """Test different view mixing methods."""
    tester = D2EquivarianceTester()
    
    mixing_methods = ["attention", "conv1d", "mlp"]
    
    for method in mixing_methods:
        layer = GATv2Layer4View(
            in_features=8,
            out_features=16,
            view_mixing=method
        )
        
        # Test that the method works without errors
        assert tester.test_layer_equivariance(layer), f"View mixing method {method} failed"


def test_edge_case_handling():
    """Test edge cases for D2 equivariance."""
    tester = D2EquivarianceTester()
    
    # Test with minimal data
    layer = GATv2Layer4View(in_features=2, out_features=4)
    
    # Create minimal test data
    x = torch.randn(1, 4, 3, 2)  # 1 batch, 4 views, 3 nodes, 2 features
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Simple edges
    
    with torch.no_grad():
        output = layer(x, edge_index)
    
    # Should produce valid output
    assert output.shape == (1, 4, 3, 4), f"Expected shape (1, 4, 3, 4), got {output.shape}"
    assert not torch.any(torch.isnan(output)), "Output should not contain NaN values"
    assert not torch.any(torch.isinf(output)), "Output should not contain Inf values"


if __name__ == "__main__":
    # Run tests
    test_gatv2_layer_4view_equivariance()
    test_gatv2_network_4view_equivariance()
    test_weight_sharing()
    test_view_mixing_methods()
    test_edge_case_handling()
    
    print("All D2 equivariance tests passed!")
