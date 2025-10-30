"""Tests for D2 Group Convolution implementation following TacticAI paper equation (8).

Tests verify that the implementation correctly implements:
H_g^(t) = (1/|G|) Σ_h g_G( H_h^(t-1) ‖ H_{g⁻¹h}^(t-1) )
"""

import torch
import pytest
from tacticai.models.gatv2_d2_corrected import D2GroupConvolutionLayer, D2GroupConvolutionNetwork
from tacticai.modules.view_ops import D2_VIEWS, apply_view_transform


def test_d2_group_convolution_structure():
    """Test that D2GroupConvolutionLayer correctly implements equation (8)."""
    batch_size, num_nodes, in_features = 2, 10, 8
    out_features = 16
    
    # Create layer
    layer = D2GroupConvolutionLayer(
        in_features=in_features,
        out_features=out_features,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create input with 4 views
    x = torch.randn(batch_size, 4, num_nodes, in_features)
    
    # Create edge index (complete graph)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Forward pass
    output = layer(x, edge_index)
    
    # Check output shape
    assert output.shape == (batch_size, 4, num_nodes, out_features)
    
    # Check that all views are processed
    assert not torch.allclose(output[:, 0], output[:, 1], atol=1e-6)
    assert not torch.allclose(output[:, 0], output[:, 2], atol=1e-6)
    assert not torch.allclose(output[:, 0], output[:, 3], atol=1e-6)


def test_d2_equivariance_property():
    """Test D2 equivariance property: H(view="horizontal") should equal H(view="id") after horizontal flip."""
    batch_size, num_nodes, in_features = 2, 10, 8
    out_features = 16
    
    # Create layer
    layer = D2GroupConvolutionLayer(
        in_features=in_features,
        out_features=out_features,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create input with 4 views
    x = torch.randn(batch_size, 4, num_nodes, in_features)
    
    # Create edge index (complete graph)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Forward pass
    output = layer(x, edge_index)
    
    # Test equivariance: H(view="horizontal") should equal H(view="id") after horizontal flip
    # Get outputs for id view (index 0) and horizontal view (index 1)
    id_output = output[:, 0, :, :]  # [B, N, out_features]
    h_output = output[:, 1, :, :]   # [B, N, out_features]
    
    # Apply horizontal flip to h_output to align with id view
    h_output_flipped = apply_view_transform(h_output, "h", xy_indices=(0, 1))
    
    # They should be close (within numerical precision)
    assert torch.allclose(id_output, h_output_flipped, atol=1e-5), \
        "D2 equivariance property violated: H(view='horizontal') != H(view='id') after horizontal flip"


def test_d2_vertical_equivariance():
    """Test D2 equivariance property for vertical flip."""
    batch_size, num_nodes, in_features = 2, 10, 8
    out_features = 16
    
    # Create layer
    layer = D2GroupConvolutionLayer(
        in_features=in_features,
        out_features=out_features,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create input with 4 views
    x = torch.randn(batch_size, 4, num_nodes, in_features)
    
    # Create edge index (complete graph)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Forward pass
    output = layer(x, edge_index)
    
    # Test equivariance: H(view="vertical") should equal H(view="id") after vertical flip
    id_output = output[:, 0, :, :]  # [B, N, out_features]
    v_output = output[:, 2, :, :]   # [B, N, out_features]
    
    # Apply vertical flip to v_output to align with id view
    v_output_flipped = apply_view_transform(v_output, "v", xy_indices=(0, 1))
    
    # They should be close (within numerical precision)
    assert torch.allclose(id_output, v_output_flipped, atol=1e-5), \
        "D2 equivariance property violated: H(view='vertical') != H(view='id') after vertical flip"


def test_d2_both_flips_equivariance():
    """Test D2 equivariance property for both flips."""
    batch_size, num_nodes, in_features = 2, 10, 8
    out_features = 16
    
    # Create layer
    layer = D2GroupConvolutionLayer(
        in_features=in_features,
        out_features=out_features,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create input with 4 views
    x = torch.randn(batch_size, 4, num_nodes, in_features)
    
    # Create edge index (complete graph)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Forward pass
    output = layer(x, edge_index)
    
    # Test equivariance: H(view="hv") should equal H(view="id") after both flips
    id_output = output[:, 0, :, :]  # [B, N, out_features]
    hv_output = output[:, 3, :, :]  # [B, N, out_features]
    
    # Apply both flips to hv_output to align with id view
    hv_output_flipped = apply_view_transform(hv_output, "hv", xy_indices=(0, 1))
    
    # They should be close (within numerical precision)
    assert torch.allclose(id_output, hv_output_flipped, atol=1e-5), \
        "D2 equivariance property violated: H(view='hv') != H(view='id') after both flips"


def test_invariant_pooling():
    """Test that invariant pooling produces the same result regardless of input view."""
    batch_size, num_nodes, in_features = 2, 10, 8
    out_features = 16
    
    # Create network
    network = D2GroupConvolutionNetwork(
        input_dim=in_features,
        hidden_dim=32,
        output_dim=out_features,
        num_layers=2,
        num_heads=2,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create input with 4 views
    x = torch.randn(batch_size, 4, num_nodes, in_features)
    
    # Create edge index (complete graph)
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Forward pass with invariant pooling
    invariant_output = network(x, edge_index)
    
    # Check output shape
    assert invariant_output.shape == (batch_size, num_nodes, out_features)
    
    # Test that invariant pooling is truly invariant
    # Create input with different view order
    x_permuted = x[:, [1, 0, 3, 2], :, :]  # Permute views
    
    # Forward pass with permuted input
    invariant_output_permuted = network(x_permuted, edge_index)
    
    # Results should be identical (within numerical precision)
    assert torch.allclose(invariant_output, invariant_output_permuted, atol=1e-5), \
        "Invariant pooling is not truly invariant to view permutations"


def test_group_inverse_computation():
    """Test that group inverse computation is correct."""
    layer = D2GroupConvolutionLayer(
        in_features=8,
        out_features=16,
        equivariant_indices=(0, 1, 2, 3),
        invariant_indices=(4, 5, 6, 7),
    )
    
    # Test all combinations of g and h
    for g in range(4):
        for h in range(4):
            g_inv_h = layer._compute_group_inverse(g, h)
            
            # Verify that g * g_inv_h = h (group multiplication)
            # This is a basic property of group theory
            if g == 0:  # g = id
                assert g_inv_h == h, f"Group inverse computation error for g={g}, h={h}"
            elif g == 1:  # g = h (horizontal flip)
                expected = [0, 1, 3, 2][h]  # h * {id, h, v, hv} = {id, h, hv, v}
                assert g_inv_h == expected, f"Group inverse computation error for g={g}, h={h}"
            elif g == 2:  # g = v (vertical flip)
                expected = [0, 3, 2, 1][h]  # v * {id, h, v, hv} = {id, hv, v, h}
                assert g_inv_h == expected, f"Group inverse computation error for g={g}, h={h}"
            elif g == 3:  # g = hv (both flips)
                expected = [0, 2, 1, 3][h]  # hv * {id, h, v, hv} = {id, v, h, hv}
                assert g_inv_h == expected, f"Group inverse computation error for g={g}, h={h}"


def test_feature_alignment():
    """Test that feature alignment correctly handles equivariant and invariant features."""
    batch_size, num_nodes, in_features = 2, 10, 8
    
    layer = D2GroupConvolutionLayer(
        in_features=in_features,
        out_features=16,
        equivariant_indices=(0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices=(4, 5, 6, 7),    # height, weight, team, ball
    )
    
    # Create features with known values
    features = torch.zeros(batch_size, num_nodes, in_features)
    features[:, :, 0] = 1.0  # x coordinate
    features[:, :, 1] = 2.0  # y coordinate
    features[:, :, 2] = 3.0  # vx velocity
    features[:, :, 3] = 4.0  # vy velocity
    features[:, :, 4] = 5.0  # height (invariant)
    features[:, :, 5] = 6.0  # weight (invariant)
    features[:, :, 6] = 7.0  # team (invariant)
    features[:, :, 7] = 8.0  # ball (invariant)
    
    # Test alignment from id view to horizontal view
    aligned_features = layer._align_features_for_view(features, 0, 1)  # id -> h
    
    # Check that equivariant features are transformed
    assert torch.allclose(aligned_features[:, :, 0], -1.0), "x coordinate should be flipped"
    assert torch.allclose(aligned_features[:, :, 1], 2.0), "y coordinate should remain the same"
    assert torch.allclose(aligned_features[:, :, 2], -3.0), "vx velocity should be flipped"
    assert torch.allclose(aligned_features[:, :, 3], 4.0), "vy velocity should remain the same"
    
    # Check that invariant features remain unchanged
    assert torch.allclose(aligned_features[:, :, 4], 5.0), "height should remain unchanged"
    assert torch.allclose(aligned_features[:, :, 5], 6.0), "weight should remain unchanged"
    assert torch.allclose(aligned_features[:, :, 6], 7.0), "team should remain unchanged"
    assert torch.allclose(aligned_features[:, :, 7], 8.0), "ball should remain unchanged"


if __name__ == "__main__":
    # Run tests
    test_d2_group_convolution_structure()
    test_d2_equivariance_property()
    test_d2_vertical_equivariance()
    test_d2_both_flips_equivariance()
    test_invariant_pooling()
    test_group_inverse_computation()
    test_feature_alignment()
    print("All tests passed!")
