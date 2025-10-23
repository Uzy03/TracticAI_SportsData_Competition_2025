"""Tests for model shapes and forward passes.

This module tests that all models produce correct output shapes.
"""

import pytest
import torch
import torch.nn as nn

from tacticai.models import GATv2Layer, GATv2Network, ReceiverHead, ShotHead, CVAEModel
from tacticai.modules.graph_builder import GraphBuilder


class TestGATv2Shapes:
    """Test GATv2 layer and network shapes."""
    
    def test_gatv2_layer_shapes(self):
        """Test GATv2 layer output shapes."""
        batch_size = 4
        num_nodes = 22
        input_dim = 8
        output_dim = 16
        heads = 4
        
        # Create layer
        layer = GATv2Layer(
            in_features=input_dim,
            out_features=output_dim,
            heads=heads,
            concat=True,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
        
        # Forward pass
        output = layer(x, edge_index)
        
        # Check shapes
        expected_shape = (num_nodes, heads * output_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_gatv2_network_shapes(self):
        """Test GATv2 network output shapes."""
        batch_size = 4
        num_nodes = 22
        input_dim = 8
        hidden_dim = 64
        output_dim = 32
        
        # Create network
        network = GATv2Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            num_heads=4,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
        
        # Forward pass without batch
        output = network(x, edge_index)
        expected_shape = (num_nodes, output_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Forward pass with batch
        batch = torch.zeros(num_nodes, dtype=torch.long)
        node_embeddings, graph_embeddings = network(x, edge_index, batch=batch)
        
        expected_node_shape = (num_nodes, output_dim)
        expected_graph_shape = (1, output_dim)
        assert node_embeddings.shape == expected_node_shape, f"Expected {expected_node_shape}, got {node_embeddings.shape}"
        assert graph_embeddings.shape == expected_graph_shape, f"Expected {expected_graph_shape}, got {graph_embeddings.shape}"


class TestHeadShapes:
    """Test task-specific head shapes."""
    
    def test_receiver_head_shapes(self):
        """Test receiver head output shapes."""
        batch_size = 4
        num_nodes = 22
        input_dim = 64
        num_classes = 22
        
        # Create head
        head = ReceiverHead(
            input_dim=input_dim,
            num_classes=num_classes,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim)
        
        # Forward pass
        output = head(x)
        
        # Check shapes
        expected_shape = (num_nodes, num_classes)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_shot_head_shapes(self):
        """Test shot head output shapes."""
        batch_size = 4
        input_dim = 64
        
        # Create head
        head = ShotHead(input_dim=input_dim)
        
        # Create input
        x = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = head(x)
        
        # Check shapes
        expected_shape = (batch_size, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"


class TestCVAEShapes:
    """Test CVAE model shapes."""
    
    def test_cvae_model_shapes(self):
        """Test CVAE model output shapes."""
        batch_size = 4
        num_nodes = 22
        input_dim = 8
        condition_dim = 8
        latent_dim = 32
        output_dim = 44  # 22 players * 2 coordinates
        
        # Create model
        model = CVAEModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        conditions = torch.randn(1, condition_dim)
        
        # Forward pass
        output, mean, log_var = model(x, edge_index, batch, conditions, training=True)
        
        # Check shapes
        expected_output_shape = (batch_size, output_dim)
        expected_mean_shape = (batch_size, latent_dim)
        expected_log_var_shape = (batch_size, latent_dim)
        
        assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
        assert mean.shape == expected_mean_shape, f"Expected {expected_mean_shape}, got {mean.shape}"
        assert log_var.shape == expected_log_var_shape, f"Expected {expected_log_var_shape}, got {log_var.shape}"


class TestGraphBuilderShapes:
    """Test graph builder output shapes."""
    
    def test_graph_builder_shapes(self):
        """Test graph builder output shapes."""
        num_players = 22
        
        # Create builder
        builder = GraphBuilder(
            graph_type="complete",
            include_edge_attrs=True,
            normalize_positions=True,
        )
        
        # Create dummy data
        node_features, edge_index, edge_attrs = builder.create_dummy_data(num_players)
        
        # Check shapes
        expected_node_shape = (num_players, 8)  # x, y, vx, vy, height, weight, ball, team
        expected_edge_shape = (2, num_players * (num_players - 1))
        expected_edge_attr_shape = (num_players * (num_players - 1), 3)  # distance, bearing, same_team
        
        assert node_features.shape == expected_node_shape, f"Expected {expected_node_shape}, got {node_features.shape}"
        assert edge_index.shape == expected_edge_shape, f"Expected {expected_edge_shape}, got {edge_index.shape}"
        assert edge_attrs.shape == expected_edge_attr_shape, f"Expected {expected_edge_attr_shape}, got {edge_attrs.shape}"


class TestBatchProcessing:
    """Test batch processing and collation."""
    
    def test_batch_shapes(self):
        """Test that batching produces correct shapes."""
        from tacticai.dataio.dataset import collate_fn
        
        # Create dummy batch
        batch_size = 3
        num_players = 22
        input_dim = 8
        
        batch = []
        for i in range(batch_size):
            # Create dummy data
            x = torch.randn(num_players, input_dim)
            edge_index = torch.randint(0, num_players, (2, num_players * (num_players - 1)))
            batch_tensor = torch.zeros(num_players, dtype=torch.long)
            
            input_data = {
                "x": x,
                "edge_index": edge_index,
                "batch": batch_tensor,
            }
            
            target = torch.randint(0, num_players, (1,))
            
            batch.append((input_data, target))
        
        # Collate batch
        batched_input, batched_targets = collate_fn(batch)
        
        # Check shapes
        expected_x_shape = (batch_size * num_players, input_dim)
        expected_batch_shape = (batch_size * num_players,)
        expected_target_shape = (batch_size,)
        
        assert batched_input["x"].shape == expected_x_shape, f"Expected {expected_x_shape}, got {batched_input['x'].shape}"
        assert batched_input["batch"].shape == expected_batch_shape, f"Expected {expected_batch_shape}, got {batched_input['batch'].shape}"
        assert batched_targets.shape == expected_target_shape, f"Expected {expected_target_shape}, got {batched_targets.shape}"


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_gatv2_gradient_flow(self):
        """Test that gradients flow through GATv2."""
        num_nodes = 22
        input_dim = 8
        hidden_dim = 16
        
        # Create network
        network = GATv2Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
        
        # Forward pass
        output = network(x, edge_index)
        
        # Compute loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradients should not be None"
        assert x.grad.shape == x.shape, f"Gradient shape mismatch: {x.grad.shape} vs {x.shape}"
    
    def test_cvae_gradient_flow(self):
        """Test that gradients flow through CVAE."""
        num_nodes = 22
        input_dim = 8
        condition_dim = 8
        latent_dim = 16
        output_dim = 44
        
        # Create model
        model = CVAEModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            output_dim=output_dim,
        )
        
        # Create input
        x = torch.randn(num_nodes, input_dim, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        conditions = torch.randn(1, condition_dim, requires_grad=True)
        
        # Forward pass
        output, mean, log_var = model(x, edge_index, batch, conditions, training=True)
        
        # Compute loss
        loss = output.sum() + mean.sum() + log_var.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        assert x.grad is not None, "Input gradients should not be None"
        assert conditions.grad is not None, "Condition gradients should not be None"


if __name__ == "__main__":
    pytest.main([__file__])
