"""Unit tests for ReceiverHead to ensure no node aggregation."""

import torch
import torch.nn as nn
from tacticai.models.mlp_heads import ReceiverHead


def test_receiver_head_no_aggregation():
    """Test that ReceiverHead does not aggregate across nodes.
    
    Ensures logits[:,i] != logits[:,j] for different nodes i and j.
    """
    input_dim = 128
    batch_size = 2
    num_nodes = 22
    
    # Create dummy node embeddings with different values per node
    # Each node has a unique embedding to ensure different logits
    H = torch.randn(batch_size, num_nodes, input_dim)
    # Make each node's embedding distinct
    for i in range(num_nodes):
        H[:, i, :] = torch.randn(batch_size, input_dim) + i * 0.1
    
    # Create head
    head = ReceiverHead(input_dim=input_dim)
    
    # Forward pass
    logits = head(H)  # [B, N]
    
    # Assertions
    assert logits.shape == (batch_size, num_nodes), \
        f"Expected logits shape [B, N], got {logits.shape}"
    
    # Check that different nodes produce different logits
    # (with high probability, given random initialization)
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                assert not torch.allclose(logits[b, i], logits[b, j], atol=1e-5), \
                    f"Nodes {i} and {j} in batch {b} produced identical logits: {logits[b, i].item():.6f}"
    
    # Check that logits have reasonable variance
    logits_std = logits.std().item()
    assert logits_std > 1e-6, \
        f"Logits std is too small ({logits_std:.6f}), possible collapse!"
    
    print(f"✓ ReceiverHead test passed: logits_std={logits_std:.6f}")


def test_receiver_head_2d_input():
    """Test ReceiverHead with 2D input [N, d]."""
    input_dim = 128
    num_nodes = 22
    
    # Create dummy node embeddings
    H = torch.randn(num_nodes, input_dim)
    for i in range(num_nodes):
        H[i, :] = torch.randn(input_dim) + i * 0.1
    
    # Create head
    head = ReceiverHead(input_dim=input_dim)
    
    # Forward pass
    logits = head(H)  # [N]
    
    # Assertions
    assert logits.shape == (num_nodes,), \
        f"Expected logits shape [N], got {logits.shape}"
    
    # Check that different nodes produce different logits
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            assert not torch.allclose(logits[i], logits[j], atol=1e-5), \
                f"Nodes {i} and {j} produced identical logits: {logits[i].item():.6f}"
    
    print(f"✓ ReceiverHead 2D input test passed")


if __name__ == "__main__":
    test_receiver_head_no_aggregation()
    test_receiver_head_2d_input()
    print("All ReceiverHead tests passed!")
