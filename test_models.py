#!/usr/bin/env python3
"""
Test script to verify both Shot Prediction and CVAE models work correctly.
"""

import torch
import sys
import os
import yaml
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tacticai.train.train_shot import create_model as create_shot_model
from tacticai.train.train_cvae import create_model as create_cvae_model
from tacticai.dataio.dataset import ShotDataset, CVAEDataset, collate_fn
from torch.utils.data import DataLoader

def test_shot_prediction():
    """Test shot prediction model."""
    print("Testing Shot Prediction Model...")
    
    # Load config
    with open('configs/shot.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = torch.device('cpu')  # Use CPU for testing
    model = create_shot_model(config, device)
    model.eval()
    
    # Create dataset
    dataset = ShotDataset('data/processed/shot_train/data.pickle', file_format='pickle')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Test one batch
    for data, targets in dataloader:
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs = model(data['x'], data['edge_index'], data['batch'])
        
        print(f"✓ Shot Prediction - Outputs: {outputs.shape}, Targets: {targets.shape}")
        assert outputs.shape == targets.shape, f"Shape mismatch: {outputs.shape} vs {targets.shape}"
        break
    
    print("✓ Shot Prediction model test passed!")

def test_cvae():
    """Test CVAE model."""
    print("Testing CVAE Model...")
    
    # Load config
    with open('configs/cvae.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = torch.device('cpu')  # Use CPU for testing
    model = create_cvae_model(config, device)
    model.eval()
    
    # Create dataset
    dataset = CVAEDataset('data/processed/cvae_train/data.pickle', file_format='pickle')
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # Test one batch
    for data, targets in dataloader:
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        with torch.no_grad():
            outputs, mean, log_var = model(
                data['x'], 
                data['edge_index'], 
                data['batch'], 
                data.get('conditions', None),
                data.get('edge_attr', None)
            )
        
        print(f"✓ CVAE - Outputs: {outputs.shape}, Targets: {targets.shape}")
        print(f"✓ CVAE - Mean: {mean.shape}, Log_var: {log_var.shape}")
        assert outputs.shape == targets.shape, f"Shape mismatch: {outputs.shape} vs {targets.shape}"
        break
    
    print("✓ CVAE model test passed!")

def test_data_loading():
    """Test data loading for both datasets."""
    print("Testing Data Loading...")
    
    # Test shot dataset
    shot_dataset = ShotDataset('data/processed/shot_train/data.pickle', file_format='pickle')
    print(f"✓ Shot dataset loaded: {len(shot_dataset)} samples")
    
    # Test CVAE dataset
    cvae_dataset = CVAEDataset('data/processed/cvae_train/data.pickle', file_format='pickle')
    print(f"✓ CVAE dataset loaded: {len(cvae_dataset)} samples")
    
    print("✓ Data loading tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("TacticAI Model Testing")
    print("=" * 60)
    
    try:
        test_data_loading()
        print()
        
        test_shot_prediction()
        print()
        
        test_cvae()
        print()
        
        print("=" * 60)
        print("🎉 All tests passed! Models are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
