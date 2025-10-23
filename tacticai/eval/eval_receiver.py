"""Evaluation script for receiver prediction task.

This script evaluates a trained GATv2 model on receiver prediction.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network, ReceiverHead
from tacticai.dataio import ReceiverDataset, create_dataloader
from tacticai.modules import (
    TopKAccuracy, Accuracy, F1Score, ECE,
    get_device, load_checkpoint, setup_logging,
)
from tacticai.modules.transforms import GroupPoolingWrapper


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create receiver prediction model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Receiver prediction model
    """
    model_config = config["model"]
    
    # Create backbone
    backbone = GATv2Network(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        output_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=0.0,  # No dropout during evaluation
        readout="mean",
        residual=True,
    )
    
    # Create head
    head = ReceiverHead(
        input_dim=model_config["hidden_dim"],
        num_classes=model_config["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        dropout=0.0,  # No dropout during evaluation
    )
    
    # Combine into full model
    model = nn.Sequential(backbone, head)
    
    # Apply D2 group pooling if enabled
    if config.get("d2", {}).get("group_pool", False):
        model = GroupPoolingWrapper(model, average_logits=True)
    
    return model.to(device)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        metrics: Metric functions
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs = model(data["x"], data["edge_index"], data["batch"])
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    # Compute metrics
    eval_metrics = {
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "top1": metrics["top1"](all_predictions, all_targets).item(),
        "top3": metrics["top3"](all_predictions, all_targets).item(),
        "top5": metrics["top5"](all_predictions, all_targets).item(),
        "f1": metrics["f1"](all_predictions, all_targets).item(),
        "ece": metrics["ece"](all_predictions, all_targets).item(),
    }
    
    return eval_metrics


def analyze_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
) -> None:
    """Analyze model predictions in detail.
    
    Args:
        model: Model to analyze
        dataloader: Data loader
        device: Device to analyze on
        num_samples: Number of samples to analyze
    """
    model.eval()
    
    print("\n=== Detailed Prediction Analysis ===")
    
    sample_count = 0
    with torch.no_grad():
        for data, targets in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs = model(data["x"], data["edge_index"], data["batch"])
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
            print(f"\nSample {sample_count + 1}:")
            print(f"Ground truth receiver: {targets[0].item()}")
            print(f"Top 5 predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                status = "✓" if idx.item() == targets[0].item() else " "
                print(f"  {i+1}. Player {idx.item():2d}: {prob.item():.4f} {status}")
            
            sample_count += 1


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate receiver prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_path", type=str, help="Path to test data (overrides config)")
    parser.add_argument("--analyze", action="store_true", help="Analyze predictions in detail")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO")
    )
    
    logger.info(f"Evaluating receiver prediction model on {device}")
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataset
    test_path = args.test_path or config["data"]["test_path"]
    test_dataset = ReceiverDataset(
        test_path,
        file_format=config["data"].get("format", "parquet")
    )
    
    # Create test data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config["eval"].get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )
    
    # Create metrics
    metrics = {
        "accuracy": Accuracy(),
        "top1": TopKAccuracy(k=1),
        "top3": TopKAccuracy(k=3),
        "top5": TopKAccuracy(k=5),
        "f1": F1Score(),
        "ece": ECE(),
    }
    
    # Evaluate model
    logger.info("Starting evaluation...")
    eval_metrics = evaluate_model(model, test_loader, device, metrics)
    
    # Log results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"Top-1 Accuracy: {eval_metrics['top1']:.4f}")
    logger.info(f"Top-3 Accuracy: {eval_metrics['top3']:.4f}")
    logger.info(f"Top-5 Accuracy: {eval_metrics['top5']:.4f}")
    logger.info(f"F1 Score: {eval_metrics['f1']:.4f}")
    logger.info(f"Expected Calibration Error: {eval_metrics['ece']:.4f}")
    
    # Analyze predictions if requested
    if args.analyze:
        analyze_predictions(model, test_loader, device, args.num_samples)
    
    # Save results
    results_path = Path(config.get("log_dir", "runs")) / "receiver_eval_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
