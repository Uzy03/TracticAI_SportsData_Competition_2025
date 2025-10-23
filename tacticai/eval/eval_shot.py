"""Evaluation script for shot prediction task with receiver conditioning.

This script evaluates a trained GATv2 model on shot prediction with receiver conditioning.
Supports both training (with GT receiver) and inference (with marginalization over receivers).
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

from tacticai.models import GATv2Network, ShotHead, ReceiverHead
from tacticai.dataio import ShotDataset, ReceiverDataset, create_dataloader
from tacticai.modules import (
    AUC, F1Score, Accuracy, ECE,
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


class ShotConditionalModel(nn.Module):
    """Shot prediction model conditioned on receiver.
    
    This model predicts shot probability given a specific receiver.
    Supports both training (with GT receiver) and inference (with marginalization).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize conditional shot model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        model_config = config["model"]
        
        # Create shared backbone
        self.backbone = GATv2Network(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=0.0,  # No dropout during evaluation
            readout="mean",
            residual=True,
        )
        
        # Create receiver head for marginalization
        self.receiver_head = ReceiverHead(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            dropout=0.0,
        )
        
        # Create conditional shot head
        self.shot_head = ShotHead(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            dropout=0.0,
        )
        
        # Apply D2 group pooling if enabled
        if config.get("d2", {}).get("group_pool", False):
            self.backbone = GroupPoolingWrapper(self.backbone, average_logits=True)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: torch.Tensor, receiver_id: torch.Tensor = None) -> torch.Tensor:
        """Forward pass for conditional shot prediction.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            receiver_id: Receiver ID for conditioning (optional)
            
        Returns:
            Shot probability logits [B]
        """
        # Get graph embeddings
        node_embeddings, graph_embeddings = self.backbone(x, edge_index, batch)
        
        if receiver_id is not None:
            # Training mode: use GT receiver
            return self.shot_head(graph_embeddings, receiver_id)
        else:
            # Inference mode: marginalize over receivers
            return self.predict_shot_marginalized(graph_embeddings)
    
    def predict_shot_given_receiver(self, graph_embeddings: torch.Tensor, 
                                   receiver_id: torch.Tensor) -> torch.Tensor:
        """Predict shot probability given specific receiver.
        
        Args:
            graph_embeddings: Graph-level embeddings [B, D]
            receiver_id: Receiver ID [B]
            
        Returns:
            Shot probability logits [B]
        """
        return self.shot_head(graph_embeddings, receiver_id)
    
    def predict_shot_marginalized(self, graph_embeddings: torch.Tensor, 
                                top_k: int = None) -> torch.Tensor:
        """Predict shot probability by marginalizing over receivers.
        
        Args:
            graph_embeddings: Graph-level embeddings [B, D]
            top_k: Number of top receivers to consider (None for all)
            
        Returns:
            Marginalized shot probability logits [B]
        """
        batch_size = graph_embeddings.size(0)
        num_classes = self.receiver_head.num_classes
        
        # Get receiver probabilities
        receiver_logits = self.receiver_head(graph_embeddings)  # [B, num_classes]
        receiver_probs = torch.softmax(receiver_logits, dim=1)  # [B, num_classes]
        
        if top_k is not None:
            # Use top-k receivers only
            _, top_receivers = torch.topk(receiver_probs, top_k, dim=1)  # [B, top_k]
            
            # Normalize probabilities for top-k
            top_receiver_probs = torch.gather(receiver_probs, 1, top_receivers)  # [B, top_k]
            top_receiver_probs = top_receiver_probs / top_receiver_probs.sum(dim=1, keepdim=True)
            
            # Compute shot probabilities for top-k receivers
            shot_logits = []
            for i in range(batch_size):
                batch_shot_logits = []
                for j in range(top_k):
                    receiver_id = top_receivers[i, j].unsqueeze(0)
                    shot_logit = self.shot_head(graph_embeddings[i:i+1], receiver_id)
                    batch_shot_logits.append(shot_logit)
                
                # Weighted sum
                batch_shot_logits = torch.stack(batch_shot_logits, dim=1)  # [1, top_k]
                weighted_shot_logit = (batch_shot_logits * top_receiver_probs[i:i+1]).sum(dim=1)
                shot_logits.append(weighted_shot_logit)
            
            return torch.cat(shot_logits, dim=0)
        else:
            # Use all receivers
            shot_logits = []
            for i in range(batch_size):
                batch_shot_logits = []
                for receiver_id in range(num_classes):
                    receiver_tensor = torch.tensor([receiver_id], device=graph_embeddings.device)
                    shot_logit = self.shot_head(graph_embeddings[i:i+1], receiver_tensor)
                    batch_shot_logits.append(shot_logit)
                
                # Weighted sum
                batch_shot_logits = torch.stack(batch_shot_logits, dim=1)  # [1, num_classes]
                weighted_shot_logit = (batch_shot_logits * receiver_probs[i:i+1]).sum(dim=1)
                shot_logits.append(weighted_shot_logit)
            
            return torch.cat(shot_logits, dim=0)


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create conditional shot prediction model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Conditional shot prediction model
    """
    model = ShotConditionalModel(config)
    return model.to(device)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    metrics: Dict[str, Any],
    use_gt_receiver: bool = False,
    top_k_receivers: int = None,
) -> Dict[str, float]:
    """Evaluate conditional shot model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        metrics: Metric functions
        use_gt_receiver: Whether to use GT receiver (training mode)
        top_k_receivers: Number of top receivers for marginalization (None for all)
        
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
            
            if use_gt_receiver and "receiver_id" in data:
                # Training mode: use GT receiver
                outputs = model(data["x"], data["edge_index"], data["batch"], data["receiver_id"])
            else:
                # Inference mode: marginalize over receivers
                outputs = model(data["x"], data["edge_index"], data["batch"])
            
            # Get probabilities
            probabilities = torch.sigmoid(outputs)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_probabilities.append(probabilities.cpu())
    
    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    # Compute metrics
    auc_roc, auc_pr = metrics["auc"](all_predictions, all_targets, compute_auc_pr=True)
    
    eval_metrics = {
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "f1": metrics["f1"](all_predictions, all_targets).item(),
        "ece": metrics["ece"](all_predictions, all_targets).item(),
    }
    
    return eval_metrics


def analyze_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    use_gt_receiver: bool = False,
) -> None:
    """Analyze model predictions in detail.
    
    Args:
        model: Model to analyze
        dataloader: Data loader
        device: Device to analyze on
        num_samples: Number of samples to analyze
        use_gt_receiver: Whether to use GT receiver
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
            
            if use_gt_receiver and "receiver_id" in data:
                outputs = model(data["x"], data["edge_index"], data["batch"], data["receiver_id"])
                gt_receiver = data["receiver_id"][0].item()
            else:
                outputs = model(data["x"], data["edge_index"], data["batch"])
                gt_receiver = None
            
            probabilities = torch.sigmoid(outputs)
            
            print(f"\nSample {sample_count + 1}:")
            print(f"Ground truth: {'Shot' if targets[0].item() > 0.5 else 'No Shot'}")
            print(f"Predicted probability: {probabilities[0].item():.4f}")
            print(f"Prediction: {'Shot' if probabilities[0].item() > 0.5 else 'No Shot'}")
            
            if gt_receiver is not None:
                print(f"GT Receiver: {gt_receiver}")
            
            sample_count += 1


def benchmark_marginalization(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k_values: list[int] = [1, 3, 5, 10],
) -> Dict[str, Dict[str, float]]:
    """Benchmark marginalization performance with different top-k values.
    
    Args:
        model: Model to benchmark
        dataloader: Data loader
        device: Device to benchmark on
        top_k_values: List of top-k values to test
        
    Returns:
        Dictionary of results for each top-k value
    """
    model.eval()
    
    results = {}
    
    for top_k in top_k_values:
        print(f"\nBenchmarking with top-{top_k} receivers...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(dataloader, desc=f"Top-{top_k}"):
                data = {k: v.to(device) for k, v in data.items()}
                targets = targets.to(device)
                
                # Use marginalization with top-k
                outputs = model.predict_shot_marginalized(
                    model.backbone(data["x"], data["edge_index"], data["batch"])[1],
                    top_k=top_k
                )
                
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        auc_roc, auc_pr = AUC()(all_predictions, all_targets, compute_auc_pr=True)
        accuracy = Accuracy()(all_predictions, all_targets)
        f1 = F1Score()(all_predictions, all_targets)
        ece = ECE()(all_predictions, all_targets)
        
        results[f"top_{top_k}"] = {
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "accuracy": accuracy.item(),
            "f1": f1.item(),
            "ece": ece.item(),
        }
        
        print(f"Top-{top_k} Results:")
        print(f"  AUC-ROC: {auc_roc.item():.4f}")
        print(f"  AUC-PR: {auc_pr.item():.4f}")
        print(f"  Accuracy: {accuracy.item():.4f}")
        print(f"  F1: {f1.item():.4f}")
        print(f"  ECE: {ece.item():.4f}")
    
    return results


def compare_training_vs_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compare training mode (GT receiver) vs inference mode (marginalization).
    
    Args:
        model: Model to compare
        dataloader: Data loader
        device: Device to compare on
        
    Returns:
        Dictionary with training and inference results
    """
    model.eval()
    
    results = {}
    
    # Training mode (GT receiver)
    print("\nEvaluating in training mode (GT receiver)...")
    train_predictions = []
    train_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Training mode"):
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            if "receiver_id" in data:
                outputs = model(data["x"], data["edge_index"], data["batch"], data["receiver_id"])
                train_predictions.append(outputs.cpu())
                train_targets.append(targets.cpu())
    
    if train_predictions:
        train_predictions = torch.cat(train_predictions, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        
        auc_roc, auc_pr = AUC()(train_predictions, train_targets, compute_auc_pr=True)
        accuracy = Accuracy()(train_predictions, train_targets)
        f1 = F1Score()(train_predictions, train_targets)
        ece = ECE()(train_predictions, train_targets)
        
        results["training"] = {
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "accuracy": accuracy.item(),
            "f1": f1.item(),
            "ece": ece.item(),
        }
    
    # Inference mode (marginalization)
    print("\nEvaluating in inference mode (marginalization)...")
    inf_predictions = []
    inf_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Inference mode"):
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs = model(data["x"], data["edge_index"], data["batch"])
            inf_predictions.append(outputs.cpu())
            inf_targets.append(targets.cpu())
    
    inf_predictions = torch.cat(inf_predictions, dim=0)
    inf_targets = torch.cat(inf_targets, dim=0)
    
    auc_roc, auc_pr = AUC()(inf_predictions, inf_targets, compute_auc_pr=True)
    accuracy = Accuracy()(inf_predictions, inf_targets)
    f1 = F1Score()(inf_predictions, inf_targets)
    ece = ECE()(inf_predictions, inf_targets)
    
    results["inference"] = {
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
        "accuracy": accuracy.item(),
        "f1": f1.item(),
        "ece": ece.item(),
    }
    
    # Print comparison
    print("\n=== Training vs Inference Comparison ===")
    if "training" in results:
        print("Training Mode (GT Receiver):")
        for metric, value in results["training"].items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nInference Mode (Marginalization):")
    for metric, value in results["inference"].items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def plot_roc_curve(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str = None,
) -> None:
    """Plot ROC curve and precision-recall curve.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        save_path: Path to save plots
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Collecting predictions"):
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs = model(data["x"], data["edge_index"], data["batch"])
            probabilities = torch.sigmoid(outputs)
            
            all_predictions.append(probabilities.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate results
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_targets, all_predictions)
    auc_roc = np.trapz(tpr, fpr)
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
    auc_pr = np.trapz(precision, recall)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall curve
    ax2.plot(recall, precision, label=f'PR Curve (AUC = {auc_pr:.3f})')
    ax2.axhline(y=np.mean(all_targets), color='k', linestyle='--', label='Random')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plots saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate conditional shot prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_path", type=str, help="Path to test data (overrides config)")
    parser.add_argument("--analyze", action="store_true", help="Analyze predictions in detail")
    parser.add_argument("--plot", action="store_true", help="Plot ROC and PR curves")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark marginalization performance")
    parser.add_argument("--compare", action="store_true", help="Compare training vs inference modes")
    parser.add_argument("--use_gt_receiver", action="store_true", help="Use GT receiver (training mode)")
    parser.add_argument("--top_k", type=int, help="Number of top receivers for marginalization")
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
    
    logger.info(f"Evaluating shot prediction model on {device}")
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataset
    test_path = args.test_path or config["data"]["test_path"]
    test_dataset = ShotDataset(
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
        "auc": AUC(),
        "accuracy": Accuracy(),
        "f1": F1Score(),
        "ece": ECE(),
    }
    
    # Evaluate model
    logger.info("Starting evaluation...")
    eval_metrics = evaluate_model(
        model, test_loader, device, metrics,
        use_gt_receiver=args.use_gt_receiver,
        top_k_receivers=args.top_k
    )
    
    # Log results
    logger.info("=== Evaluation Results ===")
    logger.info(f"AUC-ROC: {eval_metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR: {eval_metrics['auc_pr']:.4f}")
    logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {eval_metrics['f1']:.4f}")
    logger.info(f"Expected Calibration Error: {eval_metrics['ece']:.4f}")
    
    # Analyze predictions if requested
    if args.analyze:
        analyze_predictions(model, test_loader, device, args.num_samples, args.use_gt_receiver)
    
    # Benchmark marginalization if requested
    if args.benchmark:
        benchmark_results = benchmark_marginalization(model, test_loader, device)
        
        # Save benchmark results
        benchmark_path = Path(config.get("log_dir", "runs")) / "shot_benchmark_results.json"
        import json
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        logger.info(f"Benchmark results saved to {benchmark_path}")
    
    # Compare training vs inference if requested
    if args.compare:
        comparison_results = compare_training_vs_inference(model, test_loader, device)
        
        # Save comparison results
        comparison_path = Path(config.get("log_dir", "runs")) / "shot_comparison_results.json"
        import json
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        logger.info(f"Comparison results saved to {comparison_path}")
    
    # Plot curves if requested
    if args.plot:
        plot_path = Path(config.get("log_dir", "runs")) / "shot_eval_curves.png"
        plot_roc_curve(model, test_loader, device, str(plot_path))
    
    # Save results
    results_path = Path(config.get("log_dir", "runs")) / "shot_eval_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
