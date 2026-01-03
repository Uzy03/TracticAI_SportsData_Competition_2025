"""Extended evaluation metrics for TacticAI.

Implements additional metrics including AUC, shot probability changes, and constraint violations.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Area Under ROC Curve (AUC).
    
    Args:
        predictions: Predicted probabilities [N]
        targets: Ground truth labels [N]
        
    Returns:
        AUC score
    """
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    if len(np.unique(targets_np)) < 2:
        return 0.0
    
    return roc_auc_score(targets_np, predictions_np)


def compute_delta_shot_prob(
    initial_probs: torch.Tensor,
    final_probs: torch.Tensor,
) -> Dict[str, float]:
    """Compute shot probability changes (ΔP(shot)).
    
    Args:
        initial_probs: Initial shot probabilities [N]
        final_probs: Final shot probabilities [N]
        
    Returns:
        Dictionary with delta statistics
    """
    delta = final_probs - initial_probs
    
    return {
        "mean_delta": delta.mean().item(),
        "std_delta": delta.std().item(),
        "min_delta": delta.min().item(),
        "max_delta": delta.max().item(),
        "median_delta": delta.median().item(),
        "positive_ratio": (delta > 0).float().mean().item(),
    }


def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k accuracy.
    
    Args:
        predictions: Predicted logits [N, num_classes]
        targets: Ground truth labels [N]
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    if predictions.dim() != 2:
        raise ValueError("Predictions must be 2D logits")
    
    _, top_k_indices = torch.topk(predictions, k, dim=1)
    correct = torch.any(top_k_indices == targets.unsqueeze(1), dim=1)
    
    return correct.float().mean().item()


def compute_receiver_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute receiver prediction metrics.
    
    Args:
        predictions: Predicted logits [N, num_classes]
        targets: Ground truth labels [N]
        
    Returns:
        Dictionary with receiver metrics
    """
    acc_top1 = compute_top_k_accuracy(predictions, targets, k=1)
    acc_top3 = compute_top_k_accuracy(predictions, targets, k=3)
    acc_top5 = compute_top_k_accuracy(predictions, targets, k=5)
    
    return {
        "acc_top1": acc_top1,
        "acc_top3": acc_top3,
        "acc_top5": acc_top5,
    }


def compute_movement_stats(
    initial_positions: torch.Tensor,
    final_positions: torch.Tensor,
) -> Dict[str, float]:
    """Compute movement statistics.
    
    Args:
        initial_positions: Initial positions [B, N, 2]
        final_positions: Final positions [B, N, 2]
        
    Returns:
        Dictionary with movement statistics
    """
    movement = torch.norm(final_positions - initial_positions, dim=-1)  # [B, N]
    
    return {
        "total_movement": movement.sum().item(),
        "mean_movement": movement.mean().item(),
        "std_movement": movement.std().item(),
        "max_movement": movement.max().item(),
        "min_movement": movement.min().item(),
        "median_movement": movement.median().item(),
    }


def compute_distance_violations(
    positions: torch.Tensor,
    min_distance: float = 1.0,
) -> Dict[str, int]:
    """Count distance violations (collisions).
    
    Args:
        positions: Player positions [B, N, 2]
        min_distance: Minimum allowed distance between players
        
    Returns:
        Dictionary with violation counts
    """
    B, N, _ = positions.shape
    
    total_violations = 0
    max_violations_per_sample = 0
    
    for b in range(B):
        distances = torch.cdist(positions[b], positions[b])  # [N, N]
        distances = distances + torch.eye(N, device=positions.device) * float('inf')
        min_dist = torch.min(distances, dim=-1)[0]  # [N]
        
        violations = (min_dist < min_distance).sum().item()
        total_violations += violations
        max_violations_per_sample = max(max_violations_per_sample, violations)
    
    return {
        "total_violations": total_violations,
        "violations_per_sample": total_violations / B,
        "max_violations_per_sample": max_violations_per_sample,
    }


def compute_boundary_violations(
    positions: torch.Tensor,
    field_bounds: Tuple[float, float, float, float] = (-52.5, -34.0, 52.5, 34.0),
) -> Dict[str, int]:
    """Count boundary violations.
    
    Args:
        positions: Player positions [B, N, 2]
        field_bounds: Field boundaries (x_min, y_min, x_max, y_max)
        
    Returns:
        Dictionary with violation counts
    """
    x_min, y_min, x_max, y_max = field_bounds
    
    violations = (
        (positions[:, :, 0] < x_min) | (positions[:, :, 0] > x_max) |
        (positions[:, :, 1] < y_min) | (positions[:, :, 1] > y_max)
    )
    
    return {
        "total_violations": violations.sum().item(),
        "violations_per_sample": violations.sum().float().mean().item(),
    }


def compute_shot_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute shot prediction metrics.
    
    Args:
        predictions: Predicted probabilities [N]
        targets: Ground truth labels [N]
        
    Returns:
        Dictionary with shot metrics
    """
    # Convert predictions to probabilities if needed
    if predictions.dim() > 1:
        predictions = F.softmax(predictions, dim=-1)[:, 1]
    else:
        predictions = torch.sigmoid(predictions)
    
    binary_pred = (predictions > 0.5).float()
    
    # Accuracy
    accuracy = (binary_pred == targets).float().mean().item()
    
    # AUC
    auc = compute_auc(predictions, targets)
    
    # Precision, Recall, F1
    tp = ((binary_pred == 1) & (targets == 1)).sum().float()
    fp = ((binary_pred == 1) & (targets == 0)).sum().float()
    fn = ((binary_pred == 0) & (targets == 1)).sum().float()
    
    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-8)).item()
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_guided_optimization_metrics(
    initial_positions: torch.Tensor,
    final_positions: torch.Tensor,
    initial_shot_probs: torch.Tensor,
    final_shot_probs: torch.Tensor,
    field_bounds: Tuple[float, float, float, float] = (-52.5, -34.0, 52.5, 34.0),
    min_distance: float = 1.0,
) -> Dict[str, float]:
    """Compute comprehensive metrics for guided optimization.
    
    Args:
        initial_positions: Initial player positions [B, N, 2]
        final_positions: Final player positions [B, N, 2]
        initial_shot_probs: Initial shot probabilities [B]
        final_shot_probs: Final shot probabilities [B]
        field_bounds: Field boundaries
        min_distance: Minimum distance between players
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Shot probability changes
    delta_metrics = compute_delta_shot_prob(initial_shot_probs, final_shot_probs)
    metrics.update({f"delta_{k}": v for k, v in delta_metrics.items()})
    
    # Movement statistics
    movement_metrics = compute_movement_stats(initial_positions, final_positions)
    metrics.update({f"movement_{k}": v for k, v in movement_metrics.items()})
    
    # Constraint violations
    distance_violations = compute_distance_violations(final_positions, min_distance)
    metrics.update(distance_violations)
    
    boundary_violations = compute_boundary_violations(final_positions, field_bounds)
    metrics.update(boundary_violations)
    
    return metrics

