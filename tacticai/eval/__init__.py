"""Evaluation scripts for TacticAI tasks."""

from .metrics_extended import (
    compute_auc,
    compute_delta_shot_prob,
    compute_top_k_accuracy,
    compute_receiver_metrics,
    compute_movement_stats,
    compute_distance_violations,
    compute_boundary_violations,
    compute_shot_metrics,
    compute_guided_optimization_metrics,
)

__all__ = [
    "compute_auc",
    "compute_delta_shot_prob",
    "compute_top_k_accuracy",
    "compute_receiver_metrics",
    "compute_movement_stats",
    "compute_distance_violations",
    "compute_boundary_violations",
    "compute_shot_metrics",
    "compute_guided_optimization_metrics",
]
