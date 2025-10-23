"""Evaluation script for conditional shot prediction with marginalization.

This script implements the marginalization routine for shot prediction
over all possible receivers during inference.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


def marginalize_shot_predictions(
    model: torch.nn.Module,
    graph_embeddings: torch.Tensor,
    receiver_probs: Optional[torch.Tensor] = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Marginalize shot predictions over all possible receivers.
    
    Args:
        model: Trained shot prediction model
        graph_embeddings: Graph embeddings [B, embed_dim]
        receiver_probs: Receiver probability distribution [B, num_receivers] (optional)
        device: Device to run on
        
    Returns:
        Marginalized shot probabilities [B]
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    graph_embeddings = graph_embeddings.to(device)
    B = graph_embeddings.size(0)
    
    with torch.no_grad():
        # Get predictions for all receivers
        shot_predictions = model(
            graph_embeddings=graph_embeddings,
            receiver_ids=None,  # None triggers marginalization
            training=False
        )  # [B, num_receivers]
        
        if receiver_probs is not None:
            # Weight by receiver probabilities
            receiver_probs = receiver_probs.to(device)
            marginalized_probs = (shot_predictions * receiver_probs).sum(dim=-1)
        else:
            # Uniform weighting (equal probability for all receivers)
            marginalized_probs = shot_predictions.mean(dim=-1)
    
    return marginalized_probs


def evaluate_conditional_shot_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    receiver_model: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate conditional shot prediction model.
    
    Args:
        model: Trained conditional shot model
        dataloader: Test dataloader
        device: Device to run on
        receiver_model: Optional receiver prediction model for weighting
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    if receiver_model is not None:
        receiver_model.eval()
    
    all_predictions = []
    all_labels = []
    all_receiver_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get graph embeddings (assuming backbone is available)
            if hasattr(model, 'backbone'):
                node_embeddings, graph_embeddings = model.backbone(
                    batch['x'], 
                    batch['edge_index'], 
                    batch.get('edge_attr'), 
                    batch.get('batch')
                )
            else:
                # Assume graph_embeddings are provided directly
                graph_embeddings = batch['graph_embeddings']
            
            # Get receiver predictions if receiver model is provided
            receiver_probs = None
            if receiver_model is not None:
                receiver_logits = receiver_model(
                    batch['x'], 
                    batch['edge_index'], 
                    batch.get('edge_attr'), 
                    batch.get('batch')
                )
                receiver_probs = F.softmax(receiver_logits, dim=-1)
            
            # Marginalize shot predictions
            shot_probs = marginalize_shot_predictions(
                model, graph_embeddings, receiver_probs, device
            )
            
            # Store results
            all_predictions.extend(shot_probs.cpu().numpy())
            all_labels.extend(batch['shot_label'].cpu().numpy())
            
            if receiver_probs is not None:
                all_receiver_predictions.extend(receiver_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    metrics = {}
    
    # AUC-ROC
    if len(np.unique(all_labels)) > 1:
        metrics['auc_roc'] = roc_auc_score(all_labels, all_predictions)
    else:
        metrics['auc_roc'] = 0.0
    
    # Average Precision
    metrics['average_precision'] = average_precision_score(all_labels, all_predictions)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_predictions)
    
    # F1 score at optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    metrics['f1_optimal'] = f1_scores[optimal_idx]
    metrics['precision_optimal'] = precision[optimal_idx]
    metrics['recall_optimal'] = recall[optimal_idx]
    metrics['threshold_optimal'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Accuracy at optimal threshold
    binary_predictions = (all_predictions >= metrics['threshold_optimal']).astype(int)
    metrics['accuracy_optimal'] = (binary_predictions == all_labels).mean()
    
    # Additional metrics
    metrics['mean_prediction'] = all_predictions.mean()
    metrics['std_prediction'] = all_predictions.std()
    metrics['positive_rate'] = all_labels.mean()
    
    return metrics


def compare_receiver_conditioning(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Compare different receiver conditioning strategies.
    
    Args:
        model: Trained conditional shot model
        dataloader: Test dataloader
        device: Device to run on
        
    Returns:
        Dictionary comparing different conditioning strategies
    """
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Get graph embeddings
            if hasattr(model, 'backbone'):
                node_embeddings, graph_embeddings = model.backbone(
                    batch['x'], 
                    batch['edge_index'], 
                    batch.get('edge_attr'), 
                    batch.get('batch')
                )
            else:
                graph_embeddings = batch['graph_embeddings']
            
            # 1. Ground truth receiver conditioning
            gt_receiver_predictions = model(
                graph_embeddings=graph_embeddings,
                receiver_ids=batch['receiver_label'],
                training=False
            )
            
            # 2. Marginalization over all receivers
            marginal_predictions = marginalize_shot_predictions(
                model, graph_embeddings, None, device
            )
            
            # 3. Most likely receiver conditioning
            if 'receiver_logits' in batch:
                receiver_probs = F.softmax(batch['receiver_logits'], dim=-1)
                most_likely_receiver = receiver_probs.argmax(dim=-1)
                ml_receiver_predictions = model(
                    graph_embeddings=graph_embeddings,
                    receiver_ids=most_likely_receiver,
                    training=False
                )
                
                # 4. Weighted marginalization
                weighted_predictions = marginalize_shot_predictions(
                    model, graph_embeddings, receiver_probs, device
                )
                
                results['most_likely_receiver'] = {
                    'predictions': ml_receiver_predictions.cpu().numpy(),
                    'labels': batch['shot_label'].cpu().numpy()
                }
                
                results['weighted_marginalization'] = {
                    'predictions': weighted_predictions.cpu().numpy(),
                    'labels': batch['shot_label'].cpu().numpy()
                }
            
            results['ground_truth_receiver'] = {
                'predictions': gt_receiver_predictions.cpu().numpy(),
                'labels': batch['shot_label'].cpu().numpy()
            }
            
            results['uniform_marginalization'] = {
                'predictions': marginal_predictions.cpu().numpy(),
                'labels': batch['shot_label'].cpu().numpy()
            }
    
    # Compute metrics for each strategy
    strategy_metrics = {}
    for strategy, data in results.items():
        predictions = data['predictions']
        labels = data['labels']
        
        if len(np.unique(labels)) > 1:
            auc_roc = roc_auc_score(labels, predictions)
            avg_precision = average_precision_score(labels, predictions)
        else:
            auc_roc = 0.0
            avg_precision = 0.0
        
        strategy_metrics[strategy] = {
            'auc_roc': auc_roc,
            'average_precision': avg_precision,
            'mean_prediction': predictions.mean(),
            'positive_rate': labels.mean()
        }
    
    return strategy_metrics


def analyze_receiver_importance(
    model: torch.nn.Module,
    graph_embeddings: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Analyze the importance of different receivers for shot prediction.
    
    Args:
        model: Trained conditional shot model
        graph_embeddings: Graph embeddings [B, embed_dim]
        device: Device to run on
        
    Returns:
        Dictionary with receiver importance scores
    """
    model.eval()
    graph_embeddings = graph_embeddings.to(device)
    B = graph_embeddings.size(0)
    
    with torch.no_grad():
        # Get predictions for each receiver individually
        receiver_shot_probs = []
        
        for receiver_id in range(22):  # Assuming 22 players
            receiver_ids = torch.full((B,), receiver_id, device=device)
            shot_probs = model(
                graph_embeddings=graph_embeddings,
                receiver_ids=receiver_ids,
                training=False
            )
            receiver_shot_probs.append(shot_probs.cpu().numpy())
        
        receiver_shot_probs = np.array(receiver_shot_probs)  # [22, B]
        
        # Compute importance metrics
        importance_scores = {}
        
        # Mean shot probability for each receiver
        importance_scores['mean_shot_prob'] = receiver_shot_probs.mean(axis=1)
        
        # Variance in shot probability for each receiver
        importance_scores['shot_prob_variance'] = receiver_shot_probs.var(axis=1)
        
        # Difference from marginal prediction
        marginal_probs = receiver_shot_probs.mean(axis=0)
        importance_scores['deviation_from_marginal'] = np.abs(
            receiver_shot_probs - marginal_probs[np.newaxis, :]
        ).mean(axis=1)
        
        # Convert to dictionary with receiver IDs
        receiver_importance = {}
        for i in range(22):
            receiver_importance[f'receiver_{i}'] = {
                'mean_shot_prob': importance_scores['mean_shot_prob'][i],
                'shot_prob_variance': importance_scores['shot_prob_variance'][i],
                'deviation_from_marginal': importance_scores['deviation_from_marginal'][i]
            }
    
    return receiver_importance
