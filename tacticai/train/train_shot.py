"""Training script for shot prediction task.

This script trains a GATv2 model to predict shot occurrence in football matches.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network, GATv2Network4View, ShotHead, ReceiverHead
from tacticai.models.mlp_heads import ShotHeadNodeBased
from tacticai.dataio import ShotDataset, create_dataloader, create_dummy_dataset
from tacticai.modules import (
    BCELoss, AUC, F1Score, Accuracy,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping,
)
from tacticai.modules.utils import load_backbone_from_checkpoint, save_training_history_csv_shot
from tacticai.modules.transforms import RandomFlipTransform
from tacticai.modules.view_ops import apply_view_transform, D2_VIEWS
import torch.nn.functional as F


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


class ShotModelWithReceiver(nn.Module):
    """Shot prediction model with pretrained backbone and receiver conditioning.
    
    Uses pretrained receiver prediction backbone and receiver head for conditional shot prediction.
    Implements P(shot) = Σ P(shot | receiver=i) · P(receiver=i)
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        # Get D2 equivariance setting
        d2_config = config.get("d2", {})
        self.use_d2_equivariance = d2_config.get("enabled", False)
        
        pretrained_config = config.get("pretrained", {})
        model_config = config["model"]
        
        # Load pretrained backbone
        backbone_path = pretrained_config.get("backbone_path")
        if backbone_path is None:
            # Auto-select backbone checkpoint based on D2 setting
            checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
            if self.use_d2_equivariance:
                backbone_path = f"{checkpoint_dir}/receiver/backbone_d2.ckpt"
            else:
                backbone_path = f"{checkpoint_dir}/receiver/backbone_no_d2.ckpt"
        
        self.backbone, backbone_metadata = load_backbone_from_checkpoint(backbone_path, device)
        # Backbone will be frozen if freeze_backbone is True
        self.freeze_backbone = pretrained_config.get("freeze_backbone", True)  # Default to True (frozen)
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Load pretrained receiver head (frozen, for inference)
        use_receiver_for_conditioning = pretrained_config.get("use_receiver_for_conditioning", True)
        if use_receiver_for_conditioning:
            receiver_checkpoint_path = pretrained_config.get("receiver_checkpoint_path")
            if receiver_checkpoint_path is None:
                raise ValueError("pretrained.receiver_checkpoint_path must be specified when use_receiver_for_conditioning is True")
            
            receiver_checkpoint = torch.load(receiver_checkpoint_path, map_location=device)
            self.receiver_head = ReceiverHead(
                input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
                num_classes=22,  # 22 players
                dropout=0.0,  # No dropout for inference
            ).to(device)
            # Extract receiver head weights from checkpoint
            receiver_head_state = {}
            for k, v in receiver_checkpoint.get("model_state_dict", {}).items():
                if k.startswith("head."):
                    new_key = k.replace("head.", "")
                    receiver_head_state[new_key] = v
            
            if receiver_head_state:
                self.receiver_head.load_state_dict(receiver_head_state)
            self.receiver_head.requires_grad_(False)  # Freeze receiver head
        else:
            self.receiver_head = None
        
        # Shot prediction head (trainable)
        self.shot_head = ShotHeadNodeBased(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
            use_context=False,  # ノード埋め込みのみを使用（設計図に基づく）
        ).to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        receiver_probs: Optional[torch.Tensor] = None,
        use_gt_receiver: bool = False,
        gt_receiver: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch indices [N] (optional)
            receiver_probs: Receiver probability distribution [B, 22] (optional, precomputed)
            use_gt_receiver: Whether to use GT receiver (training mode)
            gt_receiver: GT receiver IDs [B] (optional, for training)
            
        Returns:
            Shot probability logit [B, 1]
        """
        # Process batch dimension
        if batch is not None:
            B = batch.max().item() + 1
        else:
            B = 1
        
        # Get node embeddings from backbone
        if self.use_d2_equivariance:
            # D2 equivariance: Create 4 views and use GATv2Network4View
            views_list = []
            for view_idx in range(len(D2_VIEWS)):
                x_view = x.clone()
                # Apply D2 reflection to coordinate-like features
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(0, 1))  # x, y
                if x_view.size(-1) > 3:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(2, 3))  # vx, vy
                if x_view.size(-1) > 8:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(7, 8))  # dx_to_kicker, dy_to_kicker
                if x_view.size(-1) > 12:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(11, 12))  # dx_to_goal, dy_to_goal
                views_list.append(x_view)
            
            # Stack views: [4, N, D] -> [B, 4, N_per_graph, D]
            x_views = torch.stack(views_list, dim=0)  # [4, N, D]
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            x_4view = x_views.view(4, B, num_nodes_per_graph, -1).permute(1, 0, 2, 3)  # [B, 4, N_per_graph, D]
            
            # Get node embeddings: [B, N_per_graph, hidden_dim]
            node_emb_4view = self.backbone(x_4view, edge_index, edge_attr)  # [B, 4, N_per_graph, hidden_dim]
            H = node_emb_4view.mean(dim=1)  # Average over 4 views: [B, N_per_graph, hidden_dim]
        else:
            # Standard mode: No D2 equivariance
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            
            # GATv2Network returns [N, hidden_dim] when batch is None, [B, N_per_graph, hidden_dim] when batch is provided
            # We need to pass batch=None to get node embeddings (not graph-level readout)
            H = self.backbone(x, edge_index, edge_attr, batch=None)  # [N, hidden_dim]
            
            # Reshape to [B, N_per_graph, hidden_dim]
            if H.dim() == 2:
                H = H.view(B, num_nodes_per_graph, -1)
            elif H.dim() == 3:
                # Already in [B, N_per_graph, hidden_dim] format
                pass
            else:
                raise ValueError(f"Unexpected H shape: {H.shape}, expected 2D or 3D")
        
        # Get receiver probabilities
        if receiver_probs is None:
            if use_gt_receiver and gt_receiver is not None:
                # Use GT receiver (one-hot distribution)
                receiver_probs = F.one_hot(gt_receiver, num_classes=22).float()  # [B, 22]
            elif self.receiver_head is not None:
                # Use pretrained receiver model
                # H: [B, N_per_graph, hidden_dim]
                # ReceiverHead can handle [B, N, D] directly (see NodeScoreHead.forward)
                receiver_logits_per_node = self.receiver_head(H)  # [B, N] - each node's score
                # Apply softmax across nodes to get probability distribution over nodes (players)
                receiver_probs = F.softmax(receiver_logits_per_node, dim=1)  # [B, 22]
            else:
                # Uniform distribution
                receiver_probs = torch.ones(B, 22, device=H.device) / 22
        
        # Shot prediction per node
        shot_logits_per_node = self.shot_head(H)  # [B, N_per_graph]
        shot_probs_per_node = torch.sigmoid(shot_logits_per_node)  # [B, N_per_graph]
        
        # Aggregate with receiver probabilities: Σ σ(s_i) × p_i
        # shot_probs_per_node: [B, N_per_graph], receiver_probs: [B, 22]
        if receiver_probs.size(1) == shot_probs_per_node.size(1):
            shot_prob = (shot_probs_per_node * receiver_probs).sum(dim=1, keepdim=True)  # [B, 1]
        elif receiver_probs.size(1) == 22 and shot_probs_per_node.size(1) == 22:
            shot_prob = (shot_probs_per_node * receiver_probs).sum(dim=1, keepdim=True)  # [B, 1]
        else:
            # Dimension mismatch: use mean pooling as fallback
            shot_prob = shot_probs_per_node.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Check for NaN in aggregated shot_prob
        if torch.isnan(shot_prob).any() or torch.isinf(shot_prob).any():
            import logging
            logger = logging.getLogger("tacticai")
            logger.warning(f"NaN/Inf in shot_prob after aggregation! shot_probs_per_node shape={shot_probs_per_node.shape}, receiver_probs shape={receiver_probs.shape}")
            # Replace NaN/Inf with 0.5 (neutral probability)
            shot_prob = torch.where(
                torch.isnan(shot_prob) | torch.isinf(shot_prob),
                torch.full_like(shot_prob, 0.5),
                shot_prob
            )
        
        # Convert probability to logit for loss computation
        # NOTE: We already have shot_logits_per_node from shot_head, but we aggregate probabilities
        # because we're using conditional probability model: P(shot) = Σ P(shot|receiver=i) * P(receiver=i)
        # After aggregation, we convert back to logit for BCE loss
        
        # Use more stable computation to avoid NaN
        epsilon = 1e-8
        shot_prob_clamped = torch.clamp(shot_prob, epsilon, 1.0 - epsilon)
        
        # More stable logit computation: use torch.logit which handles edge cases better
        shot_logit = torch.logit(shot_prob_clamped, eps=epsilon)
        
        # Clamp logit values to prevent extreme values that cause high loss
        # logit(0.01) ≈ -4.6, logit(0.99) ≈ 4.6
        # Clamp to reasonable range: [-10, 10] corresponds to prob [0.00005, 0.99995]
        shot_logit = torch.clamp(shot_logit, -10.0, 10.0)
        
        # Check for NaN/Inf and handle
        if torch.isnan(shot_logit).any() or torch.isinf(shot_logit).any():
            # Fallback: manual computation with better numerical stability
            # logit(p) = log(p) - log(1-p)
            shot_logit = torch.log(shot_prob_clamped + epsilon) - torch.log(1.0 - shot_prob_clamped + epsilon)
            # Clamp extreme values to prevent overflow
            shot_logit = torch.clamp(shot_logit, -10.0, 10.0)
        
        return shot_logit


# Backward compatibility alias
ShotModel = ShotModelWithReceiver


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create shot prediction model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Shot prediction model
    """
    # Always use ShotModelWithReceiver for consistency
    model = ShotModelWithReceiver(config, device)
    return model


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    opt_config = config["optimizer"]
    pretrained_config = config.get("pretrained", {})
    
    # Check if different learning rates are specified for backbone and head
    freeze_backbone = config.get("pretrained", {}).get("freeze_backbone", True)  # Default to True (frozen)
    if not freeze_backbone and pretrained_config.get("lr_backbone") is not None and pretrained_config.get("lr_head") is not None:
        # Separate learning rates for backbone and head (only if backbone is not frozen)
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.shot_head.parameters())
        
        param_groups = [
            {"params": backbone_params, "lr": pretrained_config["lr_backbone"]},
            {"params": head_params, "lr": pretrained_config["lr_head"]},
        ]
    
    if opt_config["type"] == "adam":
        optimizer = optim.Adam(
                param_groups,
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    else:
        # Use single learning rate for trainable parameters only
        # If backbone is frozen, only shot_head parameters will be optimized
        if freeze_backbone:
            # Only optimize shot_head (backbone is frozen)
            trainable_params = list(model.shot_head.parameters())
        else:
            # Optimize all parameters
            trainable_params = list(model.parameters())
        
        if opt_config["type"] == "adam":
            optimizer = optim.Adam(
                trainable_params,
            lr=opt_config["lr"],
                weight_decay=opt_config.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        config: Scheduler configuration
        
    Returns:
        Scheduler instance
    """
    sched_config = config.get("scheduler", {})
    
    if sched_config.get("type") == "cosine":
        return CosineAnnealingScheduler(
            optimizer,
            T_max=sched_config.get("T_max", config["train"]["epochs"]),
            eta_min=sched_config.get("eta_min", 0.0),
            warmup_epochs=sched_config.get("warmup_epochs", 0),
        )
    elif sched_config.get("type") == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get("step_size", 10),
            gamma=sched_config.get("gamma", 0.1),
        )
    else:
        return None


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
    use_amp: bool = False,
    grad_clip_enabled: bool = False,
    grad_clip_max_norm: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        metrics: Metric functions
        use_amp: Whether to use automatic mixed precision
        grad_clip_enabled: Whether to enable gradient clipping
        grad_clip_max_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_samples = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    import time
    import logging
    logger = logging.getLogger("tacticai")
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        batch_start_time = time.time()
        
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Extract edge_attr if available
        edge_attr = data.get("edge_attr", None)
        
        # Extract GT receiver if available (for training)
        use_gt_receiver = data.get("receiver_id") is not None
        gt_receiver = data.get("receiver_id", None)
        
        forward_start = time.time()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    x=data["x"],
                    edge_index=data["edge_index"],
                    edge_attr=edge_attr,
                    batch=data["batch"],
                    use_gt_receiver=use_gt_receiver,
                    gt_receiver=gt_receiver,
                )
                # outputs: [B, 1], targets: [B] or [B, 1]
                loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected (AMP)! outputs: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                logger.warning(f"targets: min={targets.min().item():.6f}, max={targets.max().item():.6f}, mean={targets.mean().item():.6f}")
                # Skip this batch
                continue
            
            scaler.scale(loss).backward()
            if grad_clip_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                x=data["x"],
                edge_index=data["edge_index"],
                edge_attr=edge_attr,
                batch=data["batch"],
                use_gt_receiver=use_gt_receiver,
                gt_receiver=gt_receiver,
            )
            # outputs: [B, 1], targets: [B] or [B, 1]
            loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss detected! outputs: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, mean={outputs.mean().item():.6f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                logger.warning(f"targets: min={targets.min().item():.6f}, max={targets.max().item():.6f}, mean={targets.mean().item():.6f}")
                # Skip this batch
                continue
            
            loss.backward()
            if grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        forward_time = batch_end_time - forward_start
        
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        
        # Collect predictions and targets for metrics (keep on GPU for efficiency)
        with torch.no_grad():
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
        
        # Log timing for first few batches
        if batch_idx < 3:
            logger.info(f"Batch {batch_idx}: forward={forward_time:.2f}s, total={batch_time:.2f}s")
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "t": f"{batch_time:.1f}s"})
    
    # Compute metrics on GPU (more efficient than CPU)
    with torch.no_grad():
        all_predictions = torch.cat(all_predictions, dim=0)  # [N, 1]
        all_targets = torch.cat(all_targets, dim=0)  # [N] or [N, 1]
        
        # Convert targets to [N] if needed
        if all_targets.dim() > 1:
            all_targets = all_targets.squeeze(-1)
        
        # For binary classification: convert logits to probabilities and apply threshold
        # Accuracy and F1 need binary predictions (0 or 1)
        probs = torch.sigmoid(all_predictions.squeeze(-1))  # [N]
        binary_preds = (probs > 0.5).long()  # [N]
        
        # Compute AUC (works with probabilities/logits)
    auc_roc, auc_pr = metrics["auc"](all_predictions, all_targets, compute_auc_pr=True)
        
        # Compute accuracy for binary classification
        accuracy = (binary_preds == all_targets).float().mean()
        
        # Compute F1 for binary classification
        tp = ((binary_preds == 1) & (all_targets == 1)).float().sum()
        fp = ((binary_preds == 1) & (all_targets == 0)).float().sum()
        fn = ((binary_preds == 0) & (all_targets == 1)).float().sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    epoch_metrics = {
            "loss": total_loss / num_samples,
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
            "accuracy": accuracy.item(),
            "f1": f1.item(),
    }
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        metrics: Metric functions
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for data, targets in progress_bar:
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            # Extract edge_attr if available
            edge_attr = data.get("edge_attr", None)
            
            outputs = model(
                x=data["x"],
                edge_index=data["edge_index"],
                edge_attr=edge_attr,
                batch=data["batch"],
                use_gt_receiver=False,  # ValidationではReceiver予測を使用
            )
            # outputs: [B, 1], targets: [B] or [B, 1]
            loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Collect predictions and targets for metrics (keep on GPU)
            all_predictions.append(outputs)
            all_targets.append(targets)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics on GPU
    all_predictions = torch.cat(all_predictions, dim=0)  # [N, 1]
    all_targets = torch.cat(all_targets, dim=0)  # [N] or [N, 1]
    
    # Convert targets to [N] if needed
    if all_targets.dim() > 1:
        all_targets = all_targets.squeeze(-1)
    
    # For binary classification: convert logits to probabilities and apply threshold
    # Accuracy and F1 need binary predictions (0 or 1)
    probs = torch.sigmoid(all_predictions.squeeze(-1))  # [N]
    binary_preds = (probs > 0.5).long()  # [N]
    
    # Compute AUC (works with probabilities/logits)
    auc_roc, auc_pr = metrics["auc"](all_predictions, all_targets, compute_auc_pr=True)
    
    # Compute accuracy for binary classification
    accuracy = (binary_preds == all_targets).float().mean()
    
    # Compute F1 for binary classification
    tp = ((binary_preds == 1) & (all_targets == 1)).float().sum()
    fp = ((binary_preds == 1) & (all_targets == 0)).float().sum()
    fn = ((binary_preds == 0) & (all_targets == 1)).float().sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    epoch_metrics = {
        "loss": total_loss / num_samples,
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
        "accuracy": accuracy.item(),
        "f1": f1.item(),
    }
    
    return epoch_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train shot prediction model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug_overfit", action="store_true", help="Debug overfit test")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging with timestamped filename
    log_dir = Path(config.get("log_dir", "runs")) / "shot"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"training_{timestamp}.log"
    logger = setup_logging(
        log_dir,
        config.get("log_level", "INFO"),
        log_file=log_filename
    )
    
    logger.info(f"Training shot prediction model on {device}")
    logger.info(f"Configuration: {config}")
    resolved_config_path = Path(args.config).resolve()
    logger.info(f"Resolved config path: {resolved_config_path}")
    
    # Create datasets
    merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
    
    if args.debug_overfit:
        # Use small dataset for overfit test (command-line flag)
        train_dataset = create_dummy_dataset("shot", num_samples=10, num_players=22)
        val_dataset = create_dummy_dataset("shot", num_samples=5, num_players=22)
        logger.info(f"[DEBUG-OVERFIT] Using dummy dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    else:
        # Check if debug_overfit is enabled in config
        debug_overfit_config = config.get("debug_overfit", {})
        use_debug_overfit = debug_overfit_config.get("enabled", False)
        
        if use_debug_overfit:
            # Load full training dataset first
            logger.info("[DEBUG-OVERFIT] Creating mini subset for overfitting test...")
            full_train_dataset = ShotDataset(
                config["data"]["train_path"],
                file_format=config["data"].get("format", "pickle")
            )
            
            # Override num_samples from command line if provided
            num_samples = debug_overfit_config.get("num_samples", 8)
            subset_seed = debug_overfit_config.get("seed", 42)
            
            # Create reproducible subset indices
            rng = np.random.RandomState(subset_seed)
            total_samples = len(full_train_dataset)
            if num_samples > total_samples:
                logger.warning(
                    f"[DEBUG-OVERFIT] Requested {num_samples} samples but only {total_samples} available. "
                    f"Using all {total_samples} samples."
                )
                num_samples = total_samples
            
            # Shuffle and select subset indices
            indices = rng.permutation(total_samples)[:num_samples]
            indices = sorted(indices.tolist())  # Sort for reproducibility
            
            logger.info(
                f"[DEBUG-OVERFIT] Selected {len(indices)} samples from {total_samples} total samples "
                f"(seed={subset_seed}, indices={indices[:5]}...{indices[-5:] if len(indices) > 10 else indices})"
            )
            
            # Create subset datasets (train=val=same samples for overfitting test)
            train_dataset = Subset(full_train_dataset, indices)
            val_dataset = Subset(full_train_dataset, indices)  # Same samples for train and val
            logger.info(
                f"[DEBUG-OVERFIT] Created subset datasets: train={len(train_dataset)}, val={len(val_dataset)} (same samples)"
            )
            merge_val_to_train = False  # Disable merge when using debug_overfit
        else:
            train_dataset_base = ShotDataset(
            config["data"]["train_path"],
                file_format=config["data"].get("format", "pickle")
            )
        
        if merge_val_to_train:
            # Load Val dataset to merge with Train
            val_path = config["data"]["val_path"]
            logger.info(f"[MERGE-VAL] Loading validation dataset to merge with train: {val_path}")
            val_dataset_base = ShotDataset(
                val_path,
                file_format=config["data"].get("format", "pickle")
            )
            logger.info(f"[MERGE-VAL] Validation dataset loaded: {len(val_dataset_base)} samples")
            
            # Merge Train and Val datasets
            train_dataset = ConcatDataset([train_dataset_base, val_dataset_base])
            logger.info(f"[MERGE-VAL] Merged train dataset: {len(train_dataset_base)} (train) + {len(val_dataset_base)} (val) = {len(train_dataset)} samples")
            
            # Set Val dataset to None (not used when merged)
            val_dataset = None
            logger.info(f"[MERGE-VAL] Val dataset set to None (using merged train dataset only)")
        else:
            # Normal mode: separate Train and Val
            train_dataset = train_dataset_base
        val_dataset = ShotDataset(
            config["data"]["val_path"],
                file_format=config["data"].get("format", "pickle")
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,  # Disable pin_memory for MPS compatibility
    )
    
    if merge_val_to_train and val_dataset is None:
        # Val dataset is None (merged to train), create dummy loader
        val_loader = None
        logger.info("[MERGE-VAL] Val loader set to None (Val merged to Train)")
    else:
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,  # Disable pin_memory for MPS compatibility
    )
    
    # Create test dataset and loader (for final evaluation)
    test_dataset = None
    test_loader = None
    if not args.debug_overfit and "test_path" in config.get("data", {}):
        try:
            test_dataset = ShotDataset(
                config["data"]["test_path"],
                file_format=config["data"].get("format", "pickle")
            )
            test_loader = create_dataloader(
                test_dataset,
                batch_size=config["eval"]["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 0),
                pin_memory=False,  # Disable pin_memory for MPS compatibility
            )
            logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
    
    if val_loader is not None:
        logger.info(f"Train dataloader: {len(train_loader)} batches, Val dataloader: {len(val_loader)} batches")
        if test_loader is not None:
            logger.info(f"Test dataloader: {len(test_loader)} batches")
    else:
        logger.info(f"Train dataloader: {len(train_loader)} batches (Val merged to Train)")
        if test_loader is not None:
            logger.info(f"Test dataloader: {len(test_loader)} batches")
    
    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function and metrics
    criterion = BCELoss(
        label_smoothing=config.get("loss", {}).get("label_smoothing", 0.0),
        pos_weight=torch.tensor(config.get("loss", {}).get("pos_weight", 1.0)) if config.get("loss", {}).get("pos_weight") else None,
    )
    
    metrics = {
        "auc": AUC(),
        "accuracy": Accuracy(),
        "f1": F1Score(),
    }
    
    # Create early stopping
    # Disable early stopping for debug_overfit mode or when Val is merged to Train
    debug_overfit_config = config.get("debug_overfit", {})
    use_debug_overfit = debug_overfit_config.get("enabled", False) or args.debug_overfit
    merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
    if use_debug_overfit:
        early_stopping_patience = 99999  # Effectively disable early stopping
        logger.info("[DEBUG-OVERFIT] Early stopping disabled (patience=99999)")
    elif merge_val_to_train:
        early_stopping_patience = 99999  # Effectively disable early stopping when Val is merged
        logger.info("[MERGE-VAL] Early stopping disabled (Val merged to Train, no validation set)")
    else:
        early_stopping_patience = config.get("early_stopping", {}).get("patience", 10)
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode="max",
        restore_best_weights=True,
    )
    
    # Training loop
    best_val_auc = 0.0
    train_history = {"loss": [], "auc_roc": [], "auc_pr": [], "accuracy": [], "f1": []}
    val_history = {"loss": [], "auc_roc": [], "auc_pr": [], "accuracy": [], "f1": []}
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint.get("metrics", {}).get("auc_roc", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Gradient clipping settings
    grad_clip_config = config.get("train", {}).get("grad_clip", {})
    grad_clip_enabled = grad_clip_config.get("enabled", False)
    grad_clip_max_norm = grad_clip_config.get("max_norm", 1.0)
    
    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics,
            use_amp=config.get("train", {}).get("amp", False),
            grad_clip_enabled=grad_clip_enabled,
            grad_clip_max_norm=grad_clip_max_norm,
        )
        
        # Validation (skip if Val is merged to Train)
        merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
        if merge_val_to_train and (val_loader is None or val_dataset is None):
            # Create dummy val_metrics when Val is merged to Train
            val_metrics = {
                "loss": 0.0,
                "auc_roc": 0.0,
                "auc_pr": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
            }
            logger.info("[MERGE-VAL] Validation skipped (Val merged to Train)")
        else:
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics)
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingScheduler):
                current_lr = scheduler.step()
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
        if merge_val_to_train:
            logger.info(f"Epoch {epoch+1}/{config['train']['epochs']} - "
                       f"Train: Loss={train_metrics['loss']:.4f}, "
                       f"AUC-ROC={train_metrics['auc_roc']:.4f}, "
                       f"AUC-PR={train_metrics['auc_pr']:.4f}, "
                       f"Acc={train_metrics['accuracy']:.4f}, "
                       f"F1={train_metrics['f1']:.4f} | "
                       f"Val: (skipped: Val merged to Train) | "
                       f"LR={current_lr:.6f}")
        else:
            logger.info(f"Epoch {epoch+1}/{config['train']['epochs']} - "
                       f"Train: Loss={train_metrics['loss']:.4f}, "
                       f"AUC-ROC={train_metrics['auc_roc']:.4f}, "
                       f"AUC-PR={train_metrics['auc_pr']:.4f}, "
                       f"Acc={train_metrics['accuracy']:.4f}, "
                       f"F1={train_metrics['f1']:.4f} | "
                       f"Val: Loss={val_metrics['loss']:.4f}, "
                       f"AUC-ROC={val_metrics['auc_roc']:.4f}, "
                       f"AUC-PR={val_metrics['auc_pr']:.4f}, "
                       f"Acc={val_metrics['accuracy']:.4f}, "
                       f"F1={val_metrics['f1']:.4f} | "
                       f"LR={current_lr:.6f}")
        
        # Update history
        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        # Save CSV history after each epoch (overwrite mode, same as receiver prediction)
        # Use debug_overfit-specific filename if enabled
        debug_overfit_config = config.get("debug_overfit", {})
        use_debug_overfit = debug_overfit_config.get("enabled", False) or args.debug_overfit
        if use_debug_overfit:
            num_samples = debug_overfit_config.get("num_samples", 8) if debug_overfit_config.get("enabled", False) else 10
            csv_filename = f"training_history_debug_overfit_n{num_samples}_{timestamp}.csv"
        else:
            csv_filename = f"training_history_{timestamp}.csv"
        csv_dir = Path(config.get("log_dir", "runs")) / "shot"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / csv_filename
        save_training_history_csv_shot(
            train_history,
            val_history,
            test_history=None,  # Test metrics will be added at the end
            filepath=csv_path
        )
        
        # Save best model (same as receiver prediction: D2 status in filename)
        merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
        use_d2 = config.get("d2", {}).get("enabled", False)
        checkpoint_filename = "best_d2.ckpt" if use_d2 else "best_no_d2.ckpt"
        
        if merge_val_to_train:
            # Use Train AUC-ROC for best model when Val is merged
            metric_for_best = train_metrics["auc_roc"]
            if metric_for_best > best_val_auc:
                best_val_auc = metric_for_best
                checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "shot" / checkpoint_filename
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    model, optimizer, epoch, train_metrics["loss"], train_metrics,
                    checkpoint_path, scheduler
                )
                logger.info(f"New best model saved with Train AUC-ROC: {best_val_auc:.4f} (D2: {use_d2})")
        else:
            # Normal mode: use Val AUC-ROC
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
                checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "shot" / checkpoint_filename
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], val_metrics,
                checkpoint_path, scheduler
            )
                logger.info(f"New best model saved with AUC-ROC: {best_val_auc:.4f} (D2: {use_d2})")
        
        # Early stopping (skip if Val is merged to Train)
        merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
        if not merge_val_to_train:
        if early_stopping(val_metrics["auc_roc"], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    merge_val_to_train = config.get("data", {}).get("merge_val_to_train", False)
    if merge_val_to_train:
        logger.info(f"Training completed. Best Train AUC-ROC: {best_val_auc:.4f} (Val merged to Train)")
    else:
    logger.info(f"Training completed. Best validation AUC-ROC: {best_val_auc:.4f}")
    
    # Evaluate on test set if available
    test_history = None
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_metrics = validate_epoch(model, test_loader, criterion, device, metrics)
        test_history = {
            "loss": test_metrics["loss"],
            "auc_roc": test_metrics["auc_roc"],
            "auc_pr": test_metrics["auc_pr"],
            "accuracy": test_metrics["accuracy"],
            "f1": test_metrics["f1"],
        }
        logger.info(
            f"Test - Loss: {test_metrics['loss']:.4f}, "
            f"AUC-ROC: {test_metrics['auc_roc']:.4f}, "
            f"AUC-PR: {test_metrics['auc_pr']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}"
        )
    
    # Save final CSV with test metrics (overwrite mode, same as receiver prediction)
    # Use debug_overfit-specific filename if enabled
    debug_overfit_config = config.get("debug_overfit", {})
    use_debug_overfit = debug_overfit_config.get("enabled", False) or args.debug_overfit
    if use_debug_overfit:
        num_samples = debug_overfit_config.get("num_samples", 8) if debug_overfit_config.get("enabled", False) else 10
        csv_filename = f"training_history_debug_overfit_n{num_samples}_{timestamp}.csv"
    else:
        csv_filename = f"training_history_{timestamp}.csv"
    csv_dir = Path(config.get("log_dir", "runs")) / "shot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / csv_filename
    save_training_history_csv_shot(
        train_history,
        val_history,
        test_history=test_history,
        filepath=csv_path
    )
    logger.info(f"Final training history saved to {csv_path}")


if __name__ == "__main__":
    main()
