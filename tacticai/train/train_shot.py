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
from torch.utils.data import DataLoader
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
from tacticai.modules.utils import load_backbone_from_checkpoint, save_training_history_csv
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
        # Backbone will be fine-tuned (not frozen) based on config
        if pretrained_config.get("freeze_backbone", False):
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
            H = self.backbone(x, edge_index, edge_attr)  # [N, hidden_dim] or [B, N_per_graph, hidden_dim]
            if H.dim() == 2:
                # Reshape to [B, N_per_graph, hidden_dim]
                H = H.view(B, num_nodes_per_graph, -1)
        
        # Get receiver probabilities
        if receiver_probs is None:
            if use_gt_receiver and gt_receiver is not None:
                # Use GT receiver (one-hot distribution)
                receiver_probs = F.one_hot(gt_receiver, num_classes=22).float()  # [B, 22]
            elif self.receiver_head is not None:
                # Use pretrained receiver model
                # H: [B, N_per_graph, hidden_dim]
                # ReceiverHead expects [B, N, hidden_dim] or [B*N, hidden_dim] and returns [B*N] (per-node scores)
                B, N, D = H.shape
                # ReceiverHead can handle [B, N, D] directly (see NodeScoreHead.forward)
                receiver_logits_per_node = self.receiver_head(H)  # [B, N] - each node's score
                # Apply softmax across nodes to get probability distribution over nodes (players)
                # receiver_logits_per_node: [B, N] where N=22 (players)
                receiver_probs = F.softmax(receiver_logits_per_node, dim=1)  # [B, 22]
            else:
                # Uniform distribution
                receiver_probs = torch.ones(B, 22, device=H.device) / 22
        
        # Shot prediction per node
        shot_logits_per_node = self.shot_head(H)  # [B, N_per_graph]
        shot_probs_per_node = torch.sigmoid(shot_logits_per_node)  # [B, N_per_graph]
        
        # Aggregate with receiver probabilities: Σ σ(s_i) × p_i
        # shot_probs_per_node: [B, N_per_graph], receiver_probs: [B, N] or [B, 22]
        # Ensure dimensions match
        if receiver_probs.size(1) == shot_probs_per_node.size(1):
            # Perfect match: direct element-wise multiplication and sum
            shot_prob = (shot_probs_per_node * receiver_probs).sum(dim=1, keepdim=True)  # [B, 1]
        elif receiver_probs.size(1) == 22 and shot_probs_per_node.size(1) == 22:
            # Both are 22 (all players)
            shot_prob = (shot_probs_per_node * receiver_probs).sum(dim=1, keepdim=True)  # [B, 1]
        else:
            # Dimension mismatch: use mean pooling as fallback
            shot_prob = shot_probs_per_node.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Convert probability to logit for loss computation
        # Use logit = log(p / (1-p)) with numerical stability
        epsilon = 1e-8
        shot_prob_clamped = torch.clamp(shot_prob, epsilon, 1.0 - epsilon)
        shot_logit = torch.log(shot_prob_clamped / (1.0 - shot_prob_clamped))
        
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
    # Check if pretrained section exists (backbone_path can be null for auto-selection)
    pretrained_config = config.get("pretrained", {})
    if pretrained_config:
        # Use ShotModelWithReceiver (handles pretrained backbone loading)
        model = ShotModelWithReceiver(config, device)
    else:
        # Fallback to old ShotModel for backward compatibility
        # Note: ShotModel is an alias for ShotModelWithReceiver, but requires device
        # For now, always use ShotModelWithReceiver
        model = ShotModelWithReceiver(config, device)
    
    # Note: D2 equivariance is handled internally in ShotModelWithReceiver
    # No need for GroupPoolingWrapper
    
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
    if pretrained_config.get("lr_backbone") is not None and pretrained_config.get("lr_head") is not None:
        # Use different learning rates for backbone and head
        if hasattr(model, 'backbone') and hasattr(model, 'shot_head'):
            # Separate parameter groups
            backbone_params = list(model.backbone.parameters())
            head_params = list(model.shot_head.parameters())
            
            if opt_config["type"] == "adam":
                optimizer = optim.Adam([
                    {'params': backbone_params, 'lr': pretrained_config["lr_backbone"]},
                    {'params': head_params, 'lr': pretrained_config["lr_head"]},
                ], weight_decay=opt_config.get("weight_decay", 1e-4))
            elif opt_config["type"] == "adamw":
                optimizer = optim.AdamW([
                    {'params': backbone_params, 'lr': pretrained_config["lr_backbone"]},
                    {'params': head_params, 'lr': pretrained_config["lr_head"]},
                ], weight_decay=opt_config.get("weight_decay", 1e-4))
            else:
                raise ValueError(f"Unknown optimizer type: {opt_config['type']}")
            return optimizer
    
    # Default: use same learning rate for all parameters
    if opt_config["type"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 1e-4),
        )
    elif opt_config["type"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 1e-4),
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
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Extract edge_attr if available
        edge_attr = data.get("edge_attr", None)
        
        # Extract GT receiver if available (for training)
        use_gt_receiver = data.get("receiver_id") is not None
        gt_receiver = data.get("receiver_id", None)
        
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
                loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            scaler.scale(loss).backward()
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
            loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions and targets for metrics
        with torch.no_grad():
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
        
        # Update progress bar (commented out since tqdm is commented out)
        # progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute AUC
    auc_roc, auc_pr = metrics["auc"](all_predictions, all_targets, compute_auc_pr=True)
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "f1": metrics["f1"](all_predictions, all_targets).item(),
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
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Validation"):
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
            loss = criterion(outputs, targets.unsqueeze(1) if targets.dim() == 1 else targets)
            
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute AUC
    auc_roc, auc_pr = metrics["auc"](all_predictions, all_targets, compute_auc_pr=True)
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "f1": metrics["f1"](all_predictions, all_targets).item(),
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
    
    # Setup logging with timestamped filename (save to runs/shot/)
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
    if args.debug_overfit:
        # Use small dataset for overfit test
        train_dataset = create_dummy_dataset("shot", num_samples=10, num_players=22)
        val_dataset = create_dummy_dataset("shot", num_samples=5, num_players=22)
    else:
        train_dataset = ShotDataset(
            config["data"]["train_path"],
            file_format=config["data"].get("format", "parquet")
        )
        val_dataset = ShotDataset(
            config["data"]["val_path"],
            file_format=config["data"].get("format", "parquet")
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,  # Disable pin_memory for MPS compatibility
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,  # Disable pin_memory for MPS compatibility
    )
    
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
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping", {}).get("patience", 10),
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
    
    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics,
            use_amp=config.get("train", {}).get("amp", False)
        )
        
        # Validation
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
        
        # Save best model
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            
            checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "shot" / "best.ckpt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], val_metrics,
                checkpoint_path, scheduler
            )
            logger.info(f"New best model saved with AUC-ROC: {best_val_auc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics["auc_roc"], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation AUC-ROC: {best_val_auc:.4f}")
    
    # Save training history (CSV format, same as receiver prediction)
    # Save to runs/shot/ directory
    csv_filename = f"training_history_{timestamp}.csv"
    csv_dir = Path(config.get("log_dir", "runs")) / "shot"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / csv_filename
    save_training_history_csv(
        train_history,
        val_history,
        test_history=None,  # Test metrics can be added later if needed
        filepath=csv_path
    )
    logger.info(f"Training history saved to {csv_path}")


if __name__ == "__main__":
    main()
