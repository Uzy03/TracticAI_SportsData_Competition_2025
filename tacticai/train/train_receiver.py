"""Training script for receiver prediction task.

This script trains a GATv2 model to predict pass receivers in football matches.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network4View, ReceiverHead
from tacticai.modules.view_ops import apply_view_transform, D2_VIEWS
from tacticai.dataio import ReceiverDataset, create_dataloader, create_dummy_dataset
from tacticai.modules import (
    CrossEntropyLoss, TopKAccuracy, Accuracy, F1Score,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping, save_training_history,
)
from tacticai.modules.transforms import RandomFlipTransform


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


class ReceiverModel(nn.Module):
    """Complete receiver prediction model with D2 equivariance."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config["model"]
        
        # Create backbone with 4-view D2 equivariance
        self.backbone = GATv2Network4View(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            readout="mean",
            residual=True,
            view_mixing="attention",
        )
        
        # Create head
        self.head = ReceiverHead(
            input_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        team: Optional[torch.Tensor] = None,
        ball: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with D2 equivariance.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            batch: Batch indices [N]
            mask: Node mask [N] where 1=valid, 0=missing (optional)
            team: Team IDs [N] where 0=attacking team, 1=defending team (optional)
            ball: Ball possession [N] where 1=has ball, 0=no ball (optional)
            
        Returns:
            Receiver predictions - shape depends on filtering:
            - If team and ball provided: [N_attacking, num_classes] (filtered)
            - Otherwise: [N, num_classes] (all nodes)
        """
        # Process each graph in batch separately (since 4-view processing is per-graph)
        B = batch.max().item() + 1 if batch is not None else 1
        
        # Collect node embeddings for each graph
        all_node_embeddings = []
        
        for b in range(B):
            batch_mask = batch == b if batch is not None else torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            x_b = x[batch_mask]  # [N_b, D]
            
            # Create 4 views for this graph: identity, horizontal flip, vertical flip, both flips
            # x coordinates are at indices 0, 1; velocity x, y at indices 2, 3
            views = []
            for view_name in D2_VIEWS:
                view_idx = D2_VIEWS.index(view_name)
                x_view = apply_view_transform(x_b, view_idx, xy_indices=(0, 1))  # Only flip x, y positions
                views.append(x_view)
            x_4view = torch.stack(views, dim=0).unsqueeze(0)  # [1, 4, N_b, D]
            
            # Build per-graph edge_index by subsetting and remapping to 0..N_b-1
            orig_idx = torch.where(batch_mask)[0]
            # Map original indices -> 0..N_b-1
            remap = -torch.ones(x.size(0), dtype=torch.long, device=x.device)
            remap[orig_idx] = torch.arange(orig_idx.numel(), device=x.device)
            src_all, dst_all = edge_index[0], edge_index[1]
            src_m = remap[src_all]
            dst_m = remap[dst_all]
            valid = (src_m >= 0) & (dst_m >= 0)
            edge_index_b = torch.stack([src_m[valid], dst_m[valid]], dim=0)
            
            # Get node embeddings from backbone: [1, 4, N_b, output_dim]
            # Note: GATv2Network4View expects [B, 4, N, D] input
            node_emb_4view = self.backbone(x_4view, edge_index_b, None)  # batch=None since we handle per-graph
            
            # Average over 4 views: [N_b, output_dim]
            node_emb = node_emb_4view[0].mean(dim=0)  # [N_b, output_dim]
            
            all_node_embeddings.append(node_emb)
        
        # Concatenate all node embeddings
        node_embeddings = torch.cat(all_node_embeddings, dim=0)  # [N_total, output_dim]
        
        # Apply mask if provided (element-wise multiplication, not pooling)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(node_embeddings)  # [N, D]
            node_embeddings = node_embeddings * mask_expanded
        
        # Get logits for all nodes
        all_logits = self.head(node_embeddings)  # [N_total, num_classes]
        
        # Filter to attacking nodes (team=0) and exclude ball owner if both provided
        if team is not None and ball is not None:
            attacking_mask = (team == 0) & (ball == 0)  # Attacking team and not ball owner
            if attacking_mask.sum() > 0:
                # Return only attacking nodes (excluding ball owner)
                return all_logits[attacking_mask]
            else:
                # Fallback: return all nodes (should not happen in practice)
                return all_logits
        else:
            # No filtering: return predictions for all nodes
            return all_logits


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create receiver prediction model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Receiver prediction model
    """
    model = ReceiverModel(config)
    
    # Apply D2 group pooling if enabled
    if config.get("d2", {}).get("group_pool", False):
        from tacticai.modules.transforms import GroupPoolingWrapper
        model = GroupPoolingWrapper(model, average_logits=True)
    
    return model.to(device)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    opt_config = config["optimizer"]
    
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
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                # Pass mask, team, ball if available
                outputs = model(
                    data["x"], 
                    data["edge_index"], 
                    data["batch"],
                    mask=data.get("mask"),
                    team=data.get("team"),
                    ball=data.get("ball"),
                )
                
                # Handle filtered outputs (only attacking nodes) or all nodes
                batch_size = data["batch"].max().item() + 1
                graph_outputs = []
                graph_targets = []
                output_idx = 0
                
                for i in range(batch_size):
                    node_mask = data["batch"] == i
                    if node_mask.any():
                        # Get attacking nodes for this graph
                        if "team" in data and "ball" in data:
                            team_i = data["team"][node_mask]
                            ball_i = data["ball"][node_mask]
                            attacking_mask = (team_i == 0) & (ball_i == 0)
                            num_attacking = attacking_mask.sum().item()
                            if num_attacking > 0:
                                graph_outputs.append(outputs[output_idx:output_idx+num_attacking])
                                # Find receiver node index within attacking nodes
                                receiver_node_idx = targets[i].item()  # Original node index
                                # Map to attacking node index
                                attacking_indices = torch.where(node_mask)[0]
                                attacking_indices_i = attacking_indices[attacking_mask]
                                if receiver_node_idx in attacking_indices_i:
                                    receiver_attacking_idx = (attacking_indices_i == receiver_node_idx).nonzero(as_tuple=True)[0][0].item()
                                    graph_targets.append(receiver_attacking_idx)
                                else:
                                    # Receiver is ball owner or defender, skip or use fallback
                                    graph_targets.append(0)
                                output_idx += num_attacking
                        else:
                            # Fallback: use first node
                            graph_outputs.append(outputs[output_idx:output_idx+1])
                            graph_targets.append(targets[i])
                            output_idx += 1
                
                if graph_outputs:
                    graph_outputs = torch.cat(graph_outputs, dim=0)
                    graph_targets = torch.tensor(graph_targets, dtype=torch.long, device=outputs.device)
                    loss = criterion(graph_outputs, graph_targets)
                else:
                    continue
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Pass mask, team, ball if available
            outputs = model(
                data["x"], 
                data["edge_index"], 
                data["batch"],
                mask=data.get("mask"),
                team=data.get("team"),
                ball=data.get("ball"),
            )
            
            # Handle filtered outputs (only attacking nodes) or all nodes
            batch_size = data["batch"].max().item() + 1
            graph_outputs = []
            graph_targets = []
            output_idx = 0
            
            for i in range(batch_size):
                node_mask = data["batch"] == i
                if node_mask.any():
                    # Get attacking nodes for this graph
                    if "team" in data and "ball" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        attacking_mask = (team_i == 0) & (ball_i == 0)
                        num_attacking = attacking_mask.sum().item()
                        if num_attacking > 0:
                            graph_outputs.append(outputs[output_idx:output_idx+num_attacking])
                            # Find receiver node index within attacking nodes
                            receiver_node_idx = targets[i].item()  # Original node index
                            # Map to attacking node index
                            attacking_indices = torch.where(node_mask)[0]
                            attacking_indices_i = attacking_indices[attacking_mask]
                            if len(attacking_indices_i) > 0 and receiver_node_idx < len(attacking_indices_i):
                                # Check if receiver is in attacking nodes
                                if receiver_node_idx in attacking_indices_i:
                                    receiver_attacking_idx = (attacking_indices_i == receiver_node_idx).nonzero(as_tuple=True)[0][0].item()
                                else:
                                    receiver_attacking_idx = 0  # Fallback
                            else:
                                receiver_attacking_idx = 0  # Fallback
                            graph_targets.append(receiver_attacking_idx)
                            output_idx += num_attacking
                    else:
                        # Fallback: use first node
                        graph_outputs.append(outputs[output_idx:output_idx+1])
                        graph_targets.append(targets[i])
                        output_idx += 1
            
            if graph_outputs:
                graph_outputs = torch.cat(graph_outputs, dim=0)
                graph_targets = torch.tensor(graph_targets, dtype=torch.long, device=outputs.device)
                loss = criterion(graph_outputs, graph_targets)
            else:
                continue
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions and targets for metrics
        with torch.no_grad():
            if 'graph_outputs' in locals():
                all_predictions.append(graph_outputs.cpu())
                all_targets.append(graph_targets.cpu())
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "top1": metrics["top1"](all_predictions, all_targets).item(),
        "top3": metrics["top3"](all_predictions, all_targets).item(),
        "top5": metrics["top5"](all_predictions, all_targets).item(),
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
            
            # Pass mask, team, ball if available
            outputs = model(
                data["x"], 
                data["edge_index"], 
                data["batch"],
                mask=data.get("mask"),
                team=data.get("team"),
                ball=data.get("ball"),
            )
            
            # Handle filtered outputs (only attacking nodes) or all nodes
            batch_size = data["batch"].max().item() + 1
            graph_outputs = []
            graph_targets = []
            output_idx = 0
            
            for i in range(batch_size):
                node_mask = data["batch"] == i
                if node_mask.any():
                    # Get attacking nodes for this graph
                    if "team" in data and "ball" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        attacking_mask = (team_i == 0) & (ball_i == 0)
                        num_attacking = attacking_mask.sum().item()
                        if num_attacking > 0:
                            graph_outputs.append(outputs[output_idx:output_idx+num_attacking])
                            # Find receiver node index within attacking nodes
                            receiver_node_idx = targets[i].item()  # Original node index
                            # Map to attacking node index
                            attacking_indices = torch.where(node_mask)[0]
                            attacking_indices_i = attacking_indices[attacking_mask]
                            if len(attacking_indices_i) > 0 and receiver_node_idx < len(attacking_indices_i):
                                # Check if receiver is in attacking nodes
                                if receiver_node_idx in attacking_indices_i:
                                    receiver_attacking_idx = (attacking_indices_i == receiver_node_idx).nonzero(as_tuple=True)[0][0].item()
                                else:
                                    receiver_attacking_idx = 0  # Fallback
                            else:
                                receiver_attacking_idx = 0  # Fallback
                            graph_targets.append(receiver_attacking_idx)
                            output_idx += num_attacking
                    else:
                        # Fallback: use first node
                        graph_outputs.append(outputs[output_idx:output_idx+1])
                        graph_targets.append(targets[i])
                        output_idx += 1
            
            if graph_outputs:
                graph_outputs = torch.cat(graph_outputs, dim=0)
                graph_targets = torch.tensor(graph_targets, dtype=torch.long, device=outputs.device)
                
                loss = criterion(graph_outputs, graph_targets)
                total_loss += loss.item()
                
                all_predictions.append(graph_outputs.cpu())
                all_targets.append(graph_targets.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": metrics["accuracy"](all_predictions, all_targets).item(),
        "top1": metrics["top1"](all_predictions, all_targets).item(),
        "top3": metrics["top3"](all_predictions, all_targets).item(),
        "top5": metrics["top5"](all_predictions, all_targets).item(),
    }
    
    return epoch_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train receiver prediction model")
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
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO")
    )
    
    logger.info(f"Training receiver prediction model on {device}")
    logger.info(f"Configuration: {config}")
    
    # Create datasets
    if args.debug_overfit:
        # Use small dataset for overfit test
        train_dataset = create_dummy_dataset("receiver", num_samples=10, num_players=22)
        val_dataset = create_dummy_dataset("receiver", num_samples=5, num_players=22)
    else:
        train_dataset = ReceiverDataset(
            config["data"]["train_path"],
            file_format=config["data"].get("format", "parquet")
        )
        val_dataset = ReceiverDataset(
            config["data"]["val_path"],
            file_format=config["data"].get("format", "parquet")
        )
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )
    
    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function and metrics
    criterion = CrossEntropyLoss(
        label_smoothing=config.get("loss", {}).get("label_smoothing", 0.0)
    )
    
    metrics = {
        "accuracy": Accuracy(),
        "top1": TopKAccuracy(k=1),
        "top3": TopKAccuracy(k=3),
        "top5": TopKAccuracy(k=5),
    }
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping", {}).get("patience", 10),
        mode="max",
        restore_best_weights=True,
    )
    
    # Training loop
    best_val_accuracy = 0.0
    train_history = {"loss": [], "accuracy": [], "top1": [], "top3": [], "top5": []}
    val_history = {"loss": [], "accuracy": [], "top1": [], "top3": [], "top5": []}
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint.get("metrics", {}).get("accuracy", 0.0)
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
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"Top-1: {train_metrics['top1']:.4f}, "
                   f"Top-3: {train_metrics['top3']:.4f}, "
                   f"Top-5: {train_metrics['top5']:.4f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"Top-1: {val_metrics['top1']:.4f}, "
                   f"Top-3: {val_metrics['top3']:.4f}, "
                   f"Top-5: {val_metrics['top5']:.4f}")
        
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Update history
        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        # Save best model
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            
            checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "receiver" / "best.ckpt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], val_metrics,
                checkpoint_path, scheduler
            )
            logger.info(f"New best model saved with accuracy: {best_val_accuracy:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics["accuracy"], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_accuracy:.4f}")
    
    # Save training history
    history_path = Path(config.get("log_dir", "runs")) / "receiver_training_history.json"
    save_training_history(
        {"train": train_history, "val": val_history},
        history_path
    )


if __name__ == "__main__":
    main()
