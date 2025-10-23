"""Training script for shot prediction task.

This script trains a GATv2 model to predict shot occurrence in football matches.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network, ShotHead
from tacticai.dataio import ShotDataset, create_dataloader, create_dummy_dataset
from tacticai.modules import (
    BCELoss, AUC, F1Score, Accuracy,
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


class ShotModel(nn.Module):
    """Complete shot prediction model."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config["model"]
        
        # Create backbone
        self.backbone = GATv2Network(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            output_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            readout="mean",
            residual=True,
        )
        
        # Create head
        self.head = ShotHead(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch indices
            
        Returns:
            Shot predictions
        """
        # Get graph embeddings from backbone
        outputs = self.backbone(x, edge_index, batch=batch)
        if isinstance(outputs, tuple):
            _, graph_embeddings = outputs
        else:
            graph_embeddings = outputs
        
        # Apply head
        return self.head(graph_embeddings)


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create shot prediction model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Shot prediction model
    """
    model = ShotModel(config)
    
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
    
    # progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(dataloader):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(data["x"], data["edge_index"], data["batch"])
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data["x"], data["edge_index"], data["batch"])
            loss = criterion(outputs, targets)
            
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
            
            outputs = model(data["x"], data["edge_index"], data["batch"])
            loss = criterion(outputs, targets)
            
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
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO")
    )
    
    logger.info(f"Training shot prediction model on {device}")
    logger.info(f"Configuration: {config}")
    
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
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"AUC-ROC: {train_metrics['auc_roc']:.4f}, "
                   f"AUC-PR: {train_metrics['auc_pr']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"AUC-ROC: {val_metrics['auc_roc']:.4f}, "
                   f"AUC-PR: {val_metrics['auc_pr']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        
        logger.info(f"Learning rate: {current_lr:.6f}")
        
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
    
    # Save training history
    history_path = Path(config.get("log_dir", "runs")) / "shot_training_history.json"
    save_training_history(
        {"train": train_history, "val": val_history},
        history_path
    )


if __name__ == "__main__":
    main()
