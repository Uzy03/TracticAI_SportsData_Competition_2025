"""Training script for CVAE tactic generation task.

This script trains a CVAE model to generate tactical formations in football matches.
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

from tacticai.models import CVAEModel
from tacticai.dataio import CVAEDataset, create_dataloader, create_dummy_dataset
from tacticai.modules import (
    CVAELoss, ReconstructionLoss, KLLoss,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping, save_training_history,
)


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
    """Create CVAE model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        CVAE model
    """
    model_config = config["model"]
    
    model = CVAEModel(
        input_dim=model_config["input_dim"],
        condition_dim=model_config["condition_dim"],
        latent_dim=model_config["latent_dim"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=model_config["dropout"],
    )
    
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
    use_amp: bool = False,
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, mean, log_var = model(
                    data["x"], data["edge_index"], data["batch"],
                    data["conditions"], training=True
                )
                
                # Reshape targets to match outputs (flatten player positions)
                batch_size = data["batch"].max().item() + 1
                targets_flat = targets.view(batch_size, -1)
                
                total_loss_batch, recon_loss, kl_loss = criterion(
                    outputs, targets_flat, mean, log_var
                )
            
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, mean, log_var = model(
                data["x"], data["edge_index"], data["batch"],
                data["conditions"], training=True
            )
            
            # Reshape targets to match outputs (flatten player positions)
            batch_size = data["batch"].max().item() + 1
            targets_flat = targets.view(batch_size, -1)
            
            total_loss_batch, recon_loss, kl_loss = criterion(
                outputs, targets_flat, mean, log_var
            )
            
            total_loss_batch.backward()
            optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss_batch.item():.4f}",
            "recon": f"{recon_loss.item():.4f}",
            "kl": f"{kl_loss.item():.4f}"
        })
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader),
    }
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model for one epoch.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Validation"):
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs, mean, log_var = model(
                data["x"], data["edge_index"], data["batch"],
                data["conditions"], training=False
            )
            
            # Reshape targets to match outputs (flatten player positions)
            batch_size = data["batch"].max().item() + 1
            targets_flat = targets.view(batch_size, -1)
            
            total_loss_batch, recon_loss, kl_loss = criterion(
                outputs, targets_flat, mean, log_var
            )
            
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    epoch_metrics = {
        "loss": total_loss / len(dataloader),
        "recon_loss": total_recon_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader),
    }
    
    return epoch_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CVAE model")
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
    
    logger.info(f"Training CVAE model on {device}")
    logger.info(f"Configuration: {config}")
    
    # Create datasets
    if args.debug_overfit:
        # Use small dataset for overfit test
        train_dataset = create_dummy_dataset("cvae", num_samples=10, num_players=22)
        val_dataset = create_dummy_dataset("cvae", num_samples=5, num_players=22)
    else:
        train_dataset = CVAEDataset(
            config["data"]["train_path"],
            file_format=config["data"].get("format", "parquet")
        )
        val_dataset = CVAEDataset(
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
    
    # Create loss function
    criterion = CVAELoss(
        recon_loss_type=config.get("loss", {}).get("recon_type", "mse"),
        beta=config.get("loss", {}).get("beta", 1.0),
        recon_weight=config.get("loss", {}).get("recon_weight", 1.0),
        kl_weight=config.get("loss", {}).get("kl_weight", 1.0),
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping", {}).get("patience", 15),
        mode="min",
        restore_best_weights=True,
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_history = {"loss": [], "recon_loss": [], "kl_loss": []}
    val_history = {"loss": [], "recon_loss": [], "kl_loss": []}
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("loss", float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_amp=config.get("train", {}).get("amp", False)
        )
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
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
                   f"Recon: {train_metrics['recon_loss']:.4f}, "
                   f"KL: {train_metrics['kl_loss']:.4f}")
        
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Recon: {val_metrics['recon_loss']:.4f}, "
                   f"KL: {val_metrics['kl_loss']:.4f}")
        
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Update history
        for key in train_history:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            
            checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "cvae" / "best.ckpt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], val_metrics,
                checkpoint_path, scheduler
            )
            logger.info(f"New best model saved with loss: {best_val_loss:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics["loss"], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save training history
    history_path = Path(config.get("log_dir", "runs")) / "cvae_training_history.json"
    save_training_history(
        {"train": train_history, "val": val_history},
        history_path
    )


if __name__ == "__main__":
    main()
