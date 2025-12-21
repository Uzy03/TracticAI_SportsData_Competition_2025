"""Training script for multi-task learning (receiver + shot prediction)."""

import argparse
import logging
import math
import time
from datetime import datetime
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
from tqdm import tqdm

from tacticai.models import MultiTaskModel
from tacticai.dataio import MultiTaskDataset, create_dataloader
from tacticai.modules import (
    CrossEntropyLoss, BCELoss, TopKAccuracy, Accuracy, F1Score, AUC,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping,
)
from tacticai.modules.utils import save_training_history_csv
from tacticai.modules.mlp_heads import mask_logits


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create multi-task model."""
    model = MultiTaskModel(config)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
    
    model.apply(init_weights)
    return model.to(device)


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer."""
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
    """Create learning rate scheduler."""
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
    criterion_receiver: nn.Module,
    criterion_shot: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
    lambda_receiver: float = 1.0,
    lambda_shot: float = 1.0,
    use_amp: bool = False,
    grad_clip_enabled: bool = False,
    grad_clip_max_norm: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_receiver_loss = 0.0
    total_shot_loss = 0.0
    total_loss = 0.0
    
    receiver_predictions = []
    receiver_targets = []
    shot_predictions = []
    shot_targets = []
    
    num_samples = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    logger = logging.getLogger("tacticai")
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets_dict) in enumerate(progress_bar):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        receiver_target = targets_dict["receiver_target"].to(device)
        shot_target = targets_dict["shot_target"].to(device)
        
        optimizer.zero_grad()
        
        edge_attr = data.get("edge_attr", None)
        mask = data.get("mask", None)
        team = data.get("team", None)
        ball = data.get("ball", None)
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    x=data["x"],
                    edge_index=data["edge_index"],
                    edge_attr=edge_attr,
                    batch=data["batch"],
                    mask=mask,
                    team=team,
                    ball=ball,
                )
        else:
            outputs = model(
                x=data["x"],
                edge_index=data["edge_index"],
                edge_attr=edge_attr,
                batch=data["batch"],
                mask=mask,
                team=team,
                ball=ball,
            )
        
        receiver_logits = outputs["receiver_logits"]  # [N_attacking, num_classes]
        shot_logit = outputs["shot_logit"]  # [B, 1]
        
        # Compute receiver loss (need to filter targets to match filtered logits)
        # TODO: Implement proper filtering logic here (similar to train_receiver.py)
        # For now, use a simplified approach
        # This is a placeholder - actual implementation needs to handle candidate masking
        receiver_loss = criterion_receiver(receiver_logits, receiver_target)
        
        # Compute shot loss
        shot_loss = criterion_shot(shot_logit, shot_target.unsqueeze(1) if shot_target.dim() == 1 else shot_target)
        
        # Combined loss
        total_batch_loss = lambda_receiver * receiver_loss + lambda_shot * shot_loss
        
        # Check for NaN/Inf
        if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
            logger.warning(f"NaN/Inf loss detected! Skipping batch {batch_idx}")
            continue
        
        if use_amp and scaler is not None:
            scaler.scale(total_batch_loss).backward()
            if grad_clip_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_batch_loss.backward()
            if grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
        
        # Accumulate metrics
        batch_size = shot_target.size(0)
        total_receiver_loss += receiver_loss.item() * batch_size
        total_shot_loss += shot_loss.item() * batch_size
        total_loss += total_batch_loss.item() * batch_size
        num_samples += batch_size
        
        # Collect predictions for metrics
        with torch.no_grad():
            receiver_predictions.append(receiver_logits.detach())
            receiver_targets.append(receiver_target.detach())
            shot_predictions.append(shot_logit.detach())
            shot_targets.append(shot_target.detach())
        
        progress_bar.set_postfix({
            "loss": f"{total_batch_loss.item():.4f}",
            "rec": f"{receiver_loss.item():.4f}",
            "shot": f"{shot_loss.item():.4f}",
        })
    
    # Compute metrics
    with torch.no_grad():
        receiver_logits_all = torch.cat(receiver_predictions, dim=0)
        receiver_targets_all = torch.cat(receiver_targets, dim=0)
        shot_logits_all = torch.cat(shot_predictions, dim=0)
        shot_targets_all = torch.cat(shot_targets, dim=0)
        
        # Receiver metrics
        receiver_top1 = metrics["receiver_top1"](receiver_logits_all, receiver_targets_all)
        receiver_top3 = metrics["receiver_top3"](receiver_logits_all, receiver_targets_all)
        
        # Shot metrics
        shot_probs = torch.sigmoid(shot_logits_all.squeeze(-1))
        shot_binary = (shot_probs > 0.5).long()
        shot_acc = (shot_binary == shot_targets_all).float().mean()
        shot_auc_roc, shot_auc_pr = metrics["shot_auc"](shot_logits_all, shot_targets_all, compute_auc_pr=True)
        shot_f1 = metrics["shot_f1"](shot_logits_all, shot_targets_all)
    
    epoch_metrics = {
        "total_loss": total_loss / num_samples,
        "receiver_loss": total_receiver_loss / num_samples,
        "shot_loss": total_shot_loss / num_samples,
        "receiver_top1": receiver_top1.item(),
        "receiver_top3": receiver_top3.item(),
        "shot_acc": shot_acc.item(),
        "shot_auc_roc": shot_auc_roc.item(),
        "shot_auc_pr": shot_auc_pr.item(),
        "shot_f1": shot_f1.item(),
    }
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_receiver: nn.Module,
    criterion_shot: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
    lambda_receiver: float = 1.0,
    lambda_shot: float = 1.0,
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_receiver_loss = 0.0
    total_shot_loss = 0.0
    total_loss = 0.0
    
    receiver_predictions = []
    receiver_targets = []
    shot_predictions = []
    shot_targets = []
    
    num_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for data, targets_dict in progress_bar:
            data = {k: v.to(device) for k, v in data.items()}
            receiver_target = targets_dict["receiver_target"].to(device)
            shot_target = targets_dict["shot_target"].to(device)
            
            edge_attr = data.get("edge_attr", None)
            mask = data.get("mask", None)
            team = data.get("team", None)
            ball = data.get("ball", None)
            
            outputs = model(
                x=data["x"],
                edge_index=data["edge_index"],
                edge_attr=edge_attr,
                batch=data["batch"],
                mask=mask,
                team=team,
                ball=ball,
            )
            
            receiver_logits = outputs["receiver_logits"]
            shot_logit = outputs["shot_logit"]
            
            receiver_loss = criterion_receiver(receiver_logits, receiver_target)
            shot_loss = criterion_shot(shot_logit, shot_target.unsqueeze(1) if shot_target.dim() == 1 else shot_target)
            total_batch_loss = lambda_receiver * receiver_loss + lambda_shot * shot_loss
            
            batch_size = shot_target.size(0)
            total_receiver_loss += receiver_loss.item() * batch_size
            total_shot_loss += shot_loss.item() * batch_size
            total_loss += total_batch_loss.item() * batch_size
            num_samples += batch_size
            
            receiver_predictions.append(receiver_logits)
            receiver_targets.append(receiver_target)
            shot_predictions.append(shot_logit)
            shot_targets.append(shot_target)
    
    # Compute metrics
    receiver_logits_all = torch.cat(receiver_predictions, dim=0)
    receiver_targets_all = torch.cat(receiver_targets, dim=0)
    shot_logits_all = torch.cat(shot_predictions, dim=0)
    shot_targets_all = torch.cat(shot_targets, dim=0)
    
    receiver_top1 = metrics["receiver_top1"](receiver_logits_all, receiver_targets_all)
    receiver_top3 = metrics["receiver_top3"](receiver_logits_all, receiver_targets_all)
    
    shot_probs = torch.sigmoid(shot_logits_all.squeeze(-1))
    shot_binary = (shot_probs > 0.5).long()
    shot_acc = (shot_binary == shot_targets_all).float().mean()
    shot_auc_roc, shot_auc_pr = metrics["shot_auc"](shot_logits_all, shot_targets_all, compute_auc_pr=True)
    shot_f1 = metrics["shot_f1"](shot_logits_all, shot_targets_all)
    
    epoch_metrics = {
        "total_loss": total_loss / num_samples,
        "receiver_loss": total_receiver_loss / num_samples,
        "shot_loss": total_shot_loss / num_samples,
        "receiver_top1": receiver_top1.item(),
        "receiver_top3": receiver_top3.item(),
        "shot_acc": shot_acc.item(),
        "shot_auc_roc": shot_auc_roc.item(),
        "shot_auc_pr": shot_auc_pr.item(),
        "shot_f1": shot_f1.item(),
    }
    
    return epoch_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train multi-task model (receiver + shot prediction)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.get("seed", 42))
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Setup logging (save to runs/receiver_shot/)
    log_dir = Path(config.get("log_dir", "runs")) / "receiver_shot"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"training_{timestamp}.log"
    logger = setup_logging(
        log_dir,
        config.get("log_level", "INFO"),
        log_file=log_filename
    )
    
    logger.info(f"Training multi-task model on {device}")
    logger.info(f"Configuration: {config}")
    
    # Check D2 configuration
    d2_config = config.get("d2", {})
    use_d2 = d2_config.get("enabled", False)
    
    # Create datasets
    train_dataset = MultiTaskDataset(
        receiver_data_path=config["data"]["receiver_train_path"],
        shot_data_path=config["data"]["shot_train_path"],
        receiver_file_format=config["data"].get("format", "pickle"),
        shot_file_format=config["data"].get("format", "pickle"),
        phase="train",
    )
    
    val_dataset = MultiTaskDataset(
        receiver_data_path=config["data"]["receiver_val_path"],
        shot_data_path=config["data"]["shot_val_path"],
        receiver_file_format=config["data"].get("format", "pickle"),
        shot_file_format=config["data"].get("format", "pickle"),
        phase="val",
    )
    
    test_dataset = MultiTaskDataset(
        receiver_data_path=config["data"]["receiver_test_path"],
        shot_data_path=config["data"]["shot_test_path"],
        receiver_file_format=config["data"].get("format", "pickle"),
        shot_file_format=config["data"].get("format", "pickle"),
        phase="test",
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
    )
    
    test_dataloader = create_dataloader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=False,
    )
    
    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create losses
    loss_config_receiver = config["loss"]["receiver"]
    criterion_receiver = CrossEntropyLoss(
        label_smoothing=loss_config_receiver.get("label_smoothing", 0.0),
        weight=loss_config_receiver.get("weight", 1.0),
    )
    
    loss_config_shot = config["loss"]["shot"]
    pos_weight = loss_config_shot.get("pos_weight", 1.0)
    if isinstance(pos_weight, (int, float)):
        pos_weight = torch.tensor([pos_weight], device=device)
    criterion_shot = BCELoss(
        label_smoothing=loss_config_shot.get("label_smoothing", 0.0),
        pos_weight=pos_weight,
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create metrics
    metrics = {
        "receiver_top1": TopKAccuracy(k=1),
        "receiver_top3": TopKAccuracy(k=3),
        "shot_auc": AUC(),
        "shot_f1": F1Score(),
    }
    
    # Task weights
    multitask_config = config.get("multitask", {})
    lambda_receiver = multitask_config.get("lambda_receiver", 1.0)
    lambda_shot = multitask_config.get("lambda_shot", 1.0)
    
    # Early stopping
    early_stopping_config = config.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get("patience", 20),
        min_delta=early_stopping_config.get("min_delta", 0.0),
        monitor=early_stopping_config.get("monitor", "val_receiver_top3"),
    )
    
    # Training history
    train_history = defaultdict(list)
    val_history = defaultdict(list)
    test_history = defaultdict(list)
    
    # Model save directory (checkpoints/receiver_shot/)
    model_save_dir = Path(config.get("model_save_dir", "checkpoints/receiver_shot"))
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint filename based on D2 setting
    checkpoint_filename = "best_d2.ckpt" if use_d2 else "best_no_d2.ckpt"
    checkpoint_path = model_save_dir / checkpoint_filename
    
    best_val_metric = float('-inf')
    
    # Training loop
    num_epochs = config["train"]["epochs"]
    use_amp = config["train"].get("amp", False)
    grad_clip_config = config["train"].get("grad_clip", {})
    grad_clip_enabled = grad_clip_config.get("enabled", False)
    grad_clip_max_norm = grad_clip_config.get("max_norm", 1.0)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_dataloader, optimizer,
            criterion_receiver, criterion_shot, device, metrics,
            lambda_receiver, lambda_shot,
            use_amp, grad_clip_enabled, grad_clip_max_norm,
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_dataloader,
            criterion_receiver, criterion_shot, device, metrics,
            lambda_receiver, lambda_shot,
        )
        
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingScheduler):
                current_lr = scheduler.step_epoch(epoch)
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.info(
            f"Train - Total: {train_metrics['total_loss']:.4f}, "
            f"Receiver: Loss={train_metrics['receiver_loss']:.4f}, Top1={train_metrics['receiver_top1']:.4f}, Top3={train_metrics['receiver_top3']:.4f} | "
            f"Shot: Loss={train_metrics['shot_loss']:.4f}, Acc={train_metrics['shot_acc']:.4f}, AUC-ROC={train_metrics['shot_auc_roc']:.4f}, F1={train_metrics['shot_f1']:.4f}"
        )
        logger.info(
            f"Val   - Total: {val_metrics['total_loss']:.4f}, "
            f"Receiver: Loss={val_metrics['receiver_loss']:.4f}, Top1={val_metrics['receiver_top1']:.4f}, Top3={val_metrics['receiver_top3']:.4f} | "
            f"Shot: Loss={val_metrics['shot_loss']:.4f}, Acc={val_metrics['shot_acc']:.4f}, AUC-ROC={val_metrics['shot_auc_roc']:.4f}, F1={val_metrics['shot_f1']:.4f}"
        )
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Update history
        for key in train_metrics:
            train_history[key].append(train_metrics[key])
            val_history[key].append(val_metrics[key])
        
        # Save CSV history
        csv_filename = f"training_history_{timestamp}.csv"
        csv_path = log_dir / csv_filename
        save_training_history_csv(
            train_history,
            val_history,
            test_history=None,
            filepath=csv_path
        )
        
        # Save best model
        monitor_metric = val_metrics[early_stopping.monitor]
        if monitor_metric > best_val_metric:
            best_val_metric = monitor_metric
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_path
            )
            logger.info(f"New best model saved with {early_stopping.monitor}: {best_val_metric:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch(
        model, test_dataloader,
        criterion_receiver, criterion_shot, device, metrics,
        lambda_receiver, lambda_shot,
    )
    
    logger.info(
        f"Test  - Total: {test_metrics['total_loss']:.4f}, "
        f"Receiver: Loss={test_metrics['receiver_loss']:.4f}, Top1={test_metrics['receiver_top1']:.4f}, Top3={test_metrics['receiver_top3']:.4f} | "
        f"Shot: Loss={test_metrics['shot_loss']:.4f}, Acc={test_metrics['shot_acc']:.4f}, AUC-ROC={test_metrics['shot_auc_roc']:.4f}, F1={test_metrics['shot_f1']:.4f}"
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

