"""Utility functions for TacticAI.

This module provides utility functions for training, evaluation, and general
purposes including seeding, device management, checkpointing, and logging.
"""

import os
import random
import logging
import json
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device.
    
    Args:
        device: Preferred device ('cpu', 'cuda', 'auto')
        
    Returns:
        PyTorch device
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        **kwargs: Additional data to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state dict into (optional)
        optimizer: Optimizer to load state dict into (optional)
        scheduler: Scheduler to load state dict into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


def setup_logging(
    log_dir: Union[str, Path],
    log_level: str = "INFO",
    log_file: str = "training.log"
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_file: Log file name
        
    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("tacticai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter with PID
    formatter = logging.Formatter(
        "%(asctime)s - PID:%(process)d - %(name)s - %(levelname)s - %(message)s"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def save_training_history(
    history: Dict[str, list],
    filepath: Union[str, Path]
) -> None:
    """Save training history to JSON file.
    
    Args:
        history: Training history dictionary
        filepath: Path to save history
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_history = {}
    for key, values in history.items():
        if isinstance(values, list):
            serializable_history[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]
        else:
            serializable_history[key] = values
    
    with open(filepath, 'w') as f:
        json.dump(serializable_history, f, indent=2)


def load_training_history(filepath: Union[str, Path]) -> Dict[str, list]:
    """Load training history from JSON file.
    
    Args:
        filepath: Path to history file
        
    Returns:
        Training history dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_training_history_csv(
    train_history: Dict[str, list],
    val_history: Dict[str, list],
    test_history: Optional[Dict[str, float]] = None,
    filepath: Union[str, Path] = "runs/training_history.csv"
) -> None:
    """Save training history to CSV file in a readable format.
    
    Args:
        train_history: Training history dictionary (e.g., {"loss": [...], "accuracy": [...], ...})
        val_history: Validation history dictionary
        test_history: Test metrics dictionary (optional, single values per metric)
        filepath: Path to save CSV file
    """
    import csv
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all epochs
    num_epochs = len(train_history.get("loss", []))
    
    # Prepare CSV rows
    rows = []
    for epoch in range(num_epochs):
        row = {
            "epoch": epoch + 1,
            "train_loss": train_history.get("loss", [0.0])[epoch] if epoch < len(train_history.get("loss", [])) else 0.0,
            "train_acc": train_history.get("accuracy", [0.0])[epoch] if epoch < len(train_history.get("accuracy", [])) else 0.0,
            "train_top1": train_history.get("top1", [0.0])[epoch] if epoch < len(train_history.get("top1", [])) else 0.0,
            "train_top3": train_history.get("top3", [0.0])[epoch] if epoch < len(train_history.get("top3", [])) else 0.0,
            "train_top5": train_history.get("top5", [0.0])[epoch] if epoch < len(train_history.get("top5", [])) else 0.0,
            "val_loss": val_history.get("loss", [0.0])[epoch] if epoch < len(val_history.get("loss", [])) else 0.0,
            "val_acc": val_history.get("accuracy", [0.0])[epoch] if epoch < len(val_history.get("accuracy", [])) else 0.0,
            "val_top1": val_history.get("top1", [0.0])[epoch] if epoch < len(val_history.get("top1", [])) else 0.0,
            "val_top3": val_history.get("top3", [0.0])[epoch] if epoch < len(val_history.get("top3", [])) else 0.0,
            "val_top5": val_history.get("top5", [0.0])[epoch] if epoch < len(val_history.get("top5", [])) else 0.0,
        }
        
        # Add test metrics if available (only for the last epoch)
        if test_history is not None and epoch == num_epochs - 1:
            row["test_loss"] = test_history.get("loss", 0.0)
            row["test_acc"] = test_history.get("accuracy", 0.0)
            row["test_top1"] = test_history.get("top1", 0.0)
            row["test_top3"] = test_history.get("top3", 0.0)
            row["test_top5"] = test_history.get("top5", 0.0)
        else:
            row["test_loss"] = ""
            row["test_acc"] = ""
            row["test_top1"] = ""
            row["test_top3"] = ""
            row["test_top5"] = ""
        
        rows.append(row)
    
    # Write CSV file (overwrite mode)
    fieldnames = [
        "epoch", "train_loss", "train_acc", "train_top1", "train_top3", "train_top5",
        "val_loss", "val_acc", "val_top1", "val_top3", "val_top5",
        "test_loss", "test_acc", "test_top1", "test_top3", "test_top5"
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class CosineAnnealingScheduler:
    """Cosine annealing learning rate scheduler.
    
    Implements cosine annealing with optional warm restarts.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-6,
    ):
        """Initialize cosine annealing scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            T_max: Maximum number of epochs
            eta_min: Minimum learning rate
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
        self.last_lr = self.base_lr

    def _apply_lr(self, lr: float) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_lr = lr

    def _compute_lr(self, epoch_idx: int) -> float:
        if epoch_idx < self.warmup_epochs:
            progress = (epoch_idx + 1) / max(1, self.warmup_epochs)
            return self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * progress

        epoch = epoch_idx - self.warmup_epochs
        T = self.T_max - self.warmup_epochs
        if T <= 0:
            T = 1
        return self.eta_min + (self.base_lr - self.eta_min) * (
            1 + np.cos(np.pi * epoch / T)
        ) / 2

    def step_epoch(self, epoch_idx: int) -> float:
        """Explicitly set learning rate for the given epoch index."""
        lr = self._compute_lr(epoch_idx)
        self._apply_lr(lr)
        self.current_epoch = epoch_idx + 1
        return lr
    
    def step(self) -> float:
        """Update learning rate.
        
        Returns:
            Current learning rate
        """
        lr = self._compute_lr(self.current_epoch)
        self._apply_lr(lr)
        self.current_epoch += 1
        return lr
    
    def get_last_lr(self) -> list[float]:
        """Get the last learning rate.
        
        Returns:
            List of learning rates for each parameter group
        """
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dict for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'warmup_epochs': self.warmup_epochs,
            'warmup_start_lr': self.warmup_start_lr,
            'last_lr': self.last_lr,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.T_max = state_dict['T_max']
        self.eta_min = state_dict['eta_min']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_start_lr = state_dict['warmup_start_lr']
        self.last_lr = state_dict.get('last_lr', self.base_lr)


class EarlyStopping:
    """Early stopping utility.
    
    Stops training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially restore weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False


def create_directory_structure(base_dir: Union[str, Path]) -> None:
    """Create standard directory structure for TacticAI project.
    
    Args:
        base_dir: Base directory for the project
    """
    base_dir = Path(base_dir)
    
    directories = [
        "data/raw",
        "data/processed", 
        "checkpoints",
        "runs",
        "logs",
        "results",
        "configs",
    ]
    
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files
    for directory in ["data/raw", "data/processed"]:
        (base_dir / directory / ".gitkeep").touch()


def get_model_size(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024),
        "total_size_mb": (param_size + buffer_size) / (1024 * 1024),
    }
