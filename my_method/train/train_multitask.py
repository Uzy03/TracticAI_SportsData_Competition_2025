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

from my_method.models import MultiTaskModel
from my_method.dataio import MultiTaskDataset, create_dataloader, collate_fn_multitask
from my_method.modules import (
    CrossEntropyLoss, BCELoss, TopKAccuracy, Accuracy, F1Score, BinaryF1, AUC,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping,
)
from my_method.modules.utils import save_training_history_csv_multitask


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
    lr = float(opt_config["lr"])
    weight_decay = float(opt_config.get("weight_decay", 1e-4))
    
    if opt_config["type"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_config["type"] == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
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
            T_max=int(sched_config.get("T_max", config["train"]["epochs"])),
            eta_min=float(sched_config.get("eta_min", 0.0)),
            warmup_epochs=int(sched_config.get("warmup_epochs", 0)),
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
    lambda_consistency: float = 0.0,
    receiver_loss_weight: float = 1.0,
    shot_loss_weight: float = 1.0,
    use_amp: bool = False,
    grad_clip_enabled: bool = False,
    grad_clip_max_norm: float = 1.0,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    total_receiver_loss_sum = 0.0
    total_shot_loss_sum = 0.0
    total_consistency_loss_sum = 0.0
    total_loss_sum = 0.0

    receiver_graphs = 0
    receiver_top1_correct = 0
    receiver_top3_correct = 0

    shot_predictions = []
    shot_targets = []
    num_samples = 0  # number of graphs (shot labels)
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
        
        receiver_logits = outputs["receiver_logits"]  # [B, N] (per-node logits)
        shot_logit = outputs["shot_logit"]  # [B, 1]
        consistency_loss = outputs.get("consistency_loss", None)
        
        # Receiver loss: candidate-masked softmax classification per graph (same as train_receiver.py)
        cand_mask = data.get("cand_mask", None)
        if cand_mask is None:
            raise ValueError("Multi-task training requires 'cand_mask' in batched input_data.")

        B = int(data["batch"].max().item() + 1) if data.get("batch") is not None else int(shot_target.size(0))
        N_total = int(data["x"].size(0))
        if N_total % B != 0:
            raise ValueError(f"N_total ({N_total}) must be divisible by batch size B ({B})")
        N = N_total // B

        if receiver_logits.dim() == 1:
            receiver_logits = receiver_logits.view(B, N)
        elif receiver_logits.dim() == 2 and receiver_logits.size(0) == B:
            pass
        else:
            raise ValueError(f"Unexpected receiver_logits shape: {receiver_logits.shape} (expected [B, N])")

        cand_mask_b = cand_mask.view(B, N).bool()
        receiver_target_b = receiver_target.view(-1)

        receiver_loss_sum = 0.0
        graphs_in_batch = 0

        for b in range(B):
            cm = cand_mask_b[b]
            Ncand = int(cm.sum().item())
            if Ncand <= 0:
                continue

            logits_b = receiver_logits[b][cm].unsqueeze(0)  # [1, Ncand]
            cand_indices = torch.arange(N, device=logits_b.device)[cm]  # [Ncand]
            target_global = int(receiver_target_b[b].item())

            # If target not in candidates, skip (shouldn't happen if dataset is prepared correctly)
            if not (0 <= target_global < N) or not cm[target_global].item():
                continue

            cand_target_idx = int((cand_indices == target_global).nonzero(as_tuple=True)[0].item())
            target_t = torch.tensor([cand_target_idx], device=logits_b.device, dtype=torch.long)

            graph_loss = criterion_receiver(logits_b, target_t)
            receiver_loss_sum += graph_loss
            graphs_in_batch += 1

            # Receiver top-k metrics in candidate space
            top1 = int(torch.argmax(logits_b, dim=1).item())
            receiver_top1_correct += int(top1 == cand_target_idx)
            k3 = min(3, Ncand)
            top3 = torch.topk(logits_b, k=k3, dim=1).indices[0].tolist()
            receiver_top3_correct += int(cand_target_idx in top3)

        if graphs_in_batch == 0:
            # No valid receiver graphs in this batch; still train shot head
            receiver_loss = torch.tensor(0.0, device=device)
        else:
            receiver_loss = receiver_loss_sum / graphs_in_batch
            receiver_graphs += graphs_in_batch

        receiver_loss = receiver_loss * float(receiver_loss_weight)
        
        # Compute shot loss
        shot_loss = criterion_shot(shot_logit, shot_target.unsqueeze(1) if shot_target.dim() == 1 else shot_target)
        shot_loss = shot_loss * float(shot_loss_weight)

        # Consistency loss (optional)
        if consistency_loss is None or float(lambda_consistency) <= 0.0:
            consistency_loss_t = torch.tensor(0.0, device=device)
        else:
            consistency_loss_t = consistency_loss
        
        # Combined loss
        total_batch_loss = (
            lambda_receiver * receiver_loss
            + lambda_shot * shot_loss
            + float(lambda_consistency) * consistency_loss_t
        )
        
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
        batch_size = int(shot_target.size(0))
        total_receiver_loss_sum += float(receiver_loss.item()) * batch_size
        total_shot_loss_sum += float(shot_loss.item()) * batch_size
        total_consistency_loss_sum += float(consistency_loss_t.item()) * batch_size
        total_loss_sum += float(total_batch_loss.item()) * batch_size
        num_samples += batch_size
        
        # Collect predictions for metrics
        with torch.no_grad():
            shot_predictions.append(shot_logit.detach())
            shot_targets.append(shot_target.detach())
        
        progress_bar.set_postfix({
            "loss": f"{total_batch_loss.item():.4f}",
            "rec": f"{receiver_loss.item():.4f}",
            "shot": f"{shot_loss.item():.4f}",
            "cons": f"{consistency_loss_t.item():.4f}",
        })
    
    # Compute shot metrics
    with torch.no_grad():
        shot_logits_all = torch.cat(shot_predictions, dim=0)
        shot_targets_all = torch.cat(shot_targets, dim=0)

        # Normalize shapes to avoid broadcasting bugs:
        # - logits: [N]
        # - targets: [N]
        if shot_logits_all.dim() == 2 and shot_logits_all.size(1) == 1:
            shot_logits_1d = shot_logits_all.squeeze(-1)
        else:
            shot_logits_1d = shot_logits_all.view(-1)

        if shot_targets_all.dim() == 2 and shot_targets_all.size(1) == 1:
            shot_targets_1d = shot_targets_all.squeeze(-1)
        else:
            shot_targets_1d = shot_targets_all.view(-1)

        shot_probs = torch.sigmoid(shot_logits_1d)
        shot_binary = (shot_probs > 0.5).long()
        shot_acc = (shot_binary == shot_targets_1d.long()).float().mean()
        shot_auc_roc, shot_auc_pr = metrics["shot_auc"](shot_logits_1d, shot_targets_1d, compute_auc_pr=True)
        shot_f1 = metrics["shot_f1"](shot_logits_1d, shot_targets_1d)
    
    epoch_metrics = {
        "total_loss": total_loss_sum / max(1, num_samples),
        "receiver_loss": total_receiver_loss_sum / max(1, num_samples),
        "shot_loss": total_shot_loss_sum / max(1, num_samples),
        "consistency_loss": total_consistency_loss_sum / max(1, num_samples),
        "receiver_top1": (receiver_top1_correct / max(1, receiver_graphs)),
        "receiver_top3": (receiver_top3_correct / max(1, receiver_graphs)),
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
    lambda_consistency: float = 0.0,
    receiver_loss_weight: float = 1.0,
    shot_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """Validate model for one epoch."""
    model.eval()
    
    total_receiver_loss_sum = 0.0
    total_shot_loss_sum = 0.0
    total_consistency_loss_sum = 0.0
    total_loss_sum = 0.0

    receiver_graphs = 0
    receiver_top1_correct = 0
    receiver_top3_correct = 0

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
            consistency_loss = outputs.get("consistency_loss", None)
            
            # Receiver loss (candidate-masked per graph)
            cand_mask = data.get("cand_mask", None)
            if cand_mask is None:
                raise ValueError("Multi-task validation requires 'cand_mask' in batched input_data.")

            B = int(data["batch"].max().item() + 1) if data.get("batch") is not None else int(shot_target.size(0))
            N_total = int(data["x"].size(0))
            if N_total % B != 0:
                raise ValueError(f"N_total ({N_total}) must be divisible by batch size B ({B})")
            N = N_total // B

            if receiver_logits.dim() == 1:
                receiver_logits = receiver_logits.view(B, N)
            elif receiver_logits.dim() == 2 and receiver_logits.size(0) == B:
                pass
            else:
                raise ValueError(f"Unexpected receiver_logits shape: {receiver_logits.shape} (expected [B, N])")

            cand_mask_b = cand_mask.view(B, N).bool()
            receiver_target_b = receiver_target.view(-1)

            receiver_loss_sum = 0.0
            graphs_in_batch = 0
            for b in range(B):
                cm = cand_mask_b[b]
                Ncand = int(cm.sum().item())
                if Ncand <= 0:
                    continue
                logits_b = receiver_logits[b][cm].unsqueeze(0)  # [1, Ncand]
                cand_indices = torch.arange(N, device=logits_b.device)[cm]
                target_global = int(receiver_target_b[b].item())
                if not (0 <= target_global < N) or not cm[target_global].item():
                    continue
                cand_target_idx = int((cand_indices == target_global).nonzero(as_tuple=True)[0].item())
                target_t = torch.tensor([cand_target_idx], device=logits_b.device, dtype=torch.long)
                graph_loss = criterion_receiver(logits_b, target_t)
                receiver_loss_sum += graph_loss
                graphs_in_batch += 1

                top1 = int(torch.argmax(logits_b, dim=1).item())
                receiver_top1_correct += int(top1 == cand_target_idx)
                k3 = min(3, Ncand)
                top3 = torch.topk(logits_b, k=k3, dim=1).indices[0].tolist()
                receiver_top3_correct += int(cand_target_idx in top3)

            if graphs_in_batch == 0:
                receiver_loss = torch.tensor(0.0, device=device)
            else:
                receiver_loss = receiver_loss_sum / graphs_in_batch
                receiver_graphs += graphs_in_batch
            receiver_loss = receiver_loss * float(receiver_loss_weight)
            shot_loss = criterion_shot(shot_logit, shot_target.unsqueeze(1) if shot_target.dim() == 1 else shot_target)
            shot_loss = shot_loss * float(shot_loss_weight)

            if consistency_loss is None or float(lambda_consistency) <= 0.0:
                consistency_loss_t = torch.tensor(0.0, device=device)
            else:
                consistency_loss_t = consistency_loss

            total_batch_loss = (
                lambda_receiver * receiver_loss
                + lambda_shot * shot_loss
                + float(lambda_consistency) * consistency_loss_t
            )
            
            batch_size = shot_target.size(0)
            total_receiver_loss_sum += float(receiver_loss.item()) * batch_size
            total_shot_loss_sum += float(shot_loss.item()) * batch_size
            total_consistency_loss_sum += float(consistency_loss_t.item()) * batch_size
            total_loss_sum += float(total_batch_loss.item()) * batch_size
            num_samples += batch_size
            
            shot_predictions.append(shot_logit)
            shot_targets.append(shot_target)
    
    # Compute metrics
    shot_logits_all = torch.cat(shot_predictions, dim=0)
    shot_targets_all = torch.cat(shot_targets, dim=0)

    # Normalize shapes to avoid broadcasting bugs
    if shot_logits_all.dim() == 2 and shot_logits_all.size(1) == 1:
        shot_logits_1d = shot_logits_all.squeeze(-1)
    else:
        shot_logits_1d = shot_logits_all.view(-1)

    if shot_targets_all.dim() == 2 and shot_targets_all.size(1) == 1:
        shot_targets_1d = shot_targets_all.squeeze(-1)
    else:
        shot_targets_1d = shot_targets_all.view(-1)

    shot_probs = torch.sigmoid(shot_logits_1d)
    shot_binary = (shot_probs > 0.5).long()
    shot_acc = (shot_binary == shot_targets_1d.long()).float().mean()
    shot_auc_roc, shot_auc_pr = metrics["shot_auc"](shot_logits_1d, shot_targets_1d, compute_auc_pr=True)
    shot_f1 = metrics["shot_f1"](shot_logits_1d, shot_targets_1d)
    
    epoch_metrics = {
        "total_loss": total_loss_sum / max(1, num_samples),
        "receiver_loss": total_receiver_loss_sum / max(1, num_samples),
        "shot_loss": total_shot_loss_sum / max(1, num_samples),
        "consistency_loss": total_consistency_loss_sum / max(1, num_samples),
        "receiver_top1": (receiver_top1_correct / max(1, receiver_graphs)),
        "receiver_top3": (receiver_top3_correct / max(1, receiver_graphs)),
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

    # Performance knobs (safe defaults for NVIDIA Ampere+; no effect on CPU)
    train_cfg = config.get("train", {})
    if device.type == "cuda":
        try:
            if bool(train_cfg.get("tf32", True)):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # PyTorch 2.x matmul precision hint
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
        except Exception:
            pass
    
    # Setup logging
    # If config includes run_name, separate outputs like:
    #   runs/my_method/receiver_shot/<run_name>/
    run_name = str(config.get("run_name", "")).strip()
    log_dir = Path(config.get("log_dir", "runs")) / "receiver_shot"
    if run_name:
        log_dir = log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"training_{timestamp}.log"
    logger = setup_logging(
        log_dir,
        config.get("log_level", "INFO"),
        log_file=log_filename
    )

    # Best-effort: log unexpected termination signals (SIGKILL cannot be caught).
    # This helps diagnose "stops with no traceback" cases (e.g., SIGTERM from scheduler/cluster).
    import atexit
    import signal
    import sys
    import faulthandler

    def _flush_logger() -> None:
        for h in getattr(logger, "handlers", []):
            try:
                h.flush()
            except Exception:
                pass

    def _handle_signal(signum, frame):  # type: ignore[no-untyped-def]
        try:
            logger.error(f"Received signal {signum}. Terminating...")
            _flush_logger()
        finally:
            raise SystemExit(128 + int(signum))

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(_sig, _handle_signal)
        except Exception:
            pass

    try:
        # Dump Python traceback to the same log file on fatal errors / signals where possible.
        with open(log_dir / log_filename, "a", encoding="utf-8") as _fh:
            faulthandler.enable(file=_fh, all_threads=True)
    except Exception:
        # Fallback: stderr
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass

    atexit.register(lambda: (logger.info("Process exiting."), _flush_logger()))
    
    logger.info(f"Training multi-task model on {device}")
    logger.info(f"Configuration: {config}")
    
    # Check D2 configuration
    d2_config = config.get("d2", {})
    use_d2 = d2_config.get("enabled", False)
    
    # Create datasets
    full_train_dataset = MultiTaskDataset(
        receiver_data_path=config["data"]["receiver_train_path"],
        shot_data_path=config["data"]["shot_train_path"],
        receiver_file_format=config["data"].get("format", "pickle"),
        shot_file_format=config["data"].get("format", "pickle"),
        phase="train",
    )

    # Debug overfit mode (config-driven) - make a small subset and use train=val=test same samples
    debug_overfit_cfg = config.get("debug_overfit", {})
    use_debug_overfit = bool(debug_overfit_cfg.get("enabled", False))
    if use_debug_overfit:
        num_samples = int(debug_overfit_cfg.get("num_samples", 8))
        subset_seed = int(debug_overfit_cfg.get("seed", 42))
        total = len(full_train_dataset)
        if total <= 0:
            raise ValueError("[DEBUG-OVERFIT] Training dataset is empty.")
        if num_samples > total:
            logger.warning(f"[DEBUG-OVERFIT] num_samples ({num_samples}) > total ({total}). Using all samples.")
            num_samples = total

        rng = np.random.RandomState(subset_seed)
        indices = rng.permutation(total)[:num_samples].tolist()
        indices = sorted(indices)

        logger.info(
            f"[DEBUG-OVERFIT] Using subset of {len(indices)}/{total} samples "
            f"(seed={subset_seed}, indices={indices[:5]}{'...' if len(indices) > 5 else ''})"
        )

        train_dataset = Subset(full_train_dataset, indices)
        val_dataset = Subset(full_train_dataset, indices)
        test_dataset = Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset
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
    
    # Create dataloaders (use multitask collate_fn)
    train_shuffle = False if use_debug_overfit else True
    num_workers = int(config.get("num_workers", 0))
    # pin_memory tends to help host->GPU transfer; persistent_workers avoids worker respawn each epoch
    pin_memory = bool(config.get("pin_memory", True)) and (device.type == "cuda")
    persistent_workers = bool(config.get("persistent_workers", True)) and (num_workers > 0)
    prefetch_factor = int(config.get("prefetch_factor", 2))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn_multitask,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn_multitask,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_fn_multitask,
    )
    
    # Create model
    model = create_model(config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create losses
    loss_config_receiver = config["loss"]["receiver"]
    # NOTE: CrossEntropyLoss.weight expects class-weight tensor (or None).
    # The YAML's loss.receiver.weight is treated as a scalar multiplier, not class weights.
    class_weights = loss_config_receiver.get("class_weights", None)
    class_weights_t = None
    if class_weights is not None:
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion_receiver = CrossEntropyLoss(
        label_smoothing=loss_config_receiver.get("label_smoothing", 0.0),
        weight=class_weights_t,
    )
    receiver_loss_weight = float(loss_config_receiver.get("weight", 1.0))
    
    loss_config_shot = config["loss"]["shot"]
    pos_weight = loss_config_shot.get("pos_weight", 1.0)
    if isinstance(pos_weight, (int, float)):
        pos_weight = torch.tensor([pos_weight], device=device)
    criterion_shot = BCELoss(
        label_smoothing=loss_config_shot.get("label_smoothing", 0.0),
        pos_weight=pos_weight,
    )
    shot_loss_weight = float(loss_config_shot.get("weight", 1.0))
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create metrics
    metrics = {
        "receiver_top1": TopKAccuracy(k=1),
        "receiver_top3": TopKAccuracy(k=3),
        "shot_auc": AUC(),
        "shot_f1": BinaryF1(threshold=0.5),
    }
    
    # Task weights
    multitask_config = config.get("multitask", {})
    lambda_receiver = multitask_config.get("lambda_receiver", 1.0)
    lambda_shot = multitask_config.get("lambda_shot", 1.0)
    lambda_consistency = multitask_config.get("lambda_consistency", 0.0)
    
    # Early stopping
    early_stopping_config = config.get("early_stopping", {})
    # Determine mode based on monitor metric (accuracy metrics should use 'max', loss should use 'min')
    monitor = early_stopping_config.get("monitor", "val_receiver_top3")
    # Remove "val_" prefix if present for accessing metrics dict
    monitor_key = monitor.replace("val_", "")
    mode = "max" if "accuracy" in monitor or "top" in monitor or "auc" in monitor or "f1" in monitor else "min"
    early_stopping = EarlyStopping(
        patience=early_stopping_config.get("patience", 20),
        min_delta=early_stopping_config.get("min_delta", 0.0),
        mode=mode,
    )
    
    # Training history
    train_history = defaultdict(list)
    val_history = defaultdict(list)
    test_history = {}  # Test metrics are saved only once at the end
    
    # Model save directory
    # If config includes run_name, separate outputs like:
    #   checkpoints/my_method/receiver_shot/<run_name>/
    model_save_dir = Path(config.get("model_save_dir", "checkpoints/receiver_shot"))
    if run_name:
        model_save_dir = model_save_dir / run_name
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint filename based on D2 setting
    checkpoint_filename = "best_d2.ckpt" if use_d2 else "best_no_d2.ckpt"
    checkpoint_path = model_save_dir / checkpoint_filename
    
    # Track best metric for checkpointing (direction depends on early_stopping mode)
    best_val_metric = float('inf') if mode == "min" else float('-inf')
    
    # Training loop
    num_epochs = config["train"]["epochs"]
    use_amp = config["train"].get("amp", False)
    grad_clip_config = config["train"].get("grad_clip", {})
    grad_clip_enabled = grad_clip_config.get("enabled", False)
    grad_clip_max_norm = grad_clip_config.get("max_norm", 1.0)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        try:
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Train
            logger.info(f"Starting training epoch {epoch+1}...")
            train_metrics = train_epoch(
                model, train_dataloader, optimizer,
                criterion_receiver, criterion_shot, device, metrics,
                lambda_receiver, lambda_shot, lambda_consistency,
                receiver_loss_weight, shot_loss_weight,
                use_amp, grad_clip_enabled, grad_clip_max_norm,
            )
            logger.info(f"Training epoch {epoch+1} completed successfully")

            # Validate
            logger.info(f"Starting validation epoch {epoch+1}...")
            val_metrics = validate_epoch(
                model, val_dataloader,
                criterion_receiver, criterion_shot, device, metrics,
                lambda_receiver, lambda_shot, lambda_consistency,
                receiver_loss_weight, shot_loss_weight,
            )
            logger.info(f"Validation epoch {epoch+1} completed successfully")

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, CosineAnnealingScheduler):
                    current_lr = scheduler.step_epoch(epoch)
                else:
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]["lr"]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # Log metrics
            logger.info(
                f"Train - Total: {train_metrics['total_loss']:.4f}, "
                f"Receiver: Loss={train_metrics['receiver_loss']:.4f}, Top1={train_metrics['receiver_top1']:.4f}, Top3={train_metrics['receiver_top3']:.4f} | "
                f"Shot: Loss={train_metrics['shot_loss']:.4f}, Acc={train_metrics['shot_acc']:.4f}, AUC-ROC={train_metrics['shot_auc_roc']:.4f}, F1={train_metrics['shot_f1']:.4f} | "
                f"Cons: {train_metrics.get('consistency_loss', 0.0):.4f}"
            )
            logger.info(
                f"Val   - Total: {val_metrics['total_loss']:.4f}, "
                f"Receiver: Loss={val_metrics['receiver_loss']:.4f}, Top1={val_metrics['receiver_top1']:.4f}, Top3={val_metrics['receiver_top3']:.4f} | "
                f"Shot: Loss={val_metrics['shot_loss']:.4f}, Acc={val_metrics['shot_acc']:.4f}, AUC-ROC={val_metrics['shot_auc_roc']:.4f}, F1={val_metrics['shot_f1']:.4f} | "
                f"Cons: {val_metrics.get('consistency_loss', 0.0):.4f}"
            )
            logger.info(f"Learning rate: {current_lr:.6f}")

            # Update history
            logger.info(f"Updating history for epoch {epoch+1}...")
            for key in train_metrics:
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])
            logger.info(f"History updated successfully for epoch {epoch+1}")

            # Save CSV history
            logger.info(f"Saving CSV history for epoch {epoch+1}...")
            csv_filename = f"training_history_{timestamp}.csv"
            csv_path = log_dir / csv_filename
            save_training_history_csv_multitask(
                train_history,
                val_history,
                test_history=None,
                filepath=csv_path,
            )
            logger.info(f"CSV history saved successfully for epoch {epoch+1}")

            # Save best model and check early stopping
            logger.info(f"Checking best model and early stopping for epoch {epoch+1}...")
            if monitor_key not in val_metrics:
                logger.error(
                    f"Monitor key '{monitor_key}' not found in val_metrics. Available keys: {list(val_metrics.keys())}"
                )
                raise KeyError(
                    f"Monitor key '{monitor_key}' not found in val_metrics. Available keys: {list(val_metrics.keys())}"
                )

            monitor_metric = val_metrics[monitor_key]

            # Check for NaN/Inf values
            if torch.isnan(torch.tensor(monitor_metric)) or torch.isinf(torch.tensor(monitor_metric)):
                logger.error(f"Monitor metric '{monitor_key}' is NaN or Inf: {monitor_metric}")
                logger.error(f"All val_metrics: {val_metrics}")
                raise ValueError(f"Monitor metric '{monitor_key}' is NaN or Inf: {monitor_metric}")

            is_better = (monitor_metric < best_val_metric) if mode == "min" else (monitor_metric > best_val_metric)
            if is_better:
                best_val_metric = monitor_metric
                logger.info(f"Saving checkpoint for epoch {epoch+1}...")
                save_checkpoint(
                    model, optimizer, epoch, val_metrics["total_loss"], val_metrics,
                    checkpoint_path, scheduler,
                )
                logger.info(f"New best model saved with {monitor}: {best_val_metric:.4f}")

            # Early stopping (pass score value and model)
            logger.info(f"Checking early stopping for epoch {epoch+1}...")
            if early_stopping(monitor_metric, model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            logger.info(f"Epoch {epoch+1} completed successfully")
            _flush_logger()
        except Exception as e:
            logger.error(f"Error occurred at epoch {epoch+1}: {type(e).__name__}: {str(e)}")
            logger.error("Traceback:", exc_info=True)
            logger.error("Saving current history before exit...")
            try:
                save_training_history_csv_multitask(
                    train_history,
                    val_history,
                    test_history=None,
                    filepath=log_dir / f"training_history_{timestamp}_error_epoch_{epoch+1}.csv",
                )
            except Exception:
                logger.error("Failed to save error history CSV.", exc_info=True)
            _flush_logger()
            raise  # keep failing fast so tmux/container shows exit status
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = validate_epoch(
        model, test_dataloader,
        criterion_receiver, criterion_shot, device, metrics,
        lambda_receiver, lambda_shot, lambda_consistency,
        receiver_loss_weight, shot_loss_weight,
    )
    
    logger.info(
        f"Test  - Total: {test_metrics['total_loss']:.4f}, "
        f"Receiver: Loss={test_metrics['receiver_loss']:.4f}, Top1={test_metrics['receiver_top1']:.4f}, Top3={test_metrics['receiver_top3']:.4f} | "
        f"Shot: Loss={test_metrics['shot_loss']:.4f}, Acc={test_metrics['shot_acc']:.4f}, AUC-ROC={test_metrics['shot_auc_roc']:.4f}, F1={test_metrics['shot_f1']:.4f}"
    )
    
    # Update test history (single values, not lists) and save final CSV
    test_history = test_metrics.copy()
    
    save_training_history_csv_multitask(
        train_history,
        val_history,
        test_history=test_history,
        filepath=log_dir / f"final_training_history_{timestamp}.csv"
    )
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

