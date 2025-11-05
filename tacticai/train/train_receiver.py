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
        edge_attr: Optional[torch.Tensor] = None,  # TacticAI spec: same_team feature
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
        # Process entire batch at once for maximum GPU utilization
        B = batch.max().item() + 1 if batch is not None else 1
        
        # Create 4 views for entire batch at once
        # x coordinates are at indices 0, 1; velocity x, y at indices 2, 3
        views_list = []
        for view_idx in range(len(D2_VIEWS)):
            x_view = apply_view_transform(x, view_idx, xy_indices=(0, 1))  # Only flip x, y positions
            views_list.append(x_view)
        
        # Stack views: [4, N_total, D] -> [B, 4, N_total, D]
        # Note: N_total is the total number of nodes across all graphs in the batch
        x_views = torch.stack(views_list, dim=0)  # [4, N_total, D]
        N_total = x.size(0)
        x_4view = x_views.view(4, N_total, -1).permute(1, 0, 2).unsqueeze(0)  # [1, 4, N_total, D]
        # Expand to [B, 4, N_total, D] - each graph in batch uses same views
        # Actually, we need to reshape properly: each graph should have its own views
        # Reshape: [4, N_total, D] -> [4, B, N_per_graph, D] -> [B, 4, N_per_graph, D]
        num_nodes_per_graph = N_total // B if B > 1 else N_total
        x_4view = x_views.view(4, B, num_nodes_per_graph, -1).permute(1, 0, 2, 3)  # [B, 4, N_per_graph, D]
        
        # Use edge_index and edge_attr (already batched correctly by collate_fn)
        # edge_index contains edges for all graphs with proper offsets
        # edge_attr contains same_team features [E, 1] (TacticAI spec)
        # Get node embeddings from backbone: [B, 4, N_per_graph, output_dim]
        node_emb_4view = self.backbone(x_4view, edge_index, edge_attr)  # [B, 4, N_per_graph, output_dim]
        
        # Average over 4 views: [B, N_per_graph, output_dim]
        node_emb_batched = node_emb_4view.mean(dim=1)  # [B, N_per_graph, output_dim]
        
        # Reshape back to [N_total, output_dim]
        node_embeddings = node_emb_batched.view(-1, node_emb_batched.size(-1))  # [N_total, output_dim]
        
        # Apply mask if provided (element-wise multiplication, not pooling)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(node_embeddings)  # [N, D]
            node_embeddings = node_embeddings * mask_expanded
        
        # Get logits for all nodes (TacticAI spec: [B, N] format)
        all_logits = self.head(node_embeddings).squeeze(-1)  # [N_total] - per-node scalar logits
        
        # TacticAI spec: cand_mask = (team_flag==ATTACK) & (is_kicker==0) & (valid_mask==1)
        # Apply cand_mask before softmax (not filtering, but masking with -1e9)
        if team is not None and ball is not None and mask is not None:
            # cand_mask: attacking team (team=0) and not ball owner (ball=0) and valid (mask=1)
            cand_mask = (team == 0) & (ball == 0) & (mask == 1)
            # Apply mask: set non-candidate logits to -1e9 (will be ~0 after softmax)
            all_logits = all_logits + (~cand_mask).float() * (-1e9)
            # Return all logits with mask applied (for softmax over candidate set)
            return all_logits
        elif team is not None and ball is not None:
            # Without mask: cand_mask = (team==0) & (ball==0)
            cand_mask = (team == 0) & (ball == 0)
            all_logits = all_logits + (~cand_mask).float() * (-1e9)
            return all_logits
        else:
            # No filtering: return all nodes (should not happen in practice)
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
    num_graphs_total = 0
    # metrics accumulators
    acc_correct = 0
    # diagnostics
    excluded_not_attacking = 0
    excluded_ball_owner = 0
    excluded_invalid = 0
    cand_counts = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        # Move data to device
        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                # Pass edge_attr, mask, team, ball if available
                outputs = model(
                    data["x"], 
                    data["edge_index"], 
                    data["batch"],
                    edge_attr=data.get("edge_attr"),
                    mask=data.get("mask"),
                    team=data.get("team"),
                    ball=data.get("ball"),
                )
                
                # TacticAI spec: outputs is [N_total] with cand_mask applied (-1e9 for non-candidates)
                # Process per graph: extract local logits and apply softmax
                batch_size = data["batch"].max().item() + 1
                graph_outputs = []
                graph_targets = []
                
                for i in range(batch_size):
                    node_mask = data["batch"] == i
                    if node_mask.any():
                        # Get local logits for this graph
                        local_logits = outputs[node_mask]  # [N_per_graph]
                        
                        # Get cand_mask for this graph (TacticAI spec)
                        if "team" in data and "ball" in data and "mask" in data:
                            team_i = data["team"][node_mask]
                            ball_i = data["ball"][node_mask]
                            mask_i = data["mask"][node_mask]
                            cand_mask = (team_i == 0) & (ball_i == 0) & (mask_i == 1)
                        elif "team" in data and "ball" in data:
                            team_i = data["team"][node_mask]
                            ball_i = data["ball"][node_mask]
                            cand_mask = (team_i == 0) & (ball_i == 0)
                        else:
                            cand_mask = torch.ones(node_mask.sum(), dtype=torch.bool, device=outputs.device)
                        
                        # Apply cand_mask: non-candidates already have -1e9 from forward
                        # Get candidate logits (for loss/metrics)
                        cand_logits = local_logits[cand_mask]  # [num_candidates]
                        
                        if cand_logits.numel() > 0:
                            receiver_node_idx = targets[i].item()  # Graph-local index (0-21)
                            # receiver_node_idx is already the local index within the graph
                            # Check if receiver is within the graph bounds
                            num_nodes_in_graph = node_mask.sum().item()
                            if receiver_node_idx < num_nodes_in_graph:
                                # receiver_node_idx is the local index within this graph
                                if cand_mask[receiver_node_idx]:
                                    # Receiver is in candidates: map to candidate index
                                    cand_indices = torch.where(cand_mask)[0]  # Local indices that are candidates
                                    receiver_cand_idx = (cand_indices == receiver_node_idx).nonzero(as_tuple=True)[0]
                                    if receiver_cand_idx.numel() > 0:
                                        receiver_cand_idx = receiver_cand_idx.item()
                                        graph_outputs.append(cand_logits)
                                        graph_targets.append(receiver_cand_idx)
                                        cand_counts.append(int(cand_mask.sum().item()))
                                    else:
                                        excluded_invalid += 1
                                else:
                                    excluded_ball_owner += 1
                            else:
                                excluded_invalid += 1
                
                # Compute loss per graph (TacticAI spec: softmax over candidates)
                batch_loss_sum = 0.0
                graphs_in_batch = 0
                for logits_b, target_b in zip(graph_outputs, graph_targets):
                    # logits_b: candidate logits [num_candidates] (already masked)
                    # Apply softmax and compute CrossEntropyLoss
                    lb = logits_b.unsqueeze(0)  # [1, num_candidates]
                    target_t = torch.tensor([target_b], dtype=torch.long, device=lb.device)
                    graph_loss = criterion(lb, target_t)
                    batch_loss_sum += graph_loss
                    # metrics
                    pred_top1 = torch.argmax(lb, dim=1)
                    acc_correct += int(pred_top1.item() == target_b)
                    top1_correct += int(pred_top1.item() == target_b)
                    k3 = min(3, lb.size(1))
                    k5 = min(5, lb.size(1))
                    top3_correct += int(target_b in torch.topk(lb, k=k3, dim=1).indices[0].tolist())
                    top5_correct += int(target_b in torch.topk(lb, k=k5, dim=1).indices[0].tolist())
                    graphs_in_batch += 1
                if graphs_in_batch == 0:
                    continue
                loss = batch_loss_sum / graphs_in_batch  # Average loss for this batch
                num_graphs_total += graphs_in_batch
                total_loss += batch_loss_sum.item()  # Accumulate total loss (not average)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Pass edge_attr, mask, team, ball if available
            outputs = model(
                data["x"], 
                data["edge_index"], 
                data["batch"],
                edge_attr=data.get("edge_attr"),
                mask=data.get("mask"),
                team=data.get("team"),
                ball=data.get("ball"),
            )
            
            # TacticAI spec: outputs is [N_total] with cand_mask applied (-1e9 for non-candidates)
            # Process per graph: extract local logits and apply softmax
            batch_size = data["batch"].max().item() + 1
            graph_outputs = []
            graph_targets = []
            
            for i in range(batch_size):
                node_mask = data["batch"] == i
                if node_mask.any():
                    # Get local logits for this graph
                    local_logits = outputs[node_mask]  # [N_per_graph]
                    
                    # Get cand_mask for this graph (TacticAI spec)
                    if "team" in data and "ball" in data and "mask" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        mask_i = data["mask"][node_mask]
                        cand_mask = (team_i == 0) & (ball_i == 0) & (mask_i == 1)
                    elif "team" in data and "ball" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        cand_mask = (team_i == 0) & (ball_i == 0)
                    else:
                        cand_mask = torch.ones(node_mask.sum(), dtype=torch.bool, device=outputs.device)
                    
                    # Apply cand_mask: non-candidates already have -1e9 from forward
                    # Get candidate logits (for loss/metrics)
                    cand_logits = local_logits[cand_mask]  # [num_candidates]
                    
                    if cand_logits.numel() > 0:
                        receiver_node_idx = targets[i].item()  # Graph-local index (0-21)
                        # receiver_node_idx is already the local index within the graph
                        # Check if receiver is within the graph bounds
                        num_nodes_in_graph = node_mask.sum().item()
                        if receiver_node_idx < num_nodes_in_graph:
                            # receiver_node_idx is the local index within this graph
                            if cand_mask[receiver_node_idx]:
                                # Receiver is in candidates: map to candidate index
                                cand_indices = torch.where(cand_mask)[0]  # Local indices that are candidates
                                receiver_cand_idx = (cand_indices == receiver_node_idx).nonzero(as_tuple=True)[0]
                                if receiver_cand_idx.numel() > 0:
                                    receiver_cand_idx = receiver_cand_idx.item()
                                    graph_outputs.append(cand_logits)
                                    graph_targets.append(receiver_cand_idx)
                                else:
                                    excluded_invalid += 1
                            else:
                                excluded_ball_owner += 1
                        else:
                            excluded_invalid += 1
            
            # Compute loss per graph (TacticAI spec: softmax over candidates)
            batch_loss_sum = 0.0
            graphs_in_batch = 0
            for logits_b, target_b in zip(graph_outputs, graph_targets):
                # logits_b: candidate logits [num_candidates] (already masked)
                # Apply softmax and compute CrossEntropyLoss
                lb = logits_b.unsqueeze(0)  # [1, num_candidates]
                target_t = torch.tensor([target_b], dtype=torch.long, device=lb.device)
                graph_loss = criterion(lb, target_t)
                batch_loss_sum += graph_loss
                # metrics
                pred_top1 = torch.argmax(lb, dim=1)
                acc_correct += int(pred_top1.item() == target_b)
                top1_correct += int(pred_top1.item() == target_b)
                k3 = min(3, lb.size(1))
                k5 = min(5, lb.size(1))
                top3_correct += int(target_b in torch.topk(lb, k=k3, dim=1).indices[0].tolist())
                top5_correct += int(target_b in torch.topk(lb, k=k5, dim=1).indices[0].tolist())
                graphs_in_batch += 1
            if graphs_in_batch == 0:
                continue
            loss = batch_loss_sum / graphs_in_batch  # Average loss for this batch
            num_graphs_total += graphs_in_batch
            total_loss += batch_loss_sum.item()  # Accumulate total loss (not average)
            
            loss.backward()
            optimizer.step()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics (手計算)
    denom = max(1, num_graphs_total)
    epoch_metrics = {
        "loss": total_loss / denom,  # Average loss per graph (not per batch)
        "accuracy": acc_correct / denom,
        "top1": top1_correct / denom,
        "top3": top3_correct / denom,
        "top5": top5_correct / denom,
    }
    if hasattr(torch.utils, 'tensorboard'):
        pass
    # simple stdout diagnostics via logger in caller
    print(f"[Train] excluded_not_attacking={excluded_not_attacking} excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} avg_cand={sum(cand_counts)/len(cand_counts) if cand_counts else 0:.2f}")
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
    logger: Optional[Any] = None,
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
    num_graphs_total = 0
    acc_correct = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    excluded_not_attacking = 0
    excluded_ball_owner = 0
    excluded_invalid = 0
    cand_counts = []
    
    with torch.no_grad():
        batch_idx = 0
        for data, targets in tqdm(dataloader, desc="Validation"):
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            # Pass edge_attr, mask, team, ball if available
            outputs = model(
                data["x"], 
                data["edge_index"], 
                data["batch"],
                edge_attr=data.get("edge_attr"),
                mask=data.get("mask"),
                team=data.get("team"),
                ball=data.get("ball"),
            )
            
            # Debug: Log model output statistics for first batch of first epoch
            if logger and batch_idx == 0 and num_graphs_total == 0:
                batch_size = data["batch"].max().item() + 1
                if batch_size > 0:
                    first_graph_mask = data["batch"] == 0
                    if first_graph_mask.any():
                        first_graph_outputs = outputs[first_graph_mask]
                        logger.info(f"Val model outputs (first batch): mean={first_graph_outputs.mean().item():.6f}, "
                                   f"std={first_graph_outputs.std().item():.6f}, "
                                   f"min={first_graph_outputs.min().item():.6f}, "
                                   f"max={first_graph_outputs.max().item():.6f}")
            
            # TacticAI spec: outputs is [N_total] with cand_mask applied (-1e9 for non-candidates)
            # Process per graph: extract local logits and apply softmax
            batch_size = data["batch"].max().item() + 1
            graph_outputs = []
            graph_targets = []
            
            for i in range(batch_size):
                node_mask = data["batch"] == i
                if node_mask.any():
                    # Get local logits for this graph
                    local_logits = outputs[node_mask]  # [N_per_graph]
                    
                    # Get cand_mask for this graph (TacticAI spec)
                    if "team" in data and "ball" in data and "mask" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        mask_i = data["mask"][node_mask]
                        cand_mask = (team_i == 0) & (ball_i == 0) & (mask_i == 1)
                    elif "team" in data and "ball" in data:
                        team_i = data["team"][node_mask]
                        ball_i = data["ball"][node_mask]
                        cand_mask = (team_i == 0) & (ball_i == 0)
                    else:
                        cand_mask = torch.ones(node_mask.sum(), dtype=torch.bool, device=outputs.device)
                    
                    # Apply cand_mask: non-candidates already have -1e9 from forward
                    # Get candidate logits (for loss/metrics)
                    cand_logits = local_logits[cand_mask]  # [num_candidates]
                    
                    if cand_logits.numel() > 0:
                        receiver_node_idx = targets[i].item()  # Graph-local index (0-21)
                        # receiver_node_idx is already the local index within the graph
                        # Check if receiver is within the graph bounds
                        num_nodes_in_graph = node_mask.sum().item()
                        if receiver_node_idx < num_nodes_in_graph:
                            # receiver_node_idx is the local index within this graph
                            if cand_mask[receiver_node_idx]:
                                # Receiver is in candidates: map to candidate index
                                cand_indices = torch.where(cand_mask)[0]  # Local indices that are candidates
                                receiver_cand_idx = (cand_indices == receiver_node_idx).nonzero(as_tuple=True)[0]
                                if receiver_cand_idx.numel() > 0:
                                    receiver_cand_idx = receiver_cand_idx.item()
                                    graph_outputs.append(cand_logits)
                                    graph_targets.append(receiver_cand_idx)
                                    cand_counts.append(int(cand_mask.sum().item()))
                                else:
                                    excluded_invalid += 1
                            else:
                                excluded_ball_owner += 1
                        else:
                            excluded_invalid += 1
            
            # Compute loss per graph (TacticAI spec: softmax over candidates)
            batch_loss_sum = 0.0
            graphs_in_batch = 0
            for logits_b, target_b in zip(graph_outputs, graph_targets):
                # logits_b: candidate logits [num_candidates] (already masked)
                # Apply softmax and compute CrossEntropyLoss
                lb = logits_b.unsqueeze(0)  # [1, num_candidates]
                target_t = torch.tensor([target_b], dtype=torch.long, device=lb.device)
                graph_loss = criterion(lb, target_t)
                batch_loss_sum += graph_loss
                # metrics
                pred_top1 = torch.argmax(lb, dim=1)
                acc_correct += int(pred_top1.item() == target_b)
                top1_correct += int(pred_top1.item() == target_b)
                k3 = min(3, lb.size(1))
                k5 = min(5, lb.size(1))
                top3_correct += int(target_b in torch.topk(lb, k=k3, dim=1).indices[0].tolist())
                top5_correct += int(target_b in torch.topk(lb, k=k5, dim=1).indices[0].tolist())
                graphs_in_batch += 1
            if graphs_in_batch > 0:
                batch_loss_val = batch_loss_sum.item()
                total_loss += batch_loss_val  # Accumulate total loss (not average)
                num_graphs_total += graphs_in_batch
                # Debug: Log first batch details to see if model outputs change
                if logger and batch_idx == 0:  # First batch
                    first_logits = graph_outputs[0] if graph_outputs else None
                    first_target = graph_targets[0] if graph_targets else None
                    if first_logits is not None:
                        logger.info(f"Val first batch: loss={batch_loss_val:.6f}, graphs={graphs_in_batch}, "
                                   f"logits_mean={first_logits.mean().item():.6f}, logits_std={first_logits.std().item():.6f}, "
                                   f"target={first_target}")
            
            batch_idx += 1
    
    denom = max(1, num_graphs_total)
    epoch_metrics = {
        "loss": total_loss / denom,  # Average loss per graph (not per batch)
        "accuracy": acc_correct / denom,
        "top1": top1_correct / denom,
        "top3": top3_correct / denom,
        "top5": top5_correct / denom,
    }
    print(f"[Val] excluded_not_attacking={excluded_not_attacking} excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} avg_cand={sum(cand_counts)/len(cand_counts) if cand_counts else 0:.2f} num_graphs={num_graphs_total}")
    
    # Debug: Log detailed information to identify why Val Loss is fixed
    if logger:
        if num_graphs_total == 0:
            logger.warning("No valid graphs in validation set!")
        else:
            logger.info(f"Val Loss calculation: total_loss={total_loss:.6f}, num_graphs={num_graphs_total}, avg_loss={total_loss/num_graphs_total:.6f}")
    
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
        pin_memory=True if str(device).startswith("cuda") else False,  # Enable for GPU
        prefetch_factor=config.get("prefetch_factor", 2),
        persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=True if str(device).startswith("cuda") else False,  # Enable for GPU
        prefetch_factor=config.get("prefetch_factor", 2),
        persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
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
        min_delta=config.get("early_stopping", {}).get("min_delta", 0.0),
        mode="max",
        restore_best_weights=True,
    )
    
    # Training loop
    best_val_top3 = 0.0
    train_history = {"loss": [], "accuracy": [], "top1": [], "top3": [], "top5": []}
    val_history = {"loss": [], "accuracy": [], "top1": [], "top3": [], "top5": []}
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_top3 = checkpoint.get("metrics", {}).get("top3", 0.0)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics,
            use_amp=config.get("train", {}).get("amp", False)
        )
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics, logger)
        
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
        
        # Save training history after each epoch (so it's available even if interrupted)
        history_path = Path(config.get("log_dir", "runs")) / "receiver_training_history.json"
        save_training_history(
            {"train": train_history, "val": val_history},
            history_path
        )
        
        # Save best model (based on Top-3 accuracy)
        if val_metrics["top3"] > best_val_top3:
            best_val_top3 = val_metrics["top3"]
            
            checkpoint_path = Path(config.get("checkpoint_dir", "checkpoints")) / "receiver" / "best.ckpt"
            save_checkpoint(
                model, optimizer, epoch, val_metrics["loss"], val_metrics,
                checkpoint_path, scheduler
            )
            logger.info(f"New best model saved with Top-3 accuracy: {best_val_top3:.4f}")
        
        # Early stopping (based on Top-3 accuracy)
        if early_stopping(val_metrics["top3"], model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Training completed. Best validation Top-3 accuracy: {best_val_top3:.4f}")


if __name__ == "__main__":
    main()
