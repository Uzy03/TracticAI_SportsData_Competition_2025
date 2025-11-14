"""Training script for receiver prediction task.

This script trains a GATv2 model to predict pass receivers in football matches.
"""

import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from tacticai.models import GATv2Network4View, ReceiverHead
from tacticai.models.mlp_heads import mask_logits
from tacticai.modules.view_ops import apply_view_transform, D2_VIEWS
from tacticai.dataio import ReceiverDataset, create_dataloader, create_dummy_dataset
from tacticai.modules import (
    CrossEntropyLoss, TopKAccuracy, Accuracy, F1Score,
    set_seed, get_device, save_checkpoint, setup_logging,
    CosineAnnealingScheduler, EarlyStopping, save_training_history,
)
from tacticai.modules.utils import save_training_history_csv
from tacticai.modules.transforms import RandomFlipTransform


EXPECTED_PREPROCESS_VERSION = "ck_improved_v2"

AUDIT_LOGGER = logging.getLogger(__name__)

try:
    from torch.amp import GradScaler as _GradScaler, autocast as _autocast

    def AMP_CTX():
        return _autocast("cuda")

    def make_scaler(use_amp: bool):
        return _GradScaler("cuda") if use_amp else None

except Exception:  # pragma: no cover - fallback path
    from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast  # type: ignore

    def AMP_CTX():
        return _autocast()

    def make_scaler(use_amp: bool):
        return _GradScaler() if use_amp else None


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


def _assert_dataset_version(dataset: ReceiverDataset, split_name: str) -> None:
    """Ensure dataset was generated with the expected preprocessing pipeline."""
    versions = getattr(dataset, "preprocess_versions", set())
    if not versions:
        raise AssertionError(
            f"{split_name} dataset is missing preprocess_version metadata. "
            "Please regenerate the data with the latest preprocessing script."
        )
    if versions != {EXPECTED_PREPROCESS_VERSION}:
        raise AssertionError(
            f"{split_name} dataset preprocess_version mismatch: found {versions}, "
            f"expected {{{EXPECTED_PREPROCESS_VERSION}}}. "
            "Regenerate data via SoccerData/preprocess_ck_improved.py."
        )


def _reshape_to_batch(
    tensor: torch.Tensor,
    batch_size: int,
    num_nodes_per_graph: int,
) -> torch.Tensor:
    """Reshape flat per-node tensor to [B, N] layout."""
    if tensor.dim() == 1:
        return tensor.reshape(batch_size, num_nodes_per_graph)
    if tensor.dim() == 2 and tensor.size(0) == batch_size:
        return tensor
    return tensor.reshape(batch_size, num_nodes_per_graph)


def build_candidate_mask(
    player_team: torch.Tensor,
    kicker_team: int,
    is_ball_owner: torch.Tensor,
    valid_mask: torch.Tensor,
    target_idx: int,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Construct a per-graph candidate mask with guaranteed ground-truth inclusion.

    Returns a tuple of (mask, status) where status is one of:
        - None: mask ready, target already in mask
        - "forced": target was missing but safely injected
        - "target_out_of_range": target index invalid
        - "empty": resulting candidate set is empty even after forcing target
    """
    device = player_team.device
    player_team = player_team.to(device=device, dtype=torch.long)
    is_ball_owner = is_ball_owner.to(device=device, dtype=torch.bool)
    valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    N = player_team.numel()
    if target_idx < 0 or target_idx >= N:
        return None, "target_out_of_range"

    cand = (player_team == kicker_team)
    cand = cand & (~is_ball_owner)
    cand = cand & valid_mask
    target_initial = bool(cand[target_idx].item())
    cand[target_idx] = True

    if cand.sum().item() < 1:
        return None, "empty"

    cand = cand.to(device=device, dtype=torch.bool)
    return cand, (None if target_initial else "forced")


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
            num_layers=model_config.get("mlp_num_layers", 2),  # Allow configurable MLP depth
        )
        self._cand_checks_done = False
    
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
        # TacticAI spec: D2 reflection applies to coordinates and velocity vectors
        # x, y at indices 0, 1; vx, vy at indices 2, 3
        # dx_to_kicker, dy_to_kicker at indices 6, 7; dx_to_goal, dy_to_goal at indices 10, 11
        # dist and angle are invariant (not flipped)
        views_list = []
        for view_idx in range(len(D2_VIEWS)):
            x_view = x.clone()
            # Apply D2 reflection to coordinate-like features
            # x, y coordinates: indices 0, 1
            x_view = apply_view_transform(x_view, view_idx, xy_indices=(0, 1))
            # vx, vy velocities: indices 2, 3
            if x_view.size(-1) > 3:
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(2, 3))
            # dx_to_kicker, dy_to_kicker: indices 6, 7
            if x_view.size(-1) > 7:
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(6, 7))
            # dx_to_goal, dy_to_goal: indices 10, 11
            if x_view.size(-1) > 11:
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(10, 11))
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
        
        # TacticAI spec: Average over 4 views: [B, N_per_graph, output_dim]
        H = node_emb_4view.mean(dim=1)  # [B, N_per_graph, output_dim]
        
        # Apply mask if provided (element-wise multiplication, not pooling)
        if mask is not None:
            # Reshape mask to [B, N_per_graph]
            mask_batched = mask.view(B, num_nodes_per_graph) if mask.dim() == 1 else mask
            if mask_batched.dim() == 1:
                mask_batched = mask_batched.unsqueeze(0).expand(B, -1)
            mask_expanded = mask_batched.unsqueeze(-1).expand_as(H)  # [B, N_per_graph, D]
            H = H * mask_expanded.float()
        
        cand_mask = None  # Candidate masks are built downstream during training/validation
        
        # Get logits for all nodes (TacticAI spec: [B, N] format)
        # TacticAI spec: Each node outputs 1 scalar logit (no node-mean/sum aggregation)
        # ReceiverHead applies Linear(d→1) point-wise to each node
        logits = self.head(H, cand_mask=cand_mask)  # [B, N_per_graph]
        
        if cand_mask is not None:
            cand_mask = cand_mask.bool()
            if not getattr(self, "_cand_checks_done", False):
                assert cand_mask.dtype == torch.bool and cand_mask.dim() == 2, "cand_mask must be bool [B, N]"
                num_cands = cand_mask.sum(dim=1)
                assert torch.all(num_cands > 0), "graph with zero candidates detected"
                cand_H = H[cand_mask]
                assert torch.isfinite(cand_H).all(), "candidate embeddings contain non-finite values"
                assert cand_H.std() > 0, "candidate embeddings collapsed (std=0)"
                self._cand_checks_done = True
        # TacticAI spec: cand_mask = (team_flag==ATTACK) & (is_kicker==0) & (valid_mask==1)
        # Apply cand_mask LAST (after logits creation) before softmax
        # Note: Masking will be done in training loop with FP32 for stability
        # (mask_logits function is called there to avoid FP16 overflow)
        
        # Reshape back to [N_total] for compatibility
        all_logits = logits.view(-1)  # [N_total]
        
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
    
    # Note: ReceiverModel already implements D2 equivariance internally (4-view average)
    # No need to wrap with GroupPoolingWrapper
    # The group_pool config is for models that don't have built-in D2 processing
    
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
    profile: bool = False,
    debug_single_sample: bool = False,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    stats = defaultdict(float)
    stats["excluded_invalid"] = 0.0
    stats["invalid_team_mismatch"] = 0.0
    stats["invalid_target_not_in_cand"] = 0.0
    stats["excluded_invalid_filter"] = 0.0
    profile_enabled = profile
    profile_total = 0.0
    profile_count = 0
    
    total_loss = 0.0
    num_graphs_total = 0
    acc_correct = 0
    excluded_not_attacking = 0
    excluded_ball_owner = 0
    excluded_invalid = 0
    cand_counts: list[int] = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    scaler = make_scaler(use_amp)
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, targets) in enumerate(progress_bar):
        iter_start = time.perf_counter() if profile_enabled else None

        data = {k: v.to(device) for k, v in data.items()}
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with AMP_CTX():
                outputs = model(
                    data["x"],
                    data["edge_index"],
                    data["batch"],
                    edge_attr=data.get("edge_attr"),
                    mask=data.get("mask"),
                    team=data.get("team"),
                    ball=data.get("ball"),
                )
        else:
            outputs = model(
                data["x"],
                data["edge_index"],
                data["batch"],
                edge_attr=data.get("edge_attr"),
                mask=data.get("mask"),
                team=data.get("team"),
                ball=data.get("ball"),
            )

        outputs = outputs.float()

        batch_size = data["batch"].max().item() + 1
        nodes_per_graph = outputs.numel() // max(1, batch_size)
        outputs = outputs.view(batch_size, nodes_per_graph)

        team_tensor = data.get("team")
        ball_tensor = data.get("ball")
        mask_tensor = data.get("mask")
        cand_tensor = data.get("cand_mask")

        team_labels = None
        kicker_team = None

        if team_tensor is not None and ball_tensor is not None:
            team_batched = _reshape_to_batch(team_tensor, batch_size, nodes_per_graph).to(
                device=outputs.device, dtype=torch.long
            )
            ball_batched = _reshape_to_batch(ball_tensor, batch_size, nodes_per_graph).to(device=outputs.device)
            mask_batched = (
                _reshape_to_batch(mask_tensor, batch_size, nodes_per_graph).to(device=outputs.device, dtype=torch.bool)
                if mask_tensor is not None
                else None
            )
            cand_batched = (
                _reshape_to_batch(cand_tensor, batch_size, nodes_per_graph).to(device=outputs.device, dtype=torch.bool)
                if cand_tensor is not None
                else None
            )

            cand_masks_list: list[torch.Tensor] = []
            team_rows: list[torch.Tensor] = []
            kicker_team_vals: list[int] = []
            valid_indices: list[int] = []

            for g in range(batch_size):
                if g >= targets.size(0):
                    stats["excluded_invalid_filter"] += 1
                    continue

                team_row = team_batched[g]
                ball_row = ball_batched[g]
                valid_row = (
                    mask_batched[g]
                    if mask_batched is not None
                    else torch.ones(nodes_per_graph, dtype=torch.bool, device=outputs.device)
                )
                ball_owner_mask = (ball_row > 0.5).to(torch.bool)

                kicker_candidates = torch.where(ball_row > 0.5)[0]
                kicker_idx = (
                    int(kicker_candidates[0].item())
                    if kicker_candidates.numel() > 0
                    else int(torch.argmax(ball_row).item())
                )
                if nodes_per_graph > 0:
                    kicker_idx = max(0, min(kicker_idx, nodes_per_graph - 1))
                kicker_team_val = int(team_row[kicker_idx].item())

                target_idx = int(targets[g].item())
                if 0 <= target_idx < team_row.size(0):
                    if int(team_row[target_idx].item()) != kicker_team_val:
                        stats["invalid_team_mismatch"] += 1

                cand_mask_single: Optional[torch.Tensor] = None
                status = "dataset" if cand_batched is not None else "rebuilt"
                if cand_batched is not None:
                    cand_mask_candidate = cand_batched[g].to(device=outputs.device, dtype=torch.bool)
                    if cand_mask_candidate.sum().item() == 0:
                        status = "empty"
                    else:
                        cand_mask_single = cand_mask_candidate.clone()
                        if 0 <= target_idx < cand_mask_single.size(0) and not cand_mask_single[target_idx]:
                            cand_mask_single[target_idx] = True
                            stats["invalid_target_not_in_cand"] += 1

                if cand_mask_single is None:
                    cand_mask_single, status = build_candidate_mask(
                        player_team=team_row,
                        kicker_team=kicker_team_val,
                        is_ball_owner=ball_owner_mask,
                        valid_mask=valid_row,
                        target_idx=target_idx,
                    )
                if cand_mask_single is None:
                    if status in ("target_out_of_range", "empty"):
                        stats["invalid_target_not_in_cand"] += 1
                    stats["excluded_invalid_filter"] += 1
                    continue

                valid_indices.append(g)
                cand_masks_list.append(cand_mask_single)
                team_rows.append(team_row)
                kicker_team_vals.append(kicker_team_val)

            if not cand_masks_list:
                continue

            cand_masks = torch.stack(cand_masks_list, dim=0).to(dtype=torch.bool)
            valid_idx_tensor = torch.tensor(valid_indices, device=outputs.device, dtype=torch.long)
            outputs = outputs.index_select(0, valid_idx_tensor)
            targets = targets.index_select(0, valid_idx_tensor)
            team_labels = torch.stack(team_rows, dim=0)
            kicker_team = torch.tensor(kicker_team_vals, device=outputs.device, dtype=team_labels.dtype)
            batch_size = outputs.size(0)
            nodes_per_graph = outputs.size(1)

        assert cand_masks.sum(dim=1).min().item() > 0, "Train cand_mask contains empty candidate set"

        masked_outputs = mask_logits(outputs, cand_masks)

        if not hasattr(train_epoch, "_printed_debug"):
            train_epoch._printed_debug = True
            try:
                print("DBG outputs shape:", tuple(outputs.shape))
                assert cand_masks.dtype == torch.bool and cand_masks.ndim == 2, "cand_mask must be bool [B, N]"
                num_cands = cand_masks.sum(dim=1)
                print("DBG num_cands per graph (first 8):", num_cands[:8].tolist())
                with torch.no_grad():
                    masked32 = masked_outputs.float()
                    per_graph_std = []
                    for b in range(min(masked32.size(0), 8)):
                        vals = masked32[b][cand_masks[b]]
                        per_graph_std.append(float(vals.std().item()) if vals.numel() > 1 else None)
                    print("DBG cand_logits std per graph (first 8):", per_graph_std)
            except Exception as exc:  # pragma: no cover - debug only
                print("DBG ERROR:", repr(exc))
                continue

        # === build per-graph outputs/targets safely ===
        target = targets
        cand_mask = cand_masks
        B, Nclass = outputs.shape
        assert cand_mask.shape == (B, Nclass), f"cand_mask shape mismatch: {cand_mask.shape} vs {(B, Nclass)}"
        assert target.shape[0] == B, f"target shape mismatch: {target.shape} vs ({B},)"

        graph_outputs: list[torch.Tensor] = []
        graph_targets: list[torch.Tensor] = []

        for b in range(B):
            cm = cand_mask[b]
            Ncand = int(cm.sum().item())
            logits_b = outputs[b][cm] if Ncand > 0 else outputs[b]
            target_global = int(target[b].item())

            cand_indices = torch.arange(outputs.size(1), device=outputs.device)[cm]
            if (cand_indices == target_global).any():
                cand_target_idx = int((cand_indices == target_global).nonzero(as_tuple=True)[0].item())
            else:
                cand_target_idx = 0
                target_team_repr = "NA"
                kicker_team_repr = "NA"
                if team_labels is not None and 0 <= target_global < team_labels.size(1):
                    target_team_repr = int(team_labels[b, target_global].item())
                if kicker_team is not None and b < kicker_team.size(0):
                    kicker_team_repr = int(kicker_team[b].item())
                
                # DEBUG: Log warning for single sample mode
                if debug_single_sample:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"[TRAIN-DEBUG-SINGLE] WARN: target {target_global} not in candidates for graph {b} "
                        f"(target_team={target_team_repr}, kicker_team={kicker_team_repr}, "
                        f"cand_true_sum={int(cm.sum().item())}, cand_indices={cand_indices.tolist()})"
                    )
                else:
                    print(
                        f"[WARN] target {target_global} not in candidates for graph {b} "
                        f"(target_team={target_team_repr}, kicker_team={kicker_team_repr}, "
                        f"cand_true_sum={int(cm.sum().item())})"
                    )
            
            # DEBUG: Log cand_target_idx calculation for single sample mode
            if debug_single_sample and b == 0:
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[TRAIN-DEBUG-SINGLE] Graph {b} target mapping:\n"
                    f"  target_global={target_global}, cand_indices={cand_indices.tolist()}\n"
                    f"  target_in_cand={(cand_indices == target_global).any().item()}\n"
                    f"  cand_target_idx={cand_target_idx}, Ncand={Ncand}"
                )

            graph_outputs.append(logits_b.unsqueeze(0))
            graph_targets.append(torch.tensor([cand_target_idx], device=outputs.device))
            cand_counts.append(Ncand)
        # === end build ===

        batch_loss_sum = 0.0
        graphs_in_batch = 0
        for logits_b, target_b in zip(graph_outputs, graph_targets):
            if logits_b.numel() == 0:
                continue
            if logits_b.ndim not in (1, 2):
                raise ValueError(f"Unexpected logits ndim: {logits_b.ndim}")
            lb = logits_b.unsqueeze(0) if logits_b.ndim == 1 else logits_b

            target_scalar = target_b.view(-1)[0]
            target_tensor = target_scalar.to(device=lb.device, dtype=torch.long)
            target_t = target_tensor.unsqueeze(0)

            graph_loss = criterion(lb, target_t)
            batch_loss_sum += graph_loss

            pred_top1 = torch.argmax(lb, dim=1)
            target_idx = int(target_tensor.item())
            
            # DEBUG: Detailed logging for single sample mode
            if debug_single_sample and graphs_in_batch == 0:
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[TRAIN-DEBUG-SINGLE] Graph {graphs_in_batch}:\n"
                    f"  logits_b.shape={lb.shape}, logits_b={lb.tolist()}\n"
                    f"  logits_b.mean()={lb.mean().item():.6f}, logits_b.std()={lb.std().item():.6f}\n"
                    f"  pred_top1={pred_top1.item()}, target_idx={target_idx}\n"
                    f"  match={pred_top1.item() == target_idx}\n"
                    f"  cand_target_idx={target_idx}, Ncand={Ncand}"
                )
            
            acc_correct += int(pred_top1.item() == target_idx)
            top1_correct += int(pred_top1.item() == target_idx)

            k3 = min(3, lb.size(1))
            k5 = min(5, lb.size(1))
            top3_indices = torch.topk(lb, k=k3, dim=1).indices[0].tolist()
            top5_indices = torch.topk(lb, k=k5, dim=1).indices[0].tolist()
            top3_correct += int(target_idx in top3_indices)
            top5_correct += int(target_idx in top5_indices)
            graphs_in_batch += 1

        if graphs_in_batch == 0:
                    continue
            
        loss = batch_loss_sum / graphs_in_batch
        num_graphs_total += graphs_in_batch
        total_loss += batch_loss_sum.item()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        postfix = {"loss": f"{loss.item():.4f}"}
        if profile_enabled and iter_start is not None:
            iter_time = time.perf_counter() - iter_start
            profile_total += iter_time
            profile_count += 1
            avg_ms = (profile_total / profile_count) * 1000.0
            postfix["t_ms"] = f"{avg_ms:.1f}"
        progress_bar.set_postfix(postfix)
    
    denom = max(1, num_graphs_total)
    epoch_metrics = {
        "loss": total_loss / denom,
        "accuracy": acc_correct / denom,
        "top1": top1_correct / denom,
        "top3": top3_correct / denom,
        "top5": top5_correct / denom,
        "excluded_invalid": float(stats["excluded_invalid"]),
        "invalid_team_mismatch": float(stats["invalid_team_mismatch"]),
        "invalid_target_not_in_cand": float(stats["invalid_target_not_in_cand"]),
        "excluded_invalid_filter": float(stats["excluded_invalid_filter"]),
    }
    if hasattr(torch.utils, "tensorboard"):
        pass
    print(
        f"[Train] excluded_not_attacking={excluded_not_attacking} "
        f"excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} "
        f"excluded_invalid_filter={int(stats['excluded_invalid_filter'])} "
        f"avg_cand={sum(cand_counts)/len(cand_counts) if cand_counts else 0:.2f}"
    )
    
    return epoch_metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: Dict[str, Any],
    logger: Optional[Any] = None,
    min_cands_eval: int = 1,
    debug_single_sample: bool = False,
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
    
    stats = defaultdict(float)
    stats["excluded_invalid"] = 0.0
    stats["invalid_team_mismatch"] = 0.0
    stats["invalid_target_not_in_cand"] = 0.0
    stats["excluded_invalid_filter"] = 0.0
    
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
            
            if outputs.dtype != torch.float32:
                outputs = outputs.float()

            batch_size = data["batch"].max().item() + 1
            nodes_per_graph = outputs.numel() // max(1, batch_size)
            outputs = outputs.view(batch_size, nodes_per_graph)

            team_tensor = data.get("team")
            ball_tensor = data.get("ball")
            mask_tensor = data.get("mask")
            cand_tensor = data.get("cand_mask")

            team_labels = None
            kicker_team = None

            def _cnt(x):
                try:
                    return int(x.sum().item())
                except Exception:
                    try:
                        return int(x)
                    except Exception:
                        return 0

            min_cands = max(1, int(min_cands_eval))

            # Debug: Check if team/ball tensors are available
            if batch_idx == 0 and logger is not None:
                logger.info(
                    f"[VAL-DBG] batch={batch_idx} team_tensor={'None' if team_tensor is None else 'OK'} "
                    f"ball_tensor={'None' if ball_tensor is None else 'OK'} "
                    f"cand_tensor={'None' if cand_tensor is None else 'OK'} batch_size={batch_size}"
                )

            if team_tensor is not None and ball_tensor is not None:
                team_batched = _reshape_to_batch(team_tensor, batch_size, nodes_per_graph).to(
                    device=outputs.device, dtype=torch.long
                )
                ball_batched = _reshape_to_batch(ball_tensor, batch_size, nodes_per_graph).to(device=outputs.device)
                mask_batched = (
                    _reshape_to_batch(mask_tensor, batch_size, nodes_per_graph).to(device=outputs.device, dtype=torch.bool)
                    if mask_tensor is not None
                    else None
                )
                cand_batched = (
                    _reshape_to_batch(cand_tensor, batch_size, nodes_per_graph).to(device=outputs.device, dtype=torch.bool)
                    if cand_tensor is not None
                    else None
                )

                cand_masks_list: list[torch.Tensor] = []
                team_rows: list[torch.Tensor] = []
                kicker_team_vals: list[int] = []
                valid_indices: list[int] = []

                for g in range(batch_size):
                    if g >= targets.size(0):
                        stats["excluded_invalid_filter"] += 1
                        continue

                    team_row = team_batched[g]
                    ball_row = ball_batched[g]
                    valid_row = (
                        mask_batched[g]
                        if mask_batched is not None
                        else torch.ones(nodes_per_graph, dtype=torch.bool, device=outputs.device)
                    )
                    ball_owner_mask = (ball_row > 0.5).to(torch.bool)

                    kicker_candidates = torch.where(ball_row > 0.5)[0]
                    kicker_idx = (
                        int(kicker_candidates[0].item())
                        if kicker_candidates.numel() > 0
                        else int(torch.argmax(ball_row).item())
                    )
                    if nodes_per_graph > 0:
                        kicker_idx = max(0, min(kicker_idx, nodes_per_graph - 1))
                    kicker_team_val = int(team_row[kicker_idx].item())

                    target_idx = int(targets[g].item())
                    
                    # DEBUG: Output target_idx for first few batches
                    if batch_idx < 3 and g < 2 and logger is not None:
                        logger.info(
                            f"[VAL-DEBUG-BATCH] batch={batch_idx}, graph={g}, target_idx={target_idx}, "
                            f"kicker_team={kicker_team_val}, target_team={int(team_row[target_idx].item()) if 0 <= target_idx < team_row.size(0) else 'N/A'}"
                        )
                    
                    if 0 <= target_idx < team_row.size(0):
                        if int(team_row[target_idx].item()) != kicker_team_val:
                            stats["invalid_team_mismatch"] += 1

                    cand_mask_single: Optional[torch.Tensor] = None
                    status = "dataset" if cand_batched is not None else "rebuilt"
                    if cand_batched is not None:
                        cand_mask_candidate = cand_batched[g].to(device=outputs.device, dtype=torch.bool)
                        if cand_mask_candidate.sum().item() > 0:
                            cand_mask_single = cand_mask_candidate.clone()
                            if 0 <= target_idx < cand_mask_single.size(0) and not cand_mask_single[target_idx]:
                                cand_mask_single[target_idx] = True
                                stats["invalid_target_not_in_cand"] += 1
                        else:
                            status = "empty"

                    if cand_mask_single is None:
                        cand_mask_single, status = build_candidate_mask(
                            player_team=team_row,
                            kicker_team=kicker_team_val,
                            is_ball_owner=ball_owner_mask,
                            valid_mask=valid_row,
                            target_idx=target_idx,
                        )

                    # DEBUG: Check cand_mask[target_idx] and output details
                    if cand_mask_single is not None and target_idx < cand_mask_single.size(0):
                        target_in_cand_mask = bool(cand_mask_single[target_idx].item())
                        cand_mask_sum = int(cand_mask_single.sum().item())
                        
                        # DEBUG: Output for first few batches
                        if batch_idx < 3 and g < 2 and logger is not None:
                            logger.info(
                                f"[VAL-DEBUG-CAND] batch={batch_idx}, graph={g}, target_idx={target_idx}, "
                                f"cand_mask[target]={target_in_cand_mask}, cand_mask.sum()={cand_mask_sum}, "
                                f"status={status}"
                            )
                        
                        # Assert that target must be in candidates
                        if not target_in_cand_mask:
                            raise AssertionError(
                                f"Target not in candidates! batch={batch_idx}, graph={g}, target={target_idx}, "
                                f"cand_mask_sum={cand_mask_sum}, cand_mask[target]={target_in_cand_mask}"
                            )

                    # Debug logging for first batch, first few graphs
                    if batch_idx == 0 and g < 3 and logger is not None:
                        logger.info(
                            f"[VAL-DBG] g={g} cand_mask_single={'None' if cand_mask_single is None else f'sum={cand_mask_single.sum().item()}'} "
                            f"status={status} min_cands={min_cands} target_idx={target_idx}"
                        )

                    if cand_mask_single is None:
                        if status in ("target_out_of_range", "empty"):
                            stats["invalid_target_not_in_cand"] += 1
                        stats["excluded_invalid_filter"] += 1
                        continue

                    if cand_mask_single.sum().item() < min_cands:
                        stats["excluded_invalid_filter"] += 1
                        continue

                    valid_indices.append(g)
                    cand_masks_list.append(cand_mask_single)
                    team_rows.append(team_row)
                    kicker_team_vals.append(kicker_team_val)

                if not cand_masks_list:
                    if logger is not None:
                        logger.warning(
                            "[VAL-FILTER] batch=%d total=%d kept=0 (all filtered) - "
                            "excluded_invalid_filter=%d, invalid_team_mismatch=%d, invalid_target_not_in_cand=%d",
                            batch_idx, batch_size,
                            stats.get("excluded_invalid_filter", 0),
                            stats.get("invalid_team_mismatch", 0),
                            stats.get("invalid_target_not_in_cand", 0),
                        )
                    continue

                cand_masks = torch.stack(cand_masks_list, dim=0).to(dtype=torch.bool)
                valid_idx_tensor = torch.tensor(valid_indices, device=targets.device, dtype=torch.long)
                outputs = outputs.index_select(0, valid_idx_tensor)
                targets = targets.index_select(0, valid_idx_tensor)
                cand_masks = cand_masks.index_select(0, valid_idx_tensor)
                team_labels = torch.stack(team_rows, dim=0)
                kicker_team = torch.tensor(kicker_team_vals, device=outputs.device, dtype=team_labels.dtype)

                if logger is not None and cand_masks.numel() > 0:
                    logger.info(
                        "[VAL-FILTER] batch=%d total=%d kept=%d avg_cand=%.2f",
                        batch_idx,
                        batch_size,
                        cand_masks.size(0),
                        cand_masks.sum(dim=1).float().mean().item(),
                    )
            else:
                if batch_idx == 0 and logger is not None:
                    logger.warning(
                        f"[VAL-DBG] batch={batch_idx} team_tensor or ball_tensor is None, using fallback cand_masks"
                    )
                cand_masks = torch.ones(batch_size, nodes_per_graph, dtype=torch.bool, device=outputs.device)

            assert cand_masks.sum(dim=1).min().item() > 0, "Validation cand_mask contains empty candidate set"

            masked_outputs = mask_logits(outputs, cand_masks)

            # === build per-graph outputs/targets safely (validation) ===
            target = targets
            cand_mask = cand_masks
            B, Nclass = outputs.shape
            assert cand_mask.shape == (B, Nclass), f"cand_mask shape mismatch: {cand_mask.shape} vs {(B, Nclass)}"
            assert target.shape[0] == B, f"target shape mismatch: {target.shape} vs ({B},)"

            graph_outputs: list[torch.Tensor] = []
            graph_targets: list[torch.Tensor] = []

            for b in range(B):
                cm = cand_mask[b]
                Ncand = int(cm.sum().item())
                logits_b = outputs[b][cm] if Ncand > 0 else outputs[b]
                target_global = int(target[b].item())
                graph_id = valid_indices[b] if b < len(valid_indices) else b

                # DEBUG: Check if target is in cand_mask
                if target_global < cm.size(0):
                    target_in_cand = bool(cm[target_global].item())
                    if not target_in_cand:
                        # Assert that target must be in candidates
                        raise AssertionError(
                            f"Target not in candidates! graph_id={graph_id}, target={target_global}, "
                            f"cand_mask_sum={Ncand}, cand_mask[target]={target_in_cand}"
                        )
                else:
                    target_in_cand = False
                    raise AssertionError(
                        f"Target index out of range! graph_id={graph_id}, target={target_global}, "
                        f"cand_mask_size={cm.size(0)}"
                    )

                cand_indices = torch.arange(outputs.size(1), device=outputs.device)[cm]
                if (cand_indices == target_global).any():
                    cand_target_idx = int((cand_indices == target_global).nonzero(as_tuple=True)[0].item())
                else:
                    cand_target_idx = 0
                    target_team_repr = "NA"
                    kicker_team_repr = "NA"
                    if team_labels is not None and 0 <= target_global < team_labels.size(1):
                        target_team_repr = int(team_labels[b, target_global].item())
                        tt = target_team_repr
                    else:
                        tt = None
                    if kicker_team is not None and b < kicker_team.size(0):
                        kicker_team_repr = int(kicker_team[b].item())
                        kt = kicker_team_repr
                    else:
                        kt = None

                    if tt is not None and kt is not None and tt != kt:
                        stats["invalid_team_mismatch"] += 1
                        if logger is not None and (int(stats["invalid_team_mismatch"]) % 50) == 1:
                            AUDIT_LOGGER.warning(
                                f"[AUDIT] team_mismatch: g={b}, tgt={target_global}, kicker_team={kt}, "
                                f"target_team={tt}, cand_true_sum={int(cm.sum().item())} (val)"
                            )
                    else:
                        stats["invalid_target_not_in_cand"] += 1
                        if logger is not None and (int(stats["invalid_target_not_in_cand"]) % 50) == 1:
                            AUDIT_LOGGER.warning(
                                f"[AUDIT] target_not_in_cand: g={b}, tgt={target_global}, kicker_team={kt}, "
                                f"cand_true_sum={int(cm.sum().item())} (val)"
                            )

                    print(
                        f"[WARN] target {target_global} not in candidates for graph {b} (val) "
                        f"(target_team={target_team_repr}, kicker_team={kicker_team_repr}, "
                        f"cand_true_sum={int(cm.sum().item())})"
                    )

                # DEBUG: Output detailed information for first batch, first sample
                if batch_idx == 0 and b == 0 and logger is not None:
                    logits_full = outputs[b]  # Full logits before masking
                    logits_masked = logits_b  # Logits after masking
                    topk_values, topk_indices = torch.topk(logits_full, k=min(5, logits_full.size(0)))
                    topk_cand_values, topk_cand_indices = torch.topk(logits_masked, k=min(5, logits_masked.size(0)))
                    
                    logger.info(
                        f"[VAL-DEBUG] batch={batch_idx}, graph={b}, graph_id={graph_id}:\n"
                        f"  target_global={target_global}, target_in_cand={target_in_cand}, cand_mask[target]={cm[target_global].item() if target_global < cm.size(0) else 'N/A'}\n"
                        f"  cand_mask.sum()={Ncand}, cand_mask.shape={cm.shape}\n"
                        f"  logits_full.shape={logits_full.shape}, logits_full.mean()={logits_full.mean().item():.6f}, logits_full.std()={logits_full.std().item():.6f}\n"
                        f"  logits_masked.shape={logits_masked.shape}, logits_masked.mean()={logits_masked.mean().item():.6f}, logits_masked.std()={logits_masked.std().item():.6f}\n"
                        f"  topk_full_indices={topk_indices.tolist()}, topk_full_values={topk_values.tolist()}\n"
                        f"  topk_cand_indices={topk_cand_indices.tolist()}, topk_cand_values={topk_cand_values.tolist()}\n"
                        f"  cand_target_idx={cand_target_idx}, target_in_topk={cand_target_idx in topk_cand_indices.tolist()}"
                    )

                graph_outputs.append(logits_b.unsqueeze(0))
                graph_targets.append(torch.tensor([cand_target_idx], device=outputs.device))
                cand_counts.append(Ncand)
            # === end build ===
            
            # Compute loss per graph (TacticAI spec: softmax over candidates)
            batch_loss_sum = 0.0
            graphs_in_batch = 0
            for logits_b, target_b in zip(graph_outputs, graph_targets):
                if logits_b.numel() == 0:
                    continue
                    
                if logits_b.ndim not in (1, 2):
                    raise ValueError(f"Unexpected logits ndim (val): {logits_b.ndim}")
                lb = logits_b.unsqueeze(0) if logits_b.ndim == 1 else logits_b

                target_scalar = target_b.view(-1)[0]
                target_tensor = target_scalar.to(device=lb.device, dtype=torch.long)
                target_t = target_tensor.unsqueeze(0)

                graph_loss = criterion(lb, target_t)
                batch_loss_sum += graph_loss

                pred_top1 = torch.argmax(lb, dim=1)
                target_idx = int(target_tensor.item())
                
                # DEBUG: Detailed logging for single sample mode
                if debug_single_sample and graphs_in_batch == 0 and logger is not None:
                    logger.info(
                        f"[VAL-DEBUG-SINGLE] Graph {graphs_in_batch}:\n"
                        f"  logits_b.shape={lb.shape}, logits_b={lb.tolist()}\n"
                        f"  logits_b.mean()={lb.mean().item():.6f}, logits_b.std()={lb.std().item():.6f}\n"
                        f"  pred_top1={pred_top1.item()}, target_idx={target_idx}\n"
                        f"  match={pred_top1.item() == target_idx}"
                    )
                
                acc_correct += int(pred_top1.item() == target_idx)
                top1_correct += int(pred_top1.item() == target_idx)

                k3 = min(3, lb.size(1))
                k5 = min(5, lb.size(1))
                top3_indices = torch.topk(lb, k=k3, dim=1).indices[0].tolist()
                top5_indices = torch.topk(lb, k=k5, dim=1).indices[0].tolist()
                top3_correct += int(target_idx in top3_indices)
                top5_correct += int(target_idx in top5_indices)

                graphs_in_batch += 1
            if graphs_in_batch > 0:
                batch_loss_val = batch_loss_sum.item()
                total_loss += batch_loss_val  # Accumulate total loss (not average)
                num_graphs_total += graphs_in_batch
                # Debug: Log first batch details to see if model outputs change
                if logger and batch_idx == 0:  # First batch
                    first_logits = graph_outputs[0] if graph_outputs else None
                    first_target = graph_targets[0] if graph_targets else None
                    if first_logits is not None and first_logits.numel() > 1:
                        # TacticAI spec: Log cand logits_std (should not be 0)
                        cand_logits_std = first_logits.std().item()
                        cand_unique_count = first_logits.unique().numel()
                        logger.info(f"Val first batch: loss={batch_loss_val:.6f}, graphs={graphs_in_batch}, "
                                   f"logits_shape={first_logits.shape}, logits={first_logits.tolist()[:5]}, "
                                   f"logits_mean={first_logits.mean().item():.6f}, logits_std={cand_logits_std:.6f}, "
                                   f"cand_unique_count={cand_unique_count}, target={first_target}")
                        if cand_logits_std < 1e-6:
                            logger.warning(f"Val first batch: cand logits_std is too small ({cand_logits_std:.6f}), possible collapse!")
                    elif logger and batch_idx == 0:
                        logger.warning(f"Val first batch: No valid logits (first_logits={first_logits is not None}, "
                                     f"numel={first_logits.numel() if first_logits is not None else 0})")
            
            # Clear graph_outputs and graph_targets for next batch
            graph_outputs = []
            graph_targets = []
            
            batch_idx += 1
    
    denom = max(1, num_graphs_total)
    epoch_metrics = {
        "loss": total_loss / denom,  # Average loss per graph (not per batch)
        "accuracy": acc_correct / denom,
        "top1": top1_correct / denom,
        "top3": top3_correct / denom,
        "top5": top5_correct / denom,
        "excluded_invalid": float(stats["excluded_invalid"]),
        "invalid_team_mismatch": float(stats["invalid_team_mismatch"]),
        "invalid_target_not_in_cand": float(stats["invalid_target_not_in_cand"]),
        "excluded_invalid_filter": float(stats["excluded_invalid_filter"]),
    }
    print(
        f"[Val] excluded_not_attacking={excluded_not_attacking} "
        f"excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} "
        f"excluded_invalid_filter={int(stats['excluded_invalid_filter'])} "
        f"avg_cand={sum(cand_counts)/len(cand_counts) if cand_counts else 0:.2f} "
        f"num_graphs={num_graphs_total}"
    )
    
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
    parser.add_argument(
        "--debug_overfit_single_sample",
        action="store_true",
        help="Debug overfit test with single sample (overrides config settings)"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Display moving-average batch time during training.",
    )
    
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
    resolved_config_path = Path(args.config).resolve()
    logger.info(f"Resolved config path: {resolved_config_path}")
    assert config["d2"]["group_pool"] is False, "STOP: group_pool must be False but True was loaded."
    
    # Create datasets
    if args.debug_overfit_single_sample:
        # Single sample overfit test mode
        logger.info("[DEBUG-OVERFIT-SINGLE] Using single sample for overfitting test...")
        full_train_dataset = ReceiverDataset(
            config["data"]["train_path"],
            file_format=config["data"].get("format", "parquet"),
            phase="train",
        )
        _assert_dataset_version(full_train_dataset, "train")
        
        if len(full_train_dataset) == 0:
            raise ValueError("[DEBUG-OVERFIT-SINGLE] Training dataset is empty!")
        
        # Use first sample only
        train_dataset = Subset(full_train_dataset, [0])
        val_dataset = Subset(full_train_dataset, [0])  # Same sample for train and val
        
        logger.info(
            f"[DEBUG-OVERFIT-SINGLE] Using first sample only: "
            f"train={len(train_dataset)} samples, val={len(val_dataset)} samples"
        )
        
        # DEBUG: Check the single sample data
        try:
            sample_data, sample_target = train_dataset[0]
            logger.info(
                f"[DEBUG-OVERFIT-SINGLE] Sample data check:\n"
                f"  target={sample_target.item()}\n"
                f"  x.shape={sample_data.get('x', torch.tensor([])).shape}\n"
                f"  team={'present' if 'team' in sample_data else 'missing'}\n"
                f"  ball={'present' if 'ball' in sample_data else 'missing'}\n"
                f"  cand_mask={'present' if 'cand_mask' in sample_data else 'missing'}"
            )
            if 'cand_mask' in sample_data:
                cand_mask = sample_data['cand_mask']
                target_idx = int(sample_target.item())
                logger.info(
                    f"[DEBUG-OVERFIT-SINGLE] cand_mask: sum={cand_mask.sum().item()}, "
                    f"shape={cand_mask.shape}, target_in_cand={cand_mask[target_idx].item() if target_idx < cand_mask.size(0) else 'N/A'}"
                )
        except Exception as e:
            logger.warning(f"[DEBUG-OVERFIT-SINGLE] Error checking sample data: {e}")
        
        # Override config for single sample overfit test
        config["train"]["batch_size"] = 1
        config["train"]["epochs"] = 500
        config["optimizer"]["lr"] = 1e-4
        config["optimizer"]["weight_decay"] = 0.0
        config["model"]["dropout"] = 0.0
        config["loss"]["label_smoothing"] = 0.0
        config["d2"]["transforms"]["hflip"] = False
        config["d2"]["transforms"]["vflip"] = False
        
        logger.info(
            f"[DEBUG-OVERFIT-SINGLE] Overridden settings: "
            f"batch_size=1, epochs=500, lr=1e-4, weight_decay=0.0, dropout=0.0, "
            f"label_smoothing=0.0, augmentation=OFF"
        )
    elif args.debug_overfit:
        # Use small dataset for overfit test
        train_dataset = create_dummy_dataset("receiver", num_samples=10, num_players=22)
        val_dataset = create_dummy_dataset("receiver", num_samples=5, num_players=22)
    else:
        # Check if debug_overfit is enabled in config
        debug_overfit_config = config.get("debug_overfit", {})
        use_debug_overfit = debug_overfit_config.get("enabled", False)
        
        if use_debug_overfit:
            # Load full training dataset first
            logger.info("[DEBUG-OVERFIT] Creating mini subset for overfitting test...")
            full_train_dataset = ReceiverDataset(
                config["data"]["train_path"],
                file_format=config["data"].get("format", "parquet"),
                phase="train",
            )
            _assert_dataset_version(full_train_dataset, "train")
            
            num_samples = debug_overfit_config.get("num_samples", 32)
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
                f"[DEBUG-OVERFIT] Created train/val datasets with {len(train_dataset)} samples each "
                f"(train=val=same samples for overfitting test)"
            )
        else:
            # Normal training mode
            train_dataset = ReceiverDataset(
                config["data"]["train_path"],
                file_format=config["data"].get("format", "parquet"),
                phase="train",
            )
            val_path = config["data"]["val_path"]
            logger.info(f"[VAL-DATASET] Loading validation dataset from: {val_path}")
            val_dataset = ReceiverDataset(
                val_path,
                file_format=config["data"].get("format", "parquet"),
                phase="val",
            )
            logger.info(f"[VAL-DATASET] Validation dataset loaded: {len(val_dataset)} samples")
            _assert_dataset_version(train_dataset, "train")
            _assert_dataset_version(val_dataset, "val")
            
            # DEBUG: Check first few validation samples
            if len(val_dataset) > 0:
                logger.info(f"[VAL-DATASET] Checking first validation sample...")
                try:
                    sample_data, sample_target = val_dataset[0]
                    logger.info(
                        f"[VAL-DATASET] First sample - target={sample_target.item()}, "
                        f"x.shape={sample_data.get('x', torch.tensor([])).shape}, "
                        f"team={'present' if 'team' in sample_data else 'missing'}, "
                        f"ball={'present' if 'ball' in sample_data else 'missing'}, "
                        f"cand_mask={'present' if 'cand_mask' in sample_data else 'missing'}"
                    )
                    if 'cand_mask' in sample_data:
                        cand_mask = sample_data['cand_mask']
                        target_idx = int(sample_target.item())
                        logger.info(
                            f"[VAL-DATASET] First sample cand_mask - sum={cand_mask.sum().item()}, "
                            f"shape={cand_mask.shape}, target_in_cand={cand_mask[target_idx].item() if target_idx < cand_mask.size(0) else 'N/A'}"
                        )
                except Exception as e:
                    logger.warning(f"[VAL-DATASET] Error checking first sample: {e}")
    
    # Create data loaders
    # Override shuffle for single sample mode
    train_shuffle = False if args.debug_overfit_single_sample else True
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=train_shuffle,
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
    
    # Create test dataset and loader (for final evaluation)
    test_dataset = None
    test_loader = None
    if not args.debug_overfit and "test_path" in config.get("data", {}):
        try:
            test_dataset = ReceiverDataset(
                config["data"]["test_path"],
                file_format=config["data"].get("format", "parquet"),
                phase="test",
            )
            _assert_dataset_version(test_dataset, "test")
            test_loader = create_dataloader(
                test_dataset,
                batch_size=config["eval"]["batch_size"],
                shuffle=False,
                num_workers=config.get("num_workers", 0),
                pin_memory=True if str(device).startswith("cuda") else False,
                prefetch_factor=config.get("prefetch_factor", 2),
                persistent_workers=config.get("persistent_workers", False) if config.get("num_workers", 0) > 0 else False,
            )
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
    
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
    # Disable early stopping for single sample debug mode
    if args.debug_overfit_single_sample:
        early_stopping_patience = 99999  # Effectively disable early stopping
        logger.info("[DEBUG-OVERFIT-SINGLE] Early stopping disabled (patience=99999)")
    else:
        early_stopping_patience = config.get("early_stopping", {}).get("patience", 10)
    
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
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
    
    # Training loop
    # Note: For debug_overfit_single_sample, epochs is set to 500 in config override
    for epoch in range(start_epoch, config["train"]["epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Learning rate will be updated after train/val via scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if debug_overfit is enabled for debug logging
        debug_overfit_config = config.get("debug_overfit", {})
        use_debug_overfit = debug_overfit_config.get("enabled", False)
        debug_logging = args.debug_overfit_single_sample or use_debug_overfit
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics,
            use_amp=config.get("train", {}).get("amp", False),
            profile=args.profile,
            debug_single_sample=debug_logging,
        )
        
        # Validation
        val_metrics = validate_epoch(
            model,
            val_loader,
            criterion,
            device,
            metrics,
            logger,
            min_cands_eval=config.get("eval", {}).get("min_cands_eval", 1),
            debug_single_sample=debug_logging,
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, CosineAnnealingScheduler):
                current_lr = scheduler.step_epoch(epoch - start_epoch)
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        # Special logging for single sample debug mode
        if args.debug_overfit_single_sample:
            logger.info(
                f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.6f}, train_top1={train_metrics['top1']:.6f}, "
                f"val_loss={val_metrics['loss']:.6f}, val_top1={val_metrics['top1']:.6f}"
            )
        else:
            logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"Top-1: {train_metrics['top1']:.4f}, "
                       f"Top-3: {train_metrics['top3']:.4f}, "
                f"Top-5: {train_metrics['top5']:.4f} "
                f"(excluded_invalid={int(train_metrics.get('excluded_invalid', 0))}, "
                f"excluded_invalid_filter={int(train_metrics.get('excluded_invalid_filter', 0))})"
            )
            
            logger.info(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"Top-1: {val_metrics['top1']:.4f}, "
                       f"Top-3: {val_metrics['top3']:.4f}, "
                f"Top-5: {val_metrics['top5']:.4f} "
                f"(excluded_invalid={int(val_metrics.get('excluded_invalid', 0))}, "
                f"excluded_invalid_filter={int(val_metrics.get('excluded_invalid_filter', 0))})"
            )

        logger.info(
            "[TRAIN-AUDIT] team_mismatch=%d, target_not_in_cand=%d, excluded_invalid_filter=%d",
            int(train_metrics.get("invalid_team_mismatch", 0)),
            int(train_metrics.get("invalid_target_not_in_cand", 0)),
            int(train_metrics.get("excluded_invalid_filter", 0)),
        )

        logger.info(
            "[VAL-AUDIT] team_mismatch=%d, target_not_in_cand=%d, excluded_invalid_filter=%d",
            int(val_metrics.get("invalid_team_mismatch", 0)),
            int(val_metrics.get("invalid_target_not_in_cand", 0)),
            int(val_metrics.get("excluded_invalid_filter", 0)),
        )
        
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
        
        # Save CSV history after each epoch (overwrite mode)
        # Use different filename for debug_overfit mode
        if args.debug_overfit_single_sample:
            csv_filename = "training_history_debug_overfit_single.csv"
        else:
            debug_overfit_config = config.get("debug_overfit", {})
            use_debug_overfit = debug_overfit_config.get("enabled", False)
            csv_filename = "training_history_debug_overfit.csv" if use_debug_overfit else "training_history.csv"
        csv_path = Path(config.get("log_dir", "runs")) / csv_filename
        save_training_history_csv(
            train_history,
            val_history,
            test_history=None,  # Test metrics will be added at the end
            filepath=csv_path
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
    
    # Final debug output for single sample mode
    if args.debug_overfit_single_sample:
        logger.info("[DEBUG-OVERFIT-SINGLE] Final epoch evaluation...")
        model.eval()
        with torch.no_grad():
            # Get the single sample
            for data, target in val_loader:
                data = {k: v.to(device) for k, v in data.items()}
                target = target.to(device)
                
                # Forward pass
                outputs = model(
                    data["x"],
                    data["edge_index"],
                    data["batch"],
                    edge_attr=data.get("edge_attr"),
                    mask=data.get("mask"),
                    team=data.get("team"),
                    ball=data.get("ball"),
                )
                
                if outputs.dtype != torch.float32:
                    outputs = outputs.float()
                
                batch_size = data["batch"].max().item() + 1
                nodes_per_graph = outputs.numel() // max(1, batch_size)
                outputs = outputs.view(batch_size, nodes_per_graph)
                
                # Get candidate mask and apply
                team_tensor = data.get("team")
                ball_tensor = data.get("ball")
                if team_tensor is not None and ball_tensor is not None:
                    team_batched = _reshape_to_batch(team_tensor, batch_size, nodes_per_graph).to(
                        device=outputs.device, dtype=torch.long
                    )
                    ball_batched = _reshape_to_batch(ball_tensor, batch_size, nodes_per_graph).to(device=outputs.device)
                    
                    team_row = team_batched[0]
                    ball_row = ball_batched[0]
                    ball_owner_mask = (ball_row > 0.5).to(torch.bool)
                    
                    kicker_candidates = torch.where(ball_row > 0.5)[0]
                    kicker_idx = (
                        int(kicker_candidates[0].item())
                        if kicker_candidates.numel() > 0
                        else int(torch.argmax(ball_row).item())
                    )
                    if nodes_per_graph > 0:
                        kicker_idx = max(0, min(kicker_idx, nodes_per_graph - 1))
                    kicker_team_val = int(team_row[kicker_idx].item())
                    
                    target_idx = int(target[0].item())
                    
                    # Build candidate mask
                    valid_row = torch.ones(nodes_per_graph, dtype=torch.bool, device=outputs.device)
                    cand_mask_single, _ = build_candidate_mask(
                        player_team=team_row,
                        kicker_team=kicker_team_val,
                        is_ball_owner=ball_owner_mask,
                        valid_mask=valid_row,
                        target_idx=target_idx,
                    )
                    
                    if cand_mask_single is not None:
                        cand_mask = cand_mask_single.unsqueeze(0)  # [1, N]
                        masked_outputs = mask_logits(outputs, cand_mask)
                        
                        # Get prediction
                        logits_masked = masked_outputs[0][cand_mask_single]
                        pred_idx = int(torch.argmax(logits_masked).item())
                        
                        # Map back to global index
                        cand_indices = torch.arange(outputs.size(1), device=outputs.device)[cand_mask_single]
                        pred_global_idx = int(cand_indices[pred_idx].item())
                        target_global_idx = target_idx
                        
                        logger.info(
                            f"[DEBUG-OVERFIT-SINGLE] Final debug sample: pred={pred_global_idx}, target={target_global_idx}, "
                            f"match={pred_global_idx == target_global_idx}"
                        )
                break  # Only process first batch
    
    # Evaluate on test set if available
    test_history = None
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        test_metrics = validate_epoch(
            model,
            test_loader,
            criterion,
            device,
            metrics,
            logger,
            min_cands_eval=config.get("eval", {}).get("min_cands_eval", 1),
        )
        test_history = {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "top1": test_metrics["top1"],
            "top3": test_metrics["top3"],
            "top5": test_metrics["top5"],
        }
        logger.info(
            f"Test - Loss: {test_metrics['loss']:.4f}, "
            f"Acc: {test_metrics['accuracy']:.4f}, "
            f"Top-1: {test_metrics['top1']:.4f}, "
            f"Top-3: {test_metrics['top3']:.4f}, "
            f"Top-5: {test_metrics['top5']:.4f}"
        )
        
        # Save final CSV with test metrics
        # Use different filename for debug_overfit mode
        if args.debug_overfit_single_sample:
            csv_filename = "training_history_debug_overfit_single.csv"
        else:
            debug_overfit_config = config.get("debug_overfit", {})
            use_debug_overfit = debug_overfit_config.get("enabled", False)
            csv_filename = "training_history_debug_overfit.csv" if use_debug_overfit else "training_history.csv"
        csv_path = Path(config.get("log_dir", "runs")) / csv_filename
        save_training_history_csv(
            train_history,
            val_history,
            test_history=test_history,
            filepath=csv_path
        )
        logger.info(f"Training history saved to {csv_path}")


if __name__ == "__main__":
    main()
