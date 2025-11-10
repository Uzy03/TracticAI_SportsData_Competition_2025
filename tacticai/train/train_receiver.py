"""Training script for receiver prediction task.

This script trains a GATv2 model to predict pass receivers in football matches.
"""

import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from tacticai.modules.transforms import RandomFlipTransform


EXPECTED_PREPROCESS_VERSION = "ck_improved_v1"

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
    team: Optional[torch.Tensor],
    ball: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    x: Optional[torch.Tensor],
    batch_size: int,
    num_nodes_per_graph: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Construct candidate mask following TacticAI spec."""
    if team is None or ball is None:
        return None

    team_batched = _reshape_to_batch(team, batch_size, num_nodes_per_graph).to(device=device)
    ball_batched = _reshape_to_batch(ball, batch_size, num_nodes_per_graph).to(device=device)
    mask_batched = (
        _reshape_to_batch(mask, batch_size, num_nodes_per_graph).to(device=device)
        if mask is not None
        else None
    )

    x_batched = None
    if x is not None:
        x_batched = x.reshape(batch_size, num_nodes_per_graph, -1).to(device=device)

    cand_mask = torch.zeros(batch_size, num_nodes_per_graph, dtype=torch.bool, device=device)

    for b in range(batch_size):
        team_row = team_batched[b]
        ball_row = ball_batched[b]

        valid_row = torch.ones(num_nodes_per_graph, dtype=torch.bool, device=device)
        if mask_batched is not None:
            valid_row &= mask_batched[b].to(dtype=torch.bool)
        if x_batched is not None and x_batched.size(-1) >= 2:
            pos_row = x_batched[b, :, :2]
            valid_row &= torch.isfinite(pos_row).all(dim=-1)

        kicker_candidates = torch.where(ball_row > 0.5)[0]
        if kicker_candidates.numel() > 0:
            kicker_idx = int(kicker_candidates[0].item())
        else:
            kicker_idx = int(torch.argmax(ball_row).item())

        kicker_idx = max(0, min(kicker_idx, num_nodes_per_graph - 1))

        team_row_int = team_row.to(dtype=torch.long)
        kicker_team_val = int(team_row_int[kicker_idx].item()) if num_nodes_per_graph > 0 else 0
        team_mask_row = (team_row_int == kicker_team_val)

        candidate_row = (team_mask_row & valid_row).to(dtype=torch.bool)
        if num_nodes_per_graph > 0:
            candidate_row[kicker_idx] = False

        if candidate_row.sum().item() == 0:
            fallback_row = valid_row.clone()
            if num_nodes_per_graph > 0:
                fallback_row[kicker_idx] = False
            if fallback_row.sum().item() == 0:
                fallback_row = valid_row
            candidate_row = fallback_row.to(dtype=torch.bool)

        cand_mask[b] = candidate_row

    return cand_mask


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
        
        cand_mask = build_candidate_mask(
            team=team,
            ball=ball,
            mask=mask,
            x=x,
            batch_size=B,
            num_nodes_per_graph=num_nodes_per_graph,
            device=x.device,
        )
        
        # Get logits for all nodes (TacticAI spec: [B, N] format)
        # TacticAI spec: Each node outputs 1 scalar logit (no node-mean/sum aggregation)
        # ReceiverHead applies Linear(d→1) point-wise to each node
        # Pass cand_mask for debug output (only first batch)
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
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()
    
    stats = defaultdict(float)
    stats["excluded_invalid"] = 0.0
    stats["invalid_team_mismatch"] = 0.0
    stats["invalid_target_not_in_cand"] = 0.0
    profile_enabled = profile
    profile_total = 0.0
    profile_count = 0
    stats["invalid_team_mismatch"] = 0.0
    stats["invalid_target_not_in_cand"] = 0.0
    
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

        cand_masks = build_candidate_mask(
            team=data.get("team"),
            ball=data.get("ball"),
            mask=data.get("mask"),
            x=data.get("x"),
            batch_size=batch_size,
            num_nodes_per_graph=nodes_per_graph,
            device=outputs.device,
        )
        if cand_masks is None:
            cand_masks = torch.ones(batch_size, nodes_per_graph, dtype=torch.bool, device=outputs.device)
        cand_masks = cand_masks.to(dtype=torch.bool)

        team_labels: Optional[torch.Tensor] = None
        ball_batched: Optional[torch.Tensor] = None
        kicker_team: Optional[torch.Tensor] = None

        team_tensor = data.get("team")
        ball_tensor = data.get("ball")
        if team_tensor is not None and ball_tensor is not None:
            team_labels = _reshape_to_batch(team_tensor, batch_size, nodes_per_graph).to(
                device=outputs.device, dtype=torch.long
            )
            ball_batched = _reshape_to_batch(ball_tensor, batch_size, nodes_per_graph).to(device=outputs.device)

            has_ball_mask = ball_batched > 0.5
            default_kicker_idx = torch.argmax(ball_batched, dim=1)
            first_ball_idx = torch.argmax(has_ball_mask.float(), dim=1)
            kicker_idx = torch.where(
                has_ball_mask.any(dim=1),
                first_ball_idx,
                default_kicker_idx,
            )
            if nodes_per_graph > 0:
                kicker_idx = kicker_idx.clamp_(0, nodes_per_graph - 1)
            kicker_team = team_labels.gather(1, kicker_idx.unsqueeze(1)).squeeze(1)

            # Enforce candidate mask alignment with kicker team and exclude kicker node
            same_team_mask = team_labels == kicker_team.unsqueeze(1)
            cand_masks = cand_masks & same_team_mask
            batch_indices = torch.arange(batch_size, device=outputs.device)
            cand_masks[batch_indices, kicker_idx] = False

            B_filter = targets.shape[0]
            valid_indices: list[int] = []
            for g in range(B_filter):
                tgt = int(targets[g].item())
                if tgt < 0 or tgt >= team_labels.size(1):
                    stats["excluded_invalid"] += 1
                    continue
                cm_g = cand_masks[g]
                tgt_in_cand = bool(cm_g[tgt].item()) if tgt < cm_g.size(0) else False
                tt = int(team_labels[g, tgt].item())
                kt = int(kicker_team[g].item())
                if tt != kt:
                    stats["invalid_team_mismatch"] += 1
                    if (int(stats["invalid_team_mismatch"]) % 50) == 1:
                        AUDIT_LOGGER.warning(
                            f"[AUDIT] team_mismatch: g={g}, tgt={tgt}, kicker_team={kt}, "
                            f"target_team={tt}, cand_true_sum={int(cand_masks[g].sum().item())}"
                        )
                    stats["excluded_invalid"] += 1
                    continue
                if not tgt_in_cand:
                    stats["invalid_target_not_in_cand"] += 1
                    if (int(stats["invalid_target_not_in_cand"]) % 50) == 1:
                        AUDIT_LOGGER.warning(
                            f"[AUDIT] target_not_in_cand: g={g}, tgt={tgt}, kicker_team={kt}, "
                            f"cand_true_sum={int(cand_masks[g].sum().item())}"
                        )
                    if 0 <= tgt < cm_g.size(0):
                        cand_masks[g, tgt] = True
                valid_indices.append(g)

            if not valid_indices:
                # NOTE: Samples where target is not a valid attacking candidate are excluded from training.
                continue

            valid_idx_tensor = torch.tensor(valid_indices, device=targets.device, dtype=torch.long)
            outputs = outputs.index_select(0, valid_idx_tensor)
            targets = targets.index_select(0, valid_idx_tensor)
            cand_masks = cand_masks.index_select(0, valid_idx_tensor)
            team_labels = team_labels.index_select(0, valid_idx_tensor)
            kicker_team = kicker_team.index_select(0, valid_idx_tensor)
            if ball_batched is not None:
                ball_batched = ball_batched.index_select(0, valid_idx_tensor)

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
                print(
                    f"[WARN] target {target_global} not in candidates for graph {b} "
                    f"(target_team={target_team_repr}, kicker_team={kicker_team_repr}, "
                    f"cand_true_sum={int(cm.sum().item())})"
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
    }
    if hasattr(torch.utils, "tensorboard"):
        pass
    print(
        f"[Train] excluded_not_attacking={excluded_not_attacking} "
        f"excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} "
        f"excluded_invalid_filter={int(stats['excluded_invalid'])} "
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

            cand_masks = build_candidate_mask(
                team=data.get("team"),
                ball=data.get("ball"),
                mask=data.get("mask"),
                x=data.get("x"),
                batch_size=batch_size,
                num_nodes_per_graph=nodes_per_graph,
                device=outputs.device,
            )
            if cand_masks is None:
                cand_masks = torch.ones(batch_size, nodes_per_graph, dtype=torch.bool, device=outputs.device)
            cand_masks = cand_masks.to(dtype=torch.bool)
            team_labels: Optional[torch.Tensor] = None
            ball_batched: Optional[torch.Tensor] = None
            kicker_team: Optional[torch.Tensor] = None

            team_tensor = data.get("team")
            ball_tensor = data.get("ball")
            if team_tensor is not None and ball_tensor is not None:
                team_labels = _reshape_to_batch(team_tensor, batch_size, nodes_per_graph).to(
                    device=outputs.device, dtype=torch.long
                )
                ball_batched = _reshape_to_batch(ball_tensor, batch_size, nodes_per_graph).to(device=outputs.device)

                has_ball_mask = ball_batched > 0.5
                default_kicker_idx = torch.argmax(ball_batched, dim=1)
                first_ball_idx = torch.argmax(has_ball_mask.float(), dim=1)
                kicker_idx = torch.where(
                    has_ball_mask.any(dim=1),
                    first_ball_idx,
                    default_kicker_idx,
                )
                if nodes_per_graph > 0:
                    kicker_idx = kicker_idx.clamp_(0, nodes_per_graph - 1)
                kicker_team = team_labels.gather(1, kicker_idx.unsqueeze(1)).squeeze(1)

                same_team_mask = team_labels == kicker_team.unsqueeze(1)
                cand_masks = cand_masks & same_team_mask
                batch_indices = torch.arange(batch_size, device=outputs.device)
                cand_masks[batch_indices, kicker_idx] = False

                B_filter = targets.shape[0]
                valid_indices: list[int] = []
                for g in range(B_filter):
                    tgt = int(targets[g].item())
                    if tgt < 0 or tgt >= team_labels.size(1):
                        stats["excluded_invalid"] += 1
                    continue
                    cm_g = cand_masks[g]
                    if tgt >= cm_g.size(0):
                        stats["excluded_invalid"] += 1
                    continue
                    tgt_in_cand = bool(cm_g[tgt].item())
                    tt = int(team_labels[g, tgt].item())
                    kt = int(kicker_team[g].item())
                    if tt != kt:
                        stats["invalid_team_mismatch"] += 1
                        if (int(stats["invalid_team_mismatch"]) % 50) == 1:
                            AUDIT_LOGGER.warning(
                                f"[AUDIT] team_mismatch: g={g}, tgt={tgt}, kicker_team={kt}, "
                                f"target_team={tt}, cand_true_sum={int(cand_masks[g].sum().item())} (val-filter)"
                            )
                        stats["excluded_invalid"] += 1
                        continue
                    if not tgt_in_cand:
                        stats["invalid_target_not_in_cand"] += 1
                        if (int(stats["invalid_target_not_in_cand"]) % 50) == 1:
                            AUDIT_LOGGER.warning(
                                f"[AUDIT] target_not_in_cand: g={g}, tgt={tgt}, kicker_team={kt}, "
                                f"cand_true_sum={int(cand_masks[g].sum().item())} (val-filter)"
                            )
                        if 0 <= tgt < cm_g.size(0):
                            cand_masks[g, tgt] = True
                        else:
                            stats["excluded_invalid"] += 1
                            continue
                    valid_indices.append(g)

                if not valid_indices:
                    continue

                valid_idx_tensor = torch.tensor(valid_indices, device=targets.device, dtype=torch.long)
                outputs = outputs.index_select(0, valid_idx_tensor)
                targets = targets.index_select(0, valid_idx_tensor)
                cand_masks = cand_masks.index_select(0, valid_idx_tensor)
                team_labels = team_labels.index_select(0, valid_idx_tensor)
                kicker_team = kicker_team.index_select(0, valid_idx_tensor)
                if ball_batched is not None:
                    ball_batched = ball_batched.index_select(0, valid_idx_tensor)

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
    }
    print(
        f"[Val] excluded_not_attacking={excluded_not_attacking} "
        f"excluded_ball_owner={excluded_ball_owner} excluded_invalid={excluded_invalid} "
        f"excluded_invalid_filter={int(stats['excluded_invalid'])} "
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
        _assert_dataset_version(train_dataset, "train")
        _assert_dataset_version(val_dataset, "val")
    
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
    
    # NOTE: debug: limit epochs to 1, restore Config["train"]["epochs"] when debugging finishes
    for epoch in range(start_epoch, min(start_epoch + 1, config["train"]["epochs"])):
        logger.info(f"Epoch {epoch+1}/{config['train']['epochs']}")
        
        # Learning rate will be updated after train/val via scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, metrics,
            use_amp=config.get("train", {}).get("amp", False),
            profile=args.profile,
        )
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, metrics, logger)
        
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
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"Top-1: {train_metrics['top1']:.4f}, "
                   f"Top-3: {train_metrics['top3']:.4f}, "
            f"Top-5: {train_metrics['top5']:.4f} "
            f"(excluded_invalid={int(train_metrics.get('excluded_invalid', 0))})"
        )
        
        logger.info(
            f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"Top-1: {val_metrics['top1']:.4f}, "
                   f"Top-3: {val_metrics['top3']:.4f}, "
            f"Top-5: {val_metrics['top5']:.4f} "
            f"(excluded_invalid={int(val_metrics.get('excluded_invalid', 0))})"
        )

        logger.info(
            "[Audit summary] train: invalid_team_mismatch=%d, invalid_target_not_in_cand=%d",
            int(train_metrics.get("invalid_team_mismatch", 0)),
            int(train_metrics.get("invalid_target_not_in_cand", 0)),
        )
        logger.info(
            "[Audit summary] val: invalid_team_mismatch=%d, invalid_target_not_in_cand=%d",
            int(val_metrics.get("invalid_team_mismatch", 0)),
            int(val_metrics.get("invalid_target_not_in_cand", 0)),
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
