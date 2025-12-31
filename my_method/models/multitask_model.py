"""Multi-task learning model for receiver and shot prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .gatv2 import GATv2Network, GATv2Network4View
from .mlp_heads import ReceiverHead, ShotHeadNodeBased
from ..modules.view_ops import D2_VIEWS, apply_view_transform


class MultiTaskModel(nn.Module):
    """Multi-task model for receiver prediction (node-level) and shot prediction (graph-level).
    
    Architecture:
        - Shared backbone (GATv2Network or GATv2Network4View)
        - Receiver head: node-level classification (22 classes)
        - Shot head: graph-level binary classification (mean pooling + MLP)
    
    The model learns:
        - Node-level representations optimized for receiver prediction
        - Graph-level representations optimized for shot prediction (for retrieval)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-task model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        model_config = config["model"]
        d2_config = config.get("d2", {})
        self.use_d2_equivariance = d2_config.get("enabled", False)
        multitask_config = config.get("multitask", {})
        # When 0, consistency is effectively disabled (and we skip its computation).
        self.lambda_consistency = float(multitask_config.get("lambda_consistency", 0.0))
        
        # Create shared backbone
        edge_dim = model_config.get("edge_dim", 1)
        if self.use_d2_equivariance:
            self.backbone = GATv2Network4View(
                input_dim=model_config["input_dim"],
                hidden_dim=model_config["hidden_dim"],
                output_dim=model_config["hidden_dim"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                dropout=model_config["dropout"],
                readout=None,  # Return node embeddings, not graph-level
                residual=True,
                view_mixing="attention",
                edge_feature_dim=edge_dim,
            )
        else:
            self.backbone = GATv2Network(
                input_dim=model_config["input_dim"],
                hidden_dim=model_config["hidden_dim"],
                output_dim=model_config["hidden_dim"],
                num_layers=model_config["num_layers"],
                num_heads=model_config["num_heads"],
                dropout=model_config["dropout"],
                readout=None,  # Return node embeddings, not graph-level
                residual=True,
                edge_feature_dim=edge_dim,
            )
        
        # Receiver head (node-level classification)
        self.receiver_head = ReceiverHead(
            input_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
            num_layers=model_config.get("mlp_num_layers", 2),
        )
        
        # Shot head (graph-level binary classification)
        # Use ShotHeadNodeBased but aggregate to graph level
        self.shot_head = ShotHeadNodeBased(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            dropout=model_config["dropout"],
            use_context=False,  # Simple aggregation without context
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        team: Optional[torch.Tensor] = None,
        ball: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task prediction.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch indices [N] (optional)
            mask: Node mask [N] (optional, for receiver prediction filtering)
            team: Team IDs [N] (optional, for receiver prediction filtering)
            ball: Ball possession [N] (optional, for receiver prediction filtering)
            
        Returns:
            Dictionary containing:
                - 'receiver_logits': Receiver prediction logits [N_attacking, num_classes] or [N, num_classes]
                - 'shot_logit': Shot prediction logit [B, 1]
        """
        B = batch.max().item() + 1 if batch is not None else 1

        consistency_loss: Optional[torch.Tensor] = None
        
        # Get node embeddings from backbone
        if self.use_d2_equivariance:
            # D2 equivariance: Create 4 views
            views_list = []
            for view_idx in range(len(D2_VIEWS)):
                x_view = x.clone()
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(0, 1))  # x, y
                if x_view.size(-1) > 3:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(2, 3))  # vx, vy
                if x_view.size(-1) > 8:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(7, 8))  # dx_to_kicker, dy_to_kicker
                if x_view.size(-1) > 12:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(11, 12))  # dx_to_goal, dy_to_goal
                views_list.append(x_view)
            
            x_views = torch.stack(views_list, dim=0)  # [4, N, D]
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            x_4view = x_views.view(4, B, num_nodes_per_graph, -1).permute(1, 0, 2, 3)  # [B, 4, N_per_graph, D]
            
            node_emb_4view = self.backbone(x_4view, edge_index, edge_attr)  # [B, 4, N_per_graph, hidden_dim]

            # Consistency regularization across the 4 views (average of pairwise MSE).
            if self.lambda_consistency > 0.0:
                V = int(node_emb_4view.size(1))
                cons = torch.tensor(0.0, device=node_emb_4view.device, dtype=node_emb_4view.dtype)
                pairs = 0
                for i in range(V):
                    for j in range(i + 1, V):
                        cons = cons + F.mse_loss(node_emb_4view[:, i], node_emb_4view[:, j], reduction="mean")
                        pairs += 1
                if pairs > 0:
                    consistency_loss = cons / float(pairs)

            H = node_emb_4view.mean(dim=1)  # Average over 4 views: [B, N_per_graph, hidden_dim]
        else:
            # Standard mode: No D2 equivariance
            H = self.backbone(x, edge_index, edge_attr, batch=None)  # [N, hidden_dim]
            
            # Reshape to [B, N_per_graph, hidden_dim]
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            if H.dim() == 2:
                H = H.view(B, num_nodes_per_graph, -1)
            elif H.dim() == 3:
                pass  # Already in [B, N_per_graph, hidden_dim] format
            else:
                raise ValueError(f"Unexpected H shape: {H.shape}, expected 2D or 3D")
        
        # Receiver prediction (node-level)
        # NOTE:
        # ReceiverHead is NodeScoreHead (per-node scalar), so output is [B, N].
        # Candidate masking / target mapping is handled in the training loop (like train_receiver.py).
        receiver_logits_nodes = self.receiver_head(H)  # [B, N]
        
        # Shot prediction (graph-level)
        # Normalize H to prevent numerical instability
        H_normalized = F.layer_norm(H, normalized_shape=(H.size(-1),), eps=1e-5)
        
        # Get shot logits per node
        shot_logits_per_node = self.shot_head(H_normalized)  # [B, N]
        
        # Aggregate to graph level (mean pooling)
        shot_logit = shot_logits_per_node.mean(dim=1, keepdim=True)  # [B, 1]
        
        out: Dict[str, torch.Tensor] = {
            'receiver_logits': receiver_logits_nodes,
            'shot_logit': shot_logit,
        }
        if consistency_loss is not None:
            out["consistency_loss"] = consistency_loss
        return out

