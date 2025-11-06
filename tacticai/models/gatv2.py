"""GATv2 (Graph Attention Network v2) implementation for TacticAI.

This module implements GATv2 layers and networks for processing football tactical graphs.
GATv2 improves upon GAT by using a more expressive attention mechanism.
Includes 4-view interaction design for D2 equivariance.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..modules.edge_features import compute_edge_features, EdgeFeatureEncoder
from ..modules.view_ops import D2_VIEWS


class GATv2Layer(nn.Module):
    """Simplified GATv2 layer implementation.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension  
        heads: Number of attention heads
        concat: Whether to concatenate or average head outputs
        dropout: Dropout probability
        negative_slope: Negative slope for LeakyReLU
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        edge_feature_dim: int = 0,
        use_edge_features: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim
        
        # Linear transformation for each head
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        
        # TacticAI spec: W1 for source nodes, W2 for destination nodes, U for edge features
        # W1 and W2 are shared across heads, but we need separate for multi-head
        # Actually, we can use W for both, but need separate projections
        # For simplicity, we'll use W for both but apply separately
        self.W1 = nn.Linear(in_features, heads * out_features, bias=False)  # For h_i
        self.W2 = nn.Linear(in_features, heads * out_features, bias=False)  # For h_j
        
        # Attention mechanism: TacticAI spec: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j + U*edge_attr_ij)
        if use_edge_features and edge_feature_dim > 0:
            # U for edge features (linear transformation)
            self.U = nn.Linear(edge_feature_dim, heads * out_features, bias=False)
            # Attention vector: a^T acts on [out_features] (W1*h_i + W2*h_j + U*edge_attr_ij)
            self.att = nn.Parameter(torch.empty(1, heads, out_features))
        else:
            # No edge features: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j)
            self.U = None
            self.att = nn.Parameter(torch.empty(1, heads, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features if concat else out_features))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        if self.U is not None:
            nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if getattr(self, "view_linear", None) is not None:
            nn.init.eye_(self.view_linear.weight)
            if self.view_linear.bias is not None:
                nn.init.constant_(self.view_linear.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass following e_ij = a^T LeakyReLU(W₁h_i + W₂h_j + Ue_ij)."""
        original_2d = x.dim() == 2
        if original_2d:
            x = x.unsqueeze(0)  # [1, N, d_in]

        B, N, _ = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0).expand(B, -1)
            mask = mask.to(device=device, dtype=torch.bool)

        src_idx, dst_idx = edge_index[0], edge_index[1]  # [E], [E]
        E = src_idx.numel()

        x_flat = x.reshape(B * N, -1)  # [B*N, d_in]
        h_src = self.W1(x_flat).view(B, N, self.heads, self.out_features)  # [B, N, H, d_out]
        h_dst = self.W2(x_flat).view(B, N, self.heads, self.out_features)  # [B, N, H, d_out]
        values = self.W(x_flat).view(B, N, self.heads, self.out_features)  # [B, N, H, d_out]

        h_i = h_src[:, src_idx, :, :]  # [B, E, H, d_out]
        h_j = h_dst[:, dst_idx, :, :]  # [B, E, H, d_out]

        if self.use_edge_features and edge_attr is not None and self.U is not None:
            edge_term = self.U(edge_attr).view(1, E, self.heads, self.out_features)  # [1, E, H, d_out]
        else:
            edge_term = 0.0

        att_input = h_i + h_j + edge_term  # [B, E, H, d_out]
        att_input = F.leaky_relu(att_input, negative_slope=self.negative_slope)
        att_logits = (att_input * self.att.view(1, 1, self.heads, self.out_features)).sum(-1)  # [B, E, H]

        dst_mask = mask[:, dst_idx]  # [B, E]
        att_logits = att_logits.masked_fill(~dst_mask.unsqueeze(-1), -1e9)

        att_weights = torch.zeros_like(att_logits)  # [B, E, H]
        for node in range(N):
            edge_mask = dst_idx == node  # [E]
            if not edge_mask.any():
                continue
            node_logits = att_logits[:, edge_mask, :]  # [B, E_node, H]
            node_valid = dst_mask[:, edge_mask].unsqueeze(-1)  # [B, E_node, 1]
            safe_logits = node_logits.masked_fill(~node_valid, -1e9)
            node_weights = torch.softmax(safe_logits, dim=1)  # [B, E_node, H]
            node_weights = node_weights * node_valid.float()
            att_weights[:, edge_mask, :] = node_weights

        if self.training and self.dropout > 0.0:
            att_weights = F.dropout(att_weights, p=self.dropout, training=True)

        value_edges = values[:, src_idx, :, :]  # [B, E, H, d_out]
        weighted_values = att_weights.unsqueeze(-1) * value_edges  # [B, E, H, d_out]

        out = torch.zeros(B, N, self.heads, self.out_features, device=device, dtype=x.dtype)
        for node in range(N):
            edge_mask = dst_idx == node
            if not edge_mask.any():
                continue
            contribution = weighted_values[:, edge_mask, :, :].sum(dim=1)  # [B, H, d_out]
            out[:, node, :, :] = contribution

        out = out * mask.unsqueeze(-1).unsqueeze(-1).float()

        if self.concat:
            out = out.reshape(B, N, self.heads * self.out_features)  # [B, N, H*d_out]
        else:
            out = out.mean(dim=2)  # [B, N, d_out]

        if self.bias is not None:
            out = out + self.bias

        att_return = att_weights if not original_2d else att_weights.squeeze(0)
        if original_2d:
            out = out.squeeze(0)

        if return_attention_weights:
            return out, att_return
        return out
    
    def _add_self_loops(
        self, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor], 
        num_nodes: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add self-loops to edge_index."""
        device = edge_index.device
        self_loops = torch.arange(num_nodes, device=device).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        if edge_attr is not None:
            # Add zero edge attributes for self-loops
            if edge_attr.dim() == 1:
                # Edge attributes are scalars
                self_edge_attr = torch.zeros(num_nodes, device=device)
            else:
                # Edge attributes are vectors
                self_edge_attr = torch.zeros(num_nodes, edge_attr.size(1), device=device)
            edge_attr = torch.cat([edge_attr, self_edge_attr], dim=0)
        
        return edge_index, edge_attr
    
    def _compute_attention(
        self, 
        h1: torch.Tensor, 
        h2: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention coefficients (TacticAI spec).
        
        TacticAI spec: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j + U*edge_attr_ij)
        
        Args:
            h1: Node features [N, heads, out_features] (from W1)
            h2: Node features [N, heads, out_features] (from W2)
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Attention scores [E, heads]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # TacticAI spec: W1*h_i + W2*h_j
        h_src = h1[src]  # [E, heads, out_features] - W1*h_i
        h_dst = h2[dst]  # [E, heads, out_features] - W2*h_j
        h_sum = h_src + h_dst  # [E, heads, out_features]
        
        # Include edge features if available: TacticAI spec: U*edge_attr_ij
        if self.use_edge_features and edge_attr is not None and self.U is not None:
            # Apply U to edge features
            edge_transformed = self.U(edge_attr)  # [E, heads * out_features]
            edge_transformed = edge_transformed.view(edge_attr.size(0), self.heads, self.out_features)  # [E, heads, out_features]
            # TacticAI spec: W1*h_i + W2*h_j + U*edge_attr_ij
            h_sum = h_sum + edge_transformed  # [E, heads, out_features]
        
        # TacticAI spec: a^T LeakyReLU(W1*h_i + W2*h_j + U*edge_attr_ij)
        att_input = F.leaky_relu(h_sum, negative_slope=self.negative_slope)  # [E, heads, out_features]
        att_scores = (self.att * att_input).sum(dim=-1)  # [E, heads]
        
        return att_scores
    
    def _aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor, 
        att_scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate neighbor features using attention weights.
        
        Args:
            h: Node features [N, heads, out_features]
            edge_index: Edge connectivity [2, E]
            att_scores: Attention scores [E, heads] (already masked)
            mask: Node mask [N] (optional)
            
        Returns:
            Aggregated features [N, heads, out_features]
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = h.size(0)
        num_heads = self.heads
        out_features = h.size(2)
        
        # Normalize attention scores (softmax over j for each destination node i)
        # Use scatter to compute softmax efficiently
        # First, compute exp(att_scores) and sum per destination
        att_exp = att_scores.exp()  # [E, heads]
        
        # Compute normalization per destination using scatter
        # For each destination, sum all incoming attention exponentials
        att_exp_sum = torch.zeros(num_nodes, num_heads, device=att_scores.device, dtype=att_scores.dtype)
        att_exp_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, num_heads), att_exp)
        
        # Normalize: att_weights = att_exp / att_exp_sum[dst]
        att_weights = att_exp / (att_exp_sum[dst] + 1e-9)  # [E, heads]
        
        # Initialize output
        out = torch.zeros(num_nodes, num_heads, out_features, device=h.device, dtype=h.dtype)
        
        # Aggregate features for each head
        for i in range(num_heads):
            # Get source features for this head
            src_features = h[src, i, :]  # [E, out_features]
            
            # Apply attention weights
            weighted_features = src_features * att_weights[:, i:i+1]  # [E, out_features]
            
            # Aggregate to destination nodes using scatter_add (much faster than loops)
            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, out_features), weighted_features)
        
        # Zero out missing nodes
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand_as(out)  # [N, heads, out_features]
            out = out * mask_expanded
        
        return out


class GATv2Layer4View(nn.Module):
    """GATv2 layer with 4-view interaction for D2 equivariance.
    
    Implements intra-view GAT followed by inter-view mixing with weight sharing.
    Each layer processes h ∈ [B, V=4, N, D] where V represents 4 views:
    - View 0: Original coordinates
    - View 1: Horizontal flip (x → -x)
    - View 2: Vertical flip (y → -y)  
    - View 3: Both flips (x → -x, y → -y)
    
    Weight sharing ensures D2 equivariance: same parameters used across all views.
    Coordinate transformations are handled implicitly through the 4-view structure.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension  
        heads: Number of attention heads
        concat: Whether to concatenate or average head outputs
        dropout: Dropout probability
        negative_slope: Negative slope for LeakyReLU
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
        view_mixing: Inter-view mixing strategy ('attention', 'conv1x1', or 'none')
        weight_sharing: Whether to use weight sharing across views (default: True)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        view_mixing: str = "attention",
        weight_sharing: bool = True,
        edge_feature_dim: int = 1,  # TacticAI spec: same_team only (default=1)
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.view_mixing = view_mixing
        self.weight_sharing = weight_sharing
        if view_mixing not in {"attention", "conv1x1", "none"}:
            raise ValueError(
                "view_mixing must be one of {'attention', 'conv1x1', 'none'}; "
                f"got {view_mixing!r}"
            )

        # Linear transformation for each head (shared across views)
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        
        # TacticAI spec: W1 for source nodes, W2 for destination nodes, U for edge features
        self.W1 = nn.Linear(in_features, heads * out_features, bias=False)  # For h_i
        self.W2 = nn.Linear(in_features, heads * out_features, bias=False)  # For h_j
        
        # Attention mechanism (TacticAI spec: includes edge features)
        # e_ij = a^T LeakyReLU(W₁ h_i + W₂ h_j + U * edge_attr_ij)
        # Shape: [1, heads, out_features] (not 2*out_features + edge_feature_dim)
        self.edge_feature_dim = edge_feature_dim
        if edge_feature_dim > 0:
            # U for edge features (linear transformation)
            self.U = nn.Linear(edge_feature_dim, heads * out_features, bias=False)
            # Attention vector: a^T acts on [out_features] (W1*h_i + W2*h_j + U*edge_attr_ij)
            self.att = nn.Parameter(torch.empty(1, heads, out_features))
        else:
            # No edge features: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j)
            self.U = None
            self.att = nn.Parameter(torch.empty(1, heads, out_features))
        self._att_embed_dim = heads * out_features if concat else out_features
        attn_heads = heads if concat else 1
        
        # Inter-view mixing operators
        if view_mixing == "attention":
            self.view_att = nn.MultiheadAttention(
                embed_dim=self._att_embed_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.view_linear = None
        elif view_mixing == "conv1x1":
            self.view_att = None
            self.view_linear = nn.Linear(4, 4, bias=True)
        else:
            self.view_att = None
            self.view_linear = None
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features if concat else out_features))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        if self.U is not None:
            nn.init.xavier_uniform_(self.U.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with 4-view interaction.
        
        Args:
            x: Node features [B, V=4, N, in_features]
                If batch processing: N is per-graph nodes, edge_index spans all graphs
            edge_index: Edge connectivity [2, E] - can be per-graph or batched
            edge_attr: Edge features [E, edge_dim] (optional)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [B, V=4, N, out_features * heads] or [B, V=4, N, out_features]
            Attention weights [E, heads] (optional)
        """
        B, V, N, _ = x.shape
        
        # Reshape for intra-view processing: [B*V, N, in_features]
        # For batched processing, N is per-graph nodes, but edge_index spans B*N nodes
        x_flat = x.view(B * V, N, -1)
        
        # Intra-view GAT processing
        # For batched edge_index, we need to handle it per view-batch combination
        # Process each (view, batch) combination separately but in parallel
        h_flat = self._intra_view_gat_batched(x_flat, edge_index, edge_attr, B, V, N)
        
        # Reshape back: [B, V, N, out_features]
        h = h_flat.view(B, V, N, -1)
        
        # Inter-view mixing
        h_mixed = self._inter_view_mixing(h)
        
        # Add bias
        if self.bias is not None:
            h_mixed = h_mixed + self.bias
        
        if return_attention_weights:
            return h_mixed, None  # TODO: Return proper attention weights
        return h_mixed
    
    def _intra_view_gat_batched(
        self, 
        x: torch.Tensor,  # [B*V, N_per_graph, in_features]
        edge_index: torch.Tensor,  # [2, E_total] - edges for all graphs
        edge_attr: Optional[torch.Tensor],
        B: int,
        V: int,
        N_per_graph: int,
    ) -> torch.Tensor:
        """Intra-view GAT processing for batched graphs.
        
        Args:
            x: Node features [B*V, N_per_graph, in_features]
            edge_index: Edge connectivity [2, E_total] spanning all B graphs
            edge_attr: Edge features [E_total, edge_dim] (optional)
            B: Number of graphs in batch
            V: Number of views (4)
            N_per_graph: Number of nodes per graph
            
        Returns:
            Output features [B*V, N_per_graph, out_features]
        """
        N_total = B * N_per_graph  # Total nodes across all graphs
        
        # Linear transformation: TacticAI spec - W1 and W2 separately
        h1 = self.W1(x)  # [B*V, N_per_graph, heads * out_features]
        h1 = h1.view(B * V, N_per_graph, self.heads, self.out_features)  # [B*V, N_per_graph, heads, out_features]
        h2 = self.W2(x)  # [B*V, N_per_graph, heads * out_features]
        h2 = h2.view(B * V, N_per_graph, self.heads, self.out_features)  # [B*V, N_per_graph, heads, out_features]
        
        # For batched processing, we need to expand h1 and h2 to [B*V, N_total, ...] to match edge_index
        # Fully vectorized: create h1_full and h2_full using scatter_add or advanced indexing
        h1_full = torch.zeros(B * V, N_total, self.heads, self.out_features, device=h1.device, dtype=h1.dtype)
        h2_full = torch.zeros(B * V, N_total, self.heads, self.out_features, device=h2.device, dtype=h2.dtype)
        # Create all indices at once (vectorized)
        # For each batch b and view v, copy h[b*V+v, :] to h_full[b*V+v, b*N_per_graph:(b+1)*N_per_graph]
        batch_indices = torch.arange(B, device=h1.device)  # [B]
        view_indices = torch.arange(V, device=h1.device)  # [V]
        # Create meshgrid: [B, V] for batch-view combinations
        b_grid, v_grid = torch.meshgrid(batch_indices, view_indices, indexing='ij')  # [B, V] each
        flat_b = b_grid.flatten()  # [B*V]
        flat_v = v_grid.flatten()  # [B*V]
        # Compute source and destination indices
        src_idx = flat_b * V + flat_v  # [B*V] - index in h
        # Destination node ranges
        dst_node_start = flat_b * N_per_graph  # [B*V]
        dst_node_end = (flat_b + 1) * N_per_graph  # [B*V]
        # Fully vectorized copy using scatter or advanced indexing
        # h is [B*V, N_per_graph, heads, out_features]
        # h_full is [B*V, N_total, heads, out_features]
        # For each row i in h_full, we need to copy h[src_idx[i]] to positions [dst_node_start[i]:dst_node_start[i]+N_per_graph]
        # Use a loop for correctness (B*V is typically small: 4*32=128)
        for i in range(B * V):
            src_idx_val = src_idx[i].item()
            dst_start = dst_node_start[i].item()
            h1_full[i, dst_start:dst_start+N_per_graph] = h1[src_idx_val]
            h2_full[i, dst_start:dst_start+N_per_graph] = h2[src_idx_val]
        
        # Self-loops are already included in edge_index (TacticAI spec: 22×22 complete graph)
        # Do not add self-loops again if they're already present
        # Note: edge_index should already contain self-loops from data preprocessing
        
        # Compute attention coefficients: TacticAI spec: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j + U*edge_attr_ij)
        att_scores = self._compute_attention_batched(h1_full, h2_full, edge_index, edge_attr, B, V)  # [E, heads]
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate (use h1_full for aggregation)
        out_full = self._aggregate(h1_full, edge_index, att_scores)  # [B*V, N_total, heads, out_features]
        
        # Extract per-graph outputs (fully vectorized)
        # Create indices for all batch-view combinations
        row_idx = torch.arange(B * V, device=out_full.device).unsqueeze(1).expand(-1, N_per_graph)  # [B*V, N_per_graph]
        col_idx = (flat_b * N_per_graph).unsqueeze(1) + torch.arange(N_per_graph, device=out_full.device).unsqueeze(0)  # [B*V, N_per_graph]
        # Extract using advanced indexing
        out = out_full[row_idx, col_idx]  # [B*V, N_per_graph, heads, out_features]
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(B * V, N_per_graph, self.heads * self.out_features)
        else:
            out = out.mean(dim=2)  # [B*V, N_per_graph, out_features]
        
        return out
    
    def _compute_attention_batched(
        self, 
        h1: torch.Tensor,  # [B*V, N_total, heads, out_features] - W1*h
        h2: torch.Tensor,  # [B*V, N_total, heads, out_features] - W2*h
        edge_index: torch.Tensor,  # [2, E_total]
        edge_attr: Optional[torch.Tensor],
        B: int,
        V: int,
    ) -> torch.Tensor:
        """Compute attention coefficients for batched graphs (TacticAI spec).
        
        TacticAI spec: e_ij = a^T LeakyReLU(W₁ h_i + W₂ h_j + U * edge_attr_ij)
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Process all views and batches in parallel
        # Reshape h1 and h2 to [B*V*N_total, heads, out_features]
        N_total = h1.size(1)
        h1_flat = h1.view(B * V * N_total, self.heads, h1.size(3))  # [B*V*N_total, heads, out_features]
        h2_flat = h2.view(B * V * N_total, self.heads, h2.size(3))  # [B*V*N_total, heads, out_features]
        
        # Get source and destination features
        h1_src = h1_flat[src]  # [E, heads, out_features] - W1*h_i
        h2_dst = h2_flat[dst]  # [E, heads, out_features] - W2*h_j
        
        # TacticAI spec: W₁ h_i + W₂ h_j
        h_sum = h1_src + h2_dst  # [E, heads, out_features]
        
        # Include edge features if available (TacticAI spec: U * edge_attr_ij)
        if edge_attr is not None and self.U is not None:
            # Apply U to edge features
            edge_transformed = self.U(edge_attr)  # [E, heads * out_features]
            edge_transformed = edge_transformed.view(edge_attr.size(0), self.heads, self.out_features)  # [E, heads, out_features]
            # TacticAI spec: W₁ h_i + W₂ h_j + U * edge_attr_ij
            h_sum = h_sum + edge_transformed  # [E, heads, out_features]
        
        # TacticAI spec: a^T LeakyReLU(W₁ h_i + W₂ h_j + U * edge_attr_ij)
        att_input = F.leaky_relu(h_sum, negative_slope=self.negative_slope)  # [E, heads, out_features]
        att_scores = (self.att * att_input).sum(dim=-1)  # [E, heads]
        
        return att_scores
    
    def _intra_view_gat(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Intra-view GAT processing (legacy single-graph version)."""
        N = x.size(1)
        
        # Linear transformation: TacticAI spec - W1 and W2 separately
        h1 = self.W1(x)  # [B*V, N, heads * out_features]
        h1 = h1.view(-1, N, self.heads, self.out_features)  # [B*V, N, heads, out_features]
        h2 = self.W2(x)  # [B*V, N, heads * out_features]
        h2 = h2.view(-1, N, self.heads, self.out_features)  # [B*V, N, heads, out_features]
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, N)
        
        # Compute attention coefficients: TacticAI spec: e_ij = a^T LeakyReLU(W1*h_i + W2*h_j + U*edge_attr_ij)
        # Reshape for _compute_attention: [B*V*N, heads, out_features]
        B_V = h1.size(0)
        h1_flat = h1.view(B_V * N, self.heads, self.out_features)
        h2_flat = h2.view(B_V * N, self.heads, self.out_features)
        att_scores = self._compute_attention(h1_flat, h2_flat, edge_index, edge_attr)  # [E, heads]
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate (use h1 for aggregation)
        out = self._aggregate(h1_flat, edge_index, att_scores)  # [B*V*N, heads, out_features]
        out = out.view(B_V, N, self.heads, self.out_features)  # [B*V, N, heads, out_features]
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, N, self.heads * self.out_features)
        else:
            out = out.mean(dim=2)  # [B*V, N, out_features]
        
        return out
    
    def _inter_view_mixing(self, h: torch.Tensor) -> torch.Tensor:
        """Inter-view mixing using specified method.
        
        Args:
            h: Node features [B, V=4, N, D]
            
        Returns:
            Mixed features [B, V=4, N, D]
        """
        B, V, N, D = h.shape
        
        if self.view_mixing == "attention" and self.view_att is not None:
            h_att = h.permute(0, 2, 1, 3).contiguous().view(B * N, V, D)
            if D != self._att_embed_dim:
                raise ValueError(
                    f"Expected feature dimension {self._att_embed_dim} for attention mixing, got {D}."
                )
            h_mixed, _ = self.view_att(h_att, h_att, h_att, need_weights=False)
            h_mixed = h_mixed.view(B, N, V, D).permute(0, 2, 1, 3).contiguous()
        elif self.view_mixing == "conv1x1" and self.view_linear is not None:
            h_lin = h.permute(0, 2, 3, 1)  # [B, N, D, V]
            h_lin = self.view_linear(h_lin)
            h_mixed = h_lin.permute(0, 3, 1, 2).contiguous()
        else:
            h_mixed = h

        if self.training and self.dropout > 0:
            h_mixed = F.dropout(h_mixed, p=self.dropout, training=True)

        return h_mixed
    
    def _add_self_loops(
        self, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor], 
        num_nodes: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add self-loops to edge_index."""
        device = edge_index.device
        self_loops = torch.arange(num_nodes, device=device).repeat(2, 1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        
        if edge_attr is not None:
            # Add zero edge attributes for self-loops
            if edge_attr.dim() == 1:
                # Edge attributes are scalars
                self_edge_attr = torch.zeros(num_nodes, device=device)
            else:
                # Edge attributes are vectors
                self_edge_attr = torch.zeros(num_nodes, edge_attr.size(1), device=device)
            edge_attr = torch.cat([edge_attr, self_edge_attr], dim=0)
        
        return edge_index, edge_attr
    
    def _compute_attention(
        self, 
        h: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention coefficients.
        
        Args:
            h: Node features [B*V, N, heads, out_features]
            edge_index: Edge connectivity [2, E]
            
        Returns:
            Attention scores [E, heads]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Concatenate source and destination features
        h_cat = torch.cat([h[:, src], h[:, dst]], dim=-1)  # [B*V, E, heads, 2 * out_features]

        # Compute attention: a^T * LeakyReLU(W * [h_i || h_j])
        att_input = F.leaky_relu(h_cat, negative_slope=self.negative_slope)
        att_scores = (self.att * att_input).sum(dim=-1)  # [B*V, E, heads]

        return att_scores
    
    def _aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor, 
        att_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbor features using attention weights.
        
        Args:
            h: Node features [B*V, N, heads, out_features]
            edge_index: Edge connectivity [2, E]
            att_scores: Attention scores [E, heads]
            
        Returns:
            Aggregated features [B*V, N, heads, out_features]
        """
        src, dst = edge_index[0], edge_index[1]
        num_samples = h.size(0)  # B*V
        num_nodes = h.size(1)  # N
        num_heads = self.heads
        out_features = h.size(3)
        
        # Normalize attention scores per destination node (like GATv2Layer)
        # att_scores is [E, heads]
        att_exp = att_scores.exp()  # [E, heads]
        
        # Initialize output
        out = torch.zeros(num_samples, num_nodes, num_heads, out_features, device=h.device, dtype=h.dtype)
        
        # For each sample, aggregate separately (edge_index spans all graphs, need to filter per sample)
        # But since edge_index is batched, we need to handle it correctly
        # Actually, edge_index is for all graphs combined, so we need to process per sample
        for sample in range(num_samples):
            # Get attention weights for this sample (same for all samples since edge_index is batched)
            # Normalize per destination node
            att_exp_sum = torch.zeros(num_nodes, num_heads, device=att_scores.device, dtype=att_scores.dtype)
            att_exp_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, num_heads), att_exp)
            att_weights = att_exp / (att_exp_sum[dst] + 1e-9)  # [E, heads]
            
            # Aggregate for each head
            for head in range(num_heads):
                src_features = h[sample, src, head, :]  # [E, out_features]
                weighted = (src_features * att_weights[:, head:head + 1]).to(h.dtype)  # [E, out_features], ensure dtype match
                
                # Aggregate to destination nodes using scatter_add
                # out[sample, :, head, :] is [N_total, out_features]
                # dst is [E_total], weighted is [E_total, out_features]
                out[sample, :, head, :].scatter_add_(0, dst.unsqueeze(-1).expand(-1, out_features), weighted)

        return out


class GATv2Network(nn.Module):
    """Multi-layer GATv2 network with residual connections and global readout.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (if None, no final projection)
        num_layers: Number of GATv2 layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        readout: Global readout method ('mean', 'sum', 'max', or None)
        residual: Whether to use residual connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        readout: str = "mean",
        residual: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.readout = readout
        self.residual = residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            concat = (i < num_layers - 1)  # Only last layer doesn't concatenate
            layer = GATv2Layer(
                in_features=hidden_dim,
                out_features=hidden_dim // num_heads if concat else hidden_dim,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
            )
            self.gat_layers.append(layer)
        
        # Output projection
        if output_dim is not None:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features [N, input_dim] or [B, N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional, for global readout)
            mask: Node mask [B, N] where 1=valid, 0=missing (optional)
            
        Returns:
            If batch is None: Node embeddings [N, output_dim]
            If batch is provided: (node_embeddings, graph_embeddings)
        """
        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim] or [B, N, hidden_dim]
        
        # GATv2 layers with residual connections
        for i, layer in enumerate(self.gat_layers):
            h_new = layer(h, edge_index, edge_attr, mask=mask)
            
            # Residual connection (except first layer)
            if self.residual and i > 0 and h_new.size(-1) == h.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            # Apply dropout and activation
            if i < len(self.gat_layers) - 1:  # No activation after last layer
                h = F.elu(h)
                h = self.dropout_layer(h)
        
        # Output projection
        node_embeddings = self.output_proj(h)  # [N, output_dim] or [B, N, output_dim]
        
        # Global readout if batch is provided
        if batch is not None:
            graph_embeddings = self._global_readout(node_embeddings, batch)
            return node_embeddings, graph_embeddings
        
        return node_embeddings
    
    def _global_readout(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute global graph embeddings.
        
        Args:
            x: Node embeddings [N, output_dim]
            batch: Batch assignment [N]
            
        Returns:
            Graph embeddings [B, output_dim]
        """
        if self.readout == "mean":
            return scatter_mean(x, batch, dim=0)
        elif self.readout == "sum":
            return scatter_sum(x, batch, dim=0)
        elif self.readout == "max":
            return scatter_max(x, batch, dim=0)[0]
        else:
            raise ValueError(f"Unknown readout method: {self.readout}")


class GATv2Network4View(nn.Module):
    """Multi-layer GATv2 network with 4-view interaction for D2 equivariance.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (if None, no final projection)
        num_layers: Number of GATv2 layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        readout: Global readout method ('mean', 'sum', 'max', or None)
        residual: Whether to use residual connections
        view_mixing: Type of inter-view mixing ('attention', 'conv1x1', 'none')
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        readout: str = "mean",
        residual: bool = True,
        view_mixing: str = "attention",
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.readout = readout
        self.residual = residual
        self.view_mixing = view_mixing
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GATv2 layers with 4-view interaction
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            concat = (i < num_layers - 1)  # Only last layer doesn't concatenate
            layer = GATv2Layer4View(
                in_features=hidden_dim,
                out_features=hidden_dim // num_heads if concat else hidden_dim,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
                view_mixing=view_mixing,
                edge_feature_dim=1,  # TacticAI spec: same_team only
                add_self_loops=False,  # Self-loops already in edge_index (22×22 complete graph)
            )
            self.gat_layers.append(layer)
        
        # Output projection
        if output_dim is not None:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with 4-view interaction.
        
        Args:
            x: Node features [B, V=4, N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional, for global readout)
            
        Returns:
            If batch is None: Node embeddings [B, V=4, N, output_dim]
            If batch is provided: (node_embeddings, graph_embeddings)
        """
        if x.dim() != 4 or x.size(1) != len(D2_VIEWS):
            raise ValueError(
                f"Expected input shaped [B, {len(D2_VIEWS)}, N, D] for four views, "
                f"got {tuple(x.shape)}."
            )
        # Input projection
        h = self.input_proj(x)  # [B, V=4, N, hidden_dim]
        
        # GATv2 layers with residual connections
        for i, layer in enumerate(self.gat_layers):
            h_new = layer(h, edge_index, edge_attr)
            
            # Residual connection (except first layer)
            if self.residual and i > 0 and h_new.size(-1) == h.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            # Apply dropout and activation
            if i < len(self.gat_layers) - 1:  # No activation after last layer
                h = F.elu(h)
                h = self.dropout_layer(h)
        
        # Output projection
        node_embeddings = self.output_proj(h)  # [B, V=4, N, output_dim]
        
        # Global readout if batch is provided
        if batch is not None:
            graph_embeddings = self._global_readout_4view(node_embeddings, batch)
            return node_embeddings, graph_embeddings
        
        return node_embeddings
    
    def _global_readout_4view(
        self, 
        x: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute global graph embeddings for 4-view data.
        
        Args:
            x: Node embeddings [B, V=4, N, output_dim]
            batch: Batch assignment [N]
            
        Returns:
            Graph embeddings [B, V=4, output_dim]
        """
        B, V, N, D = x.shape
        
        # Reshape for readout: [B*V, N, D]
        x_flat = x.view(B * V, N, D)
        
        # Expand batch for each view
        batch_expanded = batch.unsqueeze(0).expand(V, -1).contiguous().view(-1)
        
        # Apply readout
        if self.readout == "mean":
            graph_embeddings = scatter_mean(x_flat, batch_expanded, dim=0)
        elif self.readout == "sum":
            graph_embeddings = scatter_sum(x_flat, batch_expanded, dim=0)
        elif self.readout == "max":
            graph_embeddings = scatter_max(x_flat, batch_expanded, dim=0)[0]
        else:
            raise ValueError(f"Unknown readout method: {self.readout}")
        
        # Reshape back: [B, V, output_dim]
        graph_embeddings = graph_embeddings.view(B, V, D)
        
        return graph_embeddings


# Fallback implementation without torch_scatter
def scatter_mean(x, batch, dim=0):
    """Fallback scatter_mean implementation."""
    unique_batches = torch.unique(batch)
    result = []
    for b in unique_batches:
        mask = batch == b
        result.append(x[mask].mean(dim=0))
    return torch.stack(result)

def scatter_sum(x, batch, dim=0):
    """Fallback scatter_sum implementation."""
    unique_batches = torch.unique(batch)
    result = []
    for b in unique_batches:
        mask = batch == b
        result.append(x[mask].sum(dim=0))
    return torch.stack(result)

def scatter_max(x, batch, dim=0):
    """Fallback scatter_max implementation."""
    unique_batches = torch.unique(batch)
    result = []
    for b in unique_batches:
        mask = batch == b
        result.append(x[mask].max(dim=0)[0])
    return torch.stack(result), torch.stack(result)
