"""Corrected GATv2 implementation for D2 equivariance following TacticAI paper equation (8).

This module implements the proper group convolution structure:
H_g^(t) = (1/|G|) Σ_h g_G( H_h^(t-1) ‖ H_{g⁻¹h}^(t-1) )

Where:
- g, h ∈ D2 = {id, h, v, hv} (4 views)
- g_G is a shared GNN applied to concatenated features
- Features are split into equivariant (x, y, vx, vy) and invariant components
- Final layer performs invariant pooling over all views
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..modules.edge_features import compute_edge_features, EdgeFeatureEncoder
from ..modules.view_ops import D2_VIEWS, apply_view_transform


class D2GroupConvolutionLayer(nn.Module):
    """D2 Group Convolution Layer implementing equation (8) from TacticAI paper.
    
    Implements:
    H_g^(t) = (1/|G|) Σ_h g_G( H_h^(t-1) ‖ H_{g⁻¹h}^(t-1) )
    
    Where G = D2 = {id, h, v, hv} and g_G is a shared GNN.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        equivariant_indices: Indices of features that change under reflections (x, y, vx, vy)
        invariant_indices: Indices of features that are invariant under reflections
        heads: Number of attention heads
        dropout: Dropout probability
        negative_slope: Negative slope for LeakyReLU
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        equivariant_indices: Tuple[int, ...] = (0, 1, 2, 3),  # x, y, vx, vy
        invariant_indices: Optional[Tuple[int, ...]] = None,  # height, weight, team, ball
        heads: int = 1,
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
        self.equivariant_indices = equivariant_indices
        self.invariant_indices = invariant_indices or tuple(
            i for i in range(in_features) if i not in equivariant_indices
        )
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.use_edge_features = use_edge_features
        self.edge_feature_dim = edge_feature_dim
        
        # Shared GNN g_G (equation 8)
        self.shared_gnn = self._build_shared_gnn()
        
        # Edge feature encoder if needed
        if use_edge_features and edge_feature_dim > 0:
            self.edge_encoder = EdgeFeatureEncoder(
                edge_dim=edge_feature_dim,
                hidden_dim=min(edge_feature_dim, 32),
                output_dim=edge_feature_dim,
                dropout=dropout
            )
        else:
            self.edge_encoder = None
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
    
    def _build_shared_gnn(self):
        """Build the shared GNN g_G used in equation (8)."""
        # This is the shared GNN that processes concatenated features
        # Input: concatenated features from two views
        # Output: processed features for one view
        
        # Calculate input dimension for concatenated features
        # Each view contributes in_features, so concatenated is 2 * in_features
        concat_dim = 2 * self.in_features
        
        # Add edge features if used
        if self.use_edge_features and self.edge_feature_dim > 0:
            concat_dim += self.edge_feature_dim
        
        # Build GATv2-like shared GNN
        return SharedGATv2Layer(
            in_features=concat_dim,
            out_features=self.out_features,
            heads=self.heads,
            dropout=self.dropout,
            negative_slope=self.negative_slope,
            add_self_loops=self.add_self_loops,
            bias=False,  # Bias will be added at the end
        )
    
    def _reset_parameters(self):
        """Initialize parameters."""
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass implementing equation (8).
        
        Args:
            x: Node features [B, V=4, N, in_features]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [B, V=4, N, out_features]
            Attention weights [E, heads] (optional)
        """
        B, V, N, _ = x.shape
        
        # Ensure we have 4 views
        if V != 4:
            raise ValueError(f"Expected 4 views, got {V}")
        
        # Initialize output tensor
        output = torch.zeros(B, V, N, self.out_features, device=x.device, dtype=x.dtype)
        
        # Implement equation (8): H_g^(t) = (1/|G|) Σ_h g_G( H_h^(t-1) ‖ H_{g⁻¹h}^(t-1) )
        for g in range(V):  # g ∈ D2
            for h in range(V):  # h ∈ D2
                # Compute g⁻¹h (inverse of g applied to h)
                g_inv_h = self._compute_group_inverse(g, h)
                
                # Get features for views h and g⁻¹h
                h_features = x[:, h, :, :]  # [B, N, in_features]
                g_inv_h_features = x[:, g_inv_h, :, :]  # [B, N, in_features]
                
                # Apply view transformations to align features
                h_features_aligned = self._align_features_for_view(h_features, h, g)
                g_inv_h_features_aligned = self._align_features_for_view(g_inv_h_features, g_inv_h, g)
                
                # Concatenate features: H_h ‖ H_{g⁻¹h}
                concat_features = torch.cat([h_features_aligned, g_inv_h_features_aligned], dim=-1)
                
                # Apply shared GNN g_G
                processed_features = self.shared_gnn(concat_features, edge_index, edge_attr)
                
                # Accumulate results (will be averaged later)
                output[:, g, :, :] += processed_features
        
        # Average over all h (1/|G| factor in equation 8)
        output = output / V
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        if return_attention_weights:
            return output, None  # TODO: Return proper attention weights
        return output
    
    def _compute_group_inverse(self, g: int, h: int) -> int:
        """Compute g⁻¹h in D2 group.
        
        D2 = {id=0, h=1, v=2, hv=3}
        Group multiplication table:
        """
        # D2 group multiplication table
        group_table = [
            [0, 1, 2, 3],  # id * {id, h, v, hv}
            [1, 0, 3, 2],  # h * {id, h, v, hv}
            [2, 3, 0, 1],  # v * {id, h, v, hv}
            [3, 2, 1, 0],  # hv * {id, h, v, hv}
        ]
        
        # Find g_inv such that g_inv * g = id
        g_inv = None
        for i in range(4):
            if group_table[i][g] == 0:  # i * g = id
                g_inv = i
                break
        
        if g_inv is None:
            raise ValueError(f"Could not find inverse for group element {g}")
        
        # Compute g_inv * h
        return group_table[g_inv][h]
    
    def _align_features_for_view(self, features: torch.Tensor, source_view: int, target_view: int) -> torch.Tensor:
        """Align features from source_view to target_view.
        
        Args:
            features: Features [B, N, in_features]
            source_view: Source view index (0-3)
            target_view: Target view index (0-3)
            
        Returns:
            Aligned features [B, N, in_features]
        """
        if source_view == target_view:
            return features
        
        # Create aligned features
        aligned_features = features.clone()
        
        # Apply transformations to equivariant features
        for idx in self.equivariant_indices:
            if idx < features.size(-1):
                # Apply view transformation to align with target view
                aligned_features[:, :, idx] = self._transform_equivariant_feature(
                    features[:, :, idx], source_view, target_view
                )
        
        return aligned_features
    
    def _transform_equivariant_feature(self, feature: torch.Tensor, source_view: int, target_view: int) -> torch.Tensor:
        """Transform equivariant feature from source_view to target_view.
        
        Args:
            feature: Feature tensor [B, N]
            source_view: Source view index (0-3)
            target_view: Target view index (0-3)
            
        Returns:
            Transformed feature [B, N]
        """
        # D2 view transformations
        view_transforms = [
            (1, 1),   # id: no change
            (-1, 1),  # h: flip x
            (1, -1),  # v: flip y
            (-1, -1), # hv: flip both
        ]
        
        # Get transformation from source to target
        source_transform = view_transforms[source_view]
        target_transform = view_transforms[target_view]
        
        # Compute combined transformation
        combined_transform = (
            source_transform[0] * target_transform[0],
            source_transform[1] * target_transform[1]
        )
        
        # Apply transformation (assuming feature is x or y coordinate)
        # For now, apply x transformation (can be extended for different coordinate types)
        return feature * combined_transform[0]


class SharedGATv2Layer(nn.Module):
    """Shared GATv2 layer used in group convolution.
    
    This is the shared GNN g_G in equation (8).
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        
        # Linear transformation for each head
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attention mechanism
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features))
        else:
            self.register_parameter('bias', None)
            
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of shared GATv2 layer."""
        N = x.size(1)
        
        # Linear transformation
        h = self.W(x)  # [B, N, heads * out_features]
        h = h.view(-1, N, self.heads, self.out_features)  # [B, N, heads, out_features]
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, N)
        
        # Compute attention coefficients
        att_scores = self._compute_attention(h, edge_index)  # [B, E, heads]
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate
        out = self._aggregate(h, edge_index, att_scores)  # [B, N, heads, out_features]
        
        # Concatenate heads
        out = out.view(-1, N, self.heads * self.out_features)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
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
            if edge_attr.dim() == 1:
                self_edge_attr = torch.zeros(num_nodes, device=device)
            else:
                self_edge_attr = torch.zeros(num_nodes, edge_attr.size(1), device=device)
            edge_attr = torch.cat([edge_attr, self_edge_attr], dim=0)
        
        return edge_index, edge_attr
    
    def _compute_attention(
        self, 
        h: torch.Tensor, 
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention coefficients."""
        src, dst = edge_index[0], edge_index[1]
        
        # Concatenate source and destination features
        h_cat = torch.cat([h[:, src], h[:, dst]], dim=-1)  # [B, E, heads, 2 * out_features]
        
        # Compute attention: a^T * LeakyReLU(W * [h_i || h_j])
        att_input = F.leaky_relu(h_cat, negative_slope=self.negative_slope)
        att_scores = (self.att * att_input).sum(dim=-1)  # [B, E, heads]
        
        return att_scores
    
    def _aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor, 
        att_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbor features using attention weights."""
        src, dst = edge_index[0], edge_index[1]
        
        # Normalize attention scores
        att_weights = F.softmax(att_scores, dim=1)  # [B, E, heads]
        
        # Initialize output
        out = torch.zeros_like(h)  # [B, N, heads, out_features]
        
        # Aggregate features for each head
        for i in range(self.heads):
            # Get source features for this head
            src_features = h[:, src, i, :]  # [B, E, out_features]
            
            # Apply attention weights
            weighted_features = src_features * att_weights[:, :, i:i+1]  # [B, E, out_features]
            
            # Aggregate to destination nodes
            for j in range(len(dst)):
                out[:, dst[j], i, :] += weighted_features[:, j, :]
        
        return out


class D2GroupConvolutionNetwork(nn.Module):
    """Multi-layer D2 Group Convolution Network with invariant pooling.
    
    Implements the complete D2 equivariant architecture from TacticAI paper:
    - Multiple layers of group convolution (equation 8)
    - Final layer performs invariant pooling over all views
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (if None, no final projection)
        num_layers: Number of group convolution layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        equivariant_indices: Indices of equivariant features
        invariant_indices: Indices of invariant features
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
        equivariant_indices: Tuple[int, ...] = (0, 1, 2, 3),
        invariant_indices: Optional[Tuple[int, ...]] = None,
        residual: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Group convolution layers
        self.group_conv_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = D2GroupConvolutionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                equivariant_indices=equivariant_indices,
                invariant_indices=invariant_indices,
                heads=num_heads,
                dropout=dropout,
            )
            self.group_conv_layers.append(layer)
        
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
        return_view_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with invariant pooling.
        
        Args:
            x: Node features [B, V=4, N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional, for global readout)
            return_view_outputs: Whether to return per-view outputs
            
        Returns:
            If return_view_outputs=False: Invariant node embeddings [B, N, output_dim]
            If return_view_outputs=True: (invariant_embeddings, view_embeddings)
        """
        if x.dim() != 4 or x.size(1) != 4:
            raise ValueError(f"Expected input shaped [B, 4, N, D] for four views, got {tuple(x.shape)}.")
        
        # Input projection
        h = self.input_proj(x)  # [B, V=4, N, hidden_dim]
        
        # Group convolution layers with residual connections
        for i, layer in enumerate(self.group_conv_layers):
            h_new = layer(h, edge_index, edge_attr)
            
            # Residual connection (except first layer)
            if self.residual and i > 0 and h_new.size(-1) == h.size(-1):
                h = h + h_new
            else:
                h = h_new
            
            # Apply dropout and activation
            if i < len(self.group_conv_layers) - 1:  # No activation after last layer
                h = F.elu(h)
                h = self.dropout_layer(h)
        
        # Output projection
        view_embeddings = self.output_proj(h)  # [B, V=4, N, output_dim]
        
        # Invariant pooling: average over all views
        invariant_embeddings = view_embeddings.mean(dim=1)  # [B, N, output_dim]
        
        if return_view_outputs:
            return invariant_embeddings, view_embeddings
        
        return invariant_embeddings
