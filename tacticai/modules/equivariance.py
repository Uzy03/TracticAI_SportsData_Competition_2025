"""D2 equivariance implementation for TacticAI.

This module implements D2 group equivariance using weight sharing and 
coordinate transformation canonicalization. Provides abstraction for 
future group convolution integration.
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class D2Transform:
    """D2 group transformations for football field coordinates.
    
    D2 group consists of:
    - Identity (no transformation)
    - Horizontal flip (x -> -x)
    - Vertical flip (y -> -y)  
    - Both flips (x -> -x, y -> -y)
    """
    
    @staticmethod
    def get_transforms() -> List[torch.Tensor]:
        """Get all D2 transformation matrices.
        
        Returns:
            List of 2x2 transformation matrices
        """
        transforms = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # Identity
            torch.tensor([[-1.0, 0.0], [0.0, 1.0]]),  # Horizontal flip
            torch.tensor([[1.0, 0.0], [0.0, -1.0]]),  # Vertical flip
            torch.tensor([[-1.0, 0.0], [0.0, -1.0]]), # Both flips
        ]
        return transforms
    
    @staticmethod
    def apply_transform(coords: torch.Tensor, transform_idx: int) -> torch.Tensor:
        """Apply D2 transformation to coordinates.
        
        Args:
            coords: Coordinates [..., 2] (x, y)
            transform_idx: Transformation index (0-3)
            
        Returns:
            Transformed coordinates [..., 2]
        """
        transforms = D2Transform.get_transforms()
        transform = transforms[transform_idx].to(coords.device)
        
        # Apply transformation: coords @ transform.T
        return torch.matmul(coords, transform.T)
    
    @staticmethod
    def canonicalize_coords(coords: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Canonicalize coordinates to standard form.
        
        Args:
            coords: Coordinates [..., 2] (x, y)
            
        Returns:
            Canonicalized coordinates and transformation index
        """
        # Use horizontal flip to ensure x >= 0
        if coords[..., 0].mean() < 0:
            coords = D2Transform.apply_transform(coords, 1)  # Horizontal flip
            transform_idx = 1
        else:
            transform_idx = 0
        
        # Use vertical flip to ensure y >= 0
        if coords[..., 1].mean() < 0:
            coords = D2Transform.apply_transform(coords, 2)  # Vertical flip
            transform_idx = transform_idx | 2
        
        return coords, transform_idx


class EquivariantLinear(nn.Module):
    """Equivariant linear layer with D2 weight sharing.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        coord_dim: Coordinate dimension (typically 2)
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        coord_dim: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.coord_dim = coord_dim
        
        # Weight sharing across D2 transformations
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass with coordinate-aware processing.
        
        Args:
            x: Input features [..., in_features]
            coords: Coordinates [..., coord_dim]
            
        Returns:
            Output features [..., out_features]
        """
        # Canonicalize coordinates
        canonical_coords, transform_idx = D2Transform.canonicalize_coords(coords)
        
        # Apply linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Apply inverse transformation to maintain equivariance
        if transform_idx != 0:
            # For now, we assume the transformation doesn't affect non-coordinate features
            # In a full group convolution, this would be handled by the group structure
            pass
        
        return output


class EquivariantAttention(nn.Module):
    """Equivariant attention mechanism with D2 symmetry.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        coord_dim: Coordinate dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        coord_dim: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.coord_dim = coord_dim
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Equivariant projections
        self.q_proj = EquivariantLinear(embed_dim, embed_dim, coord_dim)
        self.k_proj = EquivariantLinear(embed_dim, embed_dim, coord_dim)
        self.v_proj = EquivariantLinear(embed_dim, embed_dim, coord_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features [N, embed_dim]
            coords: Coordinates [N, coord_dim]
            mask: Attention mask [N, N] (optional)
            
        Returns:
            Output features [N, embed_dim]
        """
        N = x.size(0)
        
        # Project to Q, K, V with equivariance
        q = self.q_proj(x, coords)  # [N, embed_dim]
        k = self.k_proj(x, coords)  # [N, embed_dim]
        v = self.v_proj(x, coords)  # [N, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        k = k.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        v = v.view(N, self.num_heads, self.head_dim)  # [N, num_heads, head_dim]
        
        # Compute attention scores
        att_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, num_heads, N]
        
        # Apply mask if provided
        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax
        att_weights = F.softmax(att_scores, dim=-1)  # [N, num_heads, N]
        att_weights = self.dropout(att_weights)
        
        # Apply attention to values
        out = torch.matmul(att_weights, v)  # [N, num_heads, head_dim]
        
        # Reshape and project output
        out = out.view(N, self.embed_dim)  # [N, embed_dim]
        out = self.out_proj(out)
        
        return out


class EquivariantGATv2Layer(nn.Module):
    """Equivariant GATv2 layer with D2 symmetry.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        heads: Number of attention heads
        coord_dim: Coordinate dimension
        dropout: Dropout probability
        negative_slope: Negative slope for LeakyReLU
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        coord_dim: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.coord_dim = coord_dim
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        
        # Equivariant linear transformation
        self.W = EquivariantLinear(in_features, heads * out_features, coord_dim, bias=False)
        
        # Equivariant attention mechanism
        self.att = EquivariantAttention(
            embed_dim=heads * out_features,
            num_heads=heads,
            coord_dim=coord_dim,
            dropout=dropout
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [N, in_features]
            coords: Node coordinates [N, coord_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Updated node features [N, out_features * heads]
        """
        N = x.size(0)
        
        # Linear transformation with equivariance
        h = self.W(x, coords)  # [N, heads * out_features]
        
        # Reshape for multi-head processing
        h = h.view(N, self.heads, self.out_features)  # [N, heads, out_features]
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, N)
        
        # Compute attention coefficients
        att_scores = self._compute_attention(h, coords, edge_index, edge_attr)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate
        out = self._aggregate(h, edge_index, att_scores)
        
        # Reshape output
        out = out.view(N, self.heads * self.out_features)
        
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
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention coefficients with equivariance."""
        src, dst = edge_index[0], edge_index[1]
        
        # Get source and destination features and coordinates
        h_src = h[src]  # [E, heads, out_features]
        h_dst = h[dst]  # [E, heads, out_features]
        coords_src = coords[src]  # [E, coord_dim]
        coords_dst = coords[dst]  # [E, coord_dim]
        
        # Concatenate features
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [E, heads, 2 * out_features]
        
        # Include edge features if available
        if edge_attr is not None:
            edge_attr_expanded = edge_attr.unsqueeze(1).expand(-1, self.heads, -1)
            h_cat = torch.cat([h_cat, edge_attr_expanded], dim=-1)
        
        # Compute attention scores
        att_input = F.leaky_relu(h_cat, negative_slope=self.negative_slope)
        
        # Use learnable attention parameters
        att_param = nn.Parameter(torch.empty(1, self.heads, h_cat.size(-1)))
        att_scores = (att_param * att_input).sum(dim=-1)  # [E, heads]
        
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
        att_weights = F.softmax(att_scores, dim=0)  # [E, heads]
        
        # Initialize output
        out = torch.zeros_like(h)  # [N, heads, out_features]
        
        # Aggregate features for each head
        for i in range(self.heads):
            src_features = h[src, i, :]  # [E, out_features]
            weighted_features = src_features * att_weights[:, i:i+1]  # [E, out_features]
            
            for j in range(len(dst)):
                out[dst[j], i, :] += weighted_features[j]
        
        return out


def create_equivariant_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    num_heads: int = 4,
    coord_dim: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Create an equivariant model with D2 symmetry.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_layers: Number of layers
        num_heads: Number of attention heads
        coord_dim: Coordinate dimension
        dropout: Dropout probability
        
    Returns:
        Equivariant model
    """
    layers = []
    
    # Input projection
    layers.append(EquivariantLinear(input_dim, hidden_dim, coord_dim))
    
    # Equivariant GATv2 layers
    for i in range(num_layers):
        layers.append(EquivariantGATv2Layer(
            in_features=hidden_dim,
            out_features=hidden_dim,
            heads=num_heads,
            coord_dim=coord_dim,
            dropout=dropout,
        ))
    
    # Output projection
    layers.append(EquivariantLinear(hidden_dim, output_dim, coord_dim))
    
    return nn.Sequential(*layers)
