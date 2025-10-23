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
        
        # Attention mechanism
        if use_edge_features and edge_feature_dim > 0:
            # Include edge features in attention computation
            self.att = nn.Parameter(torch.empty(1, heads, 2 * out_features + edge_feature_dim))
            # Edge feature encoder
            self.edge_encoder = EdgeFeatureEncoder(
                edge_dim=edge_feature_dim,
                hidden_dim=min(edge_feature_dim, 32),
                output_dim=edge_feature_dim,
                dropout=dropout
            )
        else:
            self.att = nn.Parameter(torch.empty(1, heads, 2 * out_features))
            self.edge_encoder = None
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features if concat else out_features))
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
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features [N, in_features]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [N, out_features * heads] or [N, out_features]
            Attention weights [E, heads] (optional)
        """
        N = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [N, heads * out_features]
        h = h.view(N, self.heads, self.out_features)  # [N, heads, out_features]
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, N)
        
        # Compute attention coefficients
        att_scores = self._compute_attention(h, edge_index, edge_attr)  # [E, heads]
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate
        out = self._aggregate(h, edge_index, att_scores)  # [N, heads, out_features]
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(N, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)  # [N, out_features]
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        if return_attention_weights:
            return out, att_scores
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
        h: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention coefficients.
        
        Args:
            h: Node features [N, heads, out_features]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Attention scores [E, heads]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Concatenate source and destination features
        h_cat = torch.cat([h[src], h[dst]], dim=-1)  # [E, heads, 2 * out_features]
        
        # Include edge features if available
        if self.use_edge_features and edge_attr is not None and self.edge_encoder is not None:
            # Encode edge features
            edge_encoded = self.edge_encoder(edge_attr)  # [E, edge_feature_dim]
            edge_encoded = edge_encoded.unsqueeze(1).expand(-1, self.heads, -1)  # [E, heads, edge_feature_dim]
            
            # Concatenate with node features
            h_cat = torch.cat([h_cat, edge_encoded], dim=-1)  # [E, heads, 2 * out_features + edge_feature_dim]
        
        # Compute attention: a^T * LeakyReLU(W * [h_i || h_j || edge_ij])
        att_input = F.leaky_relu(h_cat, negative_slope=self.negative_slope)
        att_scores = (self.att * att_input).sum(dim=-1)  # [E, heads]
        
        return att_scores
    
    def _aggregate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor, 
        att_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate neighbor features using attention weights.
        
        Args:
            h: Node features [N, heads, out_features]
            edge_index: Edge connectivity [2, E]
            att_scores: Attention scores [E, heads]
            
        Returns:
            Aggregated features [N, heads, out_features]
        """
        src, dst = edge_index[0], edge_index[1]
        
        # Normalize attention scores
        att_weights = F.softmax(att_scores, dim=0)  # [E, heads]
        
        # Initialize output
        out = torch.zeros_like(h)  # [N, heads, out_features]
        
        # Aggregate features for each head
        for i in range(self.heads):
            # Get source features for this head
            src_features = h[src, i, :]  # [E, out_features]
            
            # Apply attention weights
            weighted_features = src_features * att_weights[:, i:i+1]  # [E, out_features]
            
            # Aggregate to destination nodes
            for j in range(len(dst)):
                out[dst[j], i, :] += weighted_features[j]
        
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
        view_mixing: Type of inter-view mixing ('attention', 'conv1d', 'mlp')
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
        
        # Linear transformation for each head
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attention mechanism
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_features))
        
        # Inter-view mixing
        if view_mixing == "attention":
            self.view_att = nn.MultiheadAttention(
                embed_dim=heads * out_features if concat else out_features,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        elif view_mixing == "conv1d":
            self.view_conv = nn.Conv1d(
                in_channels=heads * out_features if concat else out_features,
                out_channels=heads * out_features if concat else out_features,
                kernel_size=4,
                padding=1,
                groups=heads if not concat else 1
            )
        elif view_mixing == "mlp":
            self.view_mlp = nn.Sequential(
                nn.Linear(heads * out_features if concat else out_features, 
                         (heads * out_features if concat else out_features) * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear((heads * out_features if concat else out_features) * 2,
                         heads * out_features if concat else out_features)
            )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(heads * out_features if concat else out_features))
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
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with 4-view interaction.
        
        Args:
            x: Node features [B, V=4, N, in_features]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features [B, V=4, N, out_features * heads] or [B, V=4, N, out_features]
            Attention weights [E, heads] (optional)
        """
        B, V, N, _ = x.shape
        
        # Reshape for intra-view processing: [B*V, N, in_features]
        x_flat = x.view(B * V, N, -1)
        
        # Intra-view GAT processing
        h_flat = self._intra_view_gat(x_flat, edge_index, edge_attr)
        
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
    
    def _intra_view_gat(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Intra-view GAT processing."""
        N = x.size(1)
        
        # Linear transformation
        h = self.W(x)  # [B*V, N, heads * out_features]
        h = h.view(-1, N, self.heads, self.out_features)  # [B*V, N, heads, out_features]
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, N)
        
        # Compute attention coefficients
        att_scores = self._compute_attention(h, edge_index, edge_attr)  # [E, heads]
        
        # Apply dropout
        if self.training and self.dropout > 0:
            att_scores = F.dropout(att_scores, p=self.dropout, training=True)
        
        # Apply attention and aggregate
        out = self._aggregate(h, edge_index, att_scores)  # [B*V, N, heads, out_features]
        
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
        
        if self.view_mixing == "attention":
            # Reshape for attention: [B*N, V, D]
            h_att = h.permute(0, 2, 1, 3).contiguous().view(B * N, V, D)
            
            # Self-attention across views
            h_mixed, _ = self.view_att(h_att, h_att, h_att)
            
            # Reshape back: [B, V, N, D]
            h_mixed = h_mixed.view(B, N, V, D).permute(0, 2, 1, 3).contiguous()
            
        elif self.view_mixing == "conv1d":
            # Reshape for conv1d: [B*N, D, V]
            h_conv = h.permute(0, 2, 3, 1).contiguous().view(B * N, D, V)
            
            # 1D convolution across views
            h_mixed = self.view_conv(h_conv)
            
            # Reshape back: [B, V, N, D]
            h_mixed = h_mixed.view(B, N, D, V).permute(0, 3, 1, 2).contiguous()
            
        elif self.view_mixing == "mlp":
            # Reshape for MLP: [B*V*N, D]
            h_mlp = h.view(B * V * N, D)
            
            # MLP processing
            h_mixed = self.view_mlp(h_mlp)
            
            # Reshape back: [B, V, N, D]
            h_mixed = h_mixed.view(B, V, N, D)
        
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
        edge_index: torch.Tensor
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
        
        # Average across batch*view dimension
        att_scores = att_scores.mean(dim=0)  # [E, heads]
        
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
        
        # Normalize attention scores
        att_weights = F.softmax(att_scores, dim=0)  # [E, heads]
        
        # Initialize output
        out = torch.zeros_like(h)  # [B*V, N, heads, out_features]
        
        # Aggregate features for each head
        for i in range(self.heads):
            # Get source features for this head
            src_features = h[:, src, i, :]  # [B*V, E, out_features]
            
            # Apply attention weights
            weighted_features = src_features * att_weights[:, i:i+1].unsqueeze(0)  # [B*V, E, out_features]
            
            # Aggregate to destination nodes
            for j in range(len(dst)):
                out[:, dst[j], i, :] += weighted_features[:, j]
        
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional, for global readout)
            
        Returns:
            If batch is None: Node embeddings [N, output_dim]
            If batch is provided: (node_embeddings, graph_embeddings)
        """
        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim]
        
        # GATv2 layers with residual connections
        for i, layer in enumerate(self.gat_layers):
            h_new = layer(h, edge_index, edge_attr)
            
            # Residual connection (except first layer)
            if self.residual and i > 0 and h_new.size(1) == h.size(1):
                h = h + h_new
            else:
                h = h_new
            
            # Apply dropout and activation
            if i < len(self.gat_layers) - 1:  # No activation after last layer
                h = F.elu(h)
                h = self.dropout_layer(h)
        
        # Output projection
        node_embeddings = self.output_proj(h)  # [N, output_dim]
        
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
        view_mixing: Type of inter-view mixing ('attention', 'conv1d', 'mlp')
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