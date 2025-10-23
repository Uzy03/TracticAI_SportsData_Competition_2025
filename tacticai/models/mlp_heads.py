"""MLP heads for different TacticAI tasks.

This module implements task-specific heads for:
- Receiver Prediction: Node classification head
- Shot Prediction: Graph classification head  
- Tactic Generation: Conditional generation head
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gatv2 import GATv2Network


class ReceiverHead(nn.Module):
    """Node classification head for receiver prediction.
    
    Predicts which player will receive the pass (22-class classification).
    
    Args:
        input_dim: Input feature dimension
        num_classes: Number of receiver classes (default: 22)
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 22,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            node_embeddings: Node embeddings [N, input_dim]
            
        Returns:
            Receiver predictions [N, num_classes]
        """
        return self.mlp(node_embeddings)


class ShotHeadConditional(nn.Module):
    """Conditional shot prediction head with receiver conditioning.
    
    Predicts shot probability conditioned on receiver information.
    Training: Uses ground truth receiver as global feature injection
    Inference: Marginalizes over all possible receivers
    
    Args:
        input_dim: Input feature dimension
        receiver_dim: Receiver embedding dimension
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        receiver_dim: int = 22,  # Number of players
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.receiver_dim = receiver_dim
        
        # Receiver embedding
        self.receiver_embedding = nn.Embedding(receiver_dim, hidden_dim)
        
        # MLP layers with receiver conditioning
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self, 
        graph_embeddings: torch.Tensor, 
        receiver_ids: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            graph_embeddings: Graph embeddings [B, input_dim]
            receiver_ids: Receiver IDs [B] (for training) or None (for inference)
            training: Whether in training mode
            
        Returns:
            Shot predictions [B, 1] or [B, receiver_dim] (for inference)
        """
        if training and receiver_ids is not None:
            # Training: use ground truth receiver
            receiver_emb = self.receiver_embedding(receiver_ids)  # [B, hidden_dim]
            x = torch.cat([graph_embeddings, receiver_emb], dim=-1)
            return self.mlp(x)  # [B, 1]
        else:
            # Inference: marginalize over all receivers
            B = graph_embeddings.size(0)
            all_receivers = torch.arange(self.receiver_dim, device=graph_embeddings.device)
            
            # Expand graph embeddings for all receivers
            graph_expanded = graph_embeddings.unsqueeze(1).expand(B, self.receiver_dim, -1)  # [B, receiver_dim, input_dim]
            graph_expanded = graph_expanded.contiguous().view(B * self.receiver_dim, -1)  # [B*receiver_dim, input_dim]
            
            # Get receiver embeddings for all receivers
            receiver_emb = self.receiver_embedding(all_receivers)  # [receiver_dim, hidden_dim]
            receiver_emb = receiver_emb.unsqueeze(0).expand(B, -1, -1)  # [B, receiver_dim, hidden_dim]
            receiver_emb = receiver_emb.contiguous().view(B * self.receiver_dim, -1)  # [B*receiver_dim, hidden_dim]
            
            # Concatenate and predict
            x = torch.cat([graph_expanded, receiver_emb], dim=-1)  # [B*receiver_dim, input_dim + hidden_dim]
            predictions = self.mlp(x)  # [B*receiver_dim, 1]
            
            # Reshape back to [B, receiver_dim]
            predictions = predictions.view(B, self.receiver_dim)
            
            return predictions


class ShotHead(nn.Module):
    """Graph classification head for shot prediction.
    
    Predicts if a shot will occur from the current tactical situation.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, graph_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            graph_embeddings: Graph embeddings [B, input_dim]
            
        Returns:
            Shot predictions [B, 1]
        """
        return self.mlp(graph_embeddings)


class CVAEHead(nn.Module):
    """Conditional generation head for tactic generation.
    
    Generates tactical formations conditioned on context (e.g., set piece situation).
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
        output_dim: Output dimension (typically 2D coordinates * num_players)
        condition_dim: Condition dimension
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        output_dim: int = 44,  # 22 players * 2 coordinates
        condition_dim: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        
        # Encoder: graph -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )
        
        # Decoder: latent + condition -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def encode(
        self, 
        graph_embeddings: torch.Tensor, 
        conditions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode graph embeddings to latent space.
        
        Args:
            graph_embeddings: Graph embeddings [B, input_dim]
            conditions: Conditions [B, condition_dim]
            
        Returns:
            Latent mean [B, latent_dim] and log variance [B, latent_dim]
        """
        x = torch.cat([graph_embeddings, conditions], dim=-1)
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        return mean, log_var
    
    def reparameterize(
        self, 
        mean: torch.Tensor, 
        log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for sampling.
        
        Args:
            mean: Latent mean [B, latent_dim]
            log_var: Log variance [B, latent_dim]
            
        Returns:
            Sampled latent [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(
        self, 
        latent: torch.Tensor, 
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent to output.
        
        Args:
            latent: Latent representation [B, latent_dim]
            conditions: Conditions [B, condition_dim]
            
        Returns:
            Generated output [B, output_dim]
        """
        x = torch.cat([latent, conditions], dim=-1)
        return self.decoder(x)
    
    def forward(
        self, 
        graph_embeddings: torch.Tensor, 
        conditions: torch.Tensor,
        training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            graph_embeddings: Graph embeddings [B, input_dim]
            conditions: Conditions [B, condition_dim]
            training: Whether in training mode
            
        Returns:
            Generated output [B, output_dim], latent mean [B, latent_dim], 
            latent log_var [B, latent_dim]
        """
        # Encode
        mean, log_var = self.encode(graph_embeddings, conditions)
        
        if training:
            # Sample from latent distribution
            latent = self.reparameterize(mean, log_var)
        else:
            # Use mean for inference
            latent = mean
        
        # Decode
        output = self.decode(latent, conditions)
        
        return output, mean, log_var


class TacticAI(nn.Module):
    """Complete TacticAI model combining GATv2 backbone with task-specific heads.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for GATv2
        num_layers: Number of GATv2 layers
        num_heads: Number of attention heads
        dropout: Dropout probability
        task: Task type ('receiver', 'shot', 'cvae')
        num_classes: Number of classes for receiver prediction
        latent_dim: Latent dimension for CVAE
        condition_dim: Condition dimension for CVAE
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        task: str = "receiver",
        num_classes: int = 22,
        latent_dim: int = 32,
        condition_dim: int = 8,
    ):
        super().__init__()
        
        self.task = task
        
        # GATv2 backbone
        self.backbone = GATv2Network(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            readout="mean",
            residual=True,
        )
        
        # Task-specific heads
        if task == "receiver":
            self.head = ReceiverHead(
                input_dim=hidden_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif task == "shot":
            self.head = ShotHead(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif task == "shot_conditional":
            self.head = ShotHeadConditional(
                input_dim=hidden_dim,
                receiver_dim=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif task == "cvae":
            self.head = CVAEHead(
                input_dim=hidden_dim,
                latent_dim=latent_dim,
                output_dim=num_classes * 2,  # 2D coordinates
                condition_dim=condition_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        conditions: Optional[torch.Tensor] = None,
        training: bool = True,
        **kwargs
    ):
        """Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional)
            conditions: Conditions for CVAE [B, condition_dim] (optional)
            training: Whether in training mode
            
        Returns:
            Task-specific outputs
        """
        # Get backbone embeddings
        if batch is not None:
            node_embeddings, graph_embeddings = self.backbone(
                x, edge_index, edge_attr, batch
            )
        else:
            node_embeddings = self.backbone(x, edge_index, edge_attr)
            graph_embeddings = None
        
        # Apply task-specific head
        if self.task == "receiver":
            return self.head(node_embeddings)
        elif self.task == "shot":
            return self.head(graph_embeddings)
        elif self.task == "shot_conditional":
            # For conditional shot prediction, we need receiver_ids
            receiver_ids = kwargs.get('receiver_ids', None)
            return self.head(graph_embeddings, receiver_ids, training)
        elif self.task == "cvae":
            return self.head(graph_embeddings, conditions, training)
        else:
            raise ValueError(f"Unknown task: {self.task}")
