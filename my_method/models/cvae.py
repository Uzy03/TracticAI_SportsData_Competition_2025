"""Conditional Variational Autoencoder (CVAE) implementation for tactic generation.

This module implements a CVAE for generating tactical formations conditioned on
context such as set piece situations, field position, etc.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gatv2 import GATv2Network


class CVAEEncoder(nn.Module):
    """CVAE Encoder: Graph -> Latent space.
    
    Encodes graph embeddings and conditions to latent mean and variance.
    
    Args:
        input_dim: Input feature dimension
        condition_dim: Condition dimension
        latent_dim: Latent space dimension
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),  # mean and log_var
        )
    
    def forward(
        self, 
        graph_embeddings: torch.Tensor, 
        conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            graph_embeddings: Graph embeddings [B, input_dim]
            conditions: Conditions [B, condition_dim]
            
        Returns:
            Latent mean [B, latent_dim] and log variance [B, latent_dim]
        """
        # Concatenate graph embeddings and conditions
        x = torch.cat([graph_embeddings, conditions], dim=-1)
        
        # Encode to latent parameters
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        
        return mean, log_var


class CVAEDecoder(nn.Module):
    """CVAE Decoder: Latent + Condition -> Output.
    
    Decodes latent representations and conditions to tactical formations.
    Now supports position + velocity output and shot guidance.
    
    Args:
        latent_dim: Latent space dimension
        condition_dim: Condition dimension
        output_dim: Output dimension (typically 4D coordinates * num_players: x, y, vx, vy)
        hidden_dim: Hidden dimension for MLP
        dropout: Dropout probability
        include_velocity: Whether to include velocity in output
        shot_guidance: Whether to include shot probability guidance
    """
    
    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        output_dim: int = 88,  # 22 players * 4 coordinates (x, y, vx, vy)
        hidden_dim: int = 128,
        dropout: float = 0.2,
        include_velocity: bool = True,
        shot_guidance: bool = True,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.include_velocity = include_velocity
        self.shot_guidance = shot_guidance
        
        # Shot guidance condition dimension (shot_prob_up/down)
        guidance_dim = 2 if shot_guidance else 0
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim + guidance_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Shot probability predictor for guidance
        if shot_guidance:
            self.shot_predictor = nn.Sequential(
                nn.Linear(output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        latent: torch.Tensor, 
        conditions: torch.Tensor,
        shot_guidance_conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            latent: Latent representation [B, latent_dim]
            conditions: Conditions [B, condition_dim]
            shot_guidance_conditions: Shot guidance conditions [B, 2] (up/down)
            
        Returns:
            Generated tactical formation [B, output_dim]
        """
        # Prepare input
        x = torch.cat([latent, conditions], dim=-1)
        
        if self.shot_guidance and shot_guidance_conditions is not None:
            x = torch.cat([x, shot_guidance_conditions], dim=-1)
        
        # Decode to tactical formation
        output = self.decoder(x)
        
        return output
    
    def forward_with_guidance(
        self, 
        latent: torch.Tensor, 
        conditions: torch.Tensor,
        shot_guidance_conditions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with shot probability prediction for guidance.
        
        Args:
            latent: Latent representation [B, latent_dim]
            conditions: Conditions [B, condition_dim]
            shot_guidance_conditions: Shot guidance conditions [B, 2] (up/down)
            
        Returns:
            Generated tactical formation [B, output_dim]
            Shot probability [B, 1] (if shot_guidance=True)
        """
        # Generate formation
        output = self.forward(latent, conditions, shot_guidance_conditions)
        
        # Predict shot probability for guidance
        shot_prob = None
        if self.shot_guidance:
            shot_prob = self.shot_predictor(output)
        
        return output, shot_prob


class CVAEModel(nn.Module):
    """Complete CVAE model for tactic generation.
    
    Combines GATv2 backbone with CVAE encoder/decoder for generating
    tactical formations conditioned on context.
    
    Args:
        input_dim: Input feature dimension
        condition_dim: Condition dimension
        latent_dim: Latent space dimension
        output_dim: Output dimension
        hidden_dim: Hidden dimension for GATv2 and MLPs
        num_layers: Number of GATv2 layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 8,
        latent_dim: int = 32,
        output_dim: int = 88,  # 22 players * 4 coordinates (x, y, vx, vy)
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        include_velocity: bool = True,
        shot_guidance: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.include_velocity = include_velocity
        self.shot_guidance = shot_guidance
        
        # GATv2 backbone for graph processing
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
        
        # CVAE encoder
        self.encoder = CVAEEncoder(
            input_dim=hidden_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        
        # CVAE decoder with velocity and shot guidance
        self.decoder = CVAEDecoder(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            include_velocity=include_velocity,
            shot_guidance=shot_guidance,
        )
    
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
    
    def encode(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        conditions: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode graph to latent space.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            conditions: Conditions [B, condition_dim]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Latent mean [B, latent_dim] and log variance [B, latent_dim]
        """
        # Get graph embeddings from backbone
        _, graph_embeddings = self.backbone(x, edge_index, edge_attr, batch)
        
        # Encode to latent space
        mean, log_var = self.encoder(graph_embeddings, conditions)
        
        return mean, log_var
    
    def decode(
        self, 
        latent: torch.Tensor, 
        conditions: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent to tactical formation.
        
        Args:
            latent: Latent representation [B, latent_dim]
            conditions: Conditions [B, condition_dim]
            
        Returns:
            Generated tactical formation [B, output_dim]
        """
        return self.decoder(latent, conditions)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        conditions: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        training: bool = True,
        shot_guidance_conditions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            conditions: Conditions [B, condition_dim]
            edge_attr: Edge features [E, edge_dim] (optional)
            training: Whether in training mode
            shot_guidance_conditions: Shot guidance conditions [B, 2] (optional)
            
        Returns:
            Generated formation [B, output_dim], latent mean [B, latent_dim], 
            latent log_var [B, latent_dim], shot probability [B, 1] (optional)
        """
        # Encode
        mean, log_var = self.encode(x, edge_index, batch, conditions, edge_attr)
        
        if training:
            # Sample from latent distribution
            latent = self.reparameterize(mean, log_var)
        else:
            # Use mean for inference
            latent = mean
        
        # Decode with guidance
        if self.shot_guidance:
            output, shot_prob = self.decoder.forward_with_guidance(
                latent, conditions, shot_guidance_conditions
            )
            return output, mean, log_var, shot_prob
        else:
            output = self.decoder(latent, conditions, shot_guidance_conditions)
            return output, mean, log_var, None
    
    def generate(
        self,
        conditions: torch.Tensor,
        num_samples: int = 1,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate tactical formations from conditions.
        
        Args:
            conditions: Conditions [B, condition_dim]
            num_samples: Number of samples per condition
            device: Device to generate on
            
        Returns:
            Generated formations [B * num_samples, output_dim]
        """
        self.eval()
        
        if device is None:
            device = next(self.parameters()).device
        
        conditions = conditions.to(device)
        batch_size = conditions.size(0)
        
        # Expand conditions for sampling
        conditions_expanded = conditions.repeat_interleave(num_samples, dim=0)
        
        # Sample from prior
        latent = torch.randn(
            batch_size * num_samples, 
            self.latent_dim, 
            device=device
        )
        
        # Decode
        with torch.no_grad():
            generated = self.decode(latent, conditions_expanded)
        
        return generated
    
    def reconstruct(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        conditions: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruct input from latent encoding.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment [N]
            conditions: Conditions [B, condition_dim]
            edge_attr: Edge features [E, edge_dim] (optional)
            
        Returns:
            Reconstructed formation [B, output_dim]
        """
        self.eval()
        
        with torch.no_grad():
            # Encode
            mean, _ = self.encode(x, edge_index, batch, conditions, edge_attr)
            
            # Decode using mean
            reconstructed = self.decode(mean, conditions)
        
        return reconstructed
