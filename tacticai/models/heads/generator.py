"""Guided generation head (CVAE) for tactic generation.

Implements a Conditional VAE that generates absolute player state at t=0:
  X_hat = (x, y, vx, vy) for 22 players.

Design reference: _docs/生成_設計図.md (案A: absolute value generation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from tacticai.models.gatv2 import GATv2Network


@dataclass(frozen=True)
class ConditionLayout:
    """Condition layout used by this generator.

    c = [ shot(1), swing_onehot(3), receiver_onehot(22) ] => 26 dims
    swing order: [in, out, short]
    """

    shot_dim: int = 1
    swing_dim: int = 3
    receiver_dim: int = 22

    @property
    def dim(self) -> int:
        return int(self.shot_dim + self.swing_dim + self.receiver_dim)


class NodePosterior(nn.Module):
    """Posterior encoder q(z|context, X_gt, c) producing per-node latent params."""

    def __init__(self, context_dim: int, condition_dim: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = context_dim + 4 + condition_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

    def forward(self, context: torch.Tensor, x_gt: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # context: [B,N,D], x_gt: [B,N,4], cond: [B,N,C]
        h = torch.cat([context, x_gt, cond], dim=-1)
        out = self.mlp(h)
        mu, logvar = out.chunk(2, dim=-1)
        return mu, logvar


class NodeDecoder(nn.Module):
    """Decoder p(X|context, z, c) producing per-node absolute state."""

    def __init__(self, context_dim: int, condition_dim: int, latent_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        in_dim = context_dim + latent_dim + condition_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, context: torch.Tensor, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # context: [B,N,D], z: [B,N,Z], cond: [B,N,C]
        h = torch.cat([context, z, cond], dim=-1)
        return self.mlp(h)  # [B,N,4]


class CVAEGenerator(nn.Module):
    """CVAE generator for tactic formation at t=0 (absolute values).

    - Backbone (GATv2) encodes node context H: [B,N,D]
    - Posterior encoder: q(z|H, X_gt, c) with per-node z: [B,N,Z]
    - Decoder: p(X|H, z, c) -> X_hat: [B,N,4]
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_players: int = 22,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.num_players = int(num_players)
        self.condition_dim = int(condition_dim)
        self.latent_dim = int(latent_dim)

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

        self.posterior = NodePosterior(
            context_dim=hidden_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.decoder = NodeDecoder(
            context_dim=hidden_dim,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _node_context(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_emb, _graph_emb = self.backbone(x, edge_index, edge_attr, batch)  # [N_total,D]
        # reshape to [B,N,D]
        bsz = int(batch.max().item() + 1) if batch is not None else 1
        n_total = int(node_emb.size(0))
        n_per = n_total // bsz
        if n_per != self.num_players:
            # fall back: reshape assuming fixed ordering per-graph
            # (dataset should be fixed 22 nodes per graph)
            self.num_players = n_per
        return node_emb.view(bsz, self.num_players, -1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        conditions: torch.Tensor,
        x_gt: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: node features [N_total,input_dim]
            edge_index: [2,E]
            batch: [N_total]
            conditions: [B,condition_dim]
            x_gt: ground truth state [B,N,4] (required when training=True)
            training: if True, sample from posterior; else sample from prior N(0,1)

        Returns:
            x_hat: [B,N,4], mu: [B,N,Z], logvar: [B,N,Z]
        """
        context = self._node_context(x, edge_index, batch, edge_attr=edge_attr)  # [B,N,D]
        B, N, _D = context.shape

        # Broadcast conditions to nodes
        cond = conditions.view(B, 1, self.condition_dim).expand(B, N, self.condition_dim)

        if training:
            if x_gt is None:
                raise ValueError("x_gt is required when training=True")
            mu, logvar = self.posterior(context, x_gt, cond)
            z = self.reparameterize(mu, logvar)
        else:
            # prior N(0,1)
            mu = torch.zeros(B, N, self.latent_dim, device=context.device, dtype=context.dtype)
            logvar = torch.zeros_like(mu)
            z = torch.randn(B, N, self.latent_dim, device=context.device, dtype=context.dtype)

        x_hat = self.decoder(context, z, cond)
        return x_hat, mu, logvar

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        conditions: torch.Tensor,
        num_samples: int = 1,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate multiple samples per condition.

        Returns: [B,num_samples,N,4]
        """
        self.eval()
        B = int(conditions.size(0))
        outs = []
        for _ in range(int(num_samples)):
            x_hat, _mu, _logvar = self.forward(
                x=x,
                edge_index=edge_index,
                batch=batch,
                conditions=conditions,
                x_gt=None,
                edge_attr=edge_attr,
                training=False,
            )
            outs.append(x_hat)
        return torch.stack(outs, dim=1)  # [B,S,N,4]


