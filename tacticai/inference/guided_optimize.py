"""Guided optimization for shot probability maximization/minimization.

This module implements gradient-based optimization to adjust tactical formations
to maximize or minimize shot probabilities.
"""

from typing import Optional, Tuple, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GuidedOptimizer:
    """Gradient-based optimizer for adjusting tactical formations.
    
    Adjusts player positions to maximize or minimize shot probabilities
    while respecting physical constraints.
    """
    
    def __init__(
        self,
        shot_model: nn.Module,
        field_bounds: Tuple[float, float, float, float] = (-52.5, -34.0, 52.5, 34.0),
        min_distance: float = 1.0,
        max_movement_per_step: float = 2.0,
        collision_penalty: float = 10.0,
    ):
        """Initialize guided optimizer.
        
        Args:
            shot_model: Model that predicts shot probability from positions
            field_bounds: Field boundaries as (x_min, y_min, x_max, y_max)
            min_distance: Minimum allowed distance between players (meters)
            max_movement_per_step: Maximum movement per optimization step (meters)
            collision_penalty: Penalty coefficient for collision constraints
        """
        self.shot_model = shot_model
        self.field_bounds = field_bounds
        self.min_distance = min_distance
        self.max_movement_per_step = max_movement_per_step
        self.collision_penalty = collision_penalty
    
    def optimize(
        self,
        X0: torch.Tensor,
        mask_move: torch.Tensor,
        mode: str = "maximize",
        steps: int = 10,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Optimize player positions to maximize/minimize shot probability.
        
        Args:
            X0: Initial positions [B, N, 2]
            mask_move: Mask indicating which players can move [B, N, 1] or [B, N]
            mode: Optimization mode ('maximize' or 'minimize')
            steps: Number of optimization steps
            lr: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Optimized positions [B, N, 2] and metrics dictionary
        """
        X = X0.clone().requires_grad_(True)
        initial_positions = X0.clone()
        
        # Ensure mask has correct shape
        if mask_move.dim() == 2:
            mask_move = mask_move.unsqueeze(-1)  # [B, N, 1]
        
        history = {
            "shot_probs": [],
            "total_movement": [],
            "violations": [],
        }
        
        optimizer = torch.optim.Adam([X], lr=lr)
        
        for step in range(steps):
            optimizer.zero_grad()
            
            # Compute shot probability
            shot_prob = self._compute_shot_probability(X)
            
            # Define objective
            if mode == "maximize":
                loss = -torch.log(shot_prob + 1e-8)
            else:  # minimize
                loss = torch.log(shot_prob + 1e-8)
            
            # Add penalty terms
            penalty = self._compute_penalties(X, initial_positions, mask_move)
            total_loss = loss + penalty
            
            # Backward pass
            total_loss.backward()
            
            # Apply gradient mask
            X.grad *= mask_move
            
            # Update
            optimizer.step()
            
            # Apply constraints
            X = self._project_constraints(X, initial_positions, mask_move)
            
            # Track metrics
            with torch.no_grad():
                movement = torch.norm(X - initial_positions, dim=-1).sum(-1).mean()
                violations = self._count_violations(X).mean()
                
                history["shot_probs"].append(shot_prob.item())
                history["total_movement"].append(movement.item())
                history["violations"].append(violations.item())
                
                if verbose and (step + 1) % max(1, steps // 5) == 0:
                    print(
                        f"Step {step+1}/{steps}: "
                        f"Shot Prob={shot_prob.item():.4f}, "
                        f"Movement={movement.item():.2f}m, "
                        f"Violations={violations.item():.1f}"
                    )
        
        # Compute final metrics
        with torch.no_grad():
            final_shot_prob = self._compute_shot_probability(X)
            initial_shot_prob = self._compute_shot_probability(X0)
            delta_p = final_shot_prob - initial_shot_prob
            
            metrics = {
                "initial_shot_prob": initial_shot_prob.item(),
                "final_shot_prob": final_shot_prob.item(),
                "delta_p": delta_p.item(),
                "total_movement": history["total_movement"][-1],
                "violations": history["violations"][-1],
                "mode": mode,
                "steps": steps,
            }
        
        return X.detach(), metrics
    
    def _compute_shot_probability(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute shot probability for given positions.
        
        Args:
            positions: Player positions [B, N, 2]
            
        Returns:
            Shot probability [B, 1]
        """
        # Reshape to match model input format
        B, N, _ = positions.shape
        
        # Assume model expects input as [B, N, features]
        # Here we need to construct the full input (positions, velocities, etc.)
        # For simplicity, we'll assume positions are the main input
        
        # Get node features from positions
        node_features = self._positions_to_features(positions)  # [B, N, F]
        
        # Get edge index for complete graph
        edge_index = self._create_complete_edge_index(N, device=positions.device)
        
        # Get batch indices
        batch = torch.arange(B, dtype=torch.long, device=positions.device).repeat_interleave(N)
        
        # Flatten for model
        node_features_flat = node_features.view(B * N, -1)
        
        # Forward pass through model
        with torch.no_grad():
            self.shot_model.eval()
            outputs = self.shot_model(node_features_flat, edge_index, batch)
            
            # Output should be [B, 1] after sigmoid
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(-1)
            
            shot_prob = torch.sigmoid(outputs) if outputs.size(-1) == 1 else F.softmax(outputs, dim=-1)[:, 0:1]
        
        return shot_prob.mean(dim=0, keepdim=True)
    
    def _positions_to_features(
        self, positions: torch.Tensor
    ) -> torch.Tensor:
        """Convert positions to node features.
        
        Args:
            positions: Player positions [B, N, 2]
            
        Returns:
            Node features [B, N, F]
        """
        B, N, _ = positions.shape
        
        # Normalize positions to [0, 1]
        x_min, y_min, x_max, y_max = self.field_bounds
        normalized_pos = positions.clone()
        normalized_pos[:, :, 0] = (positions[:, :, 0] - x_min) / (x_max - x_min)
        normalized_pos[:, :, 1] = (positions[:, :, 1] - y_min) / (y_max - y_min)
        
        # Create basic features: [pos, velocity(zeros), attributes(zeros), ball(zeros), team]
        # Assume invoice_dim = 8: [x, y, vx, vy, height, weight, ball, team]
        features = torch.zeros(B, N, 8, device=positions.device)
        
        # Add normalized positions
        features[:, :, 0:2] = normalized_pos
        
        # Add team information (alternating)
        features[:, :N//2, 7] = 0  # First team
        features[:, N//2:, 7] = 1  # Second team
        
        return features
    
    def _create_complete_edge_index(
        self, num_nodes: int, device: torch.device
    ) -> torch.Tensor:
        """Create complete graph edge index.
        
        Args:
            num_nodes: Number of nodes
            device: Device to place tensor on
            
        Returns:
            Edge index [2, E]
        """
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        return torch.tensor(edges, device=device).t().contiguous()
    
    def _compute_penalties(
        self,
        X: torch.Tensor,
        X0: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute constraint penalties.
        
        Args:
            X: Current positions [B, N, 2]
            X0: Initial positions [B, N, 2]
            mask: Movement mask [B, N, 1]
            
        Returns:
            Total penalty [1]
        """
        penalties = []
        
        # Boundary violation penalty
        boundary_penalty = self._boundary_penalty(X)
        penalties.append(boundary_penalty)
        
        # Movement distance penalty
        movement_penalty = self._movement_penalty(X, X0, mask)
        penalties.append(movement_penalty)
        
        # Collision penalty
        collision_penalty = self._collision_penalty(X)
        penalties.append(collision_penalty * self.collision_penalty)
        
        return sum(penalties)
    
    def _boundary_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """Compute boundary violation penalty.
        
        Args:
            X: Positions [B, N, 2]
            
        Returns:
            Penalty value [1]
        """
        x_min, y_min, x_max, y_max = self.field_bounds
        
        # Clamp to boundaries
        X_clamped = X.clone()
        X_clamped[:, :, 0] = torch.clamp(X[:, :, 0], x_min, x_max)
        X_clamped[:, :, 1] = torch.clamp(X[:, :, 1], y_min, y_max)
        
        # Penalty is proportional to distance outside boundaries
        violation = torch.norm(X - X_clamped, dim=-1).sum()
        
        return violation
    
    def _movement_penalty(self, X: torch.Tensor, X0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute excessive movement penalty.
        
        Args:
            X: Current positions [B, N, 2]
            X0: Initial positions [B, N, 2]
            mask: Movement mask [B, N, 1]
            
        Returns:
            Penalty value [1]
        """
        movement = torch.norm(X - X0, dim=-1, keepdim=True)  # [B, N, 1]
        excess = torch.clamp(movement * mask - self.max_movement_per_step, min=0.0)
        
        return excess.sum()
    
    def _collision_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """Compute collision penalty.
        
        Args:
            X: Positions [B, N, 2]
            
        Returns:
            Penalty value [1]
        """
        B, N, _ = X.shape
        
        # Compute pairwise distances
        distances = torch.cdist(X.view(B * N, 2), X.view(B * N, 2))  # [B*N, B*N]
        
        # Set diagonal to infinity (self-distances)
        distances = distances + torch.eye(B * N, device=X.device) * float('inf')
        
        # Penalize pairs too close
        min_distances = torch.min(distances, dim=-1)[0]  # [B*N]
        violation = torch.clamp(self.min_distance - min_distances, min=0.0)
        
        return violation.sum()
    
    def _count_violations(self, X: torch.Tensor) -> torch.Tensor:
        """Count constraint violations.
        
        Args:
            X: Positions [B, N, 2]
            
        Returns:
            Number of violations [B]
        """
        B, N, _ = X.shape
        x_min, y_min, x_max, y_max = self.field_bounds
        
        # Count boundary violations
        boundary_violations = (
            (X[:, :, 0] < x_min) | (X[:, :, 0] > x_max) |
            (X[:, :, 1] < y_min) | (X[:, :, 1] > y_max)
        ).sum(dim=-1)
        
        # Count collision violations
        collision_violations = torch.zeros(B, device=X.device)
        for b in range(B):
            distances = torch.cdist(X[b], X[b])
            distances = distances + torch.eye(N, device=X.device) * float('inf')
            min_dist = torch.min(distances, dim=-1)[0]
            collision_violations[b] = (min_dist < self.min_distance).sum()
        
        return boundary_violations + collision_violations
    
    def _project_constraints(
        self, X: torch.Tensor, X0: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Project positions to feasible space.
        
        Args:
            X: Current positions [B, N, 2]
            X0: Initial positions [B, N, 2]
            mask: Movement mask [B, N, 1]
            
        Returns:
            Projected positions [B, N, 2]
        """
        X_proj = X.clone()
        
        # Project to field boundaries
        x_min, y_min, x_max, y_max = self.field_bounds
        X_proj[:, :, 0] = torch.clamp(X[:, :, 0], x_min, x_max)
        X_proj[:, :, 1] = torch.clamp(X[:, :, 1], y_min, y_max)
        
        # Limit movement distance
        movement = torch.norm(X_proj - X0, dim=-1, keepdim=True)
        excess = torch.clamp(movement * mask - self.max_movement_per_step, min=0.0)
        scale = 1.0 - excess / (movement + 1e-8)
        scale = torch.clamp(scale, min=0.0, max=1.0)
        
        X_proj = X0 + (X_proj - X0) * scale * mask
        
        # Simple collision resolution: push players apart
        X_proj = self._resolve_collisions(X_proj)
        
        return X_proj
    
    def _resolve_collisions(self, X: torch.Tensor) -> torch.Tensor:
        """Resolve collisions by pushing players apart.
        
        Args:
            X: Positions [B, N, 2]
            
        Returns:
            Adjusted positions [B, N, 2]
        """
        X_adj = X.clone()
        B, N, _ = X.shape
        
        for b in range(B):
            for i in range(N):
                for j in range(i + 1, N):
                    dist = torch.norm(X[b, i] - X[b, j])
                    if dist < self.min_distance:
                        # Push apart
                        direction = (X[b, i] - X[b, j]) / (dist + 1e-8)
                        push = (self.min_distance - dist) / 2
                        X_adj[b, i] += direction * push
                        X_adj[b, j] -= direction * push
        
        return X_adj


def guided_optimize(
    X0: torch.Tensor,
    shot_model: nn.Module,
    mask_move: torch.Tensor,
    mode: str = "maximize",
    steps: int = 10,
    lr: float = 0.1,
    field_bounds: Tuple[float, float, float, float] = (-52.5, -34.0, 52.5, 34.0),
    min_distance: float = 1.0,
    max_movement_per_step: float = 2.0,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Optimize positions to maximize or minimize shot probability.
    
    Args:
        X0: Initial positions [B, N, 2]
        shot_model: Model predicting shot probability
        mask_move: Which players can move [B, N] or [B, N, 1]
        mode: 'maximize' or 'minimize'
        steps: Number of optimization steps
        lr: Learning rate
        field_bounds: Field boundaries (x_min, y_min, x_max, y_max)
        min_distance: Minimum distance between players
        max_movement_per_step: Maximum movement per step
        verbose: Whether to print progress
        
    Returns:
        Optimized positions and metrics
    """
    optimizer = GuidedOptimizer(
        shot_model=shot_model,
        field_bounds=field_bounds,
        min_distance=min_distance,
        max_movement_per_step=max_movement_per_step,
    )
    
    return optimizer.optimize(
        X0=X0,
        mask_move=mask_move,
        mode=mode,
        steps=steps,
        lr=lr,
        verbose=verbose,
    )

