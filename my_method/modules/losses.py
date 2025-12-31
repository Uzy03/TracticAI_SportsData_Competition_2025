"""Loss functions for TacticAI tasks.

This module implements various loss functions including classification losses,
reconstruction losses, and CVAE-specific losses.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for receiver prediction.
    
    Standard cross-entropy loss with optional label smoothing.
    """
    
    def __init__(self, label_smoothing: float = 0.0, weight: Optional[torch.Tensor] = None):
        """Initialize cross-entropy loss.
        
        Args:
            label_smoothing: Label smoothing factor
            weight: Class weights (optional)
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Loss value
        """
        if self.label_smoothing > 0:
            return F.cross_entropy(
                logits, targets, 
                weight=self.weight,
                label_smoothing=self.label_smoothing
            )
        else:
            return F.cross_entropy(logits, targets, weight=self.weight)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.
    
    Focal loss down-weights easy examples and focuses on hard examples.
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        gamma: float = 2.0, 
        reduction: str = "mean"
    ):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Loss value
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class BCELoss(nn.Module):
    """Binary cross-entropy loss for shot prediction.
    
    BCE loss with optional label smoothing and class weighting.
    """
    
    def __init__(
        self, 
        label_smoothing: float = 0.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ):
        """Initialize BCE loss.
        
        Args:
            label_smoothing: Label smoothing factor
            pos_weight: Positive class weight
            reduction: Reduction method
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.
        
        Args:
            logits: Model predictions [N, 1] or [N]
            targets: Ground truth labels [N, 1] or [N]
            
        Returns:
            Loss value
        """
        if self.label_smoothing > 0:
            # Apply label smoothing
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        return F.binary_cross_entropy_with_logits(
            logits, targets_smooth.float(),
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class KLLoss(nn.Module):
    """KL divergence loss for CVAE.
    
    KL divergence between posterior and prior distributions.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        """Initialize KL loss.
        
        Args:
            beta: Weighting factor for KL term (β-VAE)
            reduction: Reduction method
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.
        
        Args:
            mean: Posterior mean [B, latent_dim]
            log_var: Posterior log variance [B, latent_dim]
            
        Returns:
            KL loss value
        """
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp(), dim=1
        )
        
        if self.reduction == "mean":
            return self.beta * kl_loss.mean()
        elif self.reduction == "sum":
            return self.beta * kl_loss.sum()
        else:
            return self.beta * kl_loss


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for CVAE.
    
    L1 or L2 reconstruction loss between input and reconstructed output.
    """
    
    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        """Initialize reconstruction loss.
        
        Args:
            loss_type: Loss type ('mse', 'l1', 'huber')
            reduction: Reduction method
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.
        
        Args:
            pred: Predicted values [B, output_dim]
            target: Target values [B, output_dim]
            
        Returns:
            Reconstruction loss value
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.huber_loss(pred, target, reduction="none", delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ShotGuidanceLoss(nn.Module):
    """Shot probability guidance loss for CVAE.
    
    Guides the generation towards higher/lower shot probabilities
    based on guidance conditions.
    """
    
    def __init__(self, guidance_weight: float = 1.0, guidance_type: str = "score"):
        """Initialize shot guidance loss.
        
        Args:
            guidance_weight: Weight for guidance loss
            guidance_type: Type of guidance ('score', 'gradient')
        """
        super().__init__()
        self.guidance_weight = guidance_weight
        self.guidance_type = guidance_type
    
    def forward(
        self, 
        shot_prob: torch.Tensor, 
        guidance_target: torch.Tensor,
        guidance_lambda: float = 1.0
    ) -> torch.Tensor:
        """Compute shot guidance loss.
        
        Args:
            shot_prob: Predicted shot probability [B, 1]
            guidance_target: Target direction (1 for up, -1 for down) [B, 1]
            guidance_lambda: Guidance strength multiplier
            
        Returns:
            Guidance loss value
        """
        if self.guidance_type == "score":
            # Score-based guidance: encourage higher/lower probabilities
            target_prob = 0.5 + 0.5 * guidance_target * guidance_lambda
            target_prob = torch.clamp(target_prob, 0.0, 1.0)
            
            guidance_loss = F.mse_loss(shot_prob, target_prob)
            
        elif self.guidance_type == "gradient":
            # Gradient-based guidance: encourage probability changes
            if guidance_target > 0:
                # Encourage higher probabilities
                guidance_loss = -torch.log(shot_prob + 1e-8).mean()
            else:
                # Encourage lower probabilities
                guidance_loss = -torch.log(1 - shot_prob + 1e-8).mean()
        else:
            raise ValueError(f"Unknown guidance type: {self.guidance_type}")
        
        return self.guidance_weight * guidance_loss


class MovementConstraintLoss(nn.Module):
    """Movement constraint loss for CVAE.
    
    Constrains the total movement amount in generated formations.
    """
    
    def __init__(self, constraint_weight: float = 1.0, max_movement: float = 10.0):
        """Initialize movement constraint loss.
        
        Args:
            constraint_weight: Weight for constraint loss
            max_movement: Maximum allowed movement per player
        """
        super().__init__()
        self.constraint_weight = constraint_weight
        self.max_movement = max_movement
    
    def forward(self, generated_positions: torch.Tensor, original_positions: torch.Tensor) -> torch.Tensor:
        """Compute movement constraint loss.
        
        Args:
            generated_positions: Generated positions [B, N*2] (x, y for each player)
            original_positions: Original positions [B, N*2] (x, y for each player)
            
        Returns:
            Constraint loss value
        """
        # Reshape to [B, N, 2] for easier computation
        B, total_dim = generated_positions.shape
        N = total_dim // 2
        
        gen_pos = generated_positions.view(B, N, 2)
        orig_pos = original_positions.view(B, N, 2)
        
        # Compute movement distance for each player
        movement = torch.norm(gen_pos - orig_pos, dim=2)  # [B, N]
        
        # Apply constraint: penalize movements exceeding max_movement
        excess_movement = torch.clamp(movement - self.max_movement, min=0.0)
        constraint_loss = excess_movement.mean()
        
        return self.constraint_weight * constraint_loss


class VelocityConstraintLoss(nn.Module):
    """Velocity constraint loss for CVAE.
    
    Constrains the velocity magnitudes in generated formations.
    """
    
    def __init__(self, constraint_weight: float = 1.0, max_velocity: float = 5.0):
        """Initialize velocity constraint loss.
        
        Args:
            constraint_weight: Weight for constraint loss
            max_velocity: Maximum allowed velocity magnitude
        """
        super().__init__()
        self.constraint_weight = constraint_weight
        self.max_velocity = max_velocity
    
    def forward(self, generated_velocities: torch.Tensor) -> torch.Tensor:
        """Compute velocity constraint loss.
        
        Args:
            generated_velocities: Generated velocities [B, N*2] (vx, vy for each player)
            
        Returns:
            Constraint loss value
        """
        # Reshape to [B, N, 2] for easier computation
        B, total_dim = generated_velocities.shape
        N = total_dim // 2
        
        gen_vel = generated_velocities.view(B, N, 2)
        
        # Compute velocity magnitude for each player
        velocity_magnitude = torch.norm(gen_vel, dim=2)  # [B, N]
        
        # Apply constraint: penalize velocities exceeding max_velocity
        excess_velocity = torch.clamp(velocity_magnitude - self.max_velocity, min=0.0)
        constraint_loss = excess_velocity.mean()
        
        return self.constraint_weight * constraint_loss


class EnhancedCVAELoss(nn.Module):
    """Enhanced CVAE loss with guidance and constraints.
    
    Combines reconstruction loss, KL divergence, shot guidance, and movement constraints.
    """
    
    def __init__(
        self,
        recon_loss_type: str = "mse",
        beta: float = 1.0,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
        guidance_weight: float = 0.1,
        movement_weight: float = 0.1,
        velocity_weight: float = 0.1,
        guidance_type: str = "score",
        max_movement: float = 10.0,
        max_velocity: float = 5.0,
    ):
        """Initialize enhanced CVAE loss.
        
        Args:
            recon_loss_type: Reconstruction loss type
            beta: Beta parameter for KL loss
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL loss
            guidance_weight: Weight for shot guidance loss
            movement_weight: Weight for movement constraint loss
            velocity_weight: Weight for velocity constraint loss
            guidance_type: Type of shot guidance
            max_movement: Maximum allowed movement per player
            max_velocity: Maximum allowed velocity magnitude
        """
        super().__init__()
        self.recon_loss = ReconstructionLoss(recon_loss_type)
        self.kl_loss = KLLoss(beta)
        self.guidance_loss = ShotGuidanceLoss(guidance_weight, guidance_type)
        self.movement_loss = MovementConstraintLoss(movement_weight, max_movement)
        self.velocity_loss = VelocityConstraintLoss(velocity_weight, max_velocity)
        
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.guidance_weight = guidance_weight
        self.movement_weight = movement_weight
        self.velocity_weight = velocity_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        shot_prob: Optional[torch.Tensor] = None,
        guidance_target: Optional[torch.Tensor] = None,
        guidance_lambda: float = 1.0,
        original_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Compute enhanced CVAE loss.
        
        Args:
            pred: Predicted values [B, output_dim]
            target: Target values [B, output_dim]
            mean: Latent mean [B, latent_dim]
            log_var: Latent log variance [B, latent_dim]
            shot_prob: Predicted shot probability [B, 1] (optional)
            guidance_target: Target direction for guidance [B, 1] (optional)
            guidance_lambda: Guidance strength multiplier
            original_positions: Original positions for movement constraint [B, N*2] (optional)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Basic CVAE losses
        recon_loss = self.recon_loss(pred, target)
        kl_loss = self.kl_loss(mean, log_var)
        
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        loss_dict = {
            "reconstruction": recon_loss,
            "kl_divergence": kl_loss,
        }
        
        # Shot guidance loss
        if shot_prob is not None and guidance_target is not None:
            guidance_loss = self.guidance_loss(shot_prob, guidance_target, guidance_lambda)
            total_loss += guidance_loss
            loss_dict["shot_guidance"] = guidance_loss
        
        # Movement constraint loss
        if original_positions is not None:
            # Assume first half of output_dim is positions
            pred_positions = pred[:, :pred.shape[1]//2]
            movement_loss = self.movement_loss(pred_positions, original_positions)
            total_loss += movement_loss
            loss_dict["movement_constraint"] = movement_loss
        
        # Velocity constraint loss
        if pred.shape[1] > target.shape[1]:  # If velocity is included
            pred_velocities = pred[:, pred.shape[1]//2:]
            velocity_loss = self.velocity_loss(pred_velocities)
            total_loss += velocity_loss
            loss_dict["velocity_constraint"] = velocity_loss
        
        loss_dict["total"] = total_loss
        
        return total_loss, loss_dict


class CVAELoss(nn.Module):
    """Combined CVAE loss.
    
    Combines reconstruction loss and KL divergence loss.
    """
    
    def __init__(
        self,
        recon_loss_type: str = "mse",
        beta: float = 1.0,
        recon_weight: float = 1.0,
        kl_weight: float = 1.0,
    ):
        """Initialize CVAE loss.
        
        Args:
            recon_loss_type: Reconstruction loss type
            beta: Beta parameter for KL loss
            recon_weight: Weight for reconstruction loss
            kl_weight: Weight for KL loss
        """
        super().__init__()
        self.recon_loss = ReconstructionLoss(recon_loss_type)
        self.kl_loss = KLLoss(beta)
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute CVAE loss.
        
        Args:
            pred: Predicted values [B, output_dim]
            target: Target values [B, output_dim]
            mean: Latent mean [B, latent_dim]
            log_var: Latent log variance [B, latent_dim]
            
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        recon_loss = self.recon_loss(pred, target)
        kl_loss = self.kl_loss(mean, log_var)
        
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss


class CombinedLoss(nn.Module):
    """Combined loss for multi-task learning.
    
    Combines multiple loss functions with different weights.
    """
    
    def __init__(self, losses: dict, weights: Optional[dict] = None):
        """Initialize combined loss.
        
        Args:
            losses: Dictionary of loss functions
            weights: Dictionary of loss weights (optional)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        self.weights = weights or {name: 1.0 for name in losses.keys()}
    
    def forward(self, **kwargs) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            **kwargs: Arguments for loss functions
            
        Returns:
            Tuple of (total_loss, individual_losses)
        """
        individual_losses = {}
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(**kwargs)
            individual_losses[name] = loss_value
            total_loss += self.weights[name] * loss_value
        
        return total_loss, individual_losses


class AdaptiveLoss(nn.Module):
    """Adaptive loss that automatically balances multiple loss terms.
    
    Uses uncertainty weighting to automatically balance loss terms.
    """
    
    def __init__(self, num_losses: int, initial_weight: float = 1.0):
        """Initialize adaptive loss.
        
        Args:
            num_losses: Number of loss terms
            initial_weight: Initial weight value
        """
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        self.initial_weight = initial_weight
    
    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive loss.
        
        Args:
            losses: List of loss values
            
        Returns:
            Tuple of (total_loss, weighted_losses)
        """
        if len(losses) != len(self.log_vars):
            raise ValueError(f"Expected {len(self.log_vars)} losses, got {len(losses)}")
        
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        total_loss = sum(weighted_losses)
        return total_loss, torch.stack(weighted_losses)
