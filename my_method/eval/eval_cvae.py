"""Evaluation script for CVAE tactic generation task.

This script evaluates a trained CVAE model on tactic generation.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from tacticai.models import CVAEModel
from tacticai.dataio import CVAEDataset, create_dataloader
from tacticai.modules import (
    ReconstructionLoss, KLLoss,
    get_device, load_checkpoint, setup_logging,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create CVAE model.
    
    Args:
        config: Model configuration
        device: Device to place model on
        
    Returns:
        CVAE model
    """
    model_config = config["model"]
    
    model = CVAEModel(
        input_dim=model_config["input_dim"],
        condition_dim=model_config["condition_dim"],
        latent_dim=model_config["latent_dim"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        dropout=0.0,  # No dropout during evaluation
    )
    
    return model.to(device)


def evaluate_model_with_guidance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    guidance_lambda_values: list[float] = [0.0, 0.5, 1.0, 2.0],
) -> Dict[str, Any]:
    """Evaluate model with shot probability guidance.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        guidance_lambda_values: List of guidance strength values to test
        
    Returns:
        Dictionary of evaluation metrics with guidance analysis
    """
    model.eval()
    
    results = {}
    
    for guidance_lambda in guidance_lambda_values:
        print(f"\nEvaluating with guidance lambda = {guidance_lambda}")
        
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_guidance_loss = 0.0
        total_movement_loss = 0.0
        
        shot_prob_changes = []
        total_movements = []
        
        recon_loss_fn = ReconstructionLoss("mse")
        kl_loss_fn = KLLoss(beta=1.0)
        
        with torch.no_grad():
            for data, targets in tqdm(dataloader, desc=f"Lambda {guidance_lambda}"):
                # Move data to device
                data = {k: v.to(device) for k, v in data.items()}
                targets = targets.to(device)
                
                # Create guidance conditions (up/down)
                batch_size = data["batch"].max().item() + 1
                guidance_target = torch.tensor([1.0, -1.0] * (batch_size // 2 + 1))[:batch_size].unsqueeze(1).to(device)
                
                # Forward pass with guidance
                outputs, mean, log_var, shot_prob = model(
                    data["x"], data["edge_index"], data["batch"],
                    data["conditions"], training=False,
                    shot_guidance_conditions=guidance_target
                )
                
                # Reshape targets to match outputs
                targets_flat = targets.view(batch_size, -1)
                
                # Compute losses
                recon_loss = recon_loss_fn(outputs, targets_flat)
                kl_loss = kl_loss_fn(mean, log_var)
                
                # Compute guidance loss
                if shot_prob is not None:
                    guidance_loss = F.mse_loss(shot_prob, guidance_target)
                    total_guidance_loss += guidance_loss.item()
                    
                    # Track shot probability changes
                    shot_prob_changes.append(shot_prob.cpu().numpy())
                
                # Compute movement constraint
                if outputs.shape[1] > targets_flat.shape[1]:  # If velocity is included
                    pred_positions = outputs[:, :targets_flat.shape[1]]
                    movement_distances = torch.norm(
                        pred_positions.view(batch_size, -1, 2) - targets_flat.view(batch_size, -1, 2),
                        dim=2
                    )
                    total_movement = movement_distances.sum(dim=1).mean()
                    total_movements.append(total_movement.item())
                    
                    movement_loss = torch.clamp(movement_distances - 10.0, min=0.0).mean()
                    total_movement_loss += movement_loss.item()
                
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        # Compute metrics
        num_batches = len(dataloader)
        results[f"lambda_{guidance_lambda}"] = {
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "guidance_loss": total_guidance_loss / num_batches,
            "movement_loss": total_movement_loss / num_batches,
            "total_loss": (total_recon_loss + total_kl_loss + total_guidance_loss + total_movement_loss) / num_batches,
        }
        
        if shot_prob_changes:
            shot_prob_changes = np.concatenate(shot_prob_changes)
            results[f"lambda_{guidance_lambda}"]["shot_prob_mean"] = shot_prob_changes.mean()
            results[f"lambda_{guidance_lambda}"]["shot_prob_std"] = shot_prob_changes.std()
        
        if total_movements:
            results[f"lambda_{guidance_lambda}"]["total_movement_mean"] = np.mean(total_movements)
            results[f"lambda_{guidance_lambda}"]["total_movement_std"] = np.std(total_movements)
        
        print(f"  Reconstruction Loss: {results[f'lambda_{guidance_lambda}']['recon_loss']:.4f}")
        print(f"  KL Loss: {results[f'lambda_{guidance_lambda}']['kl_loss']:.4f}")
        print(f"  Guidance Loss: {results[f'lambda_{guidance_lambda}']['guidance_loss']:.4f}")
        if 'shot_prob_mean' in results[f"lambda_{guidance_lambda}"]:
            print(f"  Shot Probability: {results[f'lambda_{guidance_lambda}']['shot_prob_mean']:.4f} ± {results[f'lambda_{guidance_lambda}']['shot_prob_std']:.4f}")
        if 'total_movement_mean' in results[f"lambda_{guidance_lambda}"]:
            print(f"  Total Movement: {results[f'lambda_{guidance_lambda}']['total_movement_mean']:.4f} ± {results[f'lambda_{guidance_lambda}']['total_movement_std']:.4f}")
    
    return results


def plot_guidance_analysis(
    guidance_results: Dict[str, Any],
    save_path: str = None,
) -> None:
    """Plot guidance analysis results.
    
    Args:
        guidance_results: Results from guidance evaluation
        save_path: Path to save plots
    """
    lambda_values = []
    shot_probs = []
    total_movements = []
    recon_losses = []
    
    for key, metrics in guidance_results.items():
        if key.startswith("lambda_"):
            lambda_val = float(key.split("_")[1])
            lambda_values.append(lambda_val)
            shot_probs.append(metrics.get("shot_prob_mean", 0.5))
            total_movements.append(metrics.get("total_movement_mean", 0.0))
            recon_losses.append(metrics["recon_loss"])
    
    # Sort by lambda values
    sorted_indices = np.argsort(lambda_values)
    lambda_values = [lambda_values[i] for i in sorted_indices]
    shot_probs = [shot_probs[i] for i in sorted_indices]
    total_movements = [total_movements[i] for i in sorted_indices]
    recon_losses = [recon_losses[i] for i in sorted_indices]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Shot probability vs guidance lambda
    ax1.plot(lambda_values, shot_probs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Guidance Lambda')
    ax1.set_ylabel('Shot Probability')
    ax1.set_title('Shot Probability vs Guidance Strength')
    ax1.grid(True, alpha=0.3)
    
    # Total movement vs guidance lambda
    ax2.plot(lambda_values, total_movements, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Guidance Lambda')
    ax2.set_ylabel('Total Movement')
    ax2.set_title('Total Movement vs Guidance Strength')
    ax2.grid(True, alpha=0.3)
    
    # Reconstruction loss vs guidance lambda
    ax3.plot(lambda_values, recon_losses, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Guidance Lambda')
    ax3.set_ylabel('Reconstruction Loss')
    ax3.set_title('Reconstruction Loss vs Guidance Strength')
    ax3.grid(True, alpha=0.3)
    
    # Scatter plot: shot probability vs total movement
    ax4.scatter(shot_probs, total_movements, c=lambda_values, cmap='viridis', s=100)
    ax4.set_xlabel('Shot Probability')
    ax4.set_ylabel('Total Movement')
    ax4.set_title('Shot Probability vs Total Movement')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for lambda values
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Guidance Lambda')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Guidance analysis plots saved to {save_path}")
    
    plt.show()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    all_reconstructions = []
    all_targets = []
    
    recon_loss_fn = ReconstructionLoss("mse")
    kl_loss_fn = KLLoss(beta=1.0)
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            outputs, mean, log_var = model(
                data["x"], data["edge_index"], data["batch"],
                data["conditions"], training=False
            )
            
            # Reshape targets to match outputs (flatten player positions)
            batch_size = data["batch"].max().item() + 1
            targets_flat = targets.view(batch_size, -1)
            
            recon_loss = recon_loss_fn(outputs, targets_flat)
            kl_loss = kl_loss_fn(mean, log_var)
            
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            all_reconstructions.append(outputs.cpu())
            all_targets.append(targets_flat.cpu())
    
    # Concatenate results
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute additional metrics
    mse = torch.mean((all_reconstructions - all_targets) ** 2).item()
    mae = torch.mean(torch.abs(all_reconstructions - all_targets)).item()
    
    eval_metrics = {
        "recon_loss": total_recon_loss / len(dataloader),
        "kl_loss": total_kl_loss / len(dataloader),
        "total_loss": (total_recon_loss + total_kl_loss) / len(dataloader),
        "mse": mse,
        "mae": mae,
    }
    
    return eval_metrics


def generate_samples(
    model: nn.Module,
    conditions: torch.Tensor,
    device: torch.device,
    num_samples: int = 16,
    save_path: str = None,
) -> torch.Tensor:
    """Generate tactical formations from conditions.
    
    Args:
        model: CVAE model
        conditions: Input conditions
        device: Device to generate on
        num_samples: Number of samples to generate per condition
        save_path: Path to save generated samples
        
    Returns:
        Generated formations
    """
    model.eval()
    
    with torch.no_grad():
        generated = model.generate(
            conditions.to(device),
            num_samples=num_samples,
            device=device
        )
    
    # Reshape to [num_conditions * num_samples, num_players, 2]
    num_conditions = conditions.size(0)
    generated = generated.view(num_conditions * num_samples, -1, 2)
    
    # Convert from normalized coordinates to field coordinates
    field_length = 105.0
    field_width = 68.0
    generated[:, :, 0] *= field_length
    generated[:, :, 1] *= field_width
    
    # Plot generated formations
    if save_path:
        plot_generated_formations(generated, conditions, save_path)
    
    return generated


def plot_generated_formations(
    generated: torch.Tensor,
    conditions: torch.Tensor,
    save_path: str,
) -> None:
    """Plot generated tactical formations.
    
    Args:
        generated: Generated formations [N, num_players, 2]
        conditions: Input conditions
        save_path: Path to save plot
    """
    num_conditions = conditions.size(0)
    num_samples = generated.size(0) // num_conditions
    num_players = generated.size(1)
    
    # Create subplots
    fig, axes = plt.subplots(num_conditions, num_samples, figsize=(4 * num_samples, 3 * num_conditions))
    if num_conditions == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['red', 'blue'] * (num_players // 2)
    
    for i in range(num_conditions):
        for j in range(num_samples):
            idx = i * num_samples + j
            formation = generated[idx].numpy()
            
            ax = axes[i, j]
            
            # Plot field
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 68)
            ax.set_aspect('equal')
            
            # Plot players
            for k in range(num_players):
                team = k // (num_players // 2)
                color = colors[k]
                marker = 'o' if team == 0 else 's'
                ax.scatter(formation[k, 0], formation[k, 1], c=color, marker=marker, s=50)
            
            ax.set_title(f"Condition {i+1}, Sample {j+1}")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def reconstruct_samples(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 5,
    save_path: str = None,
) -> None:
    """Reconstruct samples and compare with originals.
    
    Args:
        model: CVAE model
        dataloader: Data loader
        device: Device to reconstruct on
        num_samples: Number of samples to reconstruct
        save_path: Path to save reconstruction plots
    """
    model.eval()
    
    sample_count = 0
    with torch.no_grad():
        for data, targets in dataloader:
            if sample_count >= num_samples:
                break
            
            # Move data to device
            data = {k: v.to(device) for k, v in data.items()}
            targets = targets.to(device)
            
            # Reconstruct
            reconstructed = model.reconstruct(
                data["x"], data["edge_index"], data["batch"], data["conditions"]
            )
            
            # Convert to field coordinates
            field_length = 105.0
            field_width = 68.0
            
            original = targets[0].numpy()  # [num_players, 2]
            original[:, 0] *= field_length
            original[:, 1] *= field_width
            
            recon = reconstructed[0].view(-1, 2).numpy()  # [num_players, 2]
            recon[:, 0] *= field_length
            recon[:, 1] *= field_width
            
            # Plot comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            colors = ['red', 'blue'] * (len(original) // 2)
            
            # Original
            ax1.set_xlim(0, 105)
            ax1.set_ylim(0, 68)
            ax1.set_aspect('equal')
            for k in range(len(original)):
                team = k // (len(original) // 2)
                color = colors[k]
                marker = 'o' if team == 0 else 's'
                ax1.scatter(original[k, 0], original[k, 1], c=color, marker=marker, s=50)
            ax1.set_title("Original Formation")
            ax1.grid(True, alpha=0.3)
            
            # Reconstructed
            ax2.set_xlim(0, 105)
            ax2.set_ylim(0, 68)
            ax2.set_aspect('equal')
            for k in range(len(recon)):
                team = k // (len(recon) // 2)
                color = colors[k]
                marker = 'o' if team == 0 else 's'
                ax2.scatter(recon[k, 0], recon[k, 1], c=color, marker=marker, s=50)
            ax2.set_title("Reconstructed Formation")
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f"Sample {sample_count + 1}")
            plt.tight_layout()
            
            if save_path:
                sample_save_path = f"{save_path}_sample_{sample_count + 1}.png"
                plt.savefig(sample_save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
            sample_count += 1


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CVAE model with guidance")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_path", type=str, help="Path to test data (overrides config)")
    parser.add_argument("--generate", action="store_true", help="Generate new formations")
    parser.add_argument("--reconstruct", action="store_true", help="Reconstruct existing formations")
    parser.add_argument("--guidance", action="store_true", help="Evaluate with shot probability guidance")
    parser.add_argument("--samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--num_reconstruct", type=int, default=5, help="Number of samples to reconstruct")
    parser.add_argument("--guidance_lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0], help="Guidance lambda values to test")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO")
    )
    
    logger.info(f"Evaluating CVAE model on {device}")
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataset
    test_path = args.test_path or config["data"]["test_path"]
    test_dataset = CVAEDataset(
        test_path,
        file_format=config["data"].get("format", "parquet")
    )
    
    # Create test data loader
    test_loader = create_dataloader(
        test_dataset,
        batch_size=config["eval"].get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    eval_metrics = evaluate_model(model, test_loader, device)
    
    # Log results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Reconstruction Loss: {eval_metrics['recon_loss']:.4f}")
    logger.info(f"KL Loss: {eval_metrics['kl_loss']:.4f}")
    logger.info(f"Total Loss: {eval_metrics['total_loss']:.4f}")
    logger.info(f"MSE: {eval_metrics['mse']:.4f}")
    logger.info(f"MAE: {eval_metrics['mae']:.4f}")
    
    # Evaluate with guidance if requested
    if args.guidance:
        logger.info("Evaluating with shot probability guidance...")
        guidance_results = evaluate_model_with_guidance(
            model, test_loader, device, args.guidance_lambdas
        )
        
        # Plot guidance analysis
        guidance_plot_path = Path(config.get("log_dir", "runs")) / "guidance_analysis.png"
        plot_guidance_analysis(guidance_results, str(guidance_plot_path))
        
        # Save guidance results
        guidance_results_path = Path(config.get("log_dir", "runs")) / "guidance_results.json"
        import json
        with open(guidance_results_path, 'w') as f:
            json.dump(guidance_results, f, indent=2)
        logger.info(f"Guidance results saved to {guidance_results_path}")
    
    # Generate samples if requested
    if args.generate:
        logger.info("Generating new formations...")
        
        # Create dummy conditions
        condition_dim = config["model"]["condition_dim"]
        conditions = torch.randn(2, condition_dim)  # 2 different conditions
        
        save_path = Path(config.get("log_dir", "runs")) / "generated_formations.png"
        generated = generate_samples(model, conditions, device, args.samples, str(save_path))
        logger.info(f"Generated {generated.size(0)} formations")
    
    # Reconstruct samples if requested
    if args.reconstruct:
        logger.info("Reconstructing existing formations...")
        
        save_path = Path(config.get("log_dir", "runs")) / "reconstructed_formations"
        reconstruct_samples(model, test_loader, device, args.num_reconstruct, str(save_path))
    
    # Save results
    results_path = Path(config.get("log_dir", "runs")) / "cvae_eval_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
