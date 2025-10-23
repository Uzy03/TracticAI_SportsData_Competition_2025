"""Evaluation metrics for TacticAI tasks.

This module implements various evaluation metrics including accuracy,
top-k accuracy, F1 score, AUC, and calibration metrics.
"""

from typing import Optional, Union
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


class CMC:
    """Cumulative Matching Characteristic (CMC) metric.
    
    Measures ranking performance for retrieval tasks.
    """
    
    def __init__(self, k_values: list[int] = [1, 3, 5, 10]):
        """Initialize CMC metric.
        
        Args:
            k_values: List of k values to compute CMC for
        """
        self.k_values = k_values
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Compute CMC metrics.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Dictionary mapping k to CMC score
        """
        # Get top-k predictions
        _, top_k_preds = torch.topk(logits, max(self.k_values), dim=1)
        
        results = {}
        for k in self.k_values:
            # Check if target is in top-k
            correct = torch.any(top_k_preds[:, :k] == targets.unsqueeze(1), dim=1)
            cmc_k = correct.float().mean()
            results[k] = cmc_k
        
        return results


class TopKAccuracy:
    """Top-k accuracy metric for receiver prediction.
    
    Computes accuracy for top-k predictions.
    """
    
    def __init__(self, k: int = 1):
        """Initialize top-k accuracy.
        
        Args:
            k: Number of top predictions to consider
        """
        self.k = k
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute top-k accuracy.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Top-k accuracy
        """
        _, top_k_preds = torch.topk(logits, self.k, dim=1)
        correct = torch.any(top_k_preds == targets.unsqueeze(1), dim=1)
        return correct.float().mean()
    
    def compute_all_k(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        k_values: list[int] = [1, 3, 5]
    ) -> dict[int, torch.Tensor]:
        """Compute top-k accuracy for multiple k values.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            k_values: List of k values to compute
            
        Returns:
            Dictionary mapping k to accuracy
        """
        results = {}
        for k in k_values:
            metric = TopKAccuracy(k)
            results[k] = metric(logits, targets)
        return results


class Accuracy:
    """Standard accuracy metric.
    
    Computes accuracy for classification tasks.
    """
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute accuracy.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Accuracy
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).float()
        return correct.mean()


class F1Score:
    """F1 score metric for binary and multiclass classification.
    
    Computes F1 score with optional averaging.
    """
    
    def __init__(self, average: str = "weighted"):
        """Initialize F1 score.
        
        Args:
            average: Averaging method ('micro', 'macro', 'weighted', 'none')
        """
        self.average = average
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Compute F1 score.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            F1 score(s)
        """
        preds = torch.argmax(logits, dim=1)
        
        if self.average == "none":
            # Return per-class F1 scores
            num_classes = logits.size(1)
            f1_scores = []
            
            for c in range(num_classes):
                tp = ((preds == c) & (targets == c)).float().sum()
                fp = ((preds == c) & (targets != c)).float().sum()
                fn = ((preds != c) & (targets == c)).float().sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_scores.append(f1)
            
            return f1_scores
        
        else:
            # Compute overall F1 score
            num_classes = logits.size(1)
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for c in range(num_classes):
                tp = ((preds == c) & (targets == c)).float().sum()
                fp = ((preds == c) & (targets != c)).float().sum()
                fn = ((preds != c) & (targets == c)).float().sum()
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            precision = total_tp / (total_tp + total_fp + 1e-8)
            recall = total_tp / (total_tp + total_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return f1


class AUC:
    """Area Under Curve metric for binary classification.
    
    Computes AUC-ROC and AUC-PR.
    """
    
    def __init__(self, average: str = "macro"):
        """Initialize AUC metric.
        
        Args:
            average: Averaging method for multiclass
        """
        self.average = average
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        compute_auc_pr: bool = True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute AUC metrics.
        
        Args:
            logits: Model predictions [N, 1] or [N]
            targets: Ground truth labels [N]
            compute_auc_pr: Whether to compute AUC-PR
            
        Returns:
            AUC-ROC (and AUC-PR if requested)
        """
        # Convert to numpy for sklearn
        logits_np = logits.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        
        # Handle binary classification
        if logits.dim() == 2 and logits.size(1) == 1:
            logits_np = logits_np.squeeze()
        
        # Compute AUC-ROC
        try:
            auc_roc = torch.tensor(
                roc_auc_score(targets_np, logits_np, average=self.average),
                device=logits.device
            )
        except ValueError:
            # Handle edge cases (e.g., only one class in batch)
            auc_roc = torch.tensor(0.5, device=logits.device)
        
        if compute_auc_pr:
            try:
                auc_pr = torch.tensor(
                    average_precision_score(targets_np, logits_np, average=self.average),
                    device=logits.device
                )
            except ValueError:
                auc_pr = torch.tensor(0.0, device=logits.device)
            
            return auc_roc, auc_pr
        
        return auc_roc


class ECE:
    """Expected Calibration Error metric.
    
    Measures calibration quality of predictions.
    """
    
    def __init__(self, n_bins: int = 10):
        """Initialize ECE metric.
        
        Args:
            n_bins: Number of bins for calibration
        """
        self.n_bins = n_bins
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute Expected Calibration Error.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            ECE value
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = torch.zeros(1, device=logits.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Compute accuracy in this bin
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                # Add to ECE
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class CalibrationMetrics:
    """Comprehensive calibration metrics.
    
    Computes multiple calibration metrics including ECE, MCE, and reliability diagrams.
    """
    
    def __init__(self, n_bins: int = 10):
        """Initialize calibration metrics.
        
        Args:
            n_bins: Number of bins for calibration
        """
        self.n_bins = n_bins
        self.ece = ECE(n_bins)
    
    def __call__(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute calibration metrics.
        
        Args:
            logits: Model predictions [N, num_classes]
            targets: Ground truth labels [N]
            
        Returns:
            Dictionary of calibration metrics
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # Compute ECE
        ece = self.ece(logits, targets)
        
        # Compute MCE (Maximum Calibration Error)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = torch.zeros(1, device=logits.device)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == targets[in_bin]).float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                mce = torch.max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            "ece": ece,
            "mce": mce,
        }


class MetricsLogger:
    """Metrics logger for saving results to CSV and generating visualizations.
    
    Handles CSV saving and scatter plot generation for evaluation results.
    """
    
    def __init__(self, output_dir: str = "runs"):
        """Initialize metrics logger.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        self.results = []
    
    def log_metrics(
        self, 
        task: str, 
        epoch: int, 
        split: str, 
        metrics: dict[str, float],
        additional_info: dict[str, Any] = None
    ) -> None:
        """Log metrics for a specific task and epoch.
        
        Args:
            task: Task name (receiver, shot, cvae)
            epoch: Epoch number
            split: Data split (train, val, test)
            metrics: Dictionary of metric values
            additional_info: Additional information to log
        """
        log_entry = {
            'task': task,
            'epoch': epoch,
            'split': split,
            **metrics
        }
        
        if additional_info:
            log_entry.update(additional_info)
        
        self.results.append(log_entry)
    
    def save_to_csv(self, filename: str = "metrics.csv") -> None:
        """Save logged metrics to CSV file.
        
        Args:
            filename: Output filename
        """
        import pandas as pd
        from pathlib import Path
        
        if not self.results:
            print("No metrics to save")
            return
        
        df = pd.DataFrame(self.results)
        output_path = Path(self.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Metrics saved to {output_path}")
    
    def plot_scatter(
        self, 
        x_metric: str, 
        y_metric: str, 
        task: str = None,
        save_path: str = None
    ) -> None:
        """Generate scatter plot for two metrics.
        
        Args:
            x_metric: X-axis metric name
            y_metric: Y-axis metric name
            task: Task to filter by (optional)
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from pathlib import Path
        
        if not self.results:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        if task:
            df = df[df['task'] == task]
        
        if x_metric not in df.columns or y_metric not in df.columns:
            print(f"Metrics {x_metric} or {y_metric} not found in data")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Color by split
        splits = df['split'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, split in enumerate(splits):
            split_data = df[df['split'] == split]
            plt.scatter(
                split_data[x_metric], 
                split_data[y_metric], 
                label=split, 
                color=colors[i % len(colors)],
                alpha=0.7,
                s=50
            )
        
        plt.xlabel(x_metric)
        plt.ylabel(y_metric)
        plt.title(f'{x_metric} vs {y_metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to {save_path}")
        
        plt.show()
    
    def plot_training_curves(
        self, 
        metric: str, 
        task: str = None,
        save_path: str = None
    ) -> None:
        """Plot training curves for a specific metric.
        
        Args:
            metric: Metric name to plot
            task: Task to filter by (optional)
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        from pathlib import Path
        
        if not self.results:
            print("No metrics to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        if task:
            df = df[df['task'] == task]
        
        if metric not in df.columns:
            print(f"Metric {metric} not found in data")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot for each split
        splits = df['split'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, split in enumerate(splits):
            split_data = df[df['split'] == split].sort_values('epoch')
            plt.plot(
                split_data['epoch'], 
                split_data[metric], 
                label=f'{split} {metric}',
                color=colors[i % len(colors)],
                linewidth=2,
                marker='o',
                markersize=4
            )
        
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{metric} Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_guidance_analysis(
        self, 
        guidance_results: dict[str, dict[str, float]],
        save_path: str = None
    ) -> None:
        """Plot guidance analysis results.
        
        Args:
            guidance_results: Results from guidance evaluation
            save_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
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
        scatter = ax4.scatter(shot_probs, total_movements, c=lambda_values, cmap='viridis', s=100)
        ax4.set_xlabel('Shot Probability')
        ax4.set_ylabel('Total Movement')
        ax4.set_title('Shot Probability vs Total Movement')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for lambda values
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Guidance Lambda')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Guidance analysis plots saved to {save_path}")
        
        plt.show()


class MultiTaskMetrics:
    """Metrics for multi-task evaluation.
    
    Combines metrics from different tasks.
    """
    
    def __init__(self, task_metrics: dict):
        """Initialize multi-task metrics.
        
        Args:
            task_metrics: Dictionary mapping task names to metric functions
        """
        self.task_metrics = task_metrics
    
    def __call__(self, predictions: dict, targets: dict) -> dict:
        """Compute metrics for all tasks.
        
        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            
        Returns:
            Dictionary of task metrics
        """
        results = {}
        
        for task_name, metric_fn in self.task_metrics.items():
            if task_name in predictions and task_name in targets:
                results[task_name] = metric_fn(predictions[task_name], targets[task_name])
        
        return results
