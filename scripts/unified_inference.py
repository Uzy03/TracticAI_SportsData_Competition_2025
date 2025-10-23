#!/usr/bin/env python3
"""Unified inference API for TacticAI tasks.

This script provides a unified interface for:
- inference_receiver: Receiver prediction
- inference_shot_marginalized: Shot prediction with marginalization
- inference_generate_k: CVAE generation with k samples

Supports 4-view processing and ablation switching via --views parameter.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from tqdm import tqdm

from tacticai.models import (
    GATv2Network, GATv2Network4View, ReceiverHead, ShotHead, CVAEModel
)
from tacticai.dataio import (
    ReceiverDataset, ShotDataset, CVAEDataset,
    create_dataloader
)
from tacticai.modules import (
    get_device, load_checkpoint, setup_logging,
    GroupPoolingWrapper
)


class UnifiedInferenceAPI:
    """Unified inference API for all TacticAI tasks.
    
    Supports 4-view processing and ablation studies.
    """
    
    def __init__(self, config_path: str, device: str = "auto"):
        """Initialize unified inference API.
        
        Args:
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.config = self._load_config(config_path)
        self.device = get_device(device)
        self.logger = self._setup_logging()
        
        # Load models
        self.receiver_model = None
        self.shot_model = None
        self.generation_model = None
        
        # View processing configuration
        self.view_config = {
            "enabled": True,
            "num_views": 4,
            "view_mixing": "attention",  # attention, conv1d, mlp
            "weight_sharing": True
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for inference."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger("UnifiedInferenceAPI")
    
    def _create_4view_model(self, base_model: nn.Module, task: str) -> nn.Module:
        """Create 4-view model wrapper.
        
        Args:
            base_model: Base model to wrap
            task: Task name (receiver, shot, generation)
            
        Returns:
            4-view model wrapper
        """
        if task == "receiver":
            return GATv2Network4View(
                input_dim=self.config["model"]["input_dim"],
                hidden_dim=self.config["model"]["hidden_dim"],
                output_dim=self.config["model"]["hidden_dim"],
                num_layers=self.config["model"]["num_layers"],
                num_heads=self.config["model"]["num_heads"],
                dropout=0.0,  # No dropout during inference
                view_mixing=self.view_config["view_mixing"],
                weight_sharing=self.view_config["weight_sharing"],
            )
        elif task == "shot":
            return GATv2Network4View(
                input_dim=self.config["model"]["input_dim"],
                hidden_dim=self.config["model"]["hidden_dim"],
                output_dim=self.config["model"]["hidden_dim"],
                num_layers=self.config["model"]["num_layers"],
                num_heads=self.config["model"]["num_heads"],
                dropout=0.0,
                view_mixing=self.view_config["view_mixing"],
                weight_sharing=self.view_config["weight_sharing"],
            )
        elif task == "generation":
            # For CVAE, we need to modify the encoder to use 4-view processing
            return base_model  # CVAE already supports 4-view processing
        
        return base_model
    
    def load_receiver_model(self, checkpoint_path: str, use_4view: bool = True) -> None:
        """Load receiver prediction model.
        
        Args:
            checkpoint_path: Path to receiver model checkpoint
            use_4view: Whether to use 4-view processing
        """
        self.logger.info(f"Loading receiver model from {checkpoint_path}")
        
        # Create base model
        base_model = GATv2Network(
            input_dim=self.config["model"]["input_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            output_dim=self.config["model"]["hidden_dim"],
            num_layers=self.config["model"]["num_layers"],
            num_heads=self.config["model"]["num_heads"],
            dropout=0.0,
            readout="mean",
            residual=True,
        )
        
        receiver_head = ReceiverHead(
            input_dim=self.config["model"]["hidden_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            num_classes=self.config["model"]["num_classes"],
            dropout=0.0,
        )
        
        if use_4view:
            # Create 4-view model
            backbone = self._create_4view_model(base_model, "receiver")
            model = nn.ModuleDict({
                'backbone': backbone,
                'head': receiver_head
            })
        else:
            model = nn.ModuleDict({
                'backbone': base_model,
                'head': receiver_head
            })
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model=model, device=self.device)
        model.eval()
        
        self.receiver_model = model
        self.logger.info("Receiver model loaded successfully")
    
    def load_shot_model(self, checkpoint_path: str, use_4view: bool = True) -> None:
        """Load shot prediction model.
        
        Args:
            checkpoint_path: Path to shot model checkpoint
            use_4view: Whether to use 4-view processing
        """
        self.logger.info(f"Loading shot model from {checkpoint_path}")
        
        # Create base model
        base_model = GATv2Network(
            input_dim=self.config["model"]["input_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            output_dim=self.config["model"]["hidden_dim"],
            num_layers=self.config["model"]["num_layers"],
            num_heads=self.config["model"]["num_heads"],
            dropout=0.0,
            readout="mean",
            residual=True,
        )
        
        shot_head = ShotHead(
            input_dim=self.config["model"]["hidden_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            dropout=0.0,
        )
        
        if use_4view:
            # Create 4-view model
            backbone = self._create_4view_model(base_model, "shot")
            model = nn.ModuleDict({
                'backbone': backbone,
                'head': shot_head
            })
        else:
            model = nn.ModuleDict({
                'backbone': base_model,
                'head': shot_head
            })
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model=model, device=self.device)
        model.eval()
        
        self.shot_model = model
        self.logger.info("Shot model loaded successfully")
    
    def load_generation_model(self, checkpoint_path: str, use_4view: bool = True) -> None:
        """Load CVAE generation model.
        
        Args:
            checkpoint_path: Path to CVAE model checkpoint
            use_4view: Whether to use 4-view processing
        """
        self.logger.info(f"Loading generation model from {checkpoint_path}")
        
        # Create CVAE model
        model = CVAEModel(
            input_dim=self.config["model"]["input_dim"],
            condition_dim=self.config["model"]["condition_dim"],
            latent_dim=self.config["model"]["latent_dim"],
            output_dim=self.config["model"]["output_dim"],
            hidden_dim=self.config["model"]["hidden_dim"],
            num_layers=self.config["model"]["num_layers"],
            num_heads=self.config["model"]["num_heads"],
            dropout=0.0,
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, model=model, device=self.device)
        model.eval()
        
        self.generation_model = model
        self.logger.info("Generation model loaded successfully")
    
    def inference_receiver(
        self, 
        data_loader: torch.utils.data.DataLoader,
        top_k: int = 5,
        save_predictions: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """Perform receiver prediction inference.
        
        Args:
            data_loader: Data loader for inference
            top_k: Number of top predictions to return
            save_predictions: Whether to save predictions
            output_path: Path to save predictions
            
        Returns:
            Dictionary containing predictions and metrics
        """
        if self.receiver_model is None:
            raise ValueError("Receiver model not loaded")
        
        self.logger.info("Starting receiver inference")
        
        all_predictions = []
        all_targets = []
        all_top_k_predictions = []
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Receiver inference"):
                data = {k: v.to(self.device) for k, v in data.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                node_embeddings, graph_embeddings = self.receiver_model['backbone'](
                    data["x"], data["edge_index"], data["batch"]
                )
                logits = self.receiver_model['head'](graph_embeddings)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                top_k_predictions = torch.topk(logits, top_k, dim=1)[1]
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
                all_top_k_predictions.append(top_k_predictions.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_top_k_predictions = torch.cat(all_top_k_predictions, dim=0)
        
        # Compute metrics
        accuracy = (all_predictions == all_targets).float().mean().item()
        top_k_accuracy = torch.any(
            all_top_k_predictions == all_targets.unsqueeze(1), dim=1
        ).float().mean().item()
        
        results = {
            "predictions": all_predictions.numpy(),
            "targets": all_targets.numpy(),
            "top_k_predictions": all_top_k_predictions.numpy(),
            "accuracy": accuracy,
            "top_k_accuracy": top_k_accuracy,
        }
        
        # Save predictions if requested
        if save_predictions and output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump({
                    "predictions": all_predictions.numpy().tolist(),
                    "targets": all_targets.numpy().tolist(),
                    "top_k_predictions": all_top_k_predictions.numpy().tolist(),
                    "metrics": {
                        "accuracy": accuracy,
                        "top_k_accuracy": top_k_accuracy,
                    }
                }, f, indent=2)
            self.logger.info(f"Predictions saved to {output_path}")
        
        self.logger.info(f"Receiver inference complete - Accuracy: {accuracy:.4f}, Top-{top_k} Accuracy: {top_k_accuracy:.4f}")
        
        return results
    
    def inference_shot_marginalized(
        self,
        data_loader: torch.utils.data.DataLoader,
        use_gt_receiver: bool = False,
        top_k_receivers: int = None,
        save_predictions: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """Perform shot prediction with marginalization.
        
        Args:
            data_loader: Data loader for inference
            use_gt_receiver: Whether to use ground truth receiver
            top_k_receivers: Number of top receivers for marginalization
            save_predictions: Whether to save predictions
            output_path: Path to save predictions
            
        Returns:
            Dictionary containing predictions and metrics
        """
        if self.shot_model is None:
            raise ValueError("Shot model not loaded")
        
        self.logger.info("Starting shot inference with marginalization")
        
        all_predictions = []
        all_targets = []
        all_receiver_predictions = []
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Shot inference"):
                data = {k: v.to(self.device) for k, v in data.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                node_embeddings, graph_embeddings = self.shot_model['backbone'](
                    data["x"], data["edge_index"], data["batch"]
                )
                
                if use_gt_receiver:
                    # Use ground truth receiver
                    receiver_id = data.get("receiver_id", None)
                    if receiver_id is not None:
                        shot_logits = self.shot_model['head'](graph_embeddings, receiver_id)
                    else:
                        # Fallback to marginalization
                        shot_logits = self._marginalize_shot_predictions(
                            graph_embeddings, top_k_receivers
                        )
                else:
                    # Use marginalization
                    shot_logits = self._marginalize_shot_predictions(
                        graph_embeddings, top_k_receivers
                    )
                
                # Get predictions
                predictions = torch.sigmoid(shot_logits) > 0.5
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        accuracy = (all_predictions == all_targets).float().mean().item()
        precision = self._compute_precision(all_predictions, all_targets)
        recall = self._compute_recall(all_predictions, all_targets)
        f1_score = self._compute_f1_score(all_predictions, all_targets)
        
        results = {
            "predictions": all_predictions.numpy(),
            "targets": all_targets.numpy(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
        
        # Save predictions if requested
        if save_predictions and output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump({
                    "predictions": all_predictions.numpy().tolist(),
                    "targets": all_targets.numpy().tolist(),
                    "metrics": {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                    }
                }, f, indent=2)
            self.logger.info(f"Predictions saved to {output_path}")
        
        self.logger.info(f"Shot inference complete - Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
        
        return results
    
    def _marginalize_shot_predictions(
        self, 
        graph_embeddings: torch.Tensor, 
        top_k_receivers: int = None
    ) -> torch.Tensor:
        """Marginalize shot predictions over receiver candidates.
        
        Args:
            graph_embeddings: Graph embeddings
            top_k_receivers: Number of top receivers for marginalization
            
        Returns:
            Marginalized shot predictions
        """
        batch_size = graph_embeddings.size(0)
        num_classes = self.config["model"]["num_classes"]
        
        # Get receiver probabilities (assuming we have a receiver head)
        if hasattr(self.shot_model, 'receiver_head'):
            receiver_logits = self.shot_model['receiver_head'](graph_embeddings)
            receiver_probs = torch.softmax(receiver_logits, dim=1)
        else:
            # Uniform distribution if no receiver head
            receiver_probs = torch.ones(batch_size, num_classes) / num_classes
            receiver_probs = receiver_probs.to(graph_embeddings.device)
        
        if top_k_receivers is not None:
            # Use top-k receivers
            _, top_receivers = torch.topk(receiver_probs, top_k_receivers, dim=1)
            top_receiver_probs = torch.gather(receiver_probs, 1, top_receivers)
            top_receiver_probs = top_receiver_probs / top_receiver_probs.sum(dim=1, keepdim=True)
            
            shot_logits = []
            for i in range(batch_size):
                batch_shot_logits = []
                for j in range(top_k_receivers):
                    receiver_id = top_receivers[i, j].unsqueeze(0)
                    shot_logit = self.shot_model['head'](graph_embeddings[i:i+1], receiver_id)
                    batch_shot_logits.append(shot_logit)
                
                batch_shot_logits = torch.stack(batch_shot_logits, dim=1)
                weighted_shot_logit = (batch_shot_logits * top_receiver_probs[i:i+1]).sum(dim=1)
                shot_logits.append(weighted_shot_logit)
            
            return torch.cat(shot_logits, dim=0)
        else:
            # Use all receivers
            shot_logits = []
            for i in range(batch_size):
                batch_shot_logits = []
                for receiver_id in range(num_classes):
                    receiver_tensor = torch.tensor([receiver_id], device=graph_embeddings.device)
                    shot_logit = self.shot_model['head'](graph_embeddings[i:i+1], receiver_tensor)
                    batch_shot_logits.append(shot_logit)
                
                batch_shot_logits = torch.stack(batch_shot_logits, dim=1)
                weighted_shot_logit = (batch_shot_logits * receiver_probs[i:i+1]).sum(dim=1)
                shot_logits.append(weighted_shot_logit)
            
            return torch.cat(shot_logits, dim=0)
    
    def inference_generate_k(
        self,
        conditions: torch.Tensor,
        num_samples: int = 16,
        guidance_conditions: Optional[torch.Tensor] = None,
        save_samples: bool = True,
        output_path: str = None
    ) -> Dict[str, Any]:
        """Generate k samples from CVAE model.
        
        Args:
            conditions: Input conditions
            num_samples: Number of samples to generate per condition
            guidance_conditions: Guidance conditions for generation
            save_samples: Whether to save generated samples
            output_path: Path to save samples
            
        Returns:
            Dictionary containing generated samples and metrics
        """
        if self.generation_model is None:
            raise ValueError("Generation model not loaded")
        
        self.logger.info(f"Generating {num_samples} samples per condition")
        
        with torch.no_grad():
            # Generate samples
            generated = self.generation_model.generate(
                conditions.to(self.device),
                num_samples=num_samples,
                device=self.device,
                guidance_conditions=guidance_conditions
            )
        
        # Reshape to [num_conditions * num_samples, num_players, 2]
        num_conditions = conditions.size(0)
        generated = generated.view(num_conditions * num_samples, -1, 2)
        
        # Convert from normalized coordinates to field coordinates
        field_length = 105.0
        field_width = 68.0
        generated[:, :, 0] *= field_length
        generated[:, :, 1] *= field_width
        
        results = {
            "generated_samples": generated.cpu().numpy(),
            "num_conditions": num_conditions,
            "num_samples": num_samples,
            "field_dimensions": {"length": field_length, "width": field_width},
        }
        
        # Save samples if requested
        if save_samples and output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump({
                    "generated_samples": generated.cpu().numpy().tolist(),
                    "num_conditions": num_conditions,
                    "num_samples": num_samples,
                    "field_dimensions": {"length": field_length, "width": field_width},
                }, f, indent=2)
            self.logger.info(f"Samples saved to {output_path}")
        
        self.logger.info(f"Generation complete - {generated.size(0)} samples generated")
        
        return results
    
    def _compute_precision(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute precision metric."""
        true_positives = ((predictions == 1) & (targets == 1)).sum().float()
        false_positives = ((predictions == 1) & (targets == 0)).sum().float()
        
        if true_positives + false_positives == 0:
            return 0.0
        
        return (true_positives / (true_positives + false_positives)).item()
    
    def _compute_recall(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute recall metric."""
        true_positives = ((predictions == 1) & (targets == 1)).sum().float()
        false_negatives = ((predictions == 0) & (targets == 1)).sum().float()
        
        if true_positives + false_negatives == 0:
            return 0.0
        
        return (true_positives / (true_positives + false_negatives)).item()
    
    def _compute_f1_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute F1 score metric."""
        precision = self._compute_precision(predictions, targets)
        recall = self._compute_recall(predictions, targets)
        
        if precision + recall == 0:
            return 0.0
        
        return (2 * precision * recall / (precision + recall)).item()


def main():
    """Main function for unified inference."""
    parser = argparse.ArgumentParser(description="Unified inference API for TacticAI tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, choices=["receiver", "shot", "generation"], 
                       required=True, help="Task to perform inference on")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, help="Path to data for inference")
    parser.add_argument("--output_path", type=str, help="Path to save results")
    parser.add_argument("--views", type=str, choices=["id", "avg", "interact"], 
                       default="interact", help="View processing mode")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k for receiver prediction")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--use_gt_receiver", action="store_true", help="Use ground truth receiver for shot prediction")
    parser.add_argument("--top_k_receivers", type=int, help="Top-k receivers for shot marginalization")
    
    args = parser.parse_args()
    
    # Create inference API
    api = UnifiedInferenceAPI(args.config)
    
    # Set view processing mode
    if args.views == "id":
        api.view_config["view_mixing"] = "identity"
    elif args.views == "avg":
        api.view_config["view_mixing"] = "conv1d"
    elif args.views == "interact":
        api.view_config["view_mixing"] = "attention"
    
    # Load appropriate model
    if args.task == "receiver":
        api.load_receiver_model(args.checkpoint, use_4view=True)
        
        # Create data loader
        dataset = ReceiverDataset(args.data_path)
        data_loader = create_dataloader(dataset, batch_size=32, shuffle=False)
        
        # Perform inference
        results = api.inference_receiver(
            data_loader, 
            top_k=args.top_k,
            save_predictions=True,
            output_path=args.output_path
        )
        
    elif args.task == "shot":
        api.load_shot_model(args.checkpoint, use_4view=True)
        
        # Create data loader
        dataset = ShotDataset(args.data_path)
        data_loader = create_dataloader(dataset, batch_size=32, shuffle=False)
        
        # Perform inference
        results = api.inference_shot_marginalized(
            data_loader,
            use_gt_receiver=args.use_gt_receiver,
            top_k_receivers=args.top_k_receivers,
            save_predictions=True,
            output_path=args.output_path
        )
        
    elif args.task == "generation":
        api.load_generation_model(args.checkpoint, use_4view=True)
        
        # Create dummy conditions
        condition_dim = api.config["model"]["condition_dim"]
        conditions = torch.randn(2, condition_dim)  # 2 different conditions
        
        # Perform inference
        results = api.inference_generate_k(
            conditions,
            num_samples=args.num_samples,
            save_samples=True,
            output_path=args.output_path
        )
    
    print(f"Inference complete for {args.task} task")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
