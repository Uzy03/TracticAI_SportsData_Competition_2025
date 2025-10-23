#!/usr/bin/env python3
"""Sequential training script for TacticAI tasks.

This script implements the sequential training protocol:
1. Receiver prediction pretraining
2. Shot prediction with frozen receiver encoder
3. CVAE generation with frozen encoder

The script handles encoder weight sharing and freezing across phases.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from tacticai.models import GATv2Network, ReceiverHead, ShotHead, CVAEModel
from tacticai.dataio import (
    ReceiverDataset, ShotDataset, CVAEDataset,
    create_dataloader
)
from tacticai.modules import (
    get_device, setup_logging, load_checkpoint, save_checkpoint,
    CrossEntropyLoss, BCELoss, EnhancedCVAELoss
)
from tacticai.train import (
    train_receiver, train_shot, train_cvae
)


class SequentialTrainer:
    """Sequential trainer for TacticAI tasks.
    
    Handles the sequential training protocol with encoder weight sharing.
    """
    
    def __init__(self, config_path: str):
        """Initialize sequential trainer.
        
        Args:
            config_path: Path to sequential training configuration
        """
        self.config = self._load_config(config_path)
        self.device = get_device(self.config["system"]["device"])
        self.logger = self._setup_logging()
        
        # Initialize shared encoder
        self.shared_encoder = None
        self.encoder_sharing_config = self.config["encoder_sharing"]
        
        # Training history
        self.training_history = {
            "receiver": [],
            "shot": [],
            "generation": []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for sequential training."""
        log_dir = Path(self.config["system"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"sequential_training_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["system"]["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger("SequentialTrainer")
        logger.info(f"Sequential training started at {timestamp}")
        logger.info(f"Log file: {log_file}")
        
        return logger
    
    def _create_shared_encoder(self) -> nn.Module:
        """Create shared encoder architecture."""
        encoder_config = self.encoder_sharing_config
        
        encoder = GATv2Network(
            input_dim=encoder_config["input_dim"],
            hidden_dim=encoder_config["hidden_dim"],
            output_dim=encoder_config["hidden_dim"],
            num_layers=encoder_config["num_layers"],
            num_heads=encoder_config["num_heads"],
            dropout=encoder_config["dropout"],
            readout="mean",
            residual=True,
        )
        
        return encoder.to(self.device)
    
    def _load_phase_config(self, phase: str) -> Dict[str, Any]:
        """Load configuration for a specific phase."""
        phase_config_path = self.config["phases"][phase]["config_file"]
        with open(phase_config_path, 'r') as f:
            phase_config = yaml.safe_load(f)
        return phase_config
    
    def _freeze_encoder(self, model: nn.Module) -> None:
        """Freeze encoder parameters in the model."""
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = False
            self.logger.info("Encoder parameters frozen")
        elif hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = False
            self.logger.info("Encoder parameters frozen")
    
    def _unfreeze_encoder(self, model: nn.Module) -> None:
        """Unfreeze encoder parameters in the model."""
        if hasattr(model, 'backbone'):
            for param in model.backbone.parameters():
                param.requires_grad = True
            self.logger.info("Encoder parameters unfrozen")
        elif hasattr(model, 'encoder'):
            for param in model.encoder.parameters():
                param.requires_grad = True
            self.logger.info("Encoder parameters unfrozen")
    
    def _save_encoder(self, model: nn.Module, save_path: str) -> None:
        """Save encoder weights."""
        encoder_state = {}
        
        if hasattr(model, 'backbone'):
            encoder_state = model.backbone.state_dict()
        elif hasattr(model, 'encoder'):
            encoder_state = model.encoder.state_dict()
        
        torch.save(encoder_state, save_path)
        self.logger.info(f"Encoder saved to {save_path}")
    
    def _load_encoder(self, model: nn.Module, load_path: str) -> None:
        """Load encoder weights."""
        encoder_state = torch.load(load_path, map_location=self.device)
        
        if hasattr(model, 'backbone'):
            model.backbone.load_state_dict(encoder_state)
        elif hasattr(model, 'encoder'):
            model.encoder.load_state_dict(encoder_state)
        
        self.logger.info(f"Encoder loaded from {load_path}")
    
    def train_receiver_phase(self) -> Dict[str, Any]:
        """Train receiver prediction model (Phase 1)."""
        self.logger.info("=== Starting Receiver Training Phase ===")
        
        phase_config = self._load_phase_config("receiver")
        phase_info = self.config["phases"]["receiver"]
        
        # Create model
        model = GATv2Network(
            input_dim=phase_config["model"]["input_dim"],
            hidden_dim=phase_config["model"]["hidden_dim"],
            output_dim=phase_config["model"]["hidden_dim"],
            num_layers=phase_config["model"]["num_layers"],
            num_heads=phase_config["model"]["num_heads"],
            dropout=phase_config["model"]["dropout"],
            readout="mean",
            residual=True,
        )
        
        receiver_head = ReceiverHead(
            input_dim=phase_config["model"]["hidden_dim"],
            hidden_dim=phase_config["model"]["hidden_dim"],
            num_classes=phase_config["model"]["num_classes"],
            dropout=phase_config["model"]["dropout"],
        )
        
        model = nn.ModuleDict({
            'backbone': model,
            'head': receiver_head
        }).to(self.device)
        
        # Create datasets
        train_dataset = ReceiverDataset(
            phase_config["data"]["train_path"],
            file_format=phase_config["data"]["format"]
        )
        val_dataset = ReceiverDataset(
            phase_config["data"]["val_path"],
            file_format=phase_config["data"]["format"]
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=phase_config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["system"]["num_workers"],
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=phase_config["eval"]["batch_size"],
            shuffle=False,
            num_workers=self.config["system"]["num_workers"],
        )
        
        # Train model
        history = train_receiver(
            model, train_loader, val_loader, phase_config, self.device
        )
        
        # Save encoder if requested
        if phase_info["save_encoder"]:
            encoder_save_path = phase_info["encoder_save_path"]
            Path(encoder_save_path).parent.mkdir(parents=True, exist_ok=True)
            self._save_encoder(model, encoder_save_path)
        
        # Save model checkpoint
        checkpoint_path = phase_info["checkpoint_path"]
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, phase_info["epochs"], history,
            checkpoint_path, self.device
        )
        
        self.training_history["receiver"] = history
        self.logger.info("=== Receiver Training Phase Complete ===")
        
        return history
    
    def train_shot_phase(self) -> Dict[str, Any]:
        """Train shot prediction model (Phase 2)."""
        self.logger.info("=== Starting Shot Training Phase ===")
        
        phase_config = self._load_phase_config("shot")
        phase_info = self.config["phases"]["shot"]
        
        # Create model
        model = GATv2Network(
            input_dim=phase_config["model"]["input_dim"],
            hidden_dim=phase_config["model"]["hidden_dim"],
            output_dim=phase_config["model"]["hidden_dim"],
            num_layers=phase_config["model"]["num_layers"],
            num_heads=phase_config["model"]["num_heads"],
            dropout=phase_config["model"]["dropout"],
            readout="mean",
            residual=True,
        )
        
        shot_head = ShotHead(
            input_dim=phase_config["model"]["hidden_dim"],
            hidden_dim=phase_config["model"]["hidden_dim"],
            dropout=phase_config["model"]["dropout"],
        )
        
        model = nn.ModuleDict({
            'backbone': model,
            'head': shot_head
        }).to(self.device)
        
        # Load encoder if requested
        if phase_info["load_encoder"]:
            self._load_encoder(model, phase_info["encoder_load_path"])
        
        # Freeze encoder if requested
        if phase_info["freeze_encoder"]:
            self._freeze_encoder(model)
        
        # Create datasets
        train_dataset = ShotDataset(
            phase_config["data"]["train_path"],
            file_format=phase_config["data"]["format"]
        )
        val_dataset = ShotDataset(
            phase_config["data"]["val_path"],
            file_format=phase_config["data"]["format"]
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=phase_config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["system"]["num_workers"],
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=phase_config["eval"]["batch_size"],
            shuffle=False,
            num_workers=self.config["system"]["num_workers"],
        )
        
        # Train model
        history = train_shot(
            model, train_loader, val_loader, phase_config, self.device
        )
        
        # Save model checkpoint
        checkpoint_path = phase_info["checkpoint_path"]
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, phase_info["epochs"], history,
            checkpoint_path, self.device
        )
        
        self.training_history["shot"] = history
        self.logger.info("=== Shot Training Phase Complete ===")
        
        return history
    
    def train_generation_phase(self) -> Dict[str, Any]:
        """Train CVAE generation model (Phase 3)."""
        self.logger.info("=== Starting Generation Training Phase ===")
        
        phase_config = self._load_phase_config("generation")
        phase_info = self.config["phases"]["generation"]
        
        # Create model
        model = CVAEModel(
            input_dim=phase_config["model"]["input_dim"],
            condition_dim=phase_config["model"]["condition_dim"],
            latent_dim=phase_config["model"]["latent_dim"],
            output_dim=phase_config["model"]["output_dim"],
            hidden_dim=phase_config["model"]["hidden_dim"],
            num_layers=phase_config["model"]["num_layers"],
            num_heads=phase_config["model"]["num_heads"],
            dropout=phase_config["model"]["dropout"],
        ).to(self.device)
        
        # Load encoder if requested
        if phase_info["load_encoder"]:
            self._load_encoder(model, phase_info["encoder_load_path"])
        
        # Freeze encoder if requested
        if phase_info["freeze_encoder"]:
            self._freeze_encoder(model)
        
        # Create datasets
        train_dataset = CVAEDataset(
            phase_config["data"]["train_path"],
            file_format=phase_config["data"]["format"]
        )
        val_dataset = CVAEDataset(
            phase_config["data"]["val_path"],
            file_format=phase_config["data"]["format"]
        )
        
        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=phase_config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["system"]["num_workers"],
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=phase_config["eval"]["batch_size"],
            shuffle=False,
            num_workers=self.config["system"]["num_workers"],
        )
        
        # Train model
        history = train_cvae(
            model, train_loader, val_loader, phase_config, self.device
        )
        
        # Save model checkpoint
        checkpoint_path = phase_info["checkpoint_path"]
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model, phase_info["epochs"], history,
            checkpoint_path, self.device
        )
        
        self.training_history["generation"] = history
        self.logger.info("=== Generation Training Phase Complete ===")
        
        return history
    
    def train_all_phases(self) -> Dict[str, Any]:
        """Train all phases sequentially."""
        self.logger.info("Starting sequential training protocol")
        
        # Phase 1: Receiver training
        receiver_history = self.train_receiver_phase()
        
        # Phase 2: Shot training
        shot_history = self.train_shot_phase()
        
        # Phase 3: Generation training
        generation_history = self.train_generation_phase()
        
        # Save complete training history
        history_path = Path(self.config["system"]["log_dir"]) / "complete_training_history.json"
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Complete training history saved to {history_path}")
        self.logger.info("Sequential training protocol completed successfully")
        
        return self.training_history


def main():
    """Main function for sequential training."""
    parser = argparse.ArgumentParser(description="Sequential training for TacticAI tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to sequential training config")
    parser.add_argument("--phase", type=str, choices=["receiver", "shot", "generation", "all"], 
                       default="all", help="Which phase to train")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SequentialTrainer(args.config)
    
    # Train specified phase(s)
    if args.phase == "all":
        trainer.train_all_phases()
    elif args.phase == "receiver":
        trainer.train_receiver_phase()
    elif args.phase == "shot":
        trainer.train_shot_phase()
    elif args.phase == "generation":
        trainer.train_generation_phase()


if __name__ == "__main__":
    main()
