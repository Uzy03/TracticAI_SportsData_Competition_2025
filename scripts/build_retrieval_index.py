"""Script to build retrieval index from receiver prediction data."""

import argparse
import logging
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset, create_dataloader
from tacticai.modules.utils import setup_logging, load_config, get_device


def main():
    """Main function to build retrieval index."""
    parser = argparse.ArgumentParser(
        description="Build retrieval index for similar CK search"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (used for model metadata)",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        help="Path to pretrained backbone checkpoint (auto-selects from config if not specified)",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        required=True,
        help="Path to save index file",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data phase to index (train/val/test)",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use Faiss for efficient similarity search",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = get_device(config.get("device", "auto"))
    
    # Setup logging
    logger = setup_logging(
        config.get("log_dir", "runs"),
        config.get("log_level", "INFO"),
    )
    
    # Determine backbone checkpoint path
    if args.backbone_checkpoint:
        backbone_checkpoint_path = args.backbone_checkpoint
    else:
        # Auto-select based on D2 setting
        d2_enabled = config.get("d2", {}).get("enabled", False)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        if d2_enabled:
            backbone_checkpoint_path = f"{checkpoint_dir}/receiver/backbone_d2.ckpt"
        else:
            backbone_checkpoint_path = f"{checkpoint_dir}/receiver/backbone_no_d2.ckpt"
        logger.info(f"Auto-selected backbone checkpoint based on D2={d2_enabled}: {backbone_checkpoint_path}")
    
    # Determine data path based on phase
    phase_to_key = {
        "train": "train_path",
        "val": "val_path",
        "test": "test_path",
    }
    data_path_key = phase_to_key[args.phase]
    data_path = config["data"][data_path_key]
    
    logger.info(f"Building retrieval index on {device}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Backbone checkpoint: {backbone_checkpoint_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output index: {args.output_index}")
    
    # Create search system
    search_system = SimilarCKSearch(
        backbone_checkpoint_path=backbone_checkpoint_path,
        device=device,
    )
    
    # Create dataset
    dataset = ReceiverDataset(
        data_path=data_path,
        file_format=config["data"].get("format", "pickle"),
        phase=args.phase,
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Get batch size from config (use eval batch size for inference)
    batch_size = config.get("eval", {}).get("batch_size", config.get("train", {}).get("batch_size", 32))
    num_workers = config.get("num_workers", 0)
    
    logger.info(f"Batch size: {batch_size}, num_workers: {num_workers}")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Not needed for CPU inference
    )
    
    # Build index
    index = search_system.build_index(
        dataloader,
        index=None,
        save_path=args.output_index,
    )
    
    logger.info(f"Index built successfully: {len(index)} embeddings")
    logger.info(f"Index saved to {args.output_index}")


if __name__ == "__main__":
    main()

