"""Test script for similar CK retrieval search functionality."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

import torch
from torch.utils.data import DataLoader

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset, create_dataloader
from tacticai.modules.utils import setup_logging
from tacticai.modules import get_device


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


def main():
    """Main function to test retrieval search."""
    parser = argparse.ArgumentParser(
        description="Test similar CK retrieval search"
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
        "--index-path",
        type=str,
        required=True,
        help="Path to index file",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Index of query sample in dataset (default: 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top similar CKs to retrieve (default: 5)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to query data (uses train data from config if not specified)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Phase of query data (default: train)",
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
    
    logger.info(f"Testing retrieval search on {device}")
    logger.info(f"Index path: {args.index_path}")
    logger.info(f"Query sample index: {args.query_index}")
    logger.info(f"Top-k: {args.top_k}")
    
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
        logger.info(f"Auto-selected backbone checkpoint: {backbone_checkpoint_path}")
    
    # Create search system
    search_system = SimilarCKSearch(
        backbone_checkpoint_path=backbone_checkpoint_path,
        device=device,
    )
    
    # Load index
    index = SimilarCKIndex(
        embedding_dim=config["model"]["hidden_dim"],
        index_path=args.index_path,
    )
    index.load(args.index_path)
    logger.info(f"Index loaded: {len(index)} embeddings")
    
    # Load query data
    if args.data_path:
        query_data_path = args.data_path
    else:
        phase_to_key = {
            "train": "train_path",
            "val": "val_path",
            "test": "test_path",
        }
        query_data_path = config["data"][phase_to_key[args.phase]]
    
    logger.info(f"Loading query data from: {query_data_path}")
    query_dataset = ReceiverDataset(
        data_path=query_data_path,
        file_format=config["data"].get("format", "pickle"),
        phase=args.phase,
    )
    
    if args.query_index >= len(query_dataset):
        logger.error(f"Query index {args.query_index} is out of range (dataset size: {len(query_dataset)})")
        return
    
    # Get query sample
    query_data_dict, query_target = query_dataset[args.query_index]
    logger.info(f"Query sample index: {args.query_index} (target receiver: {query_target.item()})")
    
    # Search for similar CKs
    logger.info(f"Searching for top-{args.top_k} similar CKs...")
    results = search_system.search_similar(
        query_data_dict,
        index=index,
        top_k=args.top_k,
    )
    
    # Display results
    print("\n" + "="*80)
    print(f"Query: Sample index {args.query_index} from {query_data_path}")
    print(f"Target receiver: {query_target.item()}")
    print("="*80)
    print(f"\nTop-{args.top_k} Similar CKs:")
    print("-"*80)
    
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        metadata = result['metadata']
        idx = result['index']
        
        print(f"\n{i}. Rank {i} (Similarity: {similarity:.4f})")
        print(f"   Index: {idx}")
        print(f"   Metadata: {metadata}")
    
    print("\n" + "="*80)
    logger.info("Search completed successfully")


if __name__ == "__main__":
    main()

