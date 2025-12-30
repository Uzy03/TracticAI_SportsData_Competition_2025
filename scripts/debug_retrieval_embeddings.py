"""Debug script to check embedding values and similarity calculations."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
import numpy as np

import torch

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset
from tacticai.modules.utils import setup_logging
from tacticai.modules import get_device


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Debug retrieval embeddings")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    logger = setup_logging(config.get("log_dir", "runs"), config.get("log_level", "INFO"))
    
    # Load backbone checkpoint path
    if args.backbone_checkpoint:
        backbone_path = args.backbone_checkpoint
    else:
        d2_enabled = config.get("d2", {}).get("enabled", False)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        model_save_dir = config.get("model_save_dir", f"{checkpoint_dir}/receiver_shot")
        if d2_enabled:
            backbone_path = f"{model_save_dir}/best_d2.ckpt"
        else:
            backbone_path = f"{model_save_dir}/best_no_d2.ckpt"
    
    # Create search system
    search_system = SimilarCKSearch(backbone_checkpoint_path=backbone_path, config=config, device=device)
    
    # Load index
    index = SimilarCKIndex(embedding_dim=config["model"]["hidden_dim"], index_path=args.index_path)
    index.load(args.index_path)
    
    print(f"Index loaded: {len(index)} embeddings")
    print(f"Embedding shape: {index.embeddings.shape}")
    print(f"Embedding dtype: {index.embeddings.dtype}")
    
    # Check index embeddings statistics
    print("\nIndex embeddings statistics:")
    print(f"  Mean: {index.embeddings.mean():.6f}")
    print(f"  Std: {index.embeddings.std():.6f}")
    print(f"  Min: {index.embeddings.min():.6f}")
    print(f"  Max: {index.embeddings.max():.6f}")
    
    # Check if embeddings are normalized
    norms = np.linalg.norm(index.embeddings, axis=1)
    print(f"\nIndex embedding norms:")
    print(f"  Mean norm: {norms.mean():.6f}")
    print(f"  Std norm: {norms.std():.6f}")
    print(f"  Min norm: {norms.min():.6f}")
    print(f"  Max norm: {norms.max():.6f}")
    print(f"  All close to 1.0? {np.allclose(norms, 1.0, atol=1e-5)}")
    
    # Check for duplicate embeddings
    unique_embeddings, unique_indices, counts = np.unique(
        index.embeddings, axis=0, return_inverse=True, return_counts=True
    )
    print(f"\nUnique embeddings: {len(unique_embeddings)} out of {len(index.embeddings)}")
    if len(unique_embeddings) < len(index.embeddings):
        print(f"  Duplicate embeddings found!")
        print(f"  Most frequent count: {counts.max()}")
        print(f"  Number of unique values with count > 1: {(counts > 1).sum()}")
    
    # Load query data (support both standard and multitask configs)
    if "train_path" in config["data"]:
        query_data_path = config["data"]["train_path"]
    elif "receiver_train_path" in config["data"]:
        query_data_path = config["data"]["receiver_train_path"]
    else:
        raise ValueError("Neither 'train_path' nor 'receiver_train_path' found in config['data']")
    query_dataset = ReceiverDataset(
        data_path=query_data_path,
        file_format=config["data"].get("format", "pickle"),
        phase="train",
    )
    
    query_data_dict, query_target = query_dataset[args.query_index]
    
    # Generate query embedding
    x = query_data_dict["x"].to(device)
    edge_index = query_data_dict["edge_index"].to(device)
    edge_attr = query_data_dict.get("edge_attr")
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    batch = query_data_dict.get("batch")
    if batch is not None:
        batch = batch.to(device)
    
    with torch.no_grad():
        query_embedding = search_system._forward_batch(x, edge_index, edge_attr, batch)
    
    query_embedding_np = query_embedding.cpu().numpy()
    
    print(f"\nQuery embedding shape: {query_embedding_np.shape}")
    print(f"Query embedding statistics:")
    print(f"  Mean: {query_embedding_np.mean():.6f}")
    print(f"  Std: {query_embedding_np.std():.6f}")
    print(f"  Min: {query_embedding_np.min():.6f}")
    print(f"  Max: {query_embedding_np.max():.6f}")
    
    # Normalize query embedding
    query_norm = np.linalg.norm(query_embedding_np, axis=1)
    print(f"\nQuery embedding norm: {query_norm}")
    
    # Normalize for cosine similarity
    query_norm = np.where(query_norm == 0, 1.0, query_norm)
    query_embedding_normalized = query_embedding_np / query_norm
    
    print(f"Query embedding after normalization:")
    print(f"  Norm: {np.linalg.norm(query_embedding_normalized, axis=1)}")
    
    # Compute similarities manually
    similarities = np.dot(query_embedding_normalized, index.embeddings.T)
    print(f"\nComputed similarities:")
    print(f"  Shape: {similarities.shape}")
    print(f"  Mean: {similarities.mean():.6f}")
    print(f"  Std: {similarities.std():.6f}")
    print(f"  Min: {similarities.min():.6f}")
    print(f"  Max: {similarities.max():.6f}")
    print(f"  Top-5 indices: {np.argsort(similarities[0])[::-1][:5]}")
    print(f"  Top-5 similarities: {np.sort(similarities[0])[::-1][:5]}")


if __name__ == "__main__":
    main()

