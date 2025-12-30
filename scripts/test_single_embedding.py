"""Test script to generate a single embedding and check its statistics."""

import argparse
import yaml
import torch
import numpy as np

from tacticai.retrieval import SimilarCKSearch
from tacticai.dataio import ReceiverDataset
from tacticai.modules.utils import setup_logging, get_device


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    
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
        if d2_enabled:
            backbone_path = f"{checkpoint_dir}/receiver/backbone_d2.ckpt"
        else:
            backbone_path = f"{checkpoint_dir}/receiver/backbone_no_d2.ckpt"
    
    # Create search system
    search_system = SimilarCKSearch(backbone_checkpoint_path=backbone_path, device=device)
    
    # Load dataset
    dataset = ReceiverDataset(
        data_path=config["data"]["train_path"],
        file_format=config["data"].get("format", "pickle"),
        phase="train",
    )
    
    # Get a single sample
    data_dict, target = dataset[args.sample_index]
    
    print(f"Sample {args.sample_index}:")
    print(f"  x shape: {data_dict['x'].shape}")
    print(f"  edge_index shape: {data_dict['edge_index'].shape}")
    print(f"  batch shape: {data_dict['batch'].shape if 'batch' in data_dict else 'None'}")
    if 'batch' in data_dict:
        print(f"  batch unique values: {torch.unique(data_dict['batch'])}")
        print(f"  batch max: {data_dict['batch'].max().item()}")
    
    # Generate embedding
    x = data_dict["x"].to(device)
    edge_index = data_dict["edge_index"].to(device)
    edge_attr = data_dict.get("edge_attr")
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)
    batch = data_dict.get("batch")
    if batch is not None:
        batch = batch.to(device)
    
    with torch.no_grad():
        embedding = search_system._forward_batch(x, edge_index, edge_attr, batch)
    
    embedding_np = embedding.cpu().numpy()
    
    print(f"\nEmbedding:")
    print(f"  Shape: {embedding_np.shape}")
    print(f"  Mean: {embedding_np.mean():.6f}")
    print(f"  Std: {embedding_np.std():.6f}")
    print(f"  Min: {embedding_np.min():.6f}")
    print(f"  Max: {embedding_np.max():.6f}")
    print(f"  Norm: {np.linalg.norm(embedding_np, axis=1) if embedding_np.ndim > 1 else np.linalg.norm(embedding_np)}")


if __name__ == "__main__":
    main()

