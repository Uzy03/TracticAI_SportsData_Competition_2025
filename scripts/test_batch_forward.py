"""Test script to check batch forwarding in retrieval search."""

import argparse
import yaml
import torch
import numpy as np

from tacticai.retrieval import SimilarCKSearch
from tacticai.dataio import ReceiverDataset, create_dataloader
from tacticai.modules.utils import setup_logging, get_device


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
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
    
    # Create dataloader with batch_size=3
    dataloader = create_dataloader(dataset, batch_size=3, shuffle=False, num_workers=0)
    
    # Get first batch
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx > 0:
            break
            
        data_dict, target = batch_data
        x = data_dict["x"]
        edge_index = data_dict["edge_index"]
        edge_attr = data_dict.get("edge_attr")
        batch = data_dict.get("batch")
        
        print(f"Batch {batch_idx}:")
        print(f"  x shape: {x.shape}")
        print(f"  edge_index shape: {edge_index.shape}")
        print(f"  batch shape: {batch.shape if batch is not None else 'None'}")
        if batch is not None:
            print(f"  batch unique values: {torch.unique(batch)}")
            print(f"  batch max: {batch.max().item()}")
            print(f"  Expected batch values: [0, 0, ..., 0, 1, 1, ..., 1, 2, 2, ..., 2] for 3 graphs")
            print(f"  Expected batch max: 2")
            
            # Count nodes per graph
            unique_batches = torch.unique(batch, sorted=True)
            print(f"  Number of unique batch IDs: {len(unique_batches)}")
            for b in unique_batches:
                mask = (batch == b)
                num_nodes = mask.sum().item()
                print(f"    Batch ID {b.item()}: {num_nodes} nodes")
        
        # Generate embeddings
        x_dev = x.to(device)
        edge_index_dev = edge_index.to(device)
        edge_attr_dev = edge_attr.to(device) if edge_attr is not None else None
        batch_dev = batch.to(device) if batch is not None else None
        
        with torch.no_grad():
            embeddings = search_system._forward_batch(x_dev, edge_index_dev, edge_attr_dev, batch_dev)
        
        print(f"\nEmbeddings:")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Expected shape: (3, 512) for 3 graphs")
        print(f"  Mean per graph: {embeddings.mean(dim=1).cpu().numpy()}")
        print(f"  Std per graph: {embeddings.std(dim=1).cpu().numpy()}")


if __name__ == "__main__":
    main()

