"""Test script to check embedding generation during index building."""

import argparse
import yaml
import torch
import numpy as np

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset, create_dataloader
from tacticai.modules.utils import setup_logging, get_device


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--num-batches", type=int, default=3)
    
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
    
    # Create dataloader
    batch_size = config.get("eval", {}).get("batch_size", config.get("train", {}).get("batch_size", 32))
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_embeddings = []
    
    print(f"Processing {args.num_batches} batches...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break
                
            data_dict, _ = batch_data
            x = data_dict["x"].to(device)
            edge_index = data_dict["edge_index"].to(device)
            edge_attr = data_dict.get("edge_attr")
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)
            batch = data_dict.get("batch")
            if batch is not None:
                batch = batch.to(device)
            
            batch_embeddings = search_system._forward_batch(x, edge_index, edge_attr, batch)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            print(f"\nBatch {batch_idx}:")
            print(f"  Embeddings shape: {batch_embeddings_np.shape}")
            print(f"  Mean per sample: {batch_embeddings_np.mean(axis=1)}")
            print(f"  Std per sample: {batch_embeddings_np.std(axis=1)}")
            print(f"  First embedding mean: {batch_embeddings_np[0].mean():.6f}, std: {batch_embeddings_np[0].std():.6f}")
            print(f"  Second embedding mean: {batch_embeddings_np[1].mean():.6f}, std: {batch_embeddings_np[1].std():.6f}")
            
            all_embeddings.append(batch_embeddings_np)
    
    if all_embeddings:
        all_embeddings_np = np.vstack(all_embeddings)
        print(f"\nAll embeddings (after vstack):")
        print(f"  Shape: {all_embeddings_np.shape}")
        print(f"  Mean: {all_embeddings_np.mean():.6f}")
        print(f"  Std: {all_embeddings_np.std():.6f}")
        print(f"  Min: {all_embeddings_np.min():.6f}")
        print(f"  Max: {all_embeddings_np.max():.6f}")
        
        # Check if embeddings are different
        print(f"\nEmbedding diversity:")
        norms = np.linalg.norm(all_embeddings_np, axis=1)
        print(f"  Norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}, Min: {norms.min():.6f}, Max: {norms.max():.6f}")
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(all_embeddings_np[:10], all_embeddings_np[:10], metric='cosine')
        print(f"  Pairwise cosine distances (first 10 samples):")
        print(f"    Mean: {distances.mean():.6f}, Std: {distances.std():.6f}")


if __name__ == "__main__":
    main()

