"""Analyze embedding diversity to understand why all embeddings are similar."""

import argparse
import yaml
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

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
    parser.add_argument("--num-samples", type=int, default=50)
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    logger = setup_logging(config.get("log_dir", "runs"), config.get("log_level", "INFO"))
    
    if args.backbone_checkpoint:
        backbone_path = args.backbone_checkpoint
    else:
        d2_enabled = config.get("d2", {}).get("enabled", False)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
        if d2_enabled:
            backbone_path = f"{checkpoint_dir}/receiver/backbone_d2.ckpt"
        else:
            backbone_path = f"{checkpoint_dir}/receiver/backbone_no_d2.ckpt"
    
    search_system = SimilarCKSearch(backbone_checkpoint_path=backbone_path, device=device)
    
    dataset = ReceiverDataset(
        data_path=config["data"]["train_path"],
        file_format=config["data"].get("format", "pickle"),
        phase="train",
    )
    
    # Generate embeddings for multiple samples
    batch_size = min(args.num_samples, 32)
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_embeddings = []
    all_targets = []
    
    print(f"Generating embeddings for {args.num_samples} samples...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if len(all_embeddings) >= args.num_samples:
                break
                
            data_dict, targets = batch_data
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
            
            # Take only what we need
            remaining = args.num_samples - len(all_embeddings)
            if remaining < batch_embeddings_np.shape[0]:
                batch_embeddings_np = batch_embeddings_np[:remaining]
                targets = targets[:remaining]
            
            all_embeddings.append(batch_embeddings_np)
            all_targets.extend(targets.cpu().numpy().tolist())
    
    embeddings = np.vstack(all_embeddings)
    
    print(f"\n{'='*80}")
    print(f"Embedding Analysis (BEFORE normalization)")
    print(f"{'='*80}")
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Mean: {embeddings.mean():.6f}")
    print(f"Std: {embeddings.std():.6f}")
    print(f"Min: {embeddings.min():.6f}")
    print(f"Max: {embeddings.max():.6f}")
    
    # Norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std: {norms.std():.6f}")
    print(f"  Min: {norms.min():.6f}")
    print(f"  Max: {norms.max():.6f}")
    
    # Pairwise cosine distances (before normalization)
    print(f"\nPairwise cosine distances (before normalization):")
    # Use a subset to avoid memory issues
    subset_size = min(50, embeddings.shape[0])
    subset = embeddings[:subset_size]
    
    # Compute pairwise cosine distances
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(subset)
    cosine_dist = 1 - cosine_sim
    # Remove diagonal (self-similarity)
    mask = ~np.eye(cosine_dist.shape[0], dtype=bool)
    pairwise_distances = cosine_dist[mask]
    
    print(f"  (Using first {subset_size} samples)")
    print(f"  Mean: {pairwise_distances.mean():.6f}")
    print(f"  Std: {pairwise_distances.std():.6f}")
    print(f"  Min: {pairwise_distances.min():.6f}")
    print(f"  Max: {pairwise_distances.max():.6f}")
    print(f"  Median: {np.median(pairwise_distances):.6f}")
    
    # After normalization
    norms_normalized = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms_normalized = np.where(norms_normalized == 0, 1.0, norms_normalized)
    embeddings_normalized = embeddings / norms_normalized
    
    print(f"\n{'='*80}")
    print(f"Embedding Analysis (AFTER normalization)")
    print(f"{'='*80}")
    
    subset_norm = embeddings_normalized[:subset_size]
    cosine_sim_norm = cosine_similarity(subset_norm)
    cosine_dist_norm = 1 - cosine_sim_norm
    pairwise_distances_norm = cosine_dist_norm[~np.eye(cosine_dist_norm.shape[0], dtype=bool)]
    
    print(f"\nPairwise cosine distances (after normalization):")
    print(f"  (Using first {subset_size} samples)")
    print(f"  Mean: {pairwise_distances_norm.mean():.6f}")
    print(f"  Std: {pairwise_distances_norm.std():.6f}")
    print(f"  Min: {pairwise_distances_norm.min():.6f}")
    print(f"  Max: {pairwise_distances_norm.max():.6f}")
    print(f"  Median: {np.median(pairwise_distances_norm):.6f}")
    
    # Check if embeddings are collapsing (very similar)
    if pairwise_distances_norm.mean() < 0.01:
        print(f"\n⚠️  WARNING: Embeddings are collapsing! Mean pairwise distance is very small.")
        print(f"   This suggests the model is producing very similar embeddings for all samples.")
    elif pairwise_distances_norm.mean() < 0.1:
        print(f"\n⚠️  WARNING: Embeddings are quite similar. Mean pairwise distance is small.")
        print(f"   The model may not be learning diverse representations.")
    else:
        print(f"\n✓ Embeddings show reasonable diversity.")
    
    # Check by target class (receiver)
    if len(set(all_targets)) > 1:
        print(f"\n{'='*80}")
        print(f"Analysis by Receiver Target")
        print(f"{'='*80}")
        
        unique_targets = sorted(set(all_targets))
        print(f"\nUnique receiver targets: {unique_targets}")
        
        for target in unique_targets[:5]:  # Check first 5 targets
            target_indices = [i for i, t in enumerate(all_targets) if t == target]
            if len(target_indices) < 2:
                continue
                
            target_embeddings = embeddings_normalized[target_indices]
            
            # Intra-class similarity (embeddings of same target)
            if len(target_indices) > 1:
                target_sim = cosine_similarity(target_embeddings)
                target_dist = 1 - target_sim
                target_pairwise = target_dist[~np.eye(target_dist.shape[0], dtype=bool)]
                
                print(f"\nTarget {target} ({len(target_indices)} samples):")
                print(f"  Intra-class pairwise distance:")
                print(f"    Mean: {target_pairwise.mean():.6f}")
                print(f"    Std: {target_pairwise.std():.6f}")


if __name__ == "__main__":
    main()

