"""Test generate_embeddings output and save to index."""

import argparse
import yaml
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
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    logger = setup_logging(config.get("log_dir", "runs"), config.get("log_level", "INFO"))
    
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
    
    batch_size = 5
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Generate embeddings (first 2 batches only)
    embeddings_list = []
    metadata_list = []
    
    import torch
    with torch.no_grad():
        search_system.backbone.eval()
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= 2:
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
            
            print(f"Batch {batch_idx} embeddings (before vstack):")
            print(f"  Shape: {batch_embeddings_np.shape}")
            print(f"  Mean: {batch_embeddings_np.mean():.6f}")
            print(f"  Std: {batch_embeddings_np.std():.6f}")
            
            embeddings_list.append(batch_embeddings_np)
            
            for i in range(batch_embeddings.shape[0]):
                metadata_list.append({
                    'batch_idx': batch_idx,
                    'sample_idx_in_batch': i,
                    'global_idx': len(metadata_list),
                })
    
    embeddings = np.vstack(embeddings_list)
    
    print(f"\nAll embeddings (after vstack, BEFORE normalization):")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    norms_before = np.linalg.norm(embeddings, axis=1)
    print(f"  Norms - Mean: {norms_before.mean():.6f}, Std: {norms_before.std():.6f}")
    
    # Now simulate what add_embeddings does
    embeddings_normalized = embeddings.copy()
    norms = np.linalg.norm(embeddings_normalized, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_normalized = embeddings_normalized / norms
    
    print(f"\nEmbeddings (AFTER normalization, as stored in index):")
    print(f"  Mean: {embeddings_normalized.mean():.6f}")
    print(f"  Std: {embeddings_normalized.std():.6f}")
    print(f"  Min: {embeddings_normalized.min():.6f}")
    print(f"  Max: {embeddings_normalized.max():.6f}")
    
    norms_after = np.linalg.norm(embeddings_normalized, axis=1)
    print(f"  Norms - Mean: {norms_after.mean():.6f}, Std: {norms_after.std():.6f}")


if __name__ == "__main__":
    main()

