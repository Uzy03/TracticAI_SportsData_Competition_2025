"""Debug the search process step by step."""

import argparse
import yaml
import numpy as np
import torch

from tacticai.retrieval import SimilarCKSearch, SimilarCKIndex
from tacticai.dataio import ReceiverDataset
from tacticai.modules.utils import setup_logging, get_device


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    
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
    
    # Load index
    index = SimilarCKIndex(
        embedding_dim=config["model"]["hidden_dim"],
        index_path=args.index_path,
    )
    index.load(args.index_path)
    
    print(f"Index embeddings (already normalized):")
    print(f"  Mean: {index.embeddings.mean():.6f}")
    print(f"  Std: {index.embeddings.std():.6f}")
    print(f"  Norms - Mean: {np.linalg.norm(index.embeddings, axis=1).mean():.6f}")
    
    # Load query
    dataset = ReceiverDataset(
        data_path=config["data"]["train_path"],
        file_format=config["data"].get("format", "pickle"),
        phase="train",
    )
    
    query_data_dict, _ = dataset[args.query_index]
    
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
    
    print(f"\nQuery embedding (BEFORE normalization):")
    print(f"  Shape: {query_embedding_np.shape}")
    print(f"  Mean: {query_embedding_np.mean():.6f}")
    print(f"  Std: {query_embedding_np.std():.6f}")
    print(f"  Norm: {np.linalg.norm(query_embedding_np, axis=1)}")
    
    # Normalize query embedding (as done in index.search)
    query_normalized = query_embedding_np.copy()
    norms = np.linalg.norm(query_normalized, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    query_normalized = query_normalized / norms
    
    print(f"\nQuery embedding (AFTER normalization):")
    print(f"  Mean: {query_normalized.mean():.6f}")
    print(f"  Std: {query_normalized.std():.6f}")
    print(f"  Norm: {np.linalg.norm(query_normalized, axis=1)}")
    
    # Compute similarities manually
    similarities = np.dot(query_normalized, index.embeddings.T)
    
    print(f"\nComputed similarities:")
    print(f"  Shape: {similarities.shape}")
    print(f"  Mean: {similarities.mean():.6f}")
    print(f"  Std: {similarities.std():.6f}")
    print(f"  Min: {similarities.min():.6f}")
    print(f"  Max: {similarities.max():.6f}")
    
    top_indices = np.argsort(similarities[0])[::-1][:5]
    print(f"\nTop-5 indices: {top_indices}")
    print(f"Top-5 similarities: {similarities[0][top_indices]}")
    
    # Compare with index.search
    results = index.search(query_embedding_np, top_k=5, normalize=True)
    print(f"\nResults from index.search:")
    for i, result in enumerate(results[0][:5], 1):
        print(f"  {i}. Index: {result['index']}, Similarity: {result['similarity']:.6f}")


if __name__ == "__main__":
    main()

