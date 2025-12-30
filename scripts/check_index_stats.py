"""Check statistics of saved index file."""

import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", type=str, required=True)
    
    args = parser.parse_args()
    
    # Load index
    with open(args.index_path, 'rb') as f:
        index_data = pickle.load(f)
    
    embeddings = index_data['embeddings']
    
    print(f"Index file: {args.index_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    print(f"\nEmbeddings statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean: {norms.mean():.6f}")
    print(f"  Std: {norms.std():.6f}")
    print(f"  Min: {norms.min():.6f}")
    print(f"  Max: {norms.max():.6f}")
    
    # Check if embeddings are all the same
    print(f"\nEmbedding diversity:")
    first_emb = embeddings[0]
    distances = np.linalg.norm(embeddings - first_emb, axis=1)
    print(f"  Distance to first embedding:")
    print(f"    Mean: {distances.mean():.6f}")
    print(f"    Std: {distances.std():.6f}")
    print(f"    Min: {distances.min():.6f}")
    print(f"    Max: {distances.max():.6f}")
    
    # Check if all embeddings are the same (within tolerance)
    if np.allclose(distances, 0, atol=1e-5):
        print(f"  WARNING: All embeddings are identical!")
    else:
        print(f"  Embeddings are different (good)")


if __name__ == "__main__":
    main()

