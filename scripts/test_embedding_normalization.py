"""Test embedding normalization."""

import numpy as np

# Simulate what happens during index building
embeddings_raw = np.random.randn(10, 512) * 17.0  # Similar to actual embeddings (std ~17)
embeddings_raw = embeddings_raw + 0.35  # Add mean offset

print("Raw embeddings (before normalization):")
print(f"  Mean: {embeddings_raw.mean():.6f}")
print(f"  Std: {embeddings_raw.std():.6f}")
print(f"  Norms - Mean: {np.linalg.norm(embeddings_raw, axis=1).mean():.6f}")

# Normalize
norms = np.linalg.norm(embeddings_raw, axis=1, keepdims=True)
embeddings_normalized = embeddings_raw / norms

print("\nNormalized embeddings (after L2 normalization):")
print(f"  Mean: {embeddings_normalized.mean():.6f}")
print(f"  Std: {embeddings_normalized.std():.6f}")
print(f"  Norms - Mean: {np.linalg.norm(embeddings_normalized, axis=1).mean():.6f}")
print(f"  Norms - Std: {np.linalg.norm(embeddings_normalized, axis=1).std():.6f}")

# Check cosine similarity
similarities = np.dot(embeddings_normalized[0:1], embeddings_normalized.T)
print(f"\nCosine similarities with first embedding:")
print(f"  Mean: {similarities.mean():.6f}")
print(f"  Std: {similarities.std():.6f}")
print(f"  Min: {similarities.min():.6f}")
print(f"  Max: {similarities.max():.6f}")

# Now check what happens if raw embeddings have very small values
embeddings_raw_small = np.random.randn(10, 512) * 0.044  # Similar to index stats
embeddings_raw_small = embeddings_raw_small + 0.000866

print("\n\nSmall raw embeddings (similar to index stats):")
print(f"  Mean: {embeddings_raw_small.mean():.6f}")
print(f"  Std: {embeddings_raw_small.std():.6f}")

norms_small = np.linalg.norm(embeddings_raw_small, axis=1, keepdims=True)
embeddings_normalized_small = embeddings_raw_small / norms_small

print("\nNormalized small embeddings:")
print(f"  Mean: {embeddings_normalized_small.mean():.6f}")
print(f"  Std: {embeddings_normalized_small.std():.6f}")
print(f"  Norms - Mean: {np.linalg.norm(embeddings_normalized_small, axis=1).mean():.6f}")

similarities_small = np.dot(embeddings_normalized_small[0:1], embeddings_normalized_small.T)
print(f"\nCosine similarities with first embedding:")
print(f"  Mean: {similarities_small.mean():.6f}")
print(f"  Std: {similarities_small.std():.6f}")
print(f"  Min: {similarities_small.min():.6f}")
print(f"  Max: {similarities_small.max():.6f}")

