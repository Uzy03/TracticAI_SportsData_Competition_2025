"""Index class for storing and managing embeddings for similar CK search."""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np
import torch
import pickle
import logging

logger = logging.getLogger(__name__)


class SimilarCKIndex:
    """Index for storing and managing embeddings for similar CK search.
    
    Stores graph-level embeddings computed from pretrained backbone models
    along with metadata for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        use_faiss: bool = False,
        index_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize SimilarCKIndex.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            use_faiss: Whether to use Faiss for efficient similarity search (optional)
            index_path: Path to save/load index (optional)
        """
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss
        self.index_path = Path(index_path) if index_path else None
        
        # Storage: embeddings as numpy array [N, embedding_dim]
        self.embeddings: Optional[np.ndarray] = None
        
        # Metadata: list of dicts, one per embedding
        # Each dict contains: {'data_id': str, 'file_path': str, ...}
        self.metadata: List[Dict[str, Any]] = []
        
        # Initialize Faiss index if requested
        if self.use_faiss:
            try:
                import faiss
                self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
                logger.info("Faiss index initialized for efficient similarity search")
            except ImportError:
                logger.warning("Faiss not available, falling back to numpy-based search")
                self.use_faiss = False
                self.faiss_index = None
        else:
            self.faiss_index = None
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> None:
        """Add embeddings to index.
        
        Args:
            embeddings: Embedding vectors [N, embedding_dim]
            metadata: List of metadata dicts, one per embedding
            normalize: Whether to L2-normalize embeddings (for cosine similarity)
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata length mismatch: {len(metadata)} metadata entries "
                f"for {embeddings.shape[0]} embeddings"
            )
        
        # Normalize embeddings for cosine similarity
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
            embeddings = embeddings / norms
        
        if self.embeddings is None:
            self.embeddings = embeddings
            self.metadata = metadata.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.metadata.extend(metadata)
        
        # Update Faiss index if available
        if self.faiss_index is not None:
            self.faiss_index.add(embeddings)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {len(self.metadata)}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10,
        normalize: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar CKs.
        
        Args:
            query_embeddings: Query embedding vectors [N, embedding_dim]
            top_k: Number of top similar CKs to return per query
            normalize: Whether to L2-normalize query embeddings
            
        Returns:
            List of lists, where each inner list contains top_k results for each query.
            Each result is a dict with keys: {'similarity': float, 'metadata': dict, 'index': int}
        """
        if self.embeddings is None or len(self.metadata) == 0:
            raise ValueError("Index is empty. Add embeddings before searching.")
        
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings[np.newaxis, :]
        
        if query_embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embeddings.shape[1]}"
            )
        
        # Normalize query embeddings
        if normalize:
            norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            query_embeddings = query_embeddings / norms
        
        num_queries = query_embeddings.shape[0]
        results = []
        
        if self.faiss_index is not None:
            # Use Faiss for efficient search
            similarities, indices = self.faiss_index.search(query_embeddings, top_k)
            
            for i in range(num_queries):
                query_results = []
                for j in range(len(indices[i])):
                    idx = indices[i][j]
                    if idx >= 0:  # Faiss returns -1 for invalid results
                        similarity = float(similarities[i][j])
                        query_results.append({
                            'similarity': similarity,
                            'metadata': self.metadata[idx].copy(),
                            'index': int(idx),
                        })
                results.append(query_results)
        else:
            # Use numpy for search (cosine similarity = dot product of normalized vectors)
            similarities = np.dot(query_embeddings, self.embeddings.T)  # [N_queries, N_index]
            
            for i in range(num_queries):
                query_similarities = similarities[i]
                top_indices = np.argsort(query_similarities)[::-1][:top_k]
                
                query_results = []
                for idx in top_indices:
                    similarity = float(query_similarities[idx])
                    query_results.append({
                        'similarity': similarity,
                        'metadata': self.metadata[idx].copy(),
                        'index': int(idx),
                    })
                results.append(query_results)
        
        return results
    
    def save(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Save index to disk.
        
        Args:
            filepath: Path to save index (uses self.index_path if None)
        """
        filepath = Path(filepath) if filepath else self.index_path
        if filepath is None:
            raise ValueError("No filepath provided and self.index_path is None")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim,
            'use_faiss': self.use_faiss,
        }
        
        # Save Faiss index separately if used
        if self.faiss_index is not None:
            faiss_path = filepath.with_suffix('.faiss')
            import faiss
            faiss.write_index(self.faiss_index, str(faiss_path))
            index_data['faiss_index_path'] = str(faiss_path)
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Load index from disk.
        
        Args:
            filepath: Path to load index from (uses self.index_path if None)
        """
        filepath = Path(filepath) if filepath else self.index_path
        if filepath is None:
            raise ValueError("No filepath provided and self.index_path is None")
        
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.embeddings = index_data['embeddings']
        self.metadata = index_data['metadata']
        self.embedding_dim = index_data['embedding_dim']
        self.use_faiss = index_data.get('use_faiss', False)
        
        # Load Faiss index if available
        if self.use_faiss and 'faiss_index_path' in index_data:
            try:
                import faiss
                self.faiss_index = faiss.read_index(index_data['faiss_index_path'])
                logger.info("Faiss index loaded successfully")
            except ImportError:
                logger.warning("Faiss not available, using numpy-based search")
                self.use_faiss = False
                self.faiss_index = None
        
        logger.info(f"Index loaded from {filepath}: {len(self.metadata)} embeddings")
    
    def __len__(self) -> int:
        """Get number of embeddings in index."""
        return len(self.metadata) if self.metadata else 0

