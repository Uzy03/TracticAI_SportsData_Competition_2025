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
        # Optional split embeddings (attacking/defending) for decomposed similarity.
        # When present, cosine similarity can be computed as weighted sum of two dot products.
        self.embeddings_att: Optional[np.ndarray] = None
        self.embeddings_def: Optional[np.ndarray] = None
        
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

    def add_embeddings_split(
        self,
        embeddings_att: np.ndarray,
        embeddings_def: np.ndarray,
        metadata: List[Dict[str, Any]],
        normalize: bool = True,
    ) -> None:
        """Add attacking/defending embeddings to index (and keep combined embeddings for compatibility)."""
        a = np.asarray(embeddings_att, dtype=np.float32)
        d = np.asarray(embeddings_def, dtype=np.float32)
        if a.shape != d.shape:
            raise ValueError(f"Split embedding shape mismatch: att={a.shape}, def={d.shape}")
        if a.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got {a.ndim}D")
        if a.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {a.shape[1]}"
            )
        if len(metadata) != a.shape[0]:
            raise ValueError(f"Metadata length mismatch: {len(metadata)} for {a.shape[0]} embeddings")

        if normalize:
            an = np.linalg.norm(a, axis=1, keepdims=True)
            dn = np.linalg.norm(d, axis=1, keepdims=True)
            an = np.where(an == 0, 1.0, an)
            dn = np.where(dn == 0, 1.0, dn)
            a = a / an
            d = d / dn

        # Store split
        if self.embeddings_att is None:
            self.embeddings_att = a
            self.embeddings_def = d
            # Initialize metadata if needed
            if self.embeddings is None:
                self.metadata = metadata.copy()
        else:
            self.embeddings_att = np.vstack([self.embeddings_att, a])
            assert self.embeddings_def is not None
            self.embeddings_def = np.vstack([self.embeddings_def, d])
            self.metadata.extend(metadata)

        # Also keep a combined embedding for backward compatibility with older code paths.
        combined = a + d
        cn = np.linalg.norm(combined, axis=1, keepdims=True)
        cn = np.where(cn == 0, 1.0, cn)
        combined = combined / cn
        if self.embeddings is None:
            self.embeddings = combined
        else:
            self.embeddings = np.vstack([self.embeddings, combined])

        # Note: Faiss index is left as-is; if you want Faiss with split similarity, extend this class.
        logger.info(f"Added {len(a)} split embeddings to index. Total: {len(self.metadata)}")

    def search_split(
        self,
        query_embeddings_att: np.ndarray,
        query_embeddings_def: np.ndarray,
        top_k: int = 10,
        normalize: bool = True,
        w_att: float = 0.5,
        w_def: float = 0.5,
    ) -> List[List[Dict[str, Any]]]:
        """Search using split embeddings (att/def), combining similarities as weighted sum."""
        if self.embeddings_att is None or self.embeddings_def is None:
            # Fallback to standard search if split index not available
            q = np.asarray(query_embeddings_att, dtype=np.float32)
            return self.search(q, top_k=top_k, normalize=normalize)
        if self.embeddings_att is None or self.embeddings_def is None:
            raise ValueError("Split embeddings not loaded.")
        qa = np.asarray(query_embeddings_att, dtype=np.float32)
        qd = np.asarray(query_embeddings_def, dtype=np.float32)
        if qa.ndim == 1:
            qa = qa[np.newaxis, :]
        if qd.ndim == 1:
            qd = qd[np.newaxis, :]
        if qa.shape != qd.shape:
            raise ValueError(f"Query split shape mismatch: att={qa.shape}, def={qd.shape}")
        if qa.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {qa.shape[1]}"
            )
        if normalize:
            an = np.linalg.norm(qa, axis=1, keepdims=True)
            dn = np.linalg.norm(qd, axis=1, keepdims=True)
            an = np.where(an == 0, 1.0, an)
            dn = np.where(dn == 0, 1.0, dn)
            qa = qa / an
            qd = qd / dn

        # Weighted cosine similarity = weighted dot products of normalized vectors
        sims = float(w_att) * np.dot(qa, self.embeddings_att.T) + float(w_def) * np.dot(qd, self.embeddings_def.T)

        results: List[List[Dict[str, Any]]] = []
        for i in range(sims.shape[0]):
            query_similarities = sims[i]
            top_indices = np.argsort(query_similarities)[::-1][:top_k]
            query_results = []
            for idx in top_indices:
                query_results.append({
                    'similarity': float(query_similarities[idx]),
                    'metadata': self.metadata[int(idx)].copy(),
                    'index': int(idx),
                })
            results.append(query_results)
        return results
    
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
            'embeddings_att': self.embeddings_att,
            'embeddings_def': self.embeddings_def,
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
        self.embeddings_att = index_data.get('embeddings_att', None)
        self.embeddings_def = index_data.get('embeddings_def', None)
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

