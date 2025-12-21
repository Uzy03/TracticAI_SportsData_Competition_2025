"""Search system for similar CK retrieval using pretrained backbone embeddings."""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging

from .index import SimilarCKIndex
from ..modules.utils import load_backbone_from_checkpoint, get_device
from ..modules.view_ops import apply_view_transform, D2_VIEWS

logger = logging.getLogger(__name__)


class SimilarCKSearch:
    """Search system for finding similar corner kicks using pretrained backbone embeddings.
    
    Generates graph-level embeddings from pretrained receiver prediction backbone
    and performs similarity search using cosine similarity.
    """
    
    def __init__(
        self,
        backbone_checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        """Initialize SimilarCKSearch.
        
        Args:
            backbone_checkpoint_path: Path to pretrained backbone checkpoint
            device: Device to run model on (auto-detect if None)
        """
        if device is None:
            device = get_device("auto")
        
        self.device = device
        
        # Load pretrained backbone
        logger.info(f"Loading backbone from {backbone_checkpoint_path}")
        self.backbone, self.metadata = load_backbone_from_checkpoint(
            backbone_checkpoint_path,
            device=device
        )
        self.backbone.eval()
        
        # Check if D2 equivariance is enabled
        self.use_d2_equivariance = self.metadata.get("use_d2_equivariance", False)
        
        logger.info(
            f"Backbone loaded: D2={self.use_d2_equivariance}, "
            f"hidden_dim={self.metadata['hidden_dim']}"
        )
    
    def generate_embeddings(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate graph-level embeddings for all samples in dataloader.
        
        Args:
            dataloader: DataLoader containing graph data
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (embeddings [N, hidden_dim], metadata_list)
        """
        from tqdm import tqdm
        
        all_embeddings = []
        all_metadata = []
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Generating embeddings") if show_progress else dataloader
            
            for batch_idx, batch_data in enumerate(iterator):
                # Handle different data formats
                if isinstance(batch_data, tuple):
                    # (data_dict, target) format
                    data_dict, _ = batch_data
                elif isinstance(batch_data, dict):
                    data_dict = batch_data
                else:
                    raise ValueError(f"Unexpected batch data type: {type(batch_data)}")
                
                # Extract graph data
                x = data_dict["x"].to(self.device)
                edge_index = data_dict["edge_index"].to(self.device)
                edge_attr = data_dict.get("edge_attr")
                if edge_attr is not None:
                    edge_attr = edge_attr.to(self.device)
                batch = data_dict.get("batch")
                if batch is not None:
                    batch = batch.to(self.device)
                
                # Generate embeddings for this batch
                batch_embeddings = self._forward_batch(
                    x, edge_index, edge_attr, batch
                )
                
                # Convert to numpy and store
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings_np)
                
                # Create metadata for each sample in batch
                batch_size = batch_embeddings.shape[0]
                for i in range(batch_size):
                    metadata = {
                        'batch_idx': batch_idx,
                        'sample_idx_in_batch': i,
                        'global_idx': len(all_metadata),
                    }
                    # Add any additional metadata from data_dict if available
                    if 'data_id' in data_dict:
                        if isinstance(data_dict['data_id'], (list, tuple)):
                            metadata['data_id'] = data_dict['data_id'][i]
                        else:
                            metadata['data_id'] = data_dict['data_id']
                    if 'file_path' in data_dict:
                        if isinstance(data_dict['file_path'], (list, tuple)):
                            metadata['file_path'] = data_dict['file_path'][i]
                        else:
                            metadata['file_path'] = data_dict['file_path']
                    
                    all_metadata.append(metadata)
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings, all_metadata
    
    def _forward_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through backbone to generate graph embeddings.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional)
            
        Returns:
            Graph-level embeddings [B, hidden_dim]
        """
        # Determine batch size
        if batch is not None:
            B = batch.max().item() + 1
        else:
            B = 1
        
        if self.use_d2_equivariance:
            # D2 equivariance: Create 4 views and use GATv2Network4View
            views_list = []
            for view_idx in range(len(D2_VIEWS)):
                x_view = x.clone()
                # Apply D2 reflection to coordinate-like features
                # Assuming x, y are at indices 0, 1 (adjust based on schema)
                x_view = apply_view_transform(x_view, view_idx, xy_indices=(0, 1))  # x, y
                if x_view.size(-1) > 3:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(2, 3))  # vx, vy
                if x_view.size(-1) > 8:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(7, 8))  # dx_to_kicker, dy_to_kicker
                if x_view.size(-1) > 12:
                    x_view = apply_view_transform(x_view, view_idx, xy_indices=(11, 12))  # dx_to_goal, dy_to_goal
                views_list.append(x_view)
            
            # Stack views: [4, N, D] -> [B, 4, N_per_graph, D]
            x_views = torch.stack(views_list, dim=0)  # [4, N, D]
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            x_4view = x_views.view(4, B, num_nodes_per_graph, -1).permute(1, 0, 2, 3)  # [B, 4, N_per_graph, D]
            
            # Get node embeddings: [B, 4, N_per_graph, hidden_dim]
            node_emb_4view = self.backbone(x_4view, edge_index, edge_attr)  # [B, 4, N_per_graph, hidden_dim]
            
            # Group Averaging: Average over 4 views to get invariant representation
            H_inv = node_emb_4view.mean(dim=1)  # [B, N_per_graph, hidden_dim]
        else:
            # Standard mode: No D2 equivariance
            # Get node embeddings - don't pass batch so we get node-level embeddings
            H = self.backbone(x, edge_index, edge_attr, batch=None)  # [N_total, hidden_dim]
            
            # Group node embeddings by graph using batch tensor
            if batch is not None:
                unique_batches = torch.unique(batch, sorted=True)
                B_actual = len(unique_batches)
                
                # Group node embeddings by graph using batch tensor
                graph_embeddings = []
                for b in range(B_actual):
                    mask = (batch == b)
                    graph_nodes = H[mask]  # [N_b, hidden_dim]
                    graph_emb = graph_nodes.mean(dim=0)  # [hidden_dim]
                    graph_embeddings.append(graph_emb)
                
                z_graph = torch.stack(graph_embeddings, dim=0)  # [B_actual, hidden_dim]
                return z_graph
            else:
                # Single graph case
                z_graph = H.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                return z_graph
        
        # Graph Pooling: Mean pooling over nodes to get graph-level embedding
        z_graph = H_inv.mean(dim=1)  # [B, hidden_dim]
        
        return z_graph
    
    def build_index(
        self,
        dataloader: DataLoader,
        index: Optional[SimilarCKIndex] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> SimilarCKIndex:
        """Build search index from dataloader.
        
        Args:
            dataloader: DataLoader containing graph data
            index: Existing index to add to (creates new if None)
            save_path: Path to save index (optional)
            
        Returns:
            SimilarCKIndex instance
        """
        # Generate embeddings
        embeddings, metadata = self.generate_embeddings(dataloader)
        
        # Create or use existing index
        if index is None:
            index = SimilarCKIndex(
                embedding_dim=embeddings.shape[1],
                index_path=save_path,
            )
        
        # Add embeddings to index (normalize=True for cosine similarity)
        # Note: embeddings from generate_embeddings are NOT normalized yet
        index.add_embeddings(embeddings, metadata, normalize=True)
        
        # Save if path provided
        if save_path:
            index.save(save_path)
        
        return index
    
    def search_similar(
        self,
        query_data: Dict[str, torch.Tensor],
        index: SimilarCKIndex,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for similar CKs given query data.
        
        Args:
            query_data: Dictionary containing query graph data
                Required keys: 'x', 'edge_index'
                Optional keys: 'edge_attr', 'batch'
            index: SimilarCKIndex to search in
            top_k: Number of top similar CKs to return
            
        Returns:
            List of search results, each containing:
            {'similarity': float, 'metadata': dict, 'index': int}
        """
        # Extract query data
        x = query_data["x"].to(self.device)
        edge_index = query_data["edge_index"].to(self.device)
        edge_attr = query_data.get("edge_attr")
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        batch = query_data.get("batch")
        if batch is not None:
            batch = batch.to(self.device)
        
        # Generate embedding for query
        with torch.no_grad():
            query_embedding = self._forward_batch(x, edge_index, edge_attr, batch)
        
        # Search in index
        query_embedding_np = query_embedding.cpu().numpy()
        results = index.search(query_embedding_np, top_k=top_k, normalize=True)
        
        # Return results for first (and only) query
        return results[0] if results else []

