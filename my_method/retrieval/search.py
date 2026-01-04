"""Search system for similar CK retrieval using pretrained backbone embeddings."""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import logging
import math

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
        config: Optional[Dict[str, Any]] = None,
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
        metadata_override = None
        if config is not None:
            # Derive backbone metadata from config for loading full checkpoints (e.g., multitask best_*.ckpt)
            model_cfg = config.get("model", {})
            d2_cfg = config.get("d2", {})
            metadata_override = {
                "input_dim": model_cfg.get("input_dim"),
                "hidden_dim": model_cfg.get("hidden_dim"),
                "num_layers": model_cfg.get("num_layers"),
                "num_heads": model_cfg.get("num_heads"),
                "dropout": model_cfg.get("dropout", 0.0),
                "edge_dim": model_cfg.get("edge_dim", 1),
                "use_d2_equivariance": d2_cfg.get("enabled", False),
                # Provide a hint for constructor selection
                "backbone_type": "GATv2Network4View" if d2_cfg.get("enabled", False) else "GATv2Network",
            }
        self.backbone, self.metadata = load_backbone_from_checkpoint(
            backbone_checkpoint_path,
            device=device,
            metadata_override=metadata_override,
        )
        self.backbone.eval()
        
        # Check if D2 equivariance is enabled
        self.use_d2_equivariance = self.metadata.get("use_d2_equivariance", False)

        # Retrieval behavior toggles (optional)
        retrieval_cfg = (config or {}).get("retrieval", {}) if isinstance(config, dict) else {}
        self.corner_canonicalize = bool(retrieval_cfg.get("corner_canonicalize", True))
        self.split_att_def = bool(retrieval_cfg.get("split_att_def", True))
        self.w_att = float(retrieval_cfg.get("w_att", 0.5))
        self.w_def = float(retrieval_cfg.get("w_def", 0.5))
        
        logger.info(
            f"Backbone loaded: D2={self.use_d2_equivariance}, "
            f"hidden_dim={self.metadata['hidden_dim']}"
        )

    @staticmethod
    def _wrap_angle_pi(theta: torch.Tensor) -> torch.Tensor:
        """Wrap angles (radians) into [-pi, pi]."""
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    def _canonicalize_corner_invariant(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Unify 4 corner locations by reflecting features so the kicker corner maps to (x~1,y~1).

        Assumes ReceiverSchema node features (dim>=16):
        - pos: (0,1), vel: (2,3)
        - rel_to_kicker: dx/dy (7,8), angle (10)
        - rel_to_goal: dx/dy (11,12), angle (14)
        - team_id (15) untouched
        """
        if x.size(-1) < 16:
            return x

        x2 = x.clone()
        ball_dim = 6

        def _apply_one(mask: Union[slice, torch.Tensor]) -> None:
            xb = x2[mask]
            if xb.numel() == 0:
                return
            ball = xb[:, ball_dim]
            if torch.isfinite(ball).any() and float(ball.max().item()) > 0.0:
                k = int(torch.argmax(ball).item())
            else:
                k = 0
            kx = float(xb[k, 0].item())
            ky = float(xb[k, 1].item())
            flip_x = (kx < 0.5)
            flip_y = (ky < 0.5)

            if flip_x:
                xb[:, 0] = 1.0 - xb[:, 0]
                xb[:, 2] = -xb[:, 2]
                xb[:, 7] = -xb[:, 7]
                xb[:, 11] = -xb[:, 11]
                th = xb[:, 10] * math.pi
                xb[:, 10] = self._wrap_angle_pi(math.pi - th) / math.pi
                th2 = xb[:, 14] * math.pi
                xb[:, 14] = self._wrap_angle_pi(math.pi - th2) / math.pi

            if flip_y:
                xb[:, 1] = 1.0 - xb[:, 1]
                xb[:, 3] = -xb[:, 3]
                xb[:, 8] = -xb[:, 8]
                xb[:, 12] = -xb[:, 12]
                th = xb[:, 10] * math.pi
                xb[:, 10] = self._wrap_angle_pi(-th) / math.pi
                th2 = xb[:, 14] * math.pi
                xb[:, 14] = self._wrap_angle_pi(-th2) / math.pi

            x2[mask] = xb

        if batch is None:
            _apply_one(slice(None))
        else:
            B = int(batch.max().item()) + 1
            for b in range(B):
                _apply_one(batch == b)

        return x2
    
    def generate_embeddings(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], List[Dict[str, Any]]]:
        """Generate graph-level embeddings for all samples in dataloader.
        
        Args:
            dataloader: DataLoader containing graph data
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (embeddings_combined [N, hidden_dim], embeddings_att [N, hidden_dim] or None,
            embeddings_def [N, hidden_dim] or None, metadata_list)
        """
        from tqdm import tqdm
        
        all_embeddings = []
        all_att = []
        all_def = []
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
                if self.split_att_def:
                    batch_embeddings, batch_att, batch_def = self._forward_batch(
                        x, edge_index, edge_attr, batch, return_split=True, corner_canonicalize=self.corner_canonicalize
                    )
                else:
                    batch_embeddings = self._forward_batch(
                        x, edge_index, edge_attr, batch, return_split=False, corner_canonicalize=self.corner_canonicalize
                    )
                    batch_att = batch_def = None
                
                # Convert to numpy and store
                batch_embeddings_np = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings_np)
                if batch_att is not None and batch_def is not None:
                    all_att.append(batch_att.cpu().numpy())
                    all_def.append(batch_def.cpu().numpy())
                
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
        emb_att = np.vstack(all_att) if len(all_att) > 0 else None
        emb_def = np.vstack(all_def) if len(all_def) > 0 else None
        
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings, emb_att, emb_def, all_metadata
    
    def _forward_batch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: Optional[torch.Tensor],
        return_split: bool = False,
        corner_canonicalize: bool = True,
    ):
        """Forward pass through backbone to generate graph embeddings.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim] (optional)
            batch: Batch assignment [N] (optional)
            
        Returns:
            Graph-level embeddings [B, hidden_dim]
        """
        # Optional canonicalization to unify 4 corner locations (helps cross-corner retrieval)
        if corner_canonicalize:
            x = self._canonicalize_corner_invariant(x, batch)

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
                
                if return_split and x.size(-1) >= 16:
                    z_att_list = []
                    z_def_list = []
                    z_comb_list = []
                    for b in range(B_actual):
                        mask = (batch == b)
                        xb = x[mask]
                        Hb = H[mask]
                        team = xb[:, 15]
                        att_mask = team < 0.5
                        def_mask = ~att_mask
                        z_att = Hb[att_mask].mean(dim=0) if att_mask.any() else Hb.mean(dim=0)
                        z_def = Hb[def_mask].mean(dim=0) if def_mask.any() else Hb.mean(dim=0)
                        z_att_list.append(z_att)
                        z_def_list.append(z_def)
                        z = float(self.w_att) * F.normalize(z_att.unsqueeze(0), dim=1) + float(self.w_def) * F.normalize(z_def.unsqueeze(0), dim=1)
                        z = F.normalize(z, dim=1).squeeze(0)
                        z_comb_list.append(z)
                    z_graph = torch.stack(z_comb_list, dim=0)
                    z_att = torch.stack(z_att_list, dim=0)
                    z_def = torch.stack(z_def_list, dim=0)
                    return z_graph, z_att, z_def

                # Original behavior: simple mean pooling
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
                if return_split and x.size(-1) >= 16:
                    team = x[:, 15]
                    att_mask = team < 0.5
                    def_mask = ~att_mask
                    z_att = H[att_mask].mean(dim=0, keepdim=True) if att_mask.any() else H.mean(dim=0, keepdim=True)
                    z_def = H[def_mask].mean(dim=0, keepdim=True) if def_mask.any() else H.mean(dim=0, keepdim=True)
                    z_graph = float(self.w_att) * F.normalize(z_att, dim=1) + float(self.w_def) * F.normalize(z_def, dim=1)
                    z_graph = F.normalize(z_graph, dim=1)
                    return z_graph, z_att, z_def

                z_graph = H.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                return z_graph
        
        # D2 path: optionally split by team
        if return_split and x.size(-1) >= 16:
            N_total = x.size(0)
            num_nodes_per_graph = N_total // B if B > 1 else N_total
            x_bg = x.view(B, num_nodes_per_graph, -1)
            team = x_bg[:, :, 15]  # [B, N]
            att_mask = team < 0.5
            def_mask = ~att_mask
            z_att_list = []
            z_def_list = []
            z_comb_list = []
            for b in range(B):
                Hb = H_inv[b]  # [N, hidden_dim]
                am = att_mask[b]
                dm = def_mask[b]
                z_att = Hb[am].mean(dim=0) if am.any() else Hb.mean(dim=0)
                z_def = Hb[dm].mean(dim=0) if dm.any() else Hb.mean(dim=0)
                z_att_list.append(z_att)
                z_def_list.append(z_def)
                z = float(self.w_att) * F.normalize(z_att.unsqueeze(0), dim=1) + float(self.w_def) * F.normalize(z_def.unsqueeze(0), dim=1)
                z = F.normalize(z, dim=1).squeeze(0)
                z_comb_list.append(z)
            z_graph = torch.stack(z_comb_list, dim=0)
            z_att = torch.stack(z_att_list, dim=0)
            z_def = torch.stack(z_def_list, dim=0)
            return z_graph, z_att, z_def

        # Original D2 behavior: mean pooling over nodes
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
        embeddings, emb_att, emb_def, metadata = self.generate_embeddings(dataloader)
        
        # Create or use existing index
        if index is None:
            index = SimilarCKIndex(
                embedding_dim=embeddings.shape[1],
                index_path=save_path,
            )
        
        # Add embeddings to index (prefer split embeddings if available)
        # Note: generate_embeddings returns NOT-yet-normalized tensors; index normalizes internally.
        if emb_att is not None and emb_def is not None:
            index.add_embeddings_split(emb_att, emb_def, metadata, normalize=True)
        else:
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
            if index.embeddings_att is not None and index.embeddings_def is not None and self.split_att_def:
                _z, z_att, z_def = self._forward_batch(
                    x,
                    edge_index,
                    edge_attr,
                    batch,
                    return_split=True,
                    corner_canonicalize=self.corner_canonicalize,
                )
                query_embedding_att = z_att.cpu().numpy()
                query_embedding_def = z_def.cpu().numpy()
            else:
                query_embedding = self._forward_batch(
                    x, edge_index, edge_attr, batch, return_split=False, corner_canonicalize=self.corner_canonicalize
                )
                query_embedding_att = query_embedding_def = None
        
        # Search in index
        if query_embedding_att is not None and query_embedding_def is not None:
            results = index.search_split(
                query_embedding_att,
                query_embedding_def,
                top_k=top_k,
                normalize=True,
                w_att=self.w_att,
                w_def=self.w_def,
            )
        else:
            query_embedding_np = query_embedding.cpu().numpy()
            results = index.search(query_embedding_np, top_k=top_k, normalize=True)
        
        # Return results for first (and only) query
        return results[0] if results else []

