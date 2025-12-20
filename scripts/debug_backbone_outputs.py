import torch
import numpy as np
from tacticai.dataio.dataset import ShotDataset
from tacticai.models.gatv2 import GATv2Network, GATv2Network4View
from tacticai.modules.utils import load_backbone_from_checkpoint
import logging
import argparse
from pathlib import Path
from torch.utils.data import Subset, DataLoader
from tacticai.dataio.schema import ShotSchema

def debug_backbone_outputs():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("debug_backbone")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Backbone (D2 disabled for simplicity in checking raw variance)
    # Using the path expected by the config
    backbone_path = "checkpoints/receiver/backbone_no_d2.ckpt" # Or d2 depending on config, let's assume no_d2 for now as debug_overfit used false
    
    if not Path(backbone_path).exists():
        logger.error(f"Backbone not found at {backbone_path}")
        return

    logger.info(f"Loading backbone from {backbone_path}")
    backbone, metadata = load_backbone_from_checkpoint(backbone_path, device)
    backbone.eval()
    
    # Load Data
    data_path = "data/processed_ck/shot_train/data.pickle"
    full_dataset = ShotDataset(data_path, file_format="pickle")
    
    # Select same 8 samples
    indices = [30, 73, 104, 143, 159, 170, 190, 252]
    subset = Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=None) # Batch size 8 to process all at once
    
    logger.info("Processing batch...")
    for data, targets in loader:
        # Move to device (manual handling since no custom collate here might return list of dicts or batched dict)
        # Actually DataLoader with default collate stacks tensors.
        # But our dataset returns dict of tensors. Default collate handles this fine.
        
        x = data["x"].to(device)
        edge_index = data["edge_index"].to(device)
        edge_attr = data.get("edge_attr")
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        batch = data.get("batch") # Might be None if not using geometric loader
        
        # If batch is None (standard DataLoader stacks, but for variable graphs we usually use Geometric loader)
        # ShotDataset returns fixed size graphs (22 nodes), so standard loader stacks them [B, N, F]
        # BUT GATv2 expects [N_total, F] + edge_index [2, E_total] + batch [N_total] usually?
        # Let's check GATv2 input signature.
        # GATv2Network forward: x, edge_index, edge_attr=None, batch=None
        
        # If we use standard DataLoader, x will be [B, 22, 16]. 
        # But GATv2 might expect flattened [B*22, 16] with batch index.
        # Let's verify how train_shot.py handles this.
        # train_shot.py uses create_dataloader which uses PyG DataLoader if available or custom?
        # It imports from tacticai.dataio.dataloader which uses torch_geometric.loader.DataLoader
        
        # Use tacticai's create_dataloader which handles imports
        from tacticai.dataio.dataloader import create_dataloader
        
        # Re-create loader
        loader = create_dataloader(subset, batch_size=8, shuffle=False)
        for data, targets in loader:
            # Note: create_dataloader returns (data, targets) tuple
            # data is a Batch object from torch_geometric if available, or a dict of stacked tensors
            
            # Check if data has 'x' attribute (PyG Batch) or is dict
            if hasattr(data, "x"):
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else None
                batch = data.batch.to(device)
            else:
                # Dict case
                x = data["x"].to(device)
                edge_index = data["edge_index"].to(device)
                edge_attr = data["edge_attr"].to(device) if "edge_attr" in data else None
                batch = data["batch"].to(device) if "batch" in data else None
            
            with torch.no_grad():
                # Forward pass through backbone
                # GATv2Network4View or GATv2Network
                # If we loaded no_d2, it's GATv2Network
                
                # Simple forward
                node_embeddings = backbone(x, edge_index, edge_attr, batch)
                
                # node_embeddings: [N_total, Dim] -> [B*22, 512]
                
                # Reshape to [B, 22, 512]
                B = 8
                N = 22
                H = node_embeddings.view(B, N, -1)
                
                # Calculate variance across the batch dimension
                # Does H vary between samples?
                
                # Mean embedding per graph
                graph_embs = H.mean(dim=1) # [B, 512]
                
                # Calculate pairwise cosine similarity between graph embeddings
                # to see how similar the 8 samples are in latent space.
                graph_embs_norm = torch.nn.functional.normalize(graph_embs, p=2, dim=1)
                similarity_matrix = torch.mm(graph_embs_norm, graph_embs_norm.t())
                
                logger.info("\nCosine Similarity Matrix between 8 samples (Graph Embeddings):")
                logger.info(f"\n{similarity_matrix.cpu().numpy()}")
                
                min_sim = similarity_matrix.min().item()
                mean_sim = similarity_matrix.mean().item()
                logger.info(f"Min Similarity: {min_sim:.4f}")
                logger.info(f"Mean Similarity: {mean_sim:.4f}")
                
                if min_sim > 0.99:
                    logger.warning("WARNING: All samples have nearly identical embeddings! Backbone is not distinguishing them.")
                else:
                    logger.info("Samples are distinguishable in latent space.")
                
                # Also check targets
                logger.info(f"Targets: {targets.numpy()}")
            
            break # Only one batch

if __name__ == "__main__":
    debug_backbone_outputs()

