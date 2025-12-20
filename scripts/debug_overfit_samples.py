import torch
import numpy as np
from tacticai.dataio.dataset import ShotDataset
from torch.utils.data import Subset
from pathlib import Path
import logging

def check_overfit_samples():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("check_samples")

    data_path = "data/processed_ck/shot_train/data.pickle"
    
    # Load dataset
    full_dataset = ShotDataset(data_path, file_format="pickle")
    
    # Same logic as training script to select 8 samples
    num_samples = 8
    subset_seed = 42
    
    rng = np.random.RandomState(subset_seed)
    total_samples = len(full_dataset)
    indices = rng.permutation(total_samples)[:num_samples]
    indices = sorted(indices.tolist())
    
    logger.info(f"Checking {len(indices)} samples: {indices}")
    
    subset = Subset(full_dataset, indices)
    
    for i, idx in enumerate(indices):
        sample = full_dataset[idx] # Get raw sample dict if possible, or processed item
        
        # ShotDataset returns (data, target)
        # We need to peek into the internal list if possible, or analyze the returned tensors
        
        # Access raw sample from internal list
        raw_sample = full_dataset.data[idx]
        
        game_id = raw_sample.get("game_id", "unknown")
        seq_id = raw_sample.get("sequence_id", "unknown")
        frame_id = raw_sample.get("frame_id", "unknown")
        shot_flag = raw_sample.get("shot_occurred", "unknown")
        
        # Check target from dataset.__getitem__
        data, target = full_dataset[idx]
        
        logger.info(f"Sample {i} (Idx {idx}): Target={target.item()}, RawShot={shot_flag}, Game={game_id}, Seq={seq_id}, Frame={frame_id}")

if __name__ == "__main__":
    check_overfit_samples()

