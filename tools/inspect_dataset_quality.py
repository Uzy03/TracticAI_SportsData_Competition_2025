import pickle
import numpy as np
import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
sys.path.append(project_root)

def inspect_dataset(file_path):
    print(f"=== Inspecting {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    if isinstance(data_dict, dict) and 'samples' in data_dict:
        data_list = data_dict['samples']
    else:
        data_list = data_dict # Fallback if it's a list directly
        
    print(f"Total samples: {len(data_list)}")
    
    # Counters
    self_pass_count = 0
    ball_kicker_dist_sum = 0
    valid_kicker_count = 0
    
    receiver_dist_sum = 0
    valid_receiver_count = 0
    
    for i, sample in enumerate(data_list):
        # Check self-pass
        kicker_id = sample.get('kicker_id')
        # Adjust key for receiver ID based on what's actually in the data
        # Usually 'receiver_id' or 'receiver'
        receiver_id = sample.get('receiver_id')
        
        if kicker_id is not None and receiver_id is not None:
            if kicker_id == receiver_id:
                self_pass_count += 1
        
        # Check distances using raw positions (if available)
        # Assuming 'x' and 'y' are columns in the dataframe or dict
        # We need to know the data structure.
        # Based on previous context, sample might be a dict with numpy arrays or a DataFrame row
        
        # Extract positions
        try:
            # Try to get positions from 'x' and 'y' keys if it's a dict of arrays
            # Or from a dataframe structure
            if 'x' in sample and 'y' in sample:
                xs = sample['x']
                ys = sample['y']
                
                # Get ball pos (if available)
                ball_x = sample.get('ball_x') # Might need to check how ball is stored
                ball_y = sample.get('ball_y')
                
                # If ball info is not direct, maybe in 'ball' column (ownership)
                # Let's skip complex distance calc for now and focus on IDs if raw coords aren't simple
                
                # If we can identify kicker index
                if 'kicker_idx' in sample:
                    k_idx = int(sample['kicker_idx'])
                    if k_idx < len(xs):
                        k_x, k_y = xs[k_idx], ys[k_idx]
                        
                        # If we have ball position columns or ball_x/y
                        # Let's assume standard TacticAI format where ball might be a separate entity or feature
                        pass

        except Exception as e:
            pass

    print(f"Self-passes (kicker == receiver): {self_pass_count} / {len(data_list)} ({self_pass_count/len(data_list)*100:.2f}%)")
    
    # Check label distribution
    receivers = []
    for sample in data_list:
        if 'receiver_node_index' in sample:
            receivers.append(sample['receiver_node_index'])
        elif 'receiver_id' in sample:
             receivers.append(sample['receiver_id']) # Might be player ID, not node index
    
    if receivers:
        receivers = np.array(receivers)
        print(f"Unique receivers: {len(np.unique(receivers))}")
        print(f"Label distribution (Top 5):")
        unique, counts = np.unique(receivers, return_counts=True)
        sorted_indices = np.argsort(-counts)
        for idx in sorted_indices[:5]:
            print(f"  ID {unique[idx]}: {counts[idx]} ({counts[idx]/len(receivers)*100:.2f}%)")

if __name__ == "__main__":
    inspect_dataset("data/processed_ck/receiver_train/data.pickle")
    inspect_dataset("data/processed_ck/receiver_val/data.pickle")

