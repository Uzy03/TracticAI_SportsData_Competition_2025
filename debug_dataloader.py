#!/usr/bin/env python3

import torch
from tacticai.dataio.dataset import ShotDataset
from tacticai.dataio import create_dataloader

# Test dataset
dataset = ShotDataset('data/processed/shot_train/data.pickle', file_format='pickle')

# Test DataLoader with explicit settings
dataloader = create_dataloader(
    dataset, 
    batch_size=2, 
    shuffle=False, 
    num_workers=0, 
    pin_memory=False
)

print('Testing DataLoader with explicit settings...')
try:
    for i, (data, targets) in enumerate(dataloader):
        print(f'Batch {i}: Success')
        print(f'  Data keys: {list(data.keys())}')
        print(f'  Targets shape: {targets.shape}')
        if i >= 2:
            break
    print('DataLoader test successful!')
except Exception as e:
    print(f'DataLoader error: {e}')
    import traceback
    traceback.print_exc()
    
    # Test individual samples
    print('\nTesting individual samples:')
    for i in range(3):
        try:
            sample = dataset[i]
            print(f'Sample {i}: Success - {type(sample)}')
        except Exception as e2:
            print(f'Sample {i}: Error - {e2}')
            import traceback
            traceback.print_exc()
