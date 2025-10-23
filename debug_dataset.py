#!/usr/bin/env python3

from tacticai.dataio.dataset import ShotDataset

# Test dataset directly
dataset = ShotDataset('data/processed/shot_train/data.pickle', file_format='pickle')

print(f'Dataset length: {len(dataset)}')
print(f'Dataset data type: {type(dataset.data)}')
print(f'First sample type: {type(dataset.data[0])}')

# Test individual sample access
try:
    sample = dataset[0]
    print(f'Sample access successful: {type(sample)}')
    print(f'Sample keys: {list(sample[0].keys())}')
    print(f'Target shape: {sample[1].shape}')
except Exception as e:
    print(f'Error accessing sample: {e}')
    import traceback
    traceback.print_exc()
    
    # Debug the raw data
    print(f'Raw data[0] type: {type(dataset.data[0])}')
    print(f'Raw data[0] content: {dataset.data[0]}')
