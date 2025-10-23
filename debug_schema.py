#!/usr/bin/env python3

import pickle
import numpy as np
from tacticai.dataio.schema import ShotSchema

# Load data
with open('data/processed/shot_train/data.pickle', 'rb') as f:
    data = pickle.load(f)

print(f'Data type: {type(data)}')
print(f'Data length: {len(data)}')

# Test first sample
sample = data[0]
print(f'Sample type: {type(sample)}')
print(f'Sample keys: {list(sample.keys())}')

# Test schema
schema = ShotSchema()
print(f'Schema position_columns: {schema.position_columns}')
print(f'Schema velocity_columns: {schema.velocity_columns}')

# Test get_node_features
try:
    node_features = schema.get_node_features(sample)
    print(f'Node features shape: {node_features.shape}')
except Exception as e:
    print(f'Error in get_node_features: {e}')
    import traceback
    traceback.print_exc()
    
    # Debug step by step
    print('\nDebugging step by step:')
    print(f'data type: {type(sample)}')
    print(f'data keys: {list(sample.keys())}')
    
    for col in schema.position_columns:
        print(f'Accessing data[{col}]: {type(sample[col])} - {sample[col][:3]}')
