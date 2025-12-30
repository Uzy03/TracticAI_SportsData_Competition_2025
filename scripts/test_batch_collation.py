"""Test script to check how DataLoader collates batch tensors."""

import torch
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, num_samples=5, nodes_per_sample=22):
        self.num_samples = num_samples
        self.nodes_per_sample = nodes_per_sample
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Simulate what ReceiverDataset returns
        x = torch.randn(self.nodes_per_sample, 16)
        batch = torch.zeros(self.nodes_per_sample, dtype=torch.long)
        return {"x": x, "batch": batch}, torch.tensor(0)


def main():
    dataset = TestDataset(num_samples=5, nodes_per_sample=22)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    for batch_idx, batch_data in enumerate(dataloader):
        data_dict, target = batch_data
        x = data_dict["x"]
        batch = data_dict["batch"]
        
        print(f"Batch {batch_idx}:")
        print(f"  x shape: {x.shape}")
        print(f"  batch shape: {batch.shape}")
        print(f"  batch unique values: {torch.unique(batch)}")
        print(f"  batch max: {batch.max().item()}")
        print(f"  Expected: batch should have values [0, 1, 2] for 3 graphs")
        print()


if __name__ == "__main__":
    main()

