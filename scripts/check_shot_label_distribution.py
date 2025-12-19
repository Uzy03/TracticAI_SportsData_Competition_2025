#!/usr/bin/env python3
"""Check label distribution for shot prediction task."""

import pickle
from pathlib import Path
from collections import Counter
import numpy as np

def check_distribution(data_path: Path, phase: str):
    """Check label distribution in a dataset."""
    print(f"\n{'='*60}")
    print(f"{phase.upper()} Dataset")
    print(f"{'='*60}")
    print(f"Path: {data_path}")
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and "samples" in data:
        samples = data["samples"]
        print(f"Total samples: {len(samples)}")
    elif isinstance(data, list):
        samples = data
        print(f"Total samples: {len(samples)}")
    else:
        print(f"Unknown data format: {type(data)}")
        return
    
    # Count shot_occurred labels
    shot_labels = []
    for sample in samples:
        if isinstance(sample, dict):
            shot_occurred = sample.get("shot_occurred", None)
            if shot_occurred is not None:
                shot_labels.append(int(shot_occurred))
    
    if not shot_labels:
        print("No shot_occurred labels found in samples")
        return
    
    # Count distribution
    counter = Counter(shot_labels)
    total = len(shot_labels)
    
    print(f"\nLabel distribution:")
    for label in sorted(counter.keys()):
        count = counter[label]
        percentage = 100.0 * count / total if total > 0 else 0.0
        label_name = "Shot (1)" if label == 1 else "No Shot (0)"
        print(f"  {label_name}: {count}/{total} ({percentage:.2f}%)")
    
    # Additional statistics
    shot_count = counter.get(1, 0)
    no_shot_count = counter.get(0, 0)
    
    if shot_count > 0 and no_shot_count > 0:
        imbalance_ratio = max(shot_count, no_shot_count) / min(shot_count, no_shot_count)
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 3.0:
            print("  ⚠️  High class imbalance detected!")
        elif imbalance_ratio > 2.0:
            print("  ⚠️  Moderate class imbalance detected")
        else:
            print("  ✓  Relatively balanced classes")


def main():
    """Main function."""
    base_dir = Path("data/processed_ck")
    
    phases = ["train", "val", "test"]
    
    for phase in phases:
        shot_path = base_dir / f"shot_{phase}" / "data.pickle"
        check_distribution(shot_path, phase)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    # Overall statistics
    all_labels = []
    for phase in phases:
        shot_path = base_dir / f"shot_{phase}" / "data.pickle"
        if shot_path.exists():
            with open(shot_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "samples" in data:
                samples = data["samples"]
            elif isinstance(data, list):
                samples = data
            else:
                continue
            
            for sample in samples:
                if isinstance(sample, dict):
                    shot_occurred = sample.get("shot_occurred", None)
                    if shot_occurred is not None:
                        all_labels.append(int(shot_occurred))
    
    if all_labels:
        counter = Counter(all_labels)
        total = len(all_labels)
        shot_count = counter.get(1, 0)
        no_shot_count = counter.get(0, 0)
        
        print(f"\nOverall distribution (all phases):")
        print(f"  Shot (1): {shot_count}/{total} ({100.0*shot_count/total:.2f}%)")
        print(f"  No Shot (0): {no_shot_count}/{total} ({100.0*no_shot_count/total:.2f}%)")
        
        if shot_count > 0 and no_shot_count > 0:
            imbalance_ratio = max(shot_count, no_shot_count) / min(shot_count, no_shot_count)
            print(f"\nOverall imbalance ratio: {imbalance_ratio:.2f}:1")


if __name__ == "__main__":
    main()

