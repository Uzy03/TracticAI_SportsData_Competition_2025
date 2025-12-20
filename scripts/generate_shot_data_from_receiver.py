#!/usr/bin/env python3
"""Generate shot prediction data from receiver prediction data.

This script takes the receiver prediction data from data/processed_ck/receiver_*
and creates shot prediction data with the same structure.
The receiver data already contains 'shot_occurred' field, so we just need to
extract and save it in the same format.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREPROCESS_VERSION = "ck_v3_tracking_based_shot_from_receiver"


def generate_shot_data_from_receiver(
    receiver_data_path: Path,
    output_path: Path,
) -> None:
    """Generate shot data from receiver data.
    
    Args:
        receiver_data_path: Path to receiver data pickle file
        output_path: Path to save shot data pickle file
    """
    logger.info(f"Loading receiver data from {receiver_data_path}")
    
    # Load receiver data
    with open(receiver_data_path, 'rb') as f:
        receiver_data = pickle.load(f)
    
    # Extract preprocess_version
    preprocess_version = receiver_data.get("preprocess_version", PREPROCESS_VERSION)
    receiver_samples = receiver_data.get("samples", [])
    
    logger.info(f"Found {len(receiver_samples)} samples")
    
    # Convert receiver samples to shot samples
    shot_samples: List[Dict[str, Any]] = []
    for sample in receiver_samples:
        # Check if shot_occurred exists
        if "shot_occurred" not in sample:
            logger.warning(f"Sample missing 'shot_occurred' field, skipping")
            continue
        
        # Create shot sample (same structure as receiver, but target is shot_occurred)
        shot_sample = dict(sample)  # Copy all fields
        # shot_occurred is already in the sample, so we're good
        
        shot_samples.append(shot_sample)
    
    logger.info(f"Generated {len(shot_samples)} shot samples")
    
    # Create output structure matching receiver data format
    shot_data = {
        "preprocess_version": preprocess_version,
        "samples": shot_samples,
    }
    
    # Save shot data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(shot_data, f)
    
    logger.info(f"Saved shot data to {output_path}")
    
    # Log statistics
    shot_occurred_count = sum(1 for s in shot_samples if s.get("shot_occurred", 0) == 1)
    logger.info(f"Shot statistics: {shot_occurred_count}/{len(shot_samples)} samples have shot_occurred=1 ({100*shot_occurred_count/len(shot_samples):.1f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate shot prediction data from receiver prediction data"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed_ck",
        help="Directory containing receiver data (default: data/processed_ck)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_ck",
        help="Directory to save shot data (default: data/processed_ck)"
    )
    parser.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Phases to process (default: train val test)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Process each phase
    for phase in args.phases:
        receiver_path = input_dir / f"receiver_{phase}" / "data.pickle"
        shot_path = output_dir / f"shot_{phase}" / "data.pickle"
        
        if not receiver_path.exists():
            logger.warning(f"Receiver data not found: {receiver_path}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {phase} phase")
        logger.info(f"{'='*60}")
        
        generate_shot_data_from_receiver(receiver_path, shot_path)
    
    logger.info("\n" + "="*60)
    logger.info("Shot data generation completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

