"""Data preprocessing script for TacticAI.

This script converts raw data files into processed format suitable for training.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
import json
import pickle
from tqdm import tqdm

from tacticai.dataio.schema import create_schema_mapping, FlexibleSchema


def load_raw_data(data_path: Path, file_format: str = "csv") -> pd.DataFrame:
    """Load raw data from files.
    
    Args:
        data_path: Path to data files
        file_format: File format (csv, parquet, json)
        
    Returns:
        Combined DataFrame
    """
    if data_path.is_file():
        # Single file
        if file_format == "csv":
            return pd.read_csv(data_path)
        elif file_format == "parquet":
            return pd.read_parquet(data_path)
        elif file_format == "json":
            return pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
    else:
        # Directory of files
        pattern = f"*.{file_format}"
        files = list(data_path.glob(pattern))
        
        if not files:
            raise ValueError(f"No {file_format} files found in {data_path}")
        
        dataframes = []
        for file_path in tqdm(files, desc="Loading files"):
            if file_format == "csv":
                df = pd.read_csv(file_path)
            elif file_format == "parquet":
                df = pd.read_parquet(file_path)
            elif file_format == "json":
                df = pd.read_json(file_path)
            
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)


def preprocess_receiver_data(df: pd.DataFrame, schema: Any) -> Dict[str, Any]:
    """Preprocess data for receiver prediction with edge and graph attributes.
    
    Args:
        df: Raw DataFrame
        schema: Data schema
        
    Returns:
        Processed data dictionary
    """
    # Group by match/frame to create samples
    if 'match_id' in df.columns and 'frame_id' in df.columns:
        grouped = df.groupby(['match_id', 'frame_id'])
    else:
        # Assume each row is a separate sample
        grouped = [(None, df.iloc[[i]]) for i in range(len(df))]
    
    processed_data = []
    
    for group_key, group_df in tqdm(grouped, desc="Processing receiver data"):
        try:
            # Extract features and targets using schema
            node_features = schema.get_node_features(group_df)
            edge_index = schema.get_edge_index(group_df)
            target = schema.get_targets(group_df)
            
            # Extract edge attributes if available
            edge_attributes = None
            if hasattr(schema, 'get_edge_attributes'):
                edge_attributes = schema.get_edge_attributes(group_df)
                if edge_attributes is not None:
                    edge_attributes = edge_attributes.numpy()
            
            # Extract graph attributes if available
            graph_attributes = None
            if hasattr(schema, 'get_graph_attributes'):
                graph_attributes = schema.get_graph_attributes(group_df)
                if graph_attributes is not None:
                    graph_attributes = graph_attributes.numpy()
            
            processed_data.append({
                'node_features': node_features.numpy(),
                'edge_index': edge_index.numpy(),
                'target': target.numpy(),
                'edge_attributes': edge_attributes,
                'graph_attributes': graph_attributes,
                'match_id': group_key[0] if isinstance(group_key, tuple) else None,
                'frame_id': group_key[1] if isinstance(group_key, tuple) else None,
            })
        except Exception as e:
            print(f"Error processing group {group_key}: {e}")
            continue
    
    return processed_data


def preprocess_shot_data(df: pd.DataFrame, schema: Any) -> Dict[str, Any]:
    """Preprocess data for shot prediction with edge and graph attributes.
    
    Args:
        df: Raw DataFrame
        schema: Data schema
        
    Returns:
        Processed data dictionary
    """
    # Group by match/frame to create samples
    if 'match_id' in df.columns and 'frame_id' in df.columns:
        grouped = df.groupby(['match_id', 'frame_id'])
    else:
        # Assume each row is a separate sample
        grouped = [(None, df.iloc[[i]]) for i in range(len(df))]
    
    processed_data = []
    
    for group_key, group_df in tqdm(grouped, desc="Processing shot data"):
        try:
            # Extract features and targets using schema
            node_features = schema.get_node_features(group_df)
            edge_index = schema.get_edge_index(group_df)
            target = schema.get_targets(group_df)
            
            # Extract edge attributes if available
            edge_attributes = None
            if hasattr(schema, 'get_edge_attributes'):
                edge_attributes = schema.get_edge_attributes(group_df)
                if edge_attributes is not None:
                    edge_attributes = edge_attributes.numpy()
            
            # Extract graph attributes if available
            graph_attributes = None
            if hasattr(schema, 'get_graph_attributes'):
                graph_attributes = schema.get_graph_attributes(group_df)
                if graph_attributes is not None:
                    graph_attributes = graph_attributes.numpy()
            
            processed_data.append({
                'node_features': node_features.numpy(),
                'edge_index': edge_index.numpy(),
                'target': target.numpy(),
                'edge_attributes': edge_attributes,
                'graph_attributes': graph_attributes,
                'match_id': group_key[0] if isinstance(group_key, tuple) else None,
                'frame_id': group_key[1] if isinstance(group_key, tuple) else None,
            })
        except Exception as e:
            print(f"Error processing group {group_key}: {e}")
            continue
    
    return processed_data


def preprocess_cvae_data(df: pd.DataFrame, schema: Any) -> Dict[str, Any]:
    """Preprocess data for CVAE with edge and graph attributes.
    
    Args:
        df: Raw DataFrame
        schema: Data schema
        
    Returns:
        Processed data dictionary
    """
    # Group by match/frame to create samples
    if 'match_id' in df.columns and 'frame_id' in df.columns:
        grouped = df.groupby(['match_id', 'frame_id'])
    else:
        # Assume each row is a separate sample
        grouped = [(None, df.iloc[[i]]) for i in range(len(df))]
    
    processed_data = []
    
    for group_key, group_df in tqdm(grouped, desc="Processing CVAE data"):
        try:
            # Extract features and targets using schema
            node_features = schema.get_node_features(group_df)
            edge_index = schema.get_edge_index(group_df)
            target = schema.get_targets(group_df)
            conditions = schema.get_conditions(group_df)
            
            # Extract edge attributes if available
            edge_attributes = None
            if hasattr(schema, 'get_edge_attributes'):
                edge_attributes = schema.get_edge_attributes(group_df)
                if edge_attributes is not None:
                    edge_attributes = edge_attributes.numpy()
            
            # Extract graph attributes if available
            graph_attributes = None
            if hasattr(schema, 'get_graph_attributes'):
                graph_attributes = schema.get_graph_attributes(group_df)
                if graph_attributes is not None:
                    graph_attributes = graph_attributes.numpy()
            
            processed_data.append({
                'node_features': node_features.numpy(),
                'edge_index': edge_index.numpy(),
                'target': target.numpy(),
                'conditions': conditions.numpy(),
                'edge_attributes': edge_attributes,
                'graph_attributes': graph_attributes,
                'match_id': group_key[0] if isinstance(group_key, tuple) else None,
                'frame_id': group_key[1] if isinstance(group_key, tuple) else None,
            })
        except Exception as e:
            print(f"Error processing group {group_key}: {e}")
            continue
    
    return processed_data


def save_processed_data(data: List[Dict[str, Any]], output_path: Path, format: str = "parquet"):
    """Save processed data.
    
    Args:
        data: Processed data list
        output_path: Output path
        format: Output format (parquet, pickle, json)
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == "parquet":
        # Convert to DataFrame and save as parquet
        df_data = []
        for item in data:
            df_data.append({
                'node_features': item['node_features'].tolist(),
                'edge_index': item['edge_index'].tolist(),
                'target': item['target'].tolist(),
                'conditions': item.get('conditions', []).tolist() if 'conditions' in item else [],
                'edge_attributes': item.get('edge_attributes', []).tolist() if item.get('edge_attributes') is not None else [],
                'graph_attributes': item.get('graph_attributes', []).tolist() if item.get('graph_attributes') is not None else [],
                'match_id': item.get('match_id'),
                'frame_id': item.get('frame_id'),
            })
        
        df = pd.DataFrame(df_data)
        df.to_parquet(output_path / "data.parquet")
    
    elif format == "pickle":
        # Save as pickle
        with open(output_path / "data.pickle", 'wb') as f:
            pickle.dump(data, f)
    
    elif format == "json":
        # Convert to JSON-serializable format
        json_data = []
        for item in data:
            json_item = {
                'node_features': item['node_features'].tolist(),
                'edge_index': item['edge_index'].tolist(),
                'target': item['target'].tolist(),
                'conditions': item.get('conditions', []).tolist() if 'conditions' in item else [],
                'edge_attributes': item.get('edge_attributes', []).tolist() if item.get('edge_attributes') is not None else [],
                'graph_attributes': item.get('graph_attributes', []).tolist() if item.get('graph_attributes') is not None else [],
                'match_id': item.get('match_id'),
                'frame_id': item.get('frame_id'),
            }
            json_data.append(json_item)
        
        with open(output_path / "data.json", 'w') as f:
            json.dump(json_data, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported output format: {format}")


def create_dummy_data(output_path: Path, task: str, num_samples: int = 100):
    """Create dummy data for testing.
    
    Args:
        output_path: Output path
        task: Task type
        num_samples: Number of samples to create
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    dummy_data = []
    num_players = 22
    
    for i in range(num_samples):
        # Generate random positions
        positions = np.random.rand(num_players, 2) * 100  # Rough field size
        
        # Generate random velocities
        velocities = np.random.randn(num_players, 2) * 2
        
        # Generate team information
        team = np.random.randint(0, 2, num_players)
        
        # Generate ball information
        ball_owner = np.random.randint(0, num_players)
        ball = np.zeros(num_players)
        ball[ball_owner] = 1
        
        # Generate player attributes
        height = np.random.rand(num_players) * 0.3 + 1.7  # 1.7-2.0m
        weight = np.random.rand(num_players) * 30 + 60    # 60-90kg
        
        # Create sample data
        sample_data = {
            'x': positions[:, 0],
            'y': positions[:, 1],
            'vx': velocities[:, 0],
            'vy': velocities[:, 1],
            'height': height,
            'weight': weight,
            'team': team,
            'ball': ball,
        }
        
        # Add task-specific targets
        if task == "receiver":
            sample_data['receiver_id'] = np.random.randint(0, num_players)
        elif task == "shot":
            sample_data['shot_occurred'] = np.random.randint(0, 2)
        elif task == "cvae":
            # Target positions (slightly different from input)
            target_pos = positions + np.random.randn(num_players, 2) * 5
            sample_data['target_x'] = target_pos[:, 0]
            sample_data['target_y'] = target_pos[:, 1]
            
            # Conditions
            conditions = np.random.randn(8)
            sample_data['condition'] = conditions
        
        dummy_data.append(sample_data)
    
    # Save as parquet
    df = pd.DataFrame(dummy_data)
    df.to_parquet(output_path / "data.parquet")
    
    print(f"Created {num_samples} dummy samples for {task} task")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess TacticAI data")
    parser.add_argument("--input", type=str, required=True, help="Input data path")
    parser.add_argument("--output", type=str, required=True, help="Output data path")
    parser.add_argument("--task", type=str, choices=["receiver", "shot", "cvae"], 
                       help="Task type (if not specified, will process all)")
    parser.add_argument("--format", type=str, default="csv", 
                       choices=["csv", "parquet", "json"], help="Input file format")
    parser.add_argument("--output_format", type=str, default="parquet",
                       choices=["parquet", "pickle", "json"], help="Output file format")
    parser.add_argument("--dummy", action="store_true", help="Create dummy data for testing")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of dummy samples")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.dummy:
        # Create dummy data
        if args.task:
            tasks = [args.task]
        else:
            tasks = ["receiver", "shot", "cvae"]
        
        for task in tasks:
            task_output_path = output_path / task
            create_dummy_data(task_output_path, task, args.num_samples)
        
        print("Dummy data creation completed!")
        return
    
    # Load raw data
    print(f"Loading data from {input_path}")
    raw_df = load_raw_data(input_path, args.format)
    print(f"Loaded {len(raw_df)} rows")
    
    # Process data for each task
    tasks = [args.task] if args.task else ["receiver", "shot", "cvae"]
    
    for task in tasks:
        print(f"\nProcessing {task} data...")
        
        # Create schema
        schema = create_schema_mapping(task)
        
        # Preprocess data
        if task == "receiver":
            processed_data = preprocess_receiver_data(raw_df, schema)
        elif task == "shot":
            processed_data = preprocess_shot_data(raw_df, schema)
        elif task == "cvae":
            processed_data = preprocess_cvae_data(raw_df, schema)
        
        print(f"Processed {len(processed_data)} samples")
        
        # Save processed data
        task_output_path = output_path / task
        save_processed_data(processed_data, task_output_path, args.output_format)
        
        print(f"Saved {task} data to {task_output_path}")
    
    print("\nPreprocessing completed!")


if __name__ == "__main__":
    main()
