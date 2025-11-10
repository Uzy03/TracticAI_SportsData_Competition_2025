#!/usr/bin/env python3
"""Preprocess SoccerData CK (Corner Kick) data to TacticAI format."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import json


def load_match_data(match_dir: Path):
    play_df = pd.read_csv(match_dir / "play.csv", encoding='utf-8')
    tracking_df = pd.read_csv(match_dir / "tracking.csv", encoding='utf-8')
    return play_df, tracking_df


def extract_frame_positions(tracking_df, frame):
    frame_data = tracking_df[tracking_df['Frame'] == frame].copy()
    if len(frame_data) == 0:
        return None
    
    ball_data = frame_data[frame_data['SysTarget'] == 7]
    player_data = frame_data[frame_data['SysTarget'] != 7]
    
    if len(player_data) != 22:
        return None
    
    player_data = player_data.sort_values(['HA', 'Y', 'X'])
    positions = player_data[['X', 'Y']].values / 100.0
    team_ids = np.clip((player_data['HA'].values - 1).astype(int), 0, 1)
    
    has_ball = np.zeros(22)
    if len(ball_data) > 0:
        ball_pos = ball_data[['X', 'Y']].values[0] / 100.0
        distances = np.linalg.norm(positions - ball_pos, axis=1)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 2.0:
            has_ball[closest_idx] = 1
    
    velocities = np.zeros((22, 2))
    heights = np.random.uniform(1.7, 2.0, 22)
    weights = np.random.uniform(60, 90, 22)
    
    return {
        'positions': positions, 'velocities': velocities, 'team_ids': team_ids,
        'has_ball': has_ball, 'heights': heights, 'weights': weights,
    }


def create_samples_from_match(match_dir):
    play_df, tracking_df = load_match_data(match_dir)
    samples = []
    
    # Extract CK (Corner Kick) actions
    ck_actions = play_df[play_df['アクション名'] == 'CK'].copy()
    
    for _, row in ck_actions.iterrows():
        frame = int(row['フレーム番号']) if pd.notna(row['フレーム番号']) else None
        if frame is None:
            continue
        
        frame_data = extract_frame_positions(tracking_df, frame)
        if frame_data is None:
            continue
        
        num_nodes = frame_data['positions'].shape[0]
        team_labels = frame_data['team_ids']

        # Identify kicker (player currently closest to ball)
        kicker_idx = int(np.argmax(frame_data['has_ball']))
        kicker_team = int(team_labels[kicker_idx])

        node_roles = np.array(["ball" if i == kicker_idx else "player" for i in range(num_nodes)], dtype=object)
        cand_mask = np.zeros(num_nodes, dtype=bool)
        for i in range(num_nodes):
            if team_labels[i] == kicker_team and node_roles[i] != "ball":
                cand_mask[i] = True

        assert cand_mask.sum() in {10, 11}, "Unexpected candidate count"
        
        # Build timeline of events after CK to find first attacking touch
        future_frames = (
            tracking_df[tracking_df['Frame'] > frame]['Frame']
            .drop_duplicates()
            .sort_values()
            .to_numpy()
        )

        max_future_frames = 150  # roughly ~5 seconds at 30Hz
        receiver_idx = None
        for future_frame in future_frames:
            if future_frame - frame > max_future_frames:
                break
            future_data = extract_frame_positions(tracking_df, future_frame)
            if future_data is None:
                continue
            future_has_ball = future_data['has_ball']
            if future_has_ball.sum() == 0:
                continue
            touch_idx = int(np.argmax(future_has_ball))
            touch_team = int(future_data['team_ids'][touch_idx])
            if touch_team == kicker_team:
                receiver_idx = touch_idx
                break

        if receiver_idx is None:
            # No valid attacking receiver found; skip this sample
            continue

        assert cand_mask[receiver_idx] is True, (
            f"Receiver index {receiver_idx} not in candidate mask "
            f"(team={team_labels[receiver_idx]}, kicker_team={kicker_team})"
        )
        
        x_normalized = (frame_data['positions'][:, 0] + 52.5) / 105.0
        y_normalized = (frame_data['positions'][:, 1] + 34.0) / 68.0
        
        sample = {
            'x': x_normalized, 'y': y_normalized,
            'vx': frame_data['velocities'][:, 0], 'vy': frame_data['velocities'][:, 1],
            'height': frame_data['heights'], 'weight': frame_data['weights'],
            'ball': frame_data['has_ball'], 'team': frame_data['team_ids'],
            'receiver_idx': receiver_idx, 'receiver_id': receiver_idx,
            'match_id': str(row['試合ID']), 'frame': frame,
            'cand_mask': cand_mask.astype(bool),
            'valid': True,
        }
        
        samples.append(sample)
    
    return samples


def process_all_matches(data_dir, output_dir, task="receiver", split_ratio=(0.7, 0.15, 0.15)):
    match_dirs = []
    for year_dir in ['2023_data', '2024_data']:
        year_path = data_dir / year_dir
        if year_path.exists():
            for match_dir in year_path.iterdir():
                if match_dir.is_dir() and (match_dir / "play.csv").exists():
                    match_dirs.append(match_dir)
    
    print(f"Found {len(match_dirs)} matches to process")
    
    all_samples = []
    for match_dir in tqdm(match_dirs, desc="Processing matches"):
        try:
            samples = create_samples_from_match(match_dir)
            all_samples.extend(samples)
        except Exception as e:
            print(f"Error processing {match_dir}: {e}")
            continue
    
    print(f"Created {len(all_samples)} samples")
    
    if len(all_samples) == 0:
        return
    
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    n_total = len(all_samples)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    splits = {
        'train': all_samples[:n_train],
        'val': all_samples[n_train:n_train+n_val],
        'test': all_samples[n_train+n_val:],
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, samples in splits.items():
        split_dir = output_dir / f"{task}_{split_name}"
        split_dir.mkdir(parents=True, exist_ok=True)
        
        with open(split_dir / "data.pickle", 'wb') as f:
            pickle.dump(samples, f)
        
        metadata = {'task': task, 'split': split_name, 'num_samples': len(samples)}
        with open(split_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(samples)} {split_name} samples to {split_dir}")


if __name__ == "__main__":
    process_all_matches(
        data_dir=Path("SoccerData"),
        output_dir=Path("data/processed_ck"),
        task="receiver",
        split_ratio=(0.7, 0.15, 0.15),
    )

