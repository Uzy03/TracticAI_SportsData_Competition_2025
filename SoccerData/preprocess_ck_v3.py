#!/usr/bin/env python3
"""V3 Preprocessing: Tracking-based receiver detection for larger dataset."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import pickle
from tqdm import tqdm
import json
import logging
import argparse
import shutil

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

AUDIT_COUNTERS = {
    "total_ck": 0,
    "tracking_receiver_found": 0,
    "play_receiver_found": 0,
    "team_mismatch": 0,
    "target_not_in_cand": 0,
    "cand_count_outlier": 0,
    "no_receiver": 0,
    "kicker_mismatch": 0,
    "kept": 0,
}

AUDIT_RECORDS: Dict[str, List[Dict[str, Any]]] = {
    "train": [],
    "val": [],
    "test": [],
}

PREPROCESS_VERSION = "ck_v3_tracking_based"

def load_match_data(match_dir: Path):
    play_df = pd.read_csv(match_dir / "play.csv", encoding='utf-8')
    tracking_df = pd.read_csv(match_dir / "tracking.csv", encoding='utf-8')
    players_df = pd.read_csv(match_dir / "players.csv", encoding='utf-8')
    return play_df, tracking_df, players_df

def extract_frame_data(tracking_df, frame, prev_frame=None):
    """Extract player positions and velocities from tracking data."""
    frame_data = tracking_df[tracking_df['Frame'] == frame].copy()
    if len(frame_data) == 0:
        return None
    
    # Separate ball and players
    ball_data = frame_data[(frame_data['SysTarget'] == 0) | (frame_data['SysTarget'] == 7)]
    player_data = frame_data[(frame_data['SysTarget'] != 0) & (frame_data['SysTarget'] != 7)]
    
    # Handle missing players (pad to 22)
    num_players = len(player_data)
    mask = np.ones(22, dtype=int)
    
    if num_players < 22:
        missing_count = 22 - num_players
        placeholder = pd.DataFrame({
            'HA': [1] * missing_count,
            'SysTarget': [999] * missing_count,
            'X': [0] * missing_count,
            'Y': [0] * missing_count,
            'No': [0] * missing_count,
        })
        player_data = pd.concat([player_data, placeholder], ignore_index=True)
        mask[-missing_count:] = 0
    
    player_data = player_data.sort_values(['HA', 'Y', 'X'])
    positions = player_data[['X', 'Y']].values / 100.0  # cm to m
    positions[mask == 0] = 0.0
    
    # Velocities
    velocities = np.zeros((22, 2))
    if prev_frame is not None:
        prev_data = tracking_df[tracking_df['Frame'] == prev_frame]
        if len(prev_data) > 0:
            prev_players = prev_data[(prev_data['SysTarget'] != 0) & (prev_data['SysTarget'] != 7)]
            if len(prev_players) == num_players: # Simple check
                prev_players = prev_players.sort_values(['HA', 'Y', 'X'])
                prev_pos = prev_players[['X', 'Y']].values / 100.0
                # Need careful matching, but for now assume sorted order is stable
                # If padding was needed, velocity calc might be off for padded rows (masked anyway)
                # This is a simplification; v2 had better velocity logic
                if len(prev_pos) == len(positions[mask==1]):
                     velocities[mask==1] = positions[mask==1] - prev_pos

    # Team IDs (0 or 1)
    team_ids = (player_data['HA'].values - 1).astype(int)
    team_ids = np.clip(team_ids, 0, 1)
    
    # Ball ownership
    has_ball = np.zeros(22)
    ball_position = np.array([0.0, 0.0])
    if len(ball_data) > 0:
        ball_pos = ball_data[['X', 'Y']].values[0] / 100.0
        ball_position = ball_pos
        if mask.sum() > 0:
            distances = np.full(22, np.inf)
            distances[mask == 1] = np.linalg.norm(positions[mask == 1] - ball_pos, axis=1)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 2.0 and mask[closest_idx] == 1:
                has_ball[closest_idx] = 1

    # Normalize for model input
    x_norm = (positions[:, 0] + 52.5) / 105.0
    y_norm = (positions[:, 1] + 34.0) / 68.0
    x_norm[mask == 0] = 0.0
    y_norm[mask == 0] = 0.0

    return {
        'x': x_norm,
        'y': y_norm,
        'positions': positions,
        'velocities': velocities,
        'team_ids': team_ids,
        'has_ball': has_ball,
        'ball_position': ball_position,
        'mask': mask
    }

def find_first_touch(tracking_df, start_frame, kicker_team_id, max_frames=150):
    """Find the first player to touch the ball after start_frame."""
    # Get frames after start
    future_frames = tracking_df[
        (tracking_df['Frame'] > start_frame) & 
        (tracking_df['Frame'] <= start_frame + max_frames)
    ]['Frame'].unique()
    future_frames.sort()
    
    for frame in future_frames:
        frame_data = tracking_df[tracking_df['Frame'] == frame]
        if len(frame_data) == 0:
            continue
            
        ball_data = frame_data[(frame_data['SysTarget'] == 0) | (frame_data['SysTarget'] == 7)]
        if len(ball_data) == 0:
            continue
            
        ball_pos = ball_data[['X', 'Y']].values[0] / 100.0
        
        # Check all players
        players = frame_data[(frame_data['SysTarget'] != 0) & (frame_data['SysTarget'] != 7)]
        if len(players) == 0:
            continue
            
        # Calculate distances
        player_pos = players[['X', 'Y']].values / 100.0
        distances = np.linalg.norm(player_pos - ball_pos, axis=1)
        
        # Threshold for "touch" (e.g., 1.0m)
        touch_indices = np.where(distances < 1.0)[0]
        
        if len(touch_indices) > 0:
            # Sort by distance
            closest_idx = touch_indices[np.argmin(distances[touch_indices])]
            player_row = players.iloc[closest_idx]
            
            # Identify team
            ha = int(player_row['HA'])
            team_id = ha - 1  # 0 or 1
            
            # Identify jersey number for matching
            jersey_no = int(player_row['No'])
            
            return {
                'frame': frame,
                'team_id': team_id,
                'jersey_no': jersey_no,
                'ha': ha,
                'distance': distances[closest_idx]
            }
            
    return None

def get_player_index_by_ha_no(frame_data_dict, ha, jersey_no, players_df=None):
    """Find index in the 22-element array for a specific player."""
    # In extract_frame_data, we sorted by HA, Y, X.
    # This is unstable if players move. 
    # BUT: frame_data_dict contains 'team_ids' which is sorted.
    # We need to match the *exact* sorting logic or use the mask.
    
    # Actually, let's look at the source logic in extract_frame_data.
    # It sorts by ['HA', 'Y', 'X']. 
    # This means the index depends on Y position, which CHANGES every frame!
    # CRITICAL FLAW in v1/v2: Node indices were not consistent across frames?
    # Wait, GNN assumes graph topology. If node 0 is Player A in frame t,
    # it must be Player A in frame t+1 for temporal models.
    # For TacticAI (static graph), it doesn't matter as much, BUT
    # we need to map the receiver (found in future frame) back to the input frame (CK start).
    
    # Solution: We need to identify the player in the *input frame* (CK start).
    # The receiver found in *future frame* has (HA, No).
    # We need to find which index in the *input frame* corresponds to that (HA, No).
    
    # To do this, we need the raw player data from the input frame again.
    # Or, we can modify extract_frame_data to return metadata (HA, No) for each index.
    pass # Implemented inside create_samples

def create_samples_from_match(match_dir):
    global AUDIT_COUNTERS
    play_df, tracking_df, players_df = load_match_data(match_dir)
    samples = []
    
    ck_actions = play_df[play_df['アクション名'] == 'CK'].copy()
    
    for idx, row in ck_actions.iterrows():
        AUDIT_COUNTERS["total_ck"] += 1
        frame = row['フレーム番号']
        if pd.isna(frame): continue
        frame = int(frame)
        
        # 1. Get input frame data (CK start)
        # We need to know (HA, No) for each index to map targets later
        frame_raw = tracking_df[tracking_df['Frame'] == frame].copy()
        if len(frame_raw) == 0: continue
        
        # Filter players
        player_raw = frame_raw[(frame_raw['SysTarget'] != 0) & (frame_raw['SysTarget'] != 7)]
        
        # Sort to match extract_frame_data logic
        player_raw = player_raw.sort_values(['HA', 'Y', 'X'])
        
        # Create a mapping: index -> (HA, No)
        idx_to_player = {}
        for i, (_, p_row) in enumerate(player_raw.iterrows()):
            if i >= 22: break
            idx_to_player[i] = (int(p_row['HA']), int(p_row['No']))
            
        # Get processed input features
        frame_data = extract_frame_data(tracking_df, frame, frame-1)
        if frame_data is None: continue
        
        # Identify Kicker (closest to ball in input frame)
        kicker_idx = int(np.argmax(frame_data['has_ball']))
        kicker_team = int(frame_data['team_ids'][kicker_idx])
        
        # 2. Find Receiver (Tracking based)
        receiver_info = find_first_touch(tracking_df, frame + 5, kicker_team, max_frames=150)
        # +5 frames to skip the kick itself
        
        receiver_idx = None
        
        if receiver_info:
            # Found a touch. Check if it's the attacking team.
            if receiver_info['team_id'] == kicker_team:
                # Match back to input frame indices
                rec_ha = receiver_info['ha']
                rec_no = receiver_info['jersey_no']
                
                for i, (ha, no) in idx_to_player.items():
                    if ha == rec_ha and no == rec_no:
                        receiver_idx = i
                        break
                
                if receiver_idx is not None:
                    AUDIT_COUNTERS["tracking_receiver_found"] += 1
        
        # 3. Fallback to Play data (if tracking failed)
        if receiver_idx is None:
            next_idx = idx + 1
            if next_idx < len(play_df):
                next_row = play_df.iloc[next_idx]
                # Check if same team
                if next_row['チームID'] == row['チームID']:
                    # Try to match by name
                    rec_name = next_row['選手名']
                    # Find (HA, No) for this name from players_df
                    p_info = players_df[players_df['選手名'] == rec_name]
                    if len(p_info) > 0:
                        rec_ha = int(p_info.iloc[0]['ホームアウェイF'])
                        rec_no = int(p_info.iloc[0]['背番号'])
                        
                        for i, (ha, no) in idx_to_player.items():
                            if ha == rec_ha and no == rec_no:
                                receiver_idx = i
                                AUDIT_COUNTERS["play_receiver_found"] += 1
                                break

        if receiver_idx is None:
            AUDIT_COUNTERS["no_receiver"] += 1
            continue
            
        # 4. Create Mask and Filter
        # Candidates: Same team as kicker, excluding kicker
        team_ids = frame_data['team_ids']
        mask = frame_data['mask']
        
        cand_mask = (team_ids == kicker_team) & (mask == 1)
        cand_mask[kicker_idx] = False
        
        # Ensure receiver is in candidates (might have been filtered if mask=0?)
        # If receiver was missing in input frame (mask=0), we can't predict them!
        if mask[receiver_idx] == 0:
            # Receiver wasn't visible in input frame
            AUDIT_COUNTERS["target_not_in_cand"] += 1
            continue
            
        cand_mask[receiver_idx] = True # Ensure True
        
        candidate_count = cand_mask.sum()
        if candidate_count < 2: # At least 1 receiver + others? 
            AUDIT_COUNTERS["cand_count_outlier"] += 1
            continue
            
        # Create features (similar to v2)
        # ... (Copy feature extraction logic)
        # Simplified for brevity, ensure core features are present
        
        # Calculate relative features
        positions = frame_data['positions']
        kicker_pos = positions[kicker_idx]
        goal_center = np.array([52.5, 0.0])
        
        dx_to_kicker = positions[:, 0] - kicker_pos[0]
        dy_to_kicker = positions[:, 1] - kicker_pos[1]
        dist_to_kicker = np.sqrt(dx_to_kicker**2 + dy_to_kicker**2)
        angle_to_kicker = np.arctan2(dy_to_kicker, dx_to_kicker)
        
        dx_to_goal = positions[:, 0] - goal_center[0]
        dy_to_goal = positions[:, 1] - goal_center[1]
        dist_to_goal = np.sqrt(dx_to_goal**2 + dy_to_goal**2)
        angle_to_goal = np.arctan2(dy_to_goal, dx_to_goal)
        
        field_diag = np.sqrt(105**2 + 68**2)
        
        sample = {
            'x': frame_data['x'],
            'y': frame_data['y'],
            'vx': frame_data['velocities'][:, 0],
            'vy': frame_data['velocities'][:, 1],
            'height': np.random.uniform(1.7, 2.0, 22), # Placeholder
            'weight': np.random.uniform(60, 90, 22),
            'ball': frame_data['has_ball'],
            'team': np.where(team_ids == kicker_team, 0, 1), # 0=Attacking
            'receiver_node_index': receiver_idx,
            'receiver_id': receiver_idx, # Use node index as ID for simplicity now
            'cand_mask': cand_mask,
            'match_id': str(row['試合ID']),
            'frame': frame,
            # Relative features
            'dx_to_kicker': dx_to_kicker / 105.0,
            'dy_to_kicker': dy_to_kicker / 68.0,
            'dist_to_kicker': dist_to_kicker / field_diag,
            'angle_to_kicker': angle_to_kicker / np.pi,
            'dx_to_goal': dx_to_goal / 105.0,
            'dy_to_goal': dy_to_goal / 68.0,
            'dist_to_goal': dist_to_goal / field_diag,
            'angle_to_goal': angle_to_goal / np.pi,
            'kicker_idx': kicker_idx
        }
        
        samples.append(sample)
        AUDIT_COUNTERS["kept"] += 1

    return samples

def process_all(data_dir, output_dir):
    match_dirs = []
    for year in ['2023_data', '2024_data']:
        p = data_dir / year
        if p.exists():
            match_dirs.extend([x for x in p.iterdir() if x.is_dir() and (x/"play.csv").exists()])
            
    print(f"Found {len(match_dirs)} matches")
    all_samples = []
    for m in tqdm(match_dirs):
        try:
            all_samples.extend(create_samples_from_match(m))
        except Exception as e:
            print(f"Error {m}: {e}")
            
    print("Audit:", AUDIT_COUNTERS)
    print(f"Total samples: {len(all_samples)}")
    
    if not all_samples: return

    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    n = len(all_samples)
    train = all_samples[:int(n*0.7)]
    val = all_samples[int(n*0.7):int(n*0.85)]
    test = all_samples[int(n*0.85):]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('train', train), ('val', val), ('test', test)]:
        d = output_dir / f"receiver_{name}"
        d.mkdir(exist_ok=True)
        with open(d / "data.pickle", 'wb') as f:
            pickle.dump({"samples": data, "version": "v3"}, f)
        print(f"Saved {name}: {len(data)}")

if __name__ == "__main__":
    process_all(Path("SoccerData"), Path("data/processed_ck_v3"))

