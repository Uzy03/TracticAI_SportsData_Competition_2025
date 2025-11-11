#!/usr/bin/env python3
"""Improved CK preprocessing with proper feature extraction."""
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
    "team_mismatch": 0,
    "target_not_in_cand": 0,
    "cand_count_outlier": 0,
    "no_receiver": 0,
    "kept": 0,
}

AUDIT_RECORDS: Dict[str, List[Dict[str, Any]]] = {
    "train": [],
    "val": [],
    "test": [],
}

AUDIT_DROPPED: List[Dict[str, Any]] = []


def _make_audit_entry(
    match_id: str,
    frame: int,
    kicker_idx: Optional[int],
    receiver_idx: Optional[int],
    team_ids: np.ndarray,
    candidate_count: Optional[int] = None,
    dropped_reason: Optional[str] = None,
) -> Dict[str, Any]:
    kicker_team = None
    if kicker_idx is not None and 0 <= kicker_idx < len(team_ids):
        kicker_team = int(team_ids[kicker_idx])

    target_team = None
    if receiver_idx is not None and 0 <= receiver_idx < len(team_ids):
        target_team = int(team_ids[receiver_idx])

    entry: Dict[str, Any] = {
        "match_id": match_id,
        "frame": int(frame) if frame is not None else None,
        "kicker_idx": int(kicker_idx) if kicker_idx is not None else None,
        "target_idx": int(receiver_idx) if receiver_idx is not None else None,
        "kicker_team": kicker_team,
        "target_team": target_team,
        "candidate_count": int(candidate_count) if candidate_count is not None else None,
        "dropped_reason": dropped_reason,
    }

    for key, value in list(entry.items()):
        if isinstance(value, np.generic):
            entry[key] = value.item()

    return entry


PREPROCESS_VERSION = "ck_improved_v2"


def load_match_data(match_dir: Path):
    play_df = pd.read_csv(match_dir / "play.csv", encoding='utf-8')
    tracking_df = pd.read_csv(match_dir / "tracking.csv", encoding='utf-8')
    players_df = pd.read_csv(match_dir / "players.csv", encoding='utf-8')
    
    # Load player profile data
    profile_path = match_dir.parent.parent / "出場選手プロフィール情報.xlsx"
    if profile_path.exists():
        try:
            profile_df = pd.read_excel(profile_path)
        except:
            profile_df = pd.DataFrame()
    else:
        profile_df = pd.DataFrame()
    
    return play_df, tracking_df, players_df, profile_df


def extract_frame_data(tracking_df, frame, prev_frame, reference_players=None):
    """Extract player positions and velocities from tracking data.
    
    Handles missing nodes by zero-padding positions/velocities and preserving static info.
    """
    frame_data = tracking_df[tracking_df['Frame'] == frame].copy()
    if len(frame_data) == 0:
        return None
    
    # Separate ball and players (ball has SysTarget=0 or 7)
    ball_data = frame_data[(frame_data['SysTarget'] == 0) | (frame_data['SysTarget'] == 7)]
    player_data = frame_data[(frame_data['SysTarget'] != 0) & (frame_data['SysTarget'] != 7)]
    
    # Always return 22 nodes, padding with zeros if needed
    num_players = len(player_data)
    mask = np.ones(22, dtype=int)  # 1=valid, 0=missing
    
    if num_players < 22:
        # Create placeholder rows for missing players
        missing_count = 22 - num_players
        placeholder = pd.DataFrame({
            'HA': [1] * missing_count,
            'SysTarget': [999] * missing_count,  # Placeholder
            'X': [0] * missing_count,
            'Y': [0] * missing_count,
            'No': [0] * missing_count,
        })
        player_data = pd.concat([player_data, placeholder], ignore_index=True)
        mask[-missing_count:] = 0  # Mark missing players
    
    # Sort players by HA and position for consistency
    player_data = player_data.sort_values(['HA', 'Y', 'X'])
    
    # Extract positions (convert from cm to meters, then normalize)
    positions = player_data[['X', 'Y']].values / 100.0  # cm to m
    
    # Zero-pad positions for missing players
    positions[mask == 0] = 0.0
    
    # Calculate velocities from previous frame
    velocities = np.zeros((22, 2))
    if prev_frame is not None:
        prev_frame_data = tracking_df[tracking_df['Frame'] == prev_frame]
        if len(prev_frame_data) > 0:
            prev_ball_data = prev_frame_data[(prev_frame_data['SysTarget'] == 0) | (prev_frame_data['SysTarget'] == 7)]
            prev_player_data = prev_frame_data[(prev_frame_data['SysTarget'] != 0) & (prev_frame_data['SysTarget'] != 7)]
            
            if len(prev_player_data) > 0:
                # Handle missing players in previous frame
                missing_prev = 0
                if len(prev_player_data) < 22:
                    missing_prev = 22 - len(prev_player_data)
                    prev_placeholder = pd.DataFrame({
                        'HA': [1] * missing_prev,
                        'SysTarget': [999] * missing_prev,
                        'X': [0] * missing_prev,
                        'Y': [0] * missing_prev,
                        'No': [0] * missing_prev,
                    })
                    prev_player_data = pd.concat([prev_player_data, prev_placeholder], ignore_index=True)
                
                # Ensure we have 22 players
                if len(prev_player_data) < 22:
                    additional = 22 - len(prev_player_data)
                    prev_placeholder2 = pd.DataFrame({
                        'HA': [1] * additional,
                        'SysTarget': [999] * additional,
                        'X': [0] * additional,
                        'Y': [0] * additional,
                        'No': [0] * additional,
                    })
                    prev_player_data = pd.concat([prev_player_data, prev_placeholder2], ignore_index=True)
                
                prev_player_data = prev_player_data.sort_values(['HA', 'Y', 'X'])
                prev_positions = prev_player_data[['X', 'Y']].values[:22] / 100.0
                
                # Calculate velocities (only for valid players)
                valid_mask = (mask == 1) & (np.any(prev_positions != 0, axis=1))
                if valid_mask.sum() > 0:
                    velocities[valid_mask] = positions[valid_mask] - prev_positions[valid_mask]
    
    # Extract team IDs (HA: 1=Home->0, 2=Away->1)
    team_ids = (player_data['HA'].values - 1).astype(int)
    team_ids = np.clip(team_ids, 0, 1)
    # Preserve team info for missing players from reference if available
    if reference_players is not None and num_players < 22:
        # Try to match missing players from reference
        pass  # TODO: Implement matching logic
    
    # Ball position and ownership (closest player within 2m)
    has_ball = np.zeros(22)
    ball_position = np.array([0, 0])
    
    if len(ball_data) > 0:
        ball_pos = ball_data[['X', 'Y']].values[0] / 100.0
        ball_position = ball_pos
        # Only consider valid players for ball ownership
        if mask.sum() > 0:
            distances = np.full(22, np.inf)
            distances[mask == 1] = np.linalg.norm(positions[mask == 1] - ball_pos, axis=1)
            closest_idx = np.argmin(distances)
            if distances[closest_idx] < 2.0 and mask[closest_idx] == 1:
                has_ball[closest_idx] = 1
    
    # Normalize positions to [0, 1]
    x_normalized = (positions[:, 0] + 52.5) / 105.0
    y_normalized = (positions[:, 1] + 34.0) / 68.0
    
    # Zero-pad normalized positions for missing players
    x_normalized[mask == 0] = 0.0
    y_normalized[mask == 0] = 0.0
    
    return {
        'x': x_normalized,
        'y': y_normalized,
        'positions': positions,
        'velocities': velocities,
        'team_ids': team_ids,
        'has_ball': has_ball,
        'ball_position': ball_position,
        'mask': mask,  # Added mask
    }


def get_player_attributes(players_df, player_names, profile_df):
    """Get player attributes (height, weight) from profile data."""
    heights = np.random.uniform(1.70, 2.00, 22)  # Placeholder
    weights = np.random.uniform(60, 90, 22)  # Placeholder
    
    # TODO: Match player names from tracking to profile data
    # This would require better data matching logic
    
    return heights, weights


def get_player_id_by_name(player_name, players_df):
    """Get player ID (選手ID) from player name via players.csv.
    
    Returns the player ID (選手ID) as an integer, or None if not found.
    """
    player_info = players_df[players_df['選手名'] == player_name]
    if len(player_info) == 0:
        return None
    
    return int(player_info.iloc[0]['選手ID'])


def get_player_index_by_id(tracking_df, frame, player_id, players_df):
    """Get player index (0-21) from player ID in a specific frame.
    
    Uses player ID -> jersey number (背番号) matching via players.csv,
    then finds the player in tracking.csv by No (jersey number) across all teams.
    
    Returns the index of the player in the 22-node array (sorted by HA, Y, X), or None if not found.
    """
    # Get frame data
    frame_data = tracking_df[tracking_df['Frame'] == frame].copy()
    if len(frame_data) == 0:
        return None
    
    # Get players (exclude ball: SysTarget != 0 and != 7)
    player_data = frame_data[(frame_data['SysTarget'] != 0) & (frame_data['SysTarget'] != 7)].copy()
    
    if len(player_data) == 0:
        return None
    
    # Try to match player ID via players.csv
    player_info = players_df[players_df['選手ID'] == player_id]
    if len(player_info) == 0:
        return None
    
    # Get team info and jersey number (背番号)
    ha = player_info.iloc[0]['ホームアウェイF']  # 1=Home, 2=Away
    jersey_no = player_info.iloc[0]['背番号']
    
    # Sort players by HA, Y, X (same as extract_frame_data)
    player_data = player_data.sort_values(['HA', 'Y', 'X']).reset_index(drop=True)
    
    # Find matching player by HA (team) and No (jersey number)
    # Note: Both teams are searched, but we use HA to disambiguate if needed
    matching = player_data[(player_data['HA'] == ha) & (player_data['No'] == jersey_no)]
    
    if len(matching) == 0:
        return None
    
    # Get the index in the sorted array
    matching_index = matching.index[0]
    
    # Return the index in the 22-node array
    return matching_index


def create_samples_from_match(match_dir):
    """Create training samples from CK actions in a match."""
    global AUDIT_COUNTERS
    play_df, tracking_df, players_df, profile_df = load_match_data(match_dir)
    paired_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    dropped_audits: List[Dict[str, Optional[int]]] = []
    
    # Extract CK (Corner Kick) actions
    ck_actions = play_df[play_df['アクション名'] == 'CK'].copy()
    
    for idx, row in ck_actions.iterrows():
        AUDIT_COUNTERS["total_ck"] += 1
        frame = row['フレーム番号']
        if pd.isna(frame):
            continue
        
        frame = int(frame)
        
        # Get previous frame for velocity calculation
        prev_frame = frame - 1
        
        # Extract frame data
        frame_data = extract_frame_data(tracking_df, frame, prev_frame)
        if frame_data is None:
            continue
        
        # Get CK player info
        ck_player_name = row.get('選手名', '')
        ck_team_id = row.get('チームID', None)
        attack_no = row.get('攻撃履歴No', None)
        
        # === CHECK: Exclude CK if first touch is by opponent ===
        # Check if the next action after CK is by the opponent team
        next_idx = idx + 1
        if next_idx < len(play_df):
            next_row = play_df.iloc[next_idx]
            next_team_id = next_row.get('チームID', None)
            
            # If next action is by different team (opponent), exclude this sample
            if next_team_id is not None and ck_team_id is not None and next_team_id != ck_team_id:
                continue  # Skip this CK sample
        
        # === LABEL 1: Receiver prediction ===
        # Get the next action after CK (the receiver)
        # Store as player ID (選手ID) instead of node index
        receiver_player_id = None
        receiver_node_index = None
        if next_idx < len(play_df):
            next_row = play_df.iloc[next_idx]
            receiver_player_name = next_row.get('選手名', '')
            
            # Get receiver player ID from name
            receiver_player_id = get_player_id_by_name(receiver_player_name, players_df)
            
            # Also get node index for potential fallback
            if receiver_player_id is not None:
                receiver_node_index = get_player_index_by_id(tracking_df, frame, receiver_player_id, players_df)
        
        # If receiver not found, try to use CK kicker as fallback
        if receiver_player_id is None:
            # Fallback: use CK kicker as receiver (they have the ball)
            ck_player_id = get_player_id_by_name(ck_player_name, players_df)
            if ck_player_id is not None:
                receiver_player_id = ck_player_id
                receiver_node_index = get_player_index_by_id(tracking_df, frame, receiver_player_id, players_df)
        
        # If still not found, use node index of ball owner as last resort
        if receiver_player_id is None and receiver_node_index is None:
            receiver_node_index = int(np.argmax(frame_data['has_ball']))
            # Try to get player ID from node index if possible (this is approximate)
            # For now, just use a placeholder or skip
            receiver_player_id = -1  # Invalid ID as fallback
        
        # === LABEL 2: Shot prediction ===
        # Check if same attack (same 攻撃履歴No) has a shot by same team
        shot_occurred = 0
        if attack_no is not None and not pd.isna(attack_no):
            # Get all actions in the same attack
            same_attack = play_df[play_df['攻撃履歴No'] == attack_no]
            # Check for shots by the same team
            shots = same_attack[same_attack['アクション名'] == 'シュート']
            if len(shots) > 0:
                # Check if any shot is by the same team as CK
                for _, shot_row in shots.iterrows():
                    shot_team_id = shot_row.get('チームID', None)
                    if shot_team_id == ck_team_id:
                        shot_occurred = 1
                        break
        
        # Compute kicker index/team from frame
        team_ids = frame_data['team_ids']  # 0 for HA=1, 1 for HA=2
        has_ball = frame_data['has_ball']
        kicker_idx = int(np.argmax(has_ball)) if has_ball.sum() > 0 else 0

        # Determine attacking team by CK kicker's HA (ホームアウェイF)
        attacking_ha = None
        if ck_player_name:
            pinfo = players_df[players_df['選手名'] == ck_player_name]
            if len(pinfo) > 0 and 'ホームアウェイF' in pinfo.columns:
                try:
                    attacking_ha = int(pinfo.iloc[0]['ホームアウェイF'])  # 1 or 2
                except Exception:
                    attacking_ha = None
        # Build team_flag: 0 for attacking team, 1 for defending team
        if attacking_ha in (1, 2):
            attacking_team_id = attacking_ha - 1  # 0 or 1
            team_flag = np.where(team_ids == attacking_team_id, 0, 1)
        else:
            attacking_team_id = int(team_ids[kicker_idx])
            team_flag = np.where(team_ids == attacking_team_id, 0, 1)

        # === Add relative features to increase feature variance ===
        # Get kicker position (ball owner or CK player)
        positions = frame_data['positions']  # [22, 2] in meters
        kicker_pos = positions[kicker_idx] if frame_data['mask'][kicker_idx] == 1 else np.array([0.0, 0.0])
        
        # Goal center position (assume attacking towards positive x, goal at x=52.5)
        # For corner kicks, goal position depends on which corner
        # Simplified: assume goal center at (52.5, 0) for right side, (-52.5, 0) for left side
        # Actually, for corner kicks, goal is at the corner's adjacent side
        # For simplicity, use (52.5, 0) as goal center (can be refined later)
        goal_center = np.array([52.5, 0.0])  # Goal center in meters
        
        # Relative features to kicker
        dx_to_kicker = positions[:, 0] - kicker_pos[0]  # [22]
        dy_to_kicker = positions[:, 1] - kicker_pos[1]  # [22]
        dist_to_kicker = np.sqrt(dx_to_kicker**2 + dy_to_kicker**2)  # [22]
        angle_to_kicker = np.arctan2(dy_to_kicker, dx_to_kicker)  # [22]
        
        # Relative features to goal center
        dx_to_goal = positions[:, 0] - goal_center[0]  # [22]
        dy_to_goal = positions[:, 1] - goal_center[1]  # [22]
        dist_to_goal = np.sqrt(dx_to_goal**2 + dy_to_goal**2)  # [22]
        angle_to_goal = np.arctan2(dy_to_goal, dx_to_goal)  # [22]
        
        # Normalize relative features (distances: divide by field diagonal, angles: keep as is)
        field_diagonal = np.sqrt(105.0**2 + 68.0**2)  # ~125.5m
        dist_to_kicker_norm = dist_to_kicker / field_diagonal
        dist_to_goal_norm = dist_to_goal / field_diagonal
        
        # Zero-pad for missing players
        mask = frame_data.get('mask', np.ones(22, dtype=int))
        dx_to_kicker[mask == 0] = 0.0
        dy_to_kicker[mask == 0] = 0.0
        dist_to_kicker_norm[mask == 0] = 0.0
        angle_to_kicker[mask == 0] = 0.0
        dx_to_goal[mask == 0] = 0.0
        dy_to_goal[mask == 0] = 0.0
        dist_to_goal_norm[mask == 0] = 0.0
        angle_to_goal[mask == 0] = 0.0
        
        # Guard: receiver index must exist
        if receiver_node_index is None:
            AUDIT_COUNTERS["no_receiver"] += 1
            dropped_audits.append(
                _make_audit_entry(match_dir.name, frame, kicker_idx, receiver_node_index, team_ids, dropped_reason="missing_receiver")
            )
            continue

        receiver_team_raw = int(team_ids[receiver_node_index])
        if receiver_team_raw != attacking_team_id:
            AUDIT_COUNTERS["team_mismatch"] += 1
            dropped_audits.append(
                _make_audit_entry(
                    match_dir.name,
                    frame,
                    kicker_idx,
                    receiver_node_index,
                    team_ids,
                    dropped_reason="team_mismatch",
                )
            )
            continue

        node_indices = np.arange(len(team_ids))
        cand_mask_bool = (
            (team_ids == attacking_team_id)
            & (node_indices != kicker_idx)
            & (mask == 1)
        )

        candidate_ids = np.where(cand_mask_bool)[0]
        candidate_count = int(candidate_ids.size)

        if candidate_count == 0:
            AUDIT_COUNTERS["cand_count_outlier"] += 1
            dropped_audits.append(
                _make_audit_entry(
                    match_dir.name,
                    frame,
                    kicker_idx,
                    receiver_node_index,
                    team_ids,
                    candidate_count=0,
                    dropped_reason="cand_count_zero",
                )
            )
            continue

        if receiver_node_index not in candidate_ids:
            AUDIT_COUNTERS["target_not_in_cand"] += 1
            dropped_audits.append(
                _make_audit_entry(
                    match_dir.name,
                    frame,
                    kicker_idx,
                    receiver_node_index,
                    team_ids,
                    candidate_count=candidate_count,
                    dropped_reason="target_not_in_cand",
                )
            )
            continue

        if candidate_count < 3 or candidate_count > 18:
            AUDIT_COUNTERS["cand_count_outlier"] += 1
            dropped_audits.append(
                _make_audit_entry(
                    match_dir.name,
                    frame,
                    kicker_idx,
                    receiver_node_index,
                    team_ids,
                    candidate_count=candidate_count,
                    dropped_reason="cand_count_outlier",
                )
            )
            continue

        # Ensure target is marked in candidate mask (should already be true)
        cand_mask_bool[receiver_node_index] = True
 
        # Create sample with relative features
        sample = {
            'x': frame_data['x'],
            'y': frame_data['y'],
            'vx': frame_data['velocities'][:, 0],
            'vy': frame_data['velocities'][:, 1],
            'height': np.random.uniform(1.70, 2.00, 22),  # Placeholder
            'weight': np.random.uniform(60, 90, 22),  # Placeholder
            'ball': frame_data['has_ball'],
            'team': team_flag,
            # Relative features to kicker
            'dx_to_kicker': dx_to_kicker / 105.0,  # Normalize by field length
            'dy_to_kicker': dy_to_kicker / 68.0,   # Normalize by field width
            'dist_to_kicker': dist_to_kicker_norm,
            'angle_to_kicker': angle_to_kicker / np.pi,  # Normalize to [-1, 1]
            # Relative features to goal
            'dx_to_goal': dx_to_goal / 105.0,
            'dy_to_goal': dy_to_goal / 68.0,
            'dist_to_goal': dist_to_goal_norm,
            'angle_to_goal': angle_to_goal / np.pi,
            'receiver_id': receiver_player_id,  # Label for receiver prediction (player ID, not node index)
            'receiver_node_index': receiver_node_index,  # Node index for debugging/reference
            'shot_occurred': shot_occurred,  # Label for shot prediction
            'match_id': str(int(row['試合ID'])),
            'frame': frame,
            'ck_player': ck_player_name,
            'attack_no': int(attack_no) if attack_no is not None and not pd.isna(attack_no) else None,
            'mask': mask.astype(np.float32),
            'cand_mask': cand_mask_bool.astype(bool),
            'candidate_ids': candidate_ids.tolist(),
            'kicker_idx': int(kicker_idx),
            'kicker_team': int(attacking_team_id),
            'target_idx': int(receiver_node_index),
        }

        audit_entry = _make_audit_entry(
            match_dir.name,
            frame,
            kicker_idx,
            receiver_node_index,
            team_ids,
            candidate_count=candidate_count,
            dropped_reason=None,
        )

        AUDIT_COUNTERS["kept"] += 1
        paired_samples.append((sample, audit_entry))
    
    return paired_samples, dropped_audits


def process_all_matches(data_dir, output_dir, task="receiver", split_ratio=(0.7, 0.15, 0.15), force: bool = False):
    """Process all matches in SoccerData directory."""
    # Reset audit state
    for key in AUDIT_COUNTERS:
        AUDIT_COUNTERS[key] = 0
    for key in AUDIT_RECORDS:
        AUDIT_RECORDS[key].clear()
    AUDIT_DROPPED.clear()

    match_dirs = []
    for year_dir in ['2023_data', '2024_data']:
        year_path = data_dir / year_dir
        if year_path.exists():
            for match_dir in year_path.iterdir():
                if match_dir.is_dir() and (match_dir / "play.csv").exists():
                    match_dirs.append(match_dir)
    
    print(f"Found {len(match_dirs)} matches to process")
    
    all_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    failed_frames = 0
    
    for match_dir in tqdm(match_dirs, desc="Processing matches"):
        try:
            pairs, dropped_audits = create_samples_from_match(match_dir)
            all_pairs.extend(pairs)
            AUDIT_DROPPED.extend(dropped_audits)
        except Exception as e:
            failed_frames += 1
            print(f"Error processing {match_dir}: {e}")
            continue
    
    print(f"Created {len(all_pairs)} samples (failed: {failed_frames} frames)")
    print(
        "[AUDIT] totals: total_ck={total} kept={kept} team_mismatch={tm} target_not_in_cand={tnc} "
        "cand_count_outlier={cco} no_receiver={nr}".format(
            total=AUDIT_COUNTERS["total_ck"],
            kept=AUDIT_COUNTERS["kept"],
            tm=AUDIT_COUNTERS["team_mismatch"],
            tnc=AUDIT_COUNTERS["target_not_in_cand"],
            cco=AUDIT_COUNTERS["cand_count_outlier"],
            nr=AUDIT_COUNTERS["no_receiver"],
        )
    )
    
    if len(all_pairs) == 0:
        print("No samples created")
        return
    
    # Split and save
    np.random.seed(42)
    np.random.shuffle(all_pairs)
    
    n_total = len(all_pairs)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    
    splits = {
        'train': all_pairs[:n_train],
        'val': all_pairs[n_train:n_train+n_val],
        'test': all_pairs[n_train+n_val:],
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, pair_list in splits.items():
        split_dir = output_dir / f"{task}_{split_name}"
        if force and split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        samples_only = [sample for sample, _ in pair_list]
        audits_only = []
        for _, audit in pair_list:
            audit_entry = dict(audit)
            audit_entry["split"] = split_name
            audits_only.append(audit_entry)
        AUDIT_RECORDS[split_name].extend(audits_only)
        
        payload = {
            "preprocess_version": PREPROCESS_VERSION,
            "samples": samples_only,
        }
        with open(split_dir / "data.pickle", 'wb') as f:
            pickle.dump(payload, f)
        
        metadata = {
            'task': task,
            'split': split_name,
            'num_samples': len(samples_only),
            'preprocess_version': PREPROCESS_VERSION,
        }
        with open(split_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(samples_only)} {split_name} samples to {split_dir}")

    # Write audit logs
    def _write_jsonl(path: Path, entries: List[Dict[str, Optional[int]]]) -> None:
        if not entries:
            return
        with open(path, 'w', encoding='utf-8') as f:
            for item in entries:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    for split_name, entries in AUDIT_RECORDS.items():
        _write_jsonl(output_dir / f"audit_{split_name}.jsonl", entries)

    if AUDIT_DROPPED:
        _write_jsonl(output_dir / "audit_dropped.jsonl", AUDIT_DROPPED)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CK data for receiver task")
    parser.add_argument("--data_dir", type=Path, default=Path("SoccerData"))
    parser.add_argument("--save_dir", type=Path, default=Path("data/processed_ck"))
    parser.add_argument("--task", type=str, default="receiver")
    parser.add_argument("--force", action="store_true", help="Overwrite existing processed data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_matches(
        data_dir=args.data_dir,
        output_dir=args.save_dir,
        task=args.task,
        split_ratio=(0.7, 0.15, 0.15),
        force=args.force,
    )

