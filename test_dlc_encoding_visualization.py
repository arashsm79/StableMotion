#!/usr/bin/env python3
"""
Test script for DLC feature encoding/decoding with visualization.
Creates a side-by-side video showing: original keypoints, normalized features, reconstructed keypoints.
"""

import torch
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from data_loaders.globsmpl_dataset import AMASSMotionLoader, MotionDataset
from data_loaders.dlc_keypoint_feats import features_to_dlc_keypoints, ungroup_dlc_features, ungroup_normalized_angle_features

def get_data_loader(encoding_type='normalized_angles', batch_size=1):
    """Create a simple data loader for testing."""
    motion_loader = AMASSMotionLoader(
        base_dir="dlc_data/dlc_data_raw", 
        disable=True,  # No normalization 
        umin_s=10.0, 
        umax_s=10.0,
        encoding_type=encoding_type
    )
    dataset = MotionDataset(motion_loader=motion_loader, split="train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

def extract_normalized_coords(features, metadata, encoding_type='trajectory'):
    """Extract normalized coordinates from features for plotting."""
    if encoding_type == 'normalized_angles':
        coords_normalized, _, _, _, _ = ungroup_normalized_angle_features(features, metadata)
        # Reshape to (T, N_animals, N_bodyparts, 2)
        T = features.shape[0]
        n_animals, n_bodyparts = metadata['n_animals'], metadata['n_bodyparts']
        coords_normalized = coords_normalized.view(T, n_animals, n_bodyparts, 2)
    elif encoding_type == 'trajectory':
        # Trajectory encoding
        trajectory, _, coords_normalized, _, _, _, _ = ungroup_dlc_features(features, metadata)
        T = features.shape[0]
        n_animals, n_bodyparts = metadata['n_animals'], metadata['n_bodyparts'] 
        coords_normalized = coords_normalized.view(T, n_animals, n_bodyparts, 2)
        coords_normalized = coords_normalized + trajectory.unsqueeze(1).unsqueeze(1)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    return coords_normalized

def get_keypoint_coords(df, animal_idx=0):
    """Extract x,y coordinates from DLC dataframe."""
    coords = []
    bodyparts = df.columns.get_level_values('bodyparts').unique()
    scorer = df.columns.get_level_values('scorer')[0]
    
    for _, row in df.iterrows():
        frame_coords = []
        for bp in bodyparts:
            x = row[scorer, bp, 'x']
            y = row[scorer, bp, 'y']
            frame_coords.append([x, y])
        coords.append(frame_coords)
    return np.array(coords), bodyparts

def create_visualization_video(original_df, features, metadata, encoding_type, height=400, output_path="test_output.mp4"):
    """Create side-by-side video: original | normalized features | reconstructed."""
    
    # Get data
    orig_coords, bodyparts = get_keypoint_coords(original_df)
    feat_coords = extract_normalized_coords(features, metadata, encoding_type)
    recon_df = features_to_dlc_keypoints(features, metadata, encoding_type)
    recon_coords, _ = get_keypoint_coords(recon_df)
    
    T, n_bodyparts = orig_coords.shape[:2]
    
    # Video settings
    width = height * 3  # 3 panels side by side
    panel_width = width // 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
    
    # Colors for bodyparts
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), 
              (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)]
    bp_colors = {i: colors[i % len(colors)] for i in range(n_bodyparts)}
    
    orig_norm = orig_coords
    recon_norm = recon_coords
    
    norm_coords_np = feat_coords[:, 0, :, :].numpy()  # Take all frames, first animal, all bodyparts
    if encoding_type == 'normalized_angles':
        # For normalized features, center them in the middle panel
        feat_norm = np.array(norm_coords_np) * (panel_width // 4) + [panel_width // 2, height // 2]  # Scale and center
    else:
        feat_norm = norm_coords_np
    
    for frame_idx in range(T):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Panel 1: Original keypoints
        for bp_idx in range(n_bodyparts):
            x, y = orig_norm[frame_idx, bp_idx]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x), int(y)), 3, bp_colors[bp_idx], -1)
        
        # Panel 2: Normalized features (centered)
        for bp_idx in range(n_bodyparts):
            x, y = feat_norm[frame_idx, bp_idx]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x + panel_width), int(y)), 3, bp_colors[bp_idx], -1)
        
        # Panel 3: Reconstructed keypoints  
        for bp_idx in range(n_bodyparts):
            x, y = recon_norm[frame_idx, bp_idx]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x + 2 * panel_width), int(y)), 3, bp_colors[bp_idx], -1)
        
        # Add panel dividers and labels
        cv2.line(frame, (panel_width, 0), (panel_width, height), (0, 0, 0), 2)
        cv2.line(frame, (2 * panel_width, 0), (2 * panel_width, height), (0, 0, 0), 2)
        cv2.putText(frame, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f'Features ({encoding_type})', (panel_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, 'Reconstructed', (2 * panel_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f'Frame: {frame_idx}', (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")

def main():
    """Main test function."""
    print("Loading data...")
    data_loader = get_data_loader()
    
    # Get a single batch
    batch = next(iter(data_loader))
    for i in range(len(batch['x'])):
        features = batch['x'][0]  # Remove batch dimension
        features = batch['x'][0].transpose(0, 1)  # Remove batch dimension and transpose back to [T, F]
        metadata = batch['metadata'][0]
        original_df = metadata['raw_df']
        
        print(f"Features shape: {features.shape}")
        print(f"Original dataframe shape: {original_df.shape}")
        print(f"Encoding type: {metadata.get('encoding_type', 'trajectory')}")
        
        encoding_type = metadata.get('encoding_type', 'trajectory')
        
        # Create visualization for current encoding
        os.makedirs("test_outputs", exist_ok=True)
        output_path = f"test_outputs/dlc_test_{encoding_type}_{i}.mp4"
        create_visualization_video(original_df, features, metadata, encoding_type, height=800, output_path=output_path)
    print("done")

if __name__ == "__main__":
    main()
