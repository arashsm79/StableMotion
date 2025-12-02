#!/usr/bin/env python3
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import argparse

def create_labeled_video(video_path, dlc_h5_path, output_path="output.mp4", keypoint_size=5):
    df = pd.read_hdf(dlc_h5_path)
    
    # Extract scorer and bodyparts from column structure
    scorer = df.columns.levels[0][0]
    bodyparts = df.columns.get_level_values(1).unique()
    
    # Create unique colors for each bodypart
    np.random.seed(42)  # For reproducibility
    colors = [(int(np.random.randint(0, 256)), int(np.random.randint(0, 256)), int(np.random.randint(0, 256))) 
              for _ in bodyparts]
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(df))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract coordinates for current frame
        coords = []
        low_likelihood = False
        
        for bodypart in bodyparts:
            x = df.loc[frame_idx, (scorer, bodypart, 'x')]
            y = df.loc[frame_idx, (scorer, bodypart, 'y')]
            likelihood = df.loc[frame_idx, (scorer, bodypart, 'likelihood')]
            
            if likelihood < 0.6:
                low_likelihood = True
                
            if not (np.isnan(x) or np.isnan(y)):
                coords.append((int(x), int(y)))
            else:
                coords.append(None)
        
        # Draw keypoints
        for i, (bodypart, coord) in enumerate(zip(bodyparts, coords)):
            if coord is not None:
                cv2.circle(frame, coord, keypoint_size, colors[i], -1, cv2.LINE_AA)
        
        # Draw bounding box
        valid_coords = [c for c in coords if c is not None]
        if valid_coords:
            xs, ys = zip(*valid_coords)
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Add frame number and label
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Label: {low_likelihood}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("dlc_h5_path") 
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--keypoint_size", type=int, default=3)
    args = parser.parse_args()
    
    create_labeled_video(args.video_path, args.dlc_h5_path, args.output, args.keypoint_size)
