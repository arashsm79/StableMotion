"""
Windowed DLC motion fix-up for entire H5 files.

Processes DLC H5 files in overlapping windows, applies motion cleanup using StableMotion,
and saves the fixed data in the same format with '_fixed' suffix.
"""

import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Import StableMotion dependencies
from utils.fixseed import fixseed
from utils.parser_util import det_args
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from utils.normalizer import Normalizer
from ema_pytorch import EMA
from sample.utils import prepare_cond_fn
from data_loaders.dlc_keypoint_feats import dlc_dataframe_to_features
from data_loaders.smpl_collate import length_to_mask

# Import the detection and fix functions
from sample.fix_globsmpl import detect_labels, fix_motion
import subprocess
import sys

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True


def process_dlc_h5_windowed(
    args,
    input_h5_path: str,
    model_path: str,
    normalizer_dir: str,
    output_h5_path: str = None,
    window_length: int = 250,
    step_size: int = 200,
    encoding_type: str = 'trajectory',
    likelihood_threshold: float = 0.6,
    device: str = 'cuda',
):
    """
    Process an entire DLC H5 file in windowed manner with StableMotion cleanup.
    """
    # Validate inputs
    if not Path(input_h5_path).exists():
        raise FileNotFoundError(f"Input H5 file not found: {input_h5_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    if not Path(normalizer_dir).exists():
        raise FileNotFoundError(f"Normalizer directory not found: {normalizer_dir}")
    
    if window_length <= 0:
        raise ValueError("window_length must be positive")
        
    if step_size <= 0:
        raise ValueError("step_size must be positive")
        
    if step_size > window_length:
        print("Warning: step_size > window_length will create gaps between windows")
    
    # Setup device and seed
    dist_util.setup_dist(device)
    device = dist_util.dev()
    
    # Load DLC dataframe
    print(f"Loading DLC data from {input_h5_path}...")
    dlc_df = pd.read_hdf(input_h5_path)
    total_frames = len(dlc_df)
    print(f"Total frames in input: {total_frames}")
    
    # Generate output path if not provided
    if output_h5_path is None:
        input_path = Path(input_h5_path)
        output_h5_path = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
    
    # Load motion normalizer
    print(f"Loading normalizer from {normalizer_dir}...")
    motion_normalizer = Normalizer(base_dir=normalizer_dir)
    motion_normalizer.add_label_channel()
    
    # Load model and diffusion
    print(f"Loading model from {model_path}...")
    args.model_path = model_path
    _model, diffusion = create_model_and_diffusion(args)
    model = EMA(_model, include_online_model=False) if args.use_ema else _model
    
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # Optional guidance
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)
    if cond_fn is not None:
        cond_fn.keywords["model"] = model
    
    # Initialize output dataframe with same structure as input
    output_df = dlc_df.copy()
    overlap_counts = np.zeros(total_frames)  # Track how many times each frame was processed
    
    print(f"Processing with window_length={window_length}, step_size={step_size}")
    print(f"Number of windows: {(total_frames - window_length) // step_size + 1}")
    
    # Process in windows
    window_start = 0
    while window_start + window_length <= total_frames:
        window_end = window_start + window_length
        
        print(f"Processing window [{window_start}:{window_end}]...")
        
        # Extract window data
        window_df = dlc_df.iloc[window_start:window_end].copy()
        
        # Convert to motion features
        motion_features, metadata = dlc_dataframe_to_features(
            window_df, 
            likelihood_threshold=likelihood_threshold,
            encoding_type=encoding_type
        )
        
        # Prepare tensors for model
        motion_features = motion_features.to(torch.float32)
        
        # Apply normalization
        motion_features_norm = motion_normalizer(motion_features)
        
        # Reshape for model: [T, F] -> [1, F, T] (batch_size=1)
        input_motions = motion_features_norm.transpose(0, 1).unsqueeze(0).to(device)
        batch_size, n_features, n_frames = input_motions.shape
        
        # Create attention mask and length tensor
        length = torch.tensor([n_frames], device=device)
        attention_mask = length_to_mask(length, device=device)
        
        # Metadata for model (batch format)
        metadata_batch = [metadata]
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()
        
        with torch.no_grad():
            # Detection pass
            det_out = detect_labels(
                model=model,
                diffusion=diffusion,
                args=args,
                input_motions=input_motions,
                length=length,
                attention_mask=attention_mask,
                motion_normalizer=motion_normalizer,
                metadata=metadata_batch,
            )
            
            if args.gtdet:
                print("use gt label")
                label = torch.from_numpy(gt_labels.copy())
            else:
                label = det_out["label"]
            re_sample_det_feats = det_out["re_sample_det_feats"]
            # re_sample_det_feats = []
            # re_sample_det_feats = input_motions
            
            # Fix motion
            fix_out = fix_motion(
                model=model,
                diffusion=diffusion,
                args=args,
                input_motions=input_motions,
                length=length,
                attention_mask=attention_mask,
                motion_normalizer=motion_normalizer,
                label=label,
                re_sample_det_feats=re_sample_det_feats,
                cond_fn=cond_fn,
                metadata=metadata_batch,
            )
            
            fixed_motion_list = fix_out["fixed_motion"]
            fixed_motion_df = fixed_motion_list[0]  # Get first (and only) item from batch
        
        # Blend/average the fixed window with existing output
        # Use only the first `step_size` frames from this window
        actual_frames_to_use = min(step_size, window_length, total_frames - window_start)
        
        # Update the output dataframe
        start_idx = window_start
        
        # For the first window, use all frames up to step_size
        # For subsequent windows, use only step_size frames to avoid overlap issues
        if window_start == 0:
            frames_to_use = min(step_size, window_length)
        else:
            frames_to_use = actual_frames_to_use
        
        # Extract the portion we want to keep from fixed window
        window_portion = fixed_motion_df.iloc[:frames_to_use]
        
        # Update output dataframe
        for col in output_df.columns:
            if col in window_portion.columns:
                output_df.iloc[start_idx:start_idx + frames_to_use, output_df.columns.get_loc(col)] = window_portion[col].values
        
        overlap_counts[start_idx:start_idx + frames_to_use] += 1
        
        # Move to next window
        window_start += step_size
        
        # Break if we would exceed the data
        if window_start >= total_frames:
            break
    
    # Handle any remaining frames at the end
    if window_start < total_frames:
        remaining_frames = total_frames - window_start
        if remaining_frames > 0:
            print(f"Processing final partial window [{window_start}:{total_frames}]... \n Remaining frames: {remaining_frames}")
            
            # For the last window, process all remaining frames
            final_window_df = dlc_df.iloc[window_start:].copy()
            
            motion_features, metadata = dlc_dataframe_to_features(
                final_window_df,
                likelihood_threshold=likelihood_threshold,
                encoding_type=encoding_type
            )
            
            motion_features = motion_features.to(torch.float32)
            motion_features_norm = motion_normalizer(motion_features)
            input_motions = motion_features_norm.transpose(0, 1).unsqueeze(0).to(device)
            
            length = torch.tensor([motion_features_norm.shape[0]], device=device)
            attention_mask = torch.ones((1, motion_features_norm.shape[0]), dtype=torch.bool, device=device)
            metadata_batch = [metadata]
            
            with torch.no_grad():
                det_out = detect_labels(
                    model=model,
                    diffusion=diffusion,
                    args=args,
                    input_motions=input_motions,
                    length=length,
                    attention_mask=attention_mask,
                    motion_normalizer=motion_normalizer,
                    metadata=metadata_batch,
                )
                
                fix_out = fix_motion(
                    model=model,
                    diffusion=diffusion,
                    args=args,
                    input_motions=input_motions,
                    length=length,
                    attention_mask=attention_mask,
                    motion_normalizer=motion_normalizer,
                    label=det_out["label"],
                    re_sample_det_feats=det_out["re_sample_det_feats"],
                    cond_fn=cond_fn,
                    metadata=metadata_batch,
                )
                
                final_fixed_df = fix_out["fixed_motion"][0]
            
            # Update final frames
            for col in output_df.columns:
                if col in final_fixed_df.columns:
                    output_df.iloc[window_start:, output_df.columns.get_loc(col)] = final_fixed_df[col].values
    
    # Save fixed dataframe
    print(f"Saving fixed data to {output_h5_path}...")
    output_df.to_hdf(output_h5_path, key='df', mode='w', format='table')
    
    print("Motion fix-up complete!")
    print(f"Input: {input_h5_path}")
    print(f"Output: {output_h5_path}")
    print(f"Original frames: {total_frames}")
    print(f"Frames processed multiple times: {np.sum(overlap_counts > 1)}")
    
    return output_h5_path


def main():
    """CLI interface for windowed DLC motion fix-up."""
    args = det_args()
    fixseed(args.seed)

    args.step_size = 200
    args.window_length = 250
    args.input_h5 = "/home/arashsm79/playground/StableMotion/dlc_data/dlc_video/LEWES_EiJ___animal_HDP-007192___arena_1___exp_stage_hab___age_period_adult_20NOVDLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30.h5"
    args.output_h5 = "/home/arashsm79/playground/StableMotion/dlc_data/dlc_video/LEWES_EiJ___animal_HDP-007192___arena_1___exp_stage_hab___age_period_adult_20NOVDLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30_fix.h5"
    args.encoding_type = "trajectory"
    
    process_dlc_h5_windowed(
        args,
        input_h5_path=args.input_h5,
        model_path=args.model_path,
        normalizer_dir=args.normalizer_dir,
        output_h5_path=args.output_h5,
        window_length=args.window_length,
        step_size=args.step_size,
        encoding_type=args.encoding_type,
        likelihood_threshold=0.6,
        device=args.device,
    )

    # Run the create_labeled_video script with the processed H5 file
    video_path = "/home/arashsm79/playground/StableMotion/dlc_data/dlc_video/LEWES_EiJ___animal_HDP-007192___arena_1___exp_stage_hab___age_period_adult_20NOV.mp4"
    processed_h5_path = args.output_h5

    cmd = [
        "uv", "run", "src/stablemotion/create_labeled_video.py",
        video_path,
        processed_h5_path,
        "--output", f'{time.strftime("%Y%m%d_%H%M%S")}_fixed.mp4',
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Video creation completed successfully!")
    print(result.stdout)


if __name__ == "__main__":
    main()
