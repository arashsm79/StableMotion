"""
Setup script for DLC StableMotion pipeline.

This script helps set up the complete pipeline for using StableMotion with DLC data:
1. Creates dataset splits
2. Computes normalization statistics
3. Sets up directory structure
4. Provides example usage

Usage:
python setup_dlc_pipeline.py --data_dir path/to/dlc/files --output_dir dataset/dlc_motion
"""

import os
import argparse
import json
from typing import Dict, List
import torch

# Imports will be resolved at runtime
import sys
sys.path.append('.')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup DLC StableMotion pipeline')
    
    parser.add_argument('--data_dir', default="dlc_data/dlc_data_raw", type=str,
                       help='Directory containing DLC CSV/pickle files')
    parser.add_argument('--split_dir', default="dlc_data/dlc_data_splits", type=str,
                       help='Output directory for processed data and metadata')
    parser.add_argument('--meta_dir', default="dlc_data/dlc_data_meta", type=str,
                       help='Output directory for metadata')
    parser.add_argument('--likelihood_threshold', default=0.6, type=float,
                       help='Minimum likelihood threshold for quality detection')
    parser.add_argument('--fps', default=25, type=int,
                       help='Video frame rate')
    parser.add_argument('--ext', default='.h5', type=str,
                       help='File extension for DLC files (.csv or .pickle)')
    parser.add_argument('--train_ratio', default=0.95, type=float,
                       help='Fraction of data for training')
    parser.add_argument('--val_ratio', default=0.01, type=float,
                       help='Fraction of data for validation')
    parser.add_argument('--max_files_for_stats', default=1000, type=int,
                       help='Maximum files to use for computing statistics')
    
    return parser.parse_args()



def create_dataset_splits(data_dir: str, ext: str, train_ratio: float, 
                         val_ratio: float, output_dir: str) -> Dict[str, List[str]]:
    """Create and save dataset splits."""
    from data_loaders.globsmpl_dataset import create_dlc_dataset_splits
    
    print("Creating dataset splits...")
    file_pattern = f"*{ext}"
    splits = create_dlc_dataset_splits(data_dir, file_pattern, train_ratio, val_ratio)
    
    print(f"Dataset splits:")
    for split_name, files in splits.items():
        print(f"  {split_name}: {len(files)} files")
        
        # Save split files
        split_file = os.path.join(output_dir, f"{split_name}.txt")
        with open(split_file, 'w') as f:
            for file_id in files:
                f.write(f"{file_id}\n")
    
    return splits


def compute_and_save_stats(data_dir: str, splits: Dict[str, List[str]], meta_dir: str, args) -> Dict[str, torch.Tensor]:
    """Compute and save normalization statistics."""
    from data_loaders.globsmpl_dataset import create_dlc_dataset_splits, AMASSMotionLoader
    from utils.normalizer import Normalizer
    
    print("Computing normalization statistics...")
    
    # Create motion loader without normalization
    motion_loader = AMASSMotionLoader(
        base_dir=data_dir,
        fps=args.fps,
        ext=args.ext,
        mode='train',
        disable=True  # Don't normalize during stats computation
    )
    
    # Use subset of training files for efficiency
    train_files = splits['train'][:args.max_files_for_stats]
    
    # Compute statistics
    mean, std = compute_dlc_normalization_stats(motion_loader, train_files)
    
    # Save statistics
    normalizer = Normalizer(base_dir=meta_dir, disable=True)
    normalizer.save(mean, std)
    
    print(f"Saved normalization stats to {meta_dir}")
    print(f"Feature dimensions: {len(mean)}")
    
    return {'mean': mean, 'std': std}



def main():
    args = parse_args()
    
    print(f"Setting up DLC StableMotion pipeline...")
    print(f"Data directory: {args.data_dir}")
    
    # Create dataset splits
    splits = create_dataset_splits(
        args.data_dir, args.ext, args.train_ratio, args.val_ratio, args.split_dir
    )
    
    # Compute normalization statistics
    # stats = compute_and_save_stats(args.data_dir, splits, args.meta_dir, args)
    # print(f'Normalization stats: mean {stats["mean"]}, std {stats["std"]}')


if __name__ == "__main__":
    main()
