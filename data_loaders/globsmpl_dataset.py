import torch
import numpy as np
import os
import random
import codecs as cs
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

from data_loaders.smpl_collate import collate_motion
from data_loaders.amasstools.globsmplrifke_feats import (
    smpldata_to_alignglobsmplrifkefeats,
)
from smplx.lbs import batch_rigid_transform
import einops
from data_loaders.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
)
from utils.normalizer import Normalizer


class AMASSMotionLoader:
    """
    Loads a SMPL motion clip, crops a segment, computes joints (if missing),
    converts to aligned Global SMPL RIFKE features, appends label channel,
    and (optionally) normalizes.

    Args:
        base_dir (str): Root folder of AMASS .npz files.
        fps (int): Frames per second of stored motions (used for crop lengths).
        disable (bool): If True, skip normalization/label-channel adjustments.
        ext (str): File extension for motion files (default: ".npz").
        umin_s (float): Min crop length (seconds).
        umax_s (float): Max crop length (seconds).
        mode (str): Split name, e.g., "train" or "test".
        **kwargs:
            normalizer_dir (str): Directory for Normalizer stats (required if not disable).
    """

    def __init__(
        self, base_dir, fps=25, disable: bool = False, ext=".h5", umin_s=5.0, umax_s=5.0, mode="train", encoding_type="normalized_angles", **kwargs
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.ext = ext
        self.umin = int(self.fps * umin_s)
        assert self.umin > 0
        self.umax = int(self.fps * umax_s)
        self.mode = mode
        self.encoding_type = encoding_type

        if not disable:
            normalizer_dir = kwargs.get("normalizer_dir", None)
            assert normalizer_dir is not None, "Please provide normalizer_dir when not disable"
            self.motion_normalizer = Normalizer(base_dir=normalizer_dir)
            self.motion_normalizer.add_label_channel()

    def __call__(self, path):
        """
        Load and process a single motion clip.

        Args:
            path (str): Relative path (without extension). May include ",start,duration" for slicing.

        Returns:
            dict: {"x": Tensor[T, F], "length": int}
        """
        data_path = path
        motion_path = os.path.join(self.base_dir, data_path + self.ext)
        dlc_data = pd.read_hdf(motion_path) # a dataframe
        initial_offset = 5_000

        # Determine crop [start:start+duration]
        mlen = len(dlc_data)
        duration = random.randint(min(self.umin, mlen), min(self.umax, mlen))
        start = random.randint(initial_offset, max(mlen - duration, 0))

        poses = dlc_data.iloc[start : start + duration]

        # Convert to features
        from data_loaders.dlc_keypoint_feats import dlc_dataframe_to_features
        motion, metadata = dlc_dataframe_to_features(poses, encoding_type=self.encoding_type)
        motion = motion.to(torch.float32)

        # Store raw dataframe in metadata for visualization
        metadata['raw_df'] = poses

        # Optional normalization
        if not self.disable:
            motion = self.motion_normalizer(motion)

        return {"x": motion, "length": len(motion), "metadata": metadata}


def read_split(path, split):
    """
    Read IDs from data_loaders/splits/{split}.txt.
    """
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


class MotionDataset(Dataset):
    """
    Thin dataset wrapper around a motion loader callable.
    """

    def __init__(self, motion_loader, split: str = "train", preload: bool = False):
        self.collate_fn = collate_motion
        self.split = split
        self.keyids = read_split("data_loaders", "train")
        self.motion_loader = motion_loader
        self.is_training = "train" in split

        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        """
        Load a single example by key id.
        """
        file_path = keyid.strip(".npy")
        motion_x_dict = self.motion_loader(path=file_path)
        x = motion_x_dict["x"]
        length = motion_x_dict["length"]
        metadata = motion_x_dict['metadata']
        metadata['keyid'] = keyid
        return {"x": x, "keyid": keyid, "length": length, "metadata": metadata}


def create_dlc_dataset_splits(base_dir: str, file_pattern: str = "*.h5", 
                             train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Create train/val/test splits from DLC files.
    
    Args:
        base_dir: Directory containing DLC files
        file_pattern: Pattern to match files (e.g., "*.csv")
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test gets remainder)
        
    Returns:
        dict: {"train": [...], "val": [...], "test": [...]} file lists
    """
    import glob
    
    # Find all matching files
    pattern_path = os.path.join(base_dir, "**", file_pattern)
    all_files = glob.glob(pattern_path, recursive=True)
    
    # Convert to relative paths without extensions
    file_ids = []
    for file_path in all_files:
        rel_path = os.path.relpath(file_path, base_dir)
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        file_ids.append(rel_path_no_ext)
    
    file_ids.sort()  # For reproducibility
    
    # Create splits
    n_files = len(file_ids)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    
    train_files = file_ids[:n_train]
    val_files = file_ids[n_train:n_train + n_val]
    test_files = file_ids[n_train + n_val:]
    
    return {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }


if __name__ == "__main__":
    from utils.normalizer import Normalizer

    # Example: compute normalization stats from corrupted-canonicalized data.
    motion_loader = AMASSMotionLoader(
        base_dir="dlc_data/dlc_data_raw",
        disable=True,
        encoding_type="normalized_angles"
    )
    motion_dataset = MotionDataset(motion_loader=motion_loader, split="train")

    motion_normalizer = Normalizer(
        base_dir="dlc_data/dlc_data_meta",
        disable=True,
    )

    # Accumulate features and compute mean/std (excluding label channel).
    # Tip: you may manually adjust the scale of root features standard deviation
    data_bank = []
    for _ in range(10):
        data_bank += [x["x"] for x in tqdm(motion_dataset)]
    motionfeats = torch.cat(data_bank)
    mean_motionfeats = motionfeats.mean(0)[:-1]
    std_motionfeats = motionfeats.std(0)[:-1]

    motion_normalizer.save(mean_motionfeats, std_motionfeats)
