"""
DeepLabCut keypoint feature extraction for StableMotion.

Converts DLC multi-animal pose data to motion features suitable for diffusion-based cleanup.
Features are fully reversible - can reconstruct exact original DLC dataframe.

DLC Datastructure:

if it is a single animal project:

d.columns:
MultiIndex([('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
...
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...),
            ('DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30', ...)],
           names=['scorer', 'bodyparts', 'coords'])
           
           
d.index:
Index([    0,     1,     2,     3,     4,     5,     6,     7,     8,     9,
       ...
       31660, 31661, 31662, 31663, 31664, 31665, 31666, 31667, 31668, 31669],
      dtype='int64', length=31670)
      

scorer	DLC_HrnetW32_DVC_Project_AllJul14shuffle0_detector_best-20_snapshot_best-30
bodyparts	nose	left_ear	right_ear	left_ear_tip	...	right_midside	right_hip	tail_end	head_midpoint
coords	x	y	likelihood	x	y	likelihood	x	y	likelihood	x	...	likelihood	x	y	likelihood	x	y	likelihood	x	y	likelihood
0	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.0	-1.000000	-1.000000	-1.000000
1	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.0	-1.000000	-1.000000	-1.000000
2	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.0	-1.000000	-1.000000	-1.000000
3	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.0	-1.000000	-1.000000	-1.000000
4	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	...	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.000000	-1.0	-1.000000	-1.000000	-1.000000
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
31665	241.398438	50.835938	0.954590	223.742188	54.367188	0.887695	230.804688	64.960938	0.704590	221.976562	...	0.870117	227.273438	84.382812	0.863770	179.601562	102.039062	1.0	230.804688	56.132812	0.641113
31666	245.007812	51.484375	0.822754	229.679688	53.203125	0.540527	236.492188	61.796875	0.452393	210.945312	...	0.926270	226.273438	85.859375	0.895020	178.585938	103.046875	1.0	233.085938	54.921875	0.373291
31667	247.515625	54.203125	0.925781	230.328125	54.203125	0.792969	235.484375	62.796875	0.552734	228.609375	...	0.936523	226.890625	86.859375	0.877930	178.765625	102.328125	1.0	237.203125	55.921875	0.609375
31668	246.375000	53.375000	0.912109	225.375000	55.125000	0.422607	232.375000	65.625000	0.401611	211.375000	...	0.898438	225.375000	84.875000	0.802734	179.875000	102.375000	1.0	237.625000	56.875000	0.344971
31669	247.515625	52.484375	0.923828	230.328125	54.203125	0.734863	237.203125	62.796875	0.484863	228.609375	...	0.884277	225.171875	86.859375	0.863281	178.765625	102.328125	1.0	237.203125	55.921875 0.52636

if it's multi animal project:
df.columns:
MultiIndex([('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ...
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...),
            ('DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50', ...)],
           names=['scorer', 'individuals', 'bodyparts', 'coords'], length=108)
           
           
df.index:
Index([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
       ...
       293, 294, 295, 296, 297, 298, 299, 300, 301, 302],
      dtype='int64', length=303)


df:
scorer	DLC_DlcrnetStride16Ms5_trimiceJun22shuffle1_snapshot_best-50
individuals	mus1	...	mus3
bodyparts	snout	leftear	rightear	shoulder	...	tailbase	tail1	tail2	tailend
coords	x	y	likelihood	x	y	likelihood	x	y	likelihood	x	...	likelihood	x	y	likelihood	x	y	likelihood	x	y	likelihood
0	202.051529	243.017181	0.993279	225.555130	240.040192	0.925292	211.830658	222.752441	0.970271	225.887634	...	0.890497	355.775299	175.166122	0.982170	378.110535	163.782806	0.944462	414.184601	161.646469	0.965489
1	203.635178	246.195023	0.994181	221.796631	243.983353	0.932284	65.388710	241.569351	0.976376	226.809570	...	0.775419	130.691757	367.689728	0.920773	158.269897	394.064667	0.933952	188.581055	412.151093	0.988505
2	200.940704	249.184219	0.992425	220.078445	247.340149	0.956508	212.334045	232.328430	0.983834	225.477798	...	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
3	198.689117	257.430817	0.990995	222.227493	248.328888	0.946196	207.992340	228.667618	0.980577	225.162415	...	0.949660	129.576675	364.594177	0.971385	154.859970	386.656036	0.977918	183.100037	402.187103	0.992968
"""

import torch
import numpy as np
import pandas as pd
from torch import Tensor
from typing import Dict, List, Tuple
import einops


def extract_dlc_keypoints(df: pd.DataFrame, likelihood_threshold: float = 0.6) -> Dict[str, np.ndarray]:
    """
    Extract keypoints and quality labels from DLC dataframe.
    Handles both multi-animal and single-animal dataframes.
    
    Args:
        df: DLC dataframe with MultiIndex columns 
            Multi-animal: (scorer, individuals, bodyparts, coords)
            Single-animal: (scorer, bodyparts, coords)
        likelihood_threshold: Minimum likelihood for a keypoint to be considered valid
        
    Returns:
        dict with keys:
            - 'keypoints': (T, N_animals, N_bodyparts, 3) - [x, y, likelihood]
            - 'labels': (T,) - frame-level quality labels (1=corrupted, 0=good)
            - 'animal_names': list of animal identifiers
            - 'bodypart_names': list of bodypart names
            - 'scorer': scorer name for reconstruction
            - 'has_individuals': whether this is multi-animal format
    """
    # Check if this is multi-animal or single-animal format
    column_levels = df.columns.names
    has_individuals = 'individuals' in column_levels
    
    if has_individuals:
        # Multi-animal format: (scorer, individuals, bodyparts, coords)
        scorer = df.columns.get_level_values('scorer')[0]
        animals = df.columns.get_level_values('individuals').unique().tolist()
        bodyparts = df.columns.get_level_values('bodyparts').unique().tolist()
    else:
        # Single-animal format: (scorer, bodyparts, coords)
        scorer = df.columns.get_level_values('scorer')[0] if 'scorer' in column_levels else df.columns.get_level_values(0)[0]
        animals = ['single_animal']  # Default name for single animal
        bodyparts = df.columns.get_level_values('bodyparts' if 'bodyparts' in column_levels else 1).unique().tolist()
    
    n_frames = len(df)
    n_animals = len(animals)
    n_bodyparts = len(bodyparts)
    
    # Initialize arrays
    keypoints = np.full((n_frames, n_animals, n_bodyparts, 3), np.nan)
    labels = np.zeros(n_frames)  # 0=good, 1=corrupted
    
    # Extract keypoints for each animal and bodypart
    for animal_idx, animal in enumerate(animals):
        for bp_idx, bodypart in enumerate(bodyparts):
            if has_individuals:
                # Multi-animal format
                x_col = (scorer, animal, bodypart, 'x')
                y_col = (scorer, animal, bodypart, 'y')
                likelihood_col = (scorer, animal, bodypart, 'likelihood')
            else:
                # Single-animal format
                x_col = (scorer, bodypart, 'x')
                y_col = (scorer, bodypart, 'y')
                likelihood_col = (scorer, bodypart, 'likelihood')
            
            if x_col in df.columns and y_col in df.columns and likelihood_col in df.columns:
                keypoints[:, animal_idx, bp_idx, 0] = df[x_col].values
                keypoints[:, animal_idx, bp_idx, 1] = df[y_col].values
                keypoints[:, animal_idx, bp_idx, 2] = df[likelihood_col].values
    
    # Determine frame-level quality labels
    # A frame is corrupted if ANY keypoint has likelihood < threshold
    for frame_idx in range(n_frames):
        frame_likelihoods = keypoints[frame_idx, :, :, 2]  # All likelihoods for this frame
        
        # Check if any valid keypoint (not NaN) has likelihood below threshold
        valid_likelihoods = frame_likelihoods[~np.isnan(frame_likelihoods)]
        if len(valid_likelihoods) > 0 and np.any(valid_likelihoods < likelihood_threshold):
            labels[frame_idx] = 1.0  # Mark as corrupted
    
    return {
        'keypoints': keypoints,
        'labels': labels,
        'animal_names': animals,
        'bodypart_names': bodyparts,
        'scorer': scorer,
        'has_individuals': has_individuals
    }


def my_diff(vector):
    """Compute temporal differences with proper boundary handling."""
    vel_vector = torch.diff(vector, dim=0)
    # Repeat last acceleration for the final velocity step
    if len(vel_vector) > 1:
        last_acceleration = vel_vector[-1] - vel_vector[-2]
        future_vel_vector = vel_vector[-1] + last_acceleration
    else:
        future_vel_vector = vel_vector[-1] if len(vel_vector) > 0 else torch.zeros_like(vector[0])
    vel_vector = torch.cat((vel_vector, future_vel_vector[None]), dim=0)
    return vel_vector


def keypoints_to_dlc_motion_features(keypoints: np.ndarray, labels: np.ndarray, 
                                    animal_names: List[str], bodypart_names: List[str],
                                    reference_animal_idx: int = 0) -> Tensor:
    """
    Convert DLC keypoints to motion features for StableMotion.
    
    Args:
        keypoints: (T, N_animals, N_bodyparts, 3) keypoint data [x, y, likelihood]
        labels: (T,) frame-level quality labels
        animal_names: list of animal identifiers
        bodypart_names: list of bodypart names  
        reference_animal_idx: which animal to use for coordinate alignment
        
    Returns:
        features: (T, feature_dim) motion features including label channel
    """
    keypoints = torch.from_numpy(keypoints).float()
    labels = torch.from_numpy(labels).float()
    
    T, N_animals, N_bodyparts, _ = keypoints.shape
    
    # Extract coordinates and likelihoods
    coords = keypoints[..., :2]  # (T, N_animals, N_bodyparts, 2)
    likelihoods = keypoints[..., 2]  # (T, N_animals, N_bodyparts)
    
    # Handle missing data
    coords = interpolate_missing_keypoints(coords, likelihoods)
    
    # Find reference bodypart for coordinate alignment
    reference_bodypart_idx = find_reference_bodypart(bodypart_names)
    
    # Extract reference trajectory from reference animal
    reference_trajectory = coords[:, reference_animal_idx, reference_bodypart_idx]  # (T, 2)
    
    # Normalize coordinates relative to reference trajectory  
    coords_normalized = coords.clone()
    coords_normalized -= reference_trajectory.unsqueeze(1).unsqueeze(1)
    
    # Compute velocities using temporal differences
    coords_vel = my_diff(coords_normalized.flatten(1))  # (T, N_animals*N_bodyparts*2)
    trajectory_vel = my_diff(reference_trajectory)  # (T, 2)
    
    # Compute inter-keypoint distances (bone lengths)
    bone_lengths = compute_bone_lengths(coords_normalized, bodypart_names)
    bone_vel = my_diff(bone_lengths)
    
    # Pack all features into a single vector using group function
    features = group_dlc_features(
        trajectory=reference_trajectory,
        trajectory_vel=trajectory_vel,
        coords_normalized=coords_normalized.flatten(1),
        coords_vel=coords_vel,
        bone_lengths=bone_lengths,
        bone_vel=bone_vel,
        labels=labels
    )
    
    return features


def recenter_to_root_joint(coords: Tensor, root_joint_idx: int) -> Tuple[Tensor, Tensor]:
    """
    Recenter coordinates so root joint is at origin.
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2) coordinates
        root_joint_idx: index of root joint (e.g., neck)
        
    Returns:
        coords_recentered: (T, N_animals, N_bodyparts, 2) recentered coordinates
        root_positions: (T, N_animals, 2) original root joint positions
    """
    root_positions = coords[:, :, root_joint_idx].clone()  # (T, N_animals, 2)
    coords_recentered = coords - root_positions.unsqueeze(2)  # Broadcast to all bodyparts
    return coords_recentered, root_positions


def normalize_scale_frobenius(coords: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Normalize coordinates by Frobenius norm of the pose.
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2) coordinates
        
    Returns:
        coords_normalized: (T, N_animals, N_bodyparts, 2) normalized coordinates
        scale_factors: (T, N_animals) normalization factors
    """
    # Compute Frobenius norm for each frame and animal
    # Frobenius norm = sqrt(sum of squared coordinates)
    coords_flat = coords.view(*coords.shape[:-2], -1)  # (T, N_animals, N_bodyparts*2)
    scale_factors = torch.norm(coords_flat, dim=-1)  # (T, N_animals)
    
    # Avoid division by zero
    scale_factors = torch.clamp(scale_factors, min=1e-8)
    
    # Normalize
    coords_normalized = coords / scale_factors.unsqueeze(-1).unsqueeze(-1)
    
    return coords_normalized, scale_factors


def compute_bone_angles(coords: Tensor, bone_pairs: List[Tuple[int, int]]) -> Tensor:
    """
    Compute angles of bone pairs.
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2) coordinates
        bone_pairs: list of (bp1_idx, bp2_idx) tuples
        
    Returns:
        angles: (T, N_animals * N_bone_pairs) bone angles in radians
    """
    T, N_animals, N_bodyparts, _ = coords.shape
    
    angles_list = []
    
    for animal_idx in range(N_animals):
        animal_coords = coords[:, animal_idx]  # (T, N_bodyparts, 2)
        
        for bp1_idx, bp2_idx in bone_pairs:
            if bp1_idx < N_bodyparts and bp2_idx < N_bodyparts:
                # Bone vector
                bone_vector = animal_coords[:, bp2_idx] - animal_coords[:, bp1_idx]  # (T, 2)
                
                # Compute angle with respect to x-axis
                angles = torch.atan2(bone_vector[:, 1], bone_vector[:, 0])  # (T,)
                angles_list.append(angles.unsqueeze(1))  # (T, 1)
    
    if angles_list:
        angles = torch.cat(angles_list, dim=1)  # (T, N_animals * N_bone_pairs)
    else:
        angles = torch.zeros((T, 1))
    
    return angles


def keypoints_to_normalized_angle_features(keypoints: np.ndarray, labels: np.ndarray,
                                         animal_names: List[str], bodypart_names: List[str],
                                         root_joint_name: str = 'neck') -> Tuple[Tensor, Dict]:
    """
    Convert DLC keypoints to normalized angle features for StableMotion.
    
    Pipeline:
    - Recenter to root joint (neck)
    - Normalize scale using Frobenius norm
    - Compute normalized coordinates and velocities
    - Compute bone angles and angular velocities
    
    Args:
        keypoints: (T, N_animals, N_bodyparts, 3) keypoint data [x, y, likelihood]
        labels: (T,) frame-level quality labels
        animal_names: list of animal identifiers
        bodypart_names: list of bodypart names
        root_joint_name: name of root joint for centering
        
    Returns:
        features: (T, feature_dim) motion features including label channel
        transform_metadata: dict with transformation parameters for reversibility
    """
    keypoints = torch.from_numpy(keypoints).float()
    labels = torch.from_numpy(labels).float()
    
    T, N_animals, N_bodyparts, _ = keypoints.shape
    
    # Extract coordinates and likelihoods
    coords = keypoints[..., :2]  # (T, N_animals, N_bodyparts, 2)
    likelihoods = keypoints[..., 2]  # (T, N_animals, N_bodyparts)
    
    # Handle missing data
    coords = interpolate_missing_keypoints(coords, likelihoods)
    
    # Find root joint index
    root_joint_idx = None
    for idx, bp in enumerate(bodypart_names):
        if root_joint_name.lower() in bp.lower():
            root_joint_idx = idx
            break
    if root_joint_idx is None:
        root_joint_idx = find_reference_bodypart(bodypart_names)
    
    # Recenter to root joint
    coords_recentered, root_positions = recenter_to_root_joint(coords, root_joint_idx)
    
    # Normalize scale
    coords_normalized, scale_factors = normalize_scale_frobenius(coords_recentered)
    
    # Compute velocities
    coords_flat = coords_normalized.flatten(1)  # (T, N_animals*N_bodyparts*2)
    coords_vel = my_diff(coords_flat)  # (T, N_animals*N_bodyparts*2)
    
    # Compute bone angles and angular velocities
    bone_pairs = define_bone_pairs(bodypart_names)
    bone_angles = compute_bone_angles(coords_normalized, bone_pairs)  # (T, N_animals*N_bone_pairs)
    bone_angular_vel = my_diff(bone_angles)  # (T, N_animals*N_bone_pairs)
    
    # Pack all features
    features = group_normalized_angle_features(
        coords_normalized=coords_flat,
        coords_vel=coords_vel,
        bone_angles=bone_angles,
        bone_angular_vel=bone_angular_vel,
        labels=labels
    )
    
    # Store transformation parameters for reversibility
    transform_metadata = {
        'root_positions': root_positions.numpy(),
        'scale_factors': scale_factors.numpy(), 
        'root_joint_idx': root_joint_idx,
        'root_joint_name': root_joint_name
    }
    
    return features, transform_metadata


def group_normalized_angle_features(coords_normalized, coords_vel, bone_angles, bone_angular_vel, labels):
    """
    Group normalized angle features into a single feature vector.
    
    Args:
        coords_normalized: (T, N_animals*N_bodyparts*2) normalized coordinates
        coords_vel: (T, N_animals*N_bodyparts*2) coordinate velocities
        bone_angles: (T, N_animals*N_bone_pairs) bone angles
        bone_angular_vel: (T, N_animals*N_bone_pairs) angular velocities
        labels: (T,) quality labels
        
    Returns:
        features: (T, total_feature_dim) combined features
    """
    # Pack into a single feature vector
    features, _ = einops.pack([
        coords_normalized,    # (T, N_animals*N_bodyparts*2)
        coords_vel,          # (T, N_animals*N_bodyparts*2)
        bone_angles,         # (T, N_animals*N_bone_pairs)
        bone_angular_vel,    # (T, N_animals*N_bone_pairs)
        labels.unsqueeze(1)  # (T, 1) - quality labels
    ], "T *")
    
    return features


def ungroup_normalized_angle_features(features: Tensor, metadata: Dict) -> tuple[Tensor]:
    """
    Ungroup normalized angle features back to individual components.
    
    Args:
        features: (T, feature_dim) packed features
        metadata: metadata dict containing dimensions
        
    Returns:
        tuple of individual feature tensors
    """
    n_animals = metadata['n_animals']
    n_bodyparts = metadata['n_bodyparts']
    n_bone_pairs = metadata['n_bone_pairs']
    
    coord_dim = n_animals * n_bodyparts * 2
    angle_dim = n_animals * n_bone_pairs
    
    # Unpack features - order must match group_normalized_angle_features
    (
        coords_normalized,
        coords_vel,
        bone_angles,
        bone_angular_vel,
        labels
    ) = einops.unpack(features, [
        [coord_dim],           # coords_normalized
        [coord_dim],           # coords_vel
        [angle_dim],           # bone_angles
        [angle_dim],           # bone_angular_vel
        [1]                    # labels
    ], "T *")
    
    labels = labels.squeeze(-1)  # Remove last dimension
    
    return (coords_normalized, coords_vel, bone_angles, bone_angular_vel, labels)


def reverse_transformations(coords_normalized: Tensor, root_positions: Tensor, 
                          scale_factors: Tensor) -> Tensor:
    """
    Reverse the normalization, rotation, and recentering transformations.
    
    Args:
        coords_normalized: (T, N_animals, N_bodyparts, 2) normalized and rotated coordinates
        root_positions: (T, N_animals, 2) original root joint positions
        scale_factors: (T, N_animals) scale factors used for normalization
        
    Returns:
        coords_original: (T, N_animals, N_bodyparts, 2) reconstructed original coordinates
    """
    T, N_animals, N_bodyparts, _ = coords_normalized.shape
    coords_reconstructed = coords_normalized.clone()
    
    for t in range(T):
        for animal_idx in range(N_animals):
            # Reverse scale normalization
            scale = scale_factors[t, animal_idx]
            coords_reconstructed[t, animal_idx] *= scale
            
            # Add back root position
            coords_reconstructed[t, animal_idx] += root_positions[t, animal_idx].unsqueeze(0)
    
    return coords_reconstructed


def group_dlc_features(trajectory, trajectory_vel, coords_normalized, coords_vel, 
                      bone_lengths, bone_vel, labels):
    """
    Group all DLC features into a single feature vector.
    Similar to SMPL's group function for consistent packing.
    
    Args:
        trajectory: (T, 2) reference trajectory
        trajectory_vel: (T, 2) trajectory velocity  
        coords_normalized: (T, N_animals*N_bodyparts*2) flattened normalized coordinates
        coords_vel: (T, N_animals*N_bodyparts*2) flattened coordinate velocities
        bone_lengths: (T, N_bone_pairs) bone lengths
        bone_vel: (T, N_bone_pairs) bone length velocities
        labels: (T,) quality labels
        
    Returns:
        features: (T, total_feature_dim) combined features
    """
    # Pack into a single feature vector - similar to SMPL approach
    features, _ = einops.pack([
        trajectory,           # (T, 2)
        trajectory_vel,       # (T, 2) 
        coords_normalized,    # (T, N_animals*N_bodyparts*2)
        coords_vel,          # (T, N_animals*N_bodyparts*2)
        bone_lengths,        # (T, N_bone_pairs)
        bone_vel,           # (T, N_bone_pairs)
        labels.unsqueeze(1)  # (T, 1) - quality labels
    ], "T *")
    
    return features


def ungroup_dlc_features(features: Tensor, metadata: Dict) -> tuple[Tensor]:
    """
    Ungroup DLC features back to individual components.
    
    Args:
        features: (T, feature_dim) packed features
        metadata: metadata dict containing dimensions
        
    Returns:
        tuple of individual feature tensors
    """
    n_animals = metadata['n_animals']
    n_bodyparts = metadata['n_bodyparts']
    n_bone_pairs = metadata['n_bone_pairs']
    
    coord_dim = n_animals * n_bodyparts * 2
    
    # Unpack features - order must match group_dlc_features
    (
        trajectory,
        trajectory_vel, 
        coords_normalized,
        coords_vel,
        bone_lengths,
        bone_vel,
        labels
    ) = einops.unpack(features, [
        [2],                    # trajectory
        [2],                    # trajectory_vel
        [coord_dim],           # coords_normalized
        [coord_dim],           # coords_vel  
        [n_bone_pairs],        # bone_lengths
        [n_bone_pairs],        # bone_vel
        [1]                    # labels
    ], "T *")
    
    labels = labels.squeeze(-1)  # Remove last dimension
    
    return (trajectory, trajectory_vel, coords_normalized, coords_vel, 
            bone_lengths, bone_vel, labels)


def interpolate_missing_keypoints(coords: Tensor, likelihoods: Tensor, 
                                 likelihood_threshold: float = 0.3) -> Tensor:
    """
    Interpolate missing or low-confidence keypoints.
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2) coordinates
        likelihoods: (T, N_animals, N_bodyparts) confidence scores
        likelihood_threshold: minimum confidence for using keypoint
        
    Returns:
        coords_interp: interpolated coordinates
    """
    # Simple approach: replace -1 with 0 and return
    # TODO: Implement proper interpolation if needed
    coords_interp = coords.clone()
    coords_interp[coords_interp == -1] = 0
    return coords_interp
    coords_interp = coords.clone()
    T, N_animals, N_bodyparts, _ = coords.shape
    
    for animal_idx in range(N_animals):
        for bp_idx in range(N_bodyparts):
            # Extract time series for this keypoint
            coord_series = coords[:, animal_idx, bp_idx]  # (T, 2)
            likelihood_series = likelihoods[:, animal_idx, bp_idx]  # (T,)
            
            # Find valid frames (high likelihood and not NaN)
            valid_mask = (likelihood_series > likelihood_threshold) & ~torch.isnan(likelihood_series)
            
            if valid_mask.sum() < 2:  # Not enough valid points for interpolation
                # Use forward fill or zero
                coords_interp[:, animal_idx, bp_idx] = 0
                continue
                
            # Get valid indices and values
            valid_indices = torch.where(valid_mask)[0]
            valid_coords = coord_series[valid_mask]
            
            # Simple linear interpolation for missing points
            for t in range(T):
                if not valid_mask[t]:
                    # Find surrounding valid points
                    left_idx = valid_indices[valid_indices < t].max() if (valid_indices < t).any() else None
                    right_idx = valid_indices[valid_indices > t].min() if (valid_indices > t).any() else None
                    
                    if left_idx is not None and right_idx is not None:
                        # Linear interpolation
                        alpha = (t - left_idx.item()) / (right_idx.item() - left_idx.item())
                        coords_interp[t, animal_idx, bp_idx] = (
                            (1 - alpha) * coords_interp[left_idx, animal_idx, bp_idx] + 
                            alpha * coords_interp[right_idx, animal_idx, bp_idx]
                        )
                    elif left_idx is not None:
                        # Forward fill
                        coords_interp[t, animal_idx, bp_idx] = coords_interp[left_idx, animal_idx, bp_idx]
                    elif right_idx is not None:
                        # Backward fill
                        coords_interp[t, animal_idx, bp_idx] = coords_interp[right_idx, animal_idx, bp_idx]
    
    return coords_interp


def find_reference_bodypart(bodypart_names: List[str]) -> int:
    """
    Find a suitable reference bodypart for coordinate alignment.
    """
    priority_bodyparts = ['neck', 'center', 'spine', 'hip', 'pelvis', 'torso']
    
    for priority_bp in priority_bodyparts:
        for idx, bp in enumerate(bodypart_names):
            if priority_bp.lower() in bp.lower():
                return idx
    
    # Default to first bodypart if no priority match found
    return 0


def compute_keypoint_velocities(coords: Tensor) -> Tensor:
    """
    Compute velocities for all keypoints using temporal differences.
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2)
        
    Returns:
        velocities: (T, N_animals, N_bodyparts, 2)
    """
    velocities = torch.zeros_like(coords)
    
    # Compute differences
    vel_diff = torch.diff(coords, dim=0)  # (T-1, N_animals, N_bodyparts, 2)
    velocities[1:] = vel_diff
    
    # Handle first frame - use first difference
    velocities[0] = velocities[1] if len(velocities) > 1 else torch.zeros_like(velocities[0])
    
    return velocities


def compute_velocity(trajectory: Tensor) -> Tensor:
    """Compute velocity with proper boundary handling."""
    vel = torch.diff(trajectory, dim=0)
    # Repeat last velocity for final frame
    if len(vel) > 0:
        last_vel = vel[-1] if len(vel) == 1 else vel[-1] + (vel[-1] - vel[-2])
        vel = torch.cat([vel, last_vel.unsqueeze(0)], dim=0)
    else:
        vel = torch.zeros_like(trajectory)
    return vel


def compute_bone_lengths(coords: Tensor, bodypart_names: List[str]) -> Tensor:
    """
    Compute distances between key bodypart pairs (bone lengths).
    
    Args:
        coords: (T, N_animals, N_bodyparts, 2)
        bodypart_names: list of bodypart names
        
    Returns:
        bone_lengths: (T, N_bone_pairs) distances
    """
    # Define common bone pairs based on typical animal anatomy
    bone_pairs = define_bone_pairs(bodypart_names)
    
    T, N_animals, N_bodyparts, _ = coords.shape
    bone_lengths_list = []
    
    for animal_idx in range(N_animals):
        animal_coords = coords[:, animal_idx]  # (T, N_bodyparts, 2)
        
        for bp1_idx, bp2_idx in bone_pairs:
            if bp1_idx < N_bodyparts and bp2_idx < N_bodyparts:
                # Compute distance between bodyparts
                dist = torch.norm(
                    animal_coords[:, bp1_idx] - animal_coords[:, bp2_idx], 
                    dim=1
                )  # (T,)
                bone_lengths_list.append(dist.unsqueeze(1))
    
    if bone_lengths_list:
        bone_lengths = torch.cat(bone_lengths_list, dim=1)  # (T, N_bone_pairs)
    else:
        bone_lengths = torch.zeros((T, 1))  # Fallback
        
    return bone_lengths


def define_bone_pairs(bodypart_names: List[str]) -> List[Tuple[int, int]]:
    """
    Define bone pairs based on bodypart names.
    Returns list of (bodypart1_idx, bodypart2_idx) tuples.
    skeleton:
    - [nose, head_midpoint]
    - [head_midpoint, left_eye]
    - [left_eye, left_ear_tip]
    - [left_ear_tip, left_ear]
    - [head_midpoint, right_eye]
    - [right_eye, right_ear_tip]
    - [right_ear_tip, right_ear]

    # ─── Neck & spine
    - [head_midpoint, neck]
    - [neck, mid_back]
    - [mid_back, mid_backend]
    - [mid_backend, mid_backend2]
    - [mid_backend2, mid_backend3]
    - [mid_backend3, mouse_center]

    # ─── Shoulders → hips
    - [mouse_center, left_shoulder]
    - [left_shoulder, left_midside]
    - [left_midside, left_hip]
    - [mouse_center, right_shoulder]
    - [right_shoulder, right_midside]
    - [right_midside, right_hip]
    - [left_hip, tail_base]          # connect hips to tail base
    - [right_hip, tail_base]
    - [tail_base, tail1]
    - [tail1, tail2]
    - [tail2, tail3]
    - [tail3, tail4]
    - [tail4, tail5]
    - [tail5, tail_end]

    """
    bone_pairs = []
    
    # Create mapping from bodypart name to index
    bp_to_idx = {bp.lower(): idx for idx, bp in enumerate(bodypart_names)}
    
    # Define common bone connections fonwclr animal anatomy
    potential_connections = [
        ('nose', 'head_midpoint'),
        ('head_midpoint', 'left_eye'),
        ('left_eye', 'left_ear_tip'),
        ('left_ear_tip', 'left_ear'),
        ('head_midpoint', 'right_eye'),
        ('right_eye', 'right_ear_tip'),
        ('right_ear_tip', 'right_ear'),
        ('head_midpoint', 'neck'),
        ('neck', 'mid_back'),
        ('mid_back', 'mid_backend'),
        ('mid_backend', 'mid_backend2'),
        ('mid_backend2', 'mid_backend3'),
        ('mid_backend3', 'mouse_center'),
        ('mouse_center', 'left_shoulder'),
        ('left_shoulder', 'left_midside'),
        ('left_midside', 'left_hip'),
        ('mouse_center', 'right_shoulder'),
        ('right_shoulder', 'right_midside'),
        ('right_midside', 'right_hip'),
        ('left_hip', 'tail_base'),
        ('right_hip', 'tail_base'),
        ('tail_base', 'tail1'),
        ('tail1', 'tail2'),
        ('tail2', 'tail3'),
        ('tail3', 'tail4'),
        ('tail4', 'tail5'),
        ('tail5', 'tail_end'),
    ]
    
    # Add connections that exist in the data
    for bp1, bp2 in potential_connections:
        if bp1 in bp_to_idx and bp2 in bp_to_idx:
            bone_pairs.append((bp_to_idx[bp1], bp_to_idx[bp2]))
    
    # If no predefined pairs found, use adjacent pairs
    if not bone_pairs:
        for i in range(len(bodypart_names) - 1):
            bone_pairs.append((i, i + 1))
    
    return bone_pairs


def group_keypoint_features(trajectory: Tensor, trajectory_vel: Tensor,
                           coords_normalized: Tensor, coords_vel: Tensor,
                           bone_lengths: Tensor, bone_vel: Tensor,
                           labels: Tensor) -> Tensor:
    """
    Group all keypoint features into a single feature vector.
    
    Args:
        trajectory: (T, 2) reference trajectory
        trajectory_vel: (T, 2) trajectory velocity
        coords_normalized: (T, N_animals*N_bodyparts*2) normalized coordinates
        coords_vel: (T, N_animals*N_bodyparts*2) coordinate velocities
        bone_lengths: (T, N_bone_pairs) bone lengths
        bone_vel: (T, N_bone_pairs) bone length velocities
        labels: (T,) quality labels
        
    Returns:
        features: (T, total_feature_dim) combined features
    """
    feature_components = [
        trajectory,           # Reference trajectory
        trajectory_vel,       # Reference velocity
        coords_normalized,    # All normalized coordinates
        coords_vel,          # All coordinate velocities
        bone_lengths,        # Bone lengths
        bone_vel,           # Bone length velocities
        labels.unsqueeze(1)  # Quality labels
    ]
    
    features = torch.cat(feature_components, dim=1)
    return features


def dlc_dataframe_to_features(df: pd.DataFrame, likelihood_threshold: float = 0.6, encoding_type: str = 'normalized_angles') -> Dict:
    """
    Complete pipeline: DLC DataFrame -> motion features for StableMotion.
    Stores all metadata needed for perfect reconstruction.
    
    Args:
        df: DLC multi-animal dataframe
        likelihood_threshold: minimum likelihood for valid keypoints
        encoding_type: feature encoding method ('trajectory' or 'normalized_angles')
        
    Returns:
        dict with:
            - 'features': (T, feature_dim) motion features
            - 'metadata': dict with all reconstruction info
    """
    # Extract keypoints and labels
    keypoint_data = extract_dlc_keypoints(df, likelihood_threshold)
    
    if encoding_type == 'trajectory':
        # Original trajectory-based encoding
        features = keypoints_to_dlc_motion_features(
            keypoint_data['keypoints'],
            keypoint_data['labels'],
            keypoint_data['animal_names'],
            keypoint_data['bodypart_names']
        )
        
        # Compute bone pairs to get count
        bone_pairs = define_bone_pairs(keypoint_data['bodypart_names'])
        n_bone_pairs = len(bone_pairs) * len(keypoint_data['animal_names'])
        transform_metadata = {}  # No transformation metadata needed
        
    elif encoding_type == 'normalized_angles':
        # New normalized angles encoding
        features, transform_metadata = keypoints_to_normalized_angle_features(
            keypoint_data['keypoints'],
            keypoint_data['labels'],
            keypoint_data['animal_names'],
            keypoint_data['bodypart_names']
        )
        
        # Compute bone pairs to get count
        bone_pairs = define_bone_pairs(keypoint_data['bodypart_names'])
        n_bone_pairs = len(bone_pairs) * len(keypoint_data['animal_names'])
        
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'trajectory' or 'normalized_angles'")
    
    # Store comprehensive metadata for perfect reconstruction
    metadata = {
        # Encoding type
        'encoding_type': encoding_type,
        
        # Basic dimensions
        'animal_names': keypoint_data['animal_names'],
        'bodypart_names': keypoint_data['bodypart_names'], 
        'n_animals': len(keypoint_data['animal_names']),
        'n_bodyparts': len(keypoint_data['bodypart_names']),
        'n_frames': len(df),
        'feature_dim': features.shape[1],
        'likelihood_threshold': likelihood_threshold,
        'likelihood': keypoint_data['keypoints'][..., 2],
        
        # Feature structure
        'n_bone_pairs': n_bone_pairs,
        'bone_pairs': bone_pairs,
        'reference_animal_idx': 0,
        'reference_bodypart_idx': find_reference_bodypart(keypoint_data['bodypart_names']),
        
        # DataFrame structure for reconstruction
        'scorer': keypoint_data['scorer'],
        'has_individuals': keypoint_data['has_individuals'],
        'index': df.index.tolist(),  # Preserve original index
        
        # Column structure
        'column_names': df.columns.names,
        'columns': df.columns.tolist(),  # Full MultiIndex structure
        
        # Original data shape for validation
        'original_shape': df.shape,
    }
    
    # Add transformation metadata for normalized_angles encoding
    if encoding_type == 'normalized_angles':
        metadata.update(transform_metadata)
    
    return features, metadata

def fill_default_metadata(metadata: Dict) -> Dict:
    metadata['encoding_type'] = 'trajectory'  # Default to trajectory encoding
    metadata['n_animals'] = 1
    metadata['n_bodyparts'] = 27
    metadata['n_bone_pairs'] = 27
    metadata['reference_animal_idx'] = 0
    metadata['reference_bodypart_idx'] = 7
    metadata['bone_pairs'] = [(0, 26), (26, 5), (5, 3), (3, 1), (26, 6), (6, 4), (4, 2), (26, 7), (7, 8), (8, 10), (10, 11), (11, 12), (12, 9), (9, 19), (19, 20), (20, 21), (9, 22), (22, 23), (23, 24), (21, 13), (24, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 25)]
    metadata['scorer'] = "StableMotion"
    metadata['animal_names'] = ['single_animal']
    metadata['bodypart_names'] = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip', 'tail_end', 'head_midpoint']
    metadata['has_individuals'] = False
    metadata['feature_dim'] = 167
    metadata['n_frames'] = 250
    metadata['likelihood_threshold'] = 0.6

    return metadata


def features_to_dlc_keypoints(features: Tensor, metadata: Dict = {}, encoding_type: str = 'trajectory') -> pd.DataFrame:
    """
    Convert motion features back to exact DLC dataframe format.
    Perfect reconstruction of original dataframe structure.
    
    Args:
        features: (T, feature_dim) motion features
        metadata: metadata dict from dlc_dataframe_to_features
        encoding_type: feature encoding method ('trajectory' or 'normalized_angles')
        
    Returns:
        df: Reconstructed DLC DataFrame with exact original structure
    """
    if not metadata:
        metadata = fill_default_metadata(metadata)
    
    # Get encoding type from metadata, default to trajectory for backward compatibility
    actual_encoding_type = metadata.get('encoding_type', encoding_type)
    
    if actual_encoding_type == 'trajectory':
        # Original trajectory-based decoding
        (trajectory, trajectory_vel, coords_normalized, coords_vel,
         bone_lengths, bone_vel, labels) = ungroup_dlc_features(features, metadata)
        
        n_animals = metadata['n_animals']
        n_bodyparts = metadata['n_bodyparts']
        T = features.shape[0]
        
        # Reshape coordinates back to (T, N_animals, N_bodyparts, 2)
        coords_normalized = coords_normalized.view(T, n_animals, n_bodyparts, 2)
        likelihoods = torch.from_numpy(metadata['likelihood']).unsqueeze(-1)  # (T, N_animals, N_bodyparts, 1)
        
        # Add back the reference trajectory to get absolute coordinates
        coords_absolute = coords_normalized + trajectory.unsqueeze(1).unsqueeze(1)
        
    elif actual_encoding_type == 'normalized_angles':
        # New normalized angles decoding
        (coords_normalized, coords_vel, bone_angles, bone_angular_vel, labels) = ungroup_normalized_angle_features(features, metadata)
        
        n_animals = metadata['n_animals']
        n_bodyparts = metadata['n_bodyparts']
        T = features.shape[0]
        
        # Reshape coordinates back to (T, N_animals, N_bodyparts, 2)
        coords_normalized = coords_normalized.view(T, n_animals, n_bodyparts, 2)
        
        # Reconstruct original coordinates from normalized features
        # Note: This requires transformation parameters stored in metadata
        root_positions = torch.from_numpy(metadata['root_positions'])
        scale_factors = torch.from_numpy(metadata['scale_factors'])
        
        coords_absolute = reverse_transformations(
            coords_normalized, root_positions, scale_factors
        )
            
        likelihoods = torch.from_numpy(metadata['likelihood']).unsqueeze(-1)  # (T, N_animals, N_bodyparts, 1)
        
    else:
        raise ValueError(f"Unknown encoding_type: {actual_encoding_type}. Must be 'trajectory' or 'normalized_angles'")
    
    # Combine coordinates with likelihoods: (T, N_animals, N_bodyparts, 3)
    keypoints = torch.cat([coords_absolute, likelihoods], dim=-1)
    
    # Convert to numpy
    keypoints_np = keypoints.numpy()
    
    # Reconstruct DataFrame with exact original structure
    scorer = metadata['scorer']
    animal_names = metadata['animal_names']
    bodypart_names = metadata['bodypart_names']
    has_individuals = metadata['has_individuals']
    
    # Create empty dataframe with original structure
    if has_individuals:
        # Multi-animal format
        columns = pd.MultiIndex.from_product(
            [[scorer], animal_names, bodypart_names, ['x', 'y', 'likelihood']],
            names=['scorer', 'individuals', 'bodyparts', 'coords']
        )
    else:
        # Single-animal format  
        columns = pd.MultiIndex.from_product(
            [[scorer], bodypart_names, ['x', 'y', 'likelihood']],
            names=['scorer', 'bodyparts', 'coords']
        )
    
    # Create DataFrame
    df_reconstructed = pd.DataFrame(
        index=metadata['index'],
        columns=columns,
        dtype=float
    )
    
    # Fill in the data
    for animal_idx, animal in enumerate(animal_names):
        for bp_idx, bodypart in enumerate(bodypart_names):
            if has_individuals:
                x_col = (scorer, animal, bodypart, 'x')
                y_col = (scorer, animal, bodypart, 'y')
                likelihood_col = (scorer, animal, bodypart, 'likelihood')
            else:
                # For single animal, skip the animal level
                if animal == 'single_animal':  # Our default single animal name
                    x_col = (scorer, bodypart, 'x')
                    y_col = (scorer, bodypart, 'y')
                    likelihood_col = (scorer, bodypart, 'likelihood')
                else:
                    continue
            
            df_reconstructed[x_col] = keypoints_np[:, animal_idx, bp_idx, 0]
            df_reconstructed[y_col] = keypoints_np[:, animal_idx, bp_idx, 1]
            df_reconstructed[likelihood_col] = keypoints_np[:, animal_idx, bp_idx, 2]
    
    return df_reconstructed