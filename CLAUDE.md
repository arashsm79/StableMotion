# StableMotion - Claude Assistant Reference

## Project Overview

**StableMotion** is a diffusion-based model for cleaning up corrupted motion capture data. Unlike traditional methods that require paired clean-corrupted training data, StableMotion learns to fix motion artifacts directly from unpaired corrupted motion data using a novel training approach.

### Key Innovation
- Trains motion cleanup models without requiring clean reference data
- Uses diffusion models with inpainting techniques to detect and fix motion artifacts
- Works directly with raw mocap data containing natural corruptions and artifacts

## Architecture & Components

### Core Model Architecture
- **Model Type**: Diffusion Transformer (DiT) adapted from Stable Audio
- **Base Model**: `StableMotionDiTModel` in `model/stablemotion.py`
- **Input Features**: 233-dimensional Global SMPL RIFKE features + label channel
- **Architecture**: 
  - Transformer blocks with self-attention and feed-forward layers
  - Adaptive layer normalization with time/mode conditioning
  - Rotary position embeddings for 1D sequences
  - Support for gradient checkpointing and mixed precision training

### Motion Representation
- **Format**: Global SMPL RIFKE (Rotation-Invariant Features with Kinematic Embeddings)
- **Features**: Body pose, translation, joint positions, velocities, accelerations
- **Label Channel**: Binary quality indicators for each frame
- **Normalization**: Per-feature standardization with mean/std statistics

### Training Pipeline
1. **Data Preprocessing**: Convert SMPL data to aligned Global SMPL RIFKE features
2. **Corruption Simulation**: Inject realistic motion artifacts for training data
3. **Diffusion Training**: Learn to denoise and inpaint motion sequences
4. **Quality Detection**: Simultaneously learn frame-level quality assessment

## Key Files & Directories

### Model & Training
- `model/stablemotion.py` - Core diffusion transformer architecture
- `train/train_stablemotion_smpl_glob.py` - Main training script
- `train/training_loop_smpl.py` - Training loop with EMA, mixed precision, checkpointing
- `utils/model_util.py` - Model and diffusion creation utilities

### Data Processing
- `data_loaders/globsmpl_dataset.py` - Main dataset loader for Global SMPL features
- `data_loaders/corrupting_globsmpl_dataset.py` - Dataset with synthetic corruption injection
- `data_loaders/amasstools/` - AMASS preprocessing utilities
  - `globsmplrifke_feats.py` - Feature extraction from SMPL data
  - `geometry.py` - Rotation and transformation utilities
- `data_loaders/dataset_utils.py` - Motion artifact detection and injection

### Inference & Evaluation
- `sample/fix_globsmpl.py` - Main inference script for motion cleanup
- `sample/utils.py` - Sampling utilities and cleanup strategies
- `eval/eval_motion.py` - Motion quality evaluation metrics
- `eval/eval_scripts.py` - Evaluation pipeline runner

### Diffusion Framework
- `diffusion/gaussian_diffusion.py` - Core diffusion implementation
- `diffusion/nn.py` - Neural network utilities
- `diffusion/losses.py` - Loss functions for training
- `diffusion/respace.py` - Timestep spacing utilities

### Utilities
- `utils/parser_util.py` - Command-line argument parsing
- `utils/config.py` - Configuration management
- `utils/normalizer.py` - Feature normalization
- `utils/dist_util.py` - Distributed training utilities

## Environment Setup

### Python Environment
```bash
conda create --name stablemotion python=3.11.8
conda activate stablemotion
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch 2.4.1 with CUDA support
- Diffusers 0.30.2 for diffusion utilities
- SMPLX for human body modeling
- PyTorch Lightning for training infrastructure
- EMA-PyTorch for exponential moving averages

### External Dependencies
1. **SMPL Models**: Required for body mesh reconstruction
   - Download SMPL+H models following TEMOS README
   - Place in `data_loaders/amasstools/deps/`

2. **TMR Models**: Required for evaluation metrics
   - Download TMR pretrained models
   - Place in `tmr_models/` directory
   - Update config path as described in README

## Usage Examples

### Training a Model
```bash
python -m train.train_stablemotion_smpl_glob \
  --save_dir save/my_model \
  --data_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --normalizer_dir dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --batch_size 64 \
  --lr 1e-4 \
  --num_steps 1000000 \
  --train_platform_type TensorboardPlatform
```

### Motion Cleanup Inference
```bash
# Basic cleanup
python -m sample.fix_globsmpl \
  --model_path save/my_model/ema001000000.pt \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --normalizer_dir dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --output_dir ./output/cleanup_results

# Advanced cleanup with ensemble and adaptive scheduling
python -m sample.fix_globsmpl \
  --model_path save/my_model/ema001000000.pt \
  --testdata_dir dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --normalizer_dir dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano \
  --output_dir ./output/cleanup_ensemble \
  --ensemble \
  --ProbDetNum 5 \
  --enable_sits
```

### Data Preprocessing
```bash
# Create corrupted training data
python -m data_loaders.corrupting_globsmpl_dataset --mode train
python -m data_loaders.corrupting_globsmpl_dataset --mode test
```

## Key Technical Details

### Motion Features (Global SMPL RIFKE)
- **Dimensionality**: 232 motion features + 1 label channel = 233 total
- **Content**:
  - Root translation trajectory (2D, Z-aligned)
  - Root orientation decomposition (Z/Y/X Euler angles)
  - Body joint rotations (6D representation)
  - Joint positions relative to pelvis
  - Velocities and accelerations of key components
  - Foot contact and trajectory information

### Training Strategy
- **Objective**: Learn to denoise motion sequences while preserving quality labels
- **Corruption Types**: Foot sliding, jitter, missing frames, pose artifacts
- **Inpainting**: Selective reconstruction of corrupted frames while preserving good ones
- **Detection**: Simultaneous learning of frame-level quality assessment

### Inference Pipeline
1. **Detection Pass**: Identify potentially corrupted frames using label channel prediction
2. **Mask Construction**: Build inpainting mask with slight dilation around detected issues
3. **Cleanup Pass**: Reconstruct only the masked (problematic) regions
4. **Optional Ensemble**: Average multiple cleanup attempts for robustness

### Performance Features
- **Mixed Precision**: Automatic mixed precision with GradScaler
- **EMA**: Exponential moving averages for stable model weights
- **Gradient Clipping**: Prevents training instabilities
- **FlashAttention**: Optimized attention computation when available
- **Gradient Checkpointing**: Memory-efficient training for large models

## Data Representation & Processing Pipeline

### Current Data Format: Global SMPL RIFKE Features

StableMotion uses a carefully designed 233-dimensional feature representation called "Global SMPL RIFKE" (Rotation-Invariant Features with Kinematic Embeddings) that encodes human motion in a way that's both expressive and suitable for diffusion-based processing.

#### Feature Breakdown (232 motion features + 1 label channel = 233 total)

**Input Requirements:**
- SMPL pose parameters: `(T, 66)` - 22 joints × 3 axis-angle values (no hands)
- Translation: `(T, 3)` - Root joint global translation
- Joints: `(T, 24, 3)` - 3D joint positions (computed from SMPL if missing)
- Labels: `(T,)` - Binary quality labels per frame (0=good, 1=corrupted)

**Feature Components:**
1. **Root Features (5 dims)**:
   - `root_grav_axis (1)`: Height of pelvis above ground
   - `trajectory (2)`: 2D root translation in canonical facing direction
   - `rotZ_2d (2)`: Normalized 2D vector representing Z-rotation

2. **Pose Features (132 dims)**:
   - `poses_local (22×6=132)`: 6D rotation representations for all joints
   - Global orientation decomposed into Y and X rotations (Z-rotation handled separately)
   - Local body pose representations in canonical coordinate frame

3. **Joint Features (69 dims)**:
   - `joints_local (23×3=69)`: 3D joint positions relative to pelvis (pelvis excluded)
   - Normalized to pelvis coordinate system
   - Ground height removed, trajectory-aligned

4. **Foot Contact Features (12 dims)**:
   - `foot_global (4×3=12)`: Global positions of 4 foot joints
   - Critical for detecting foot sliding artifacts

5. **Velocity Features (14 dims)**:
   - `vel_foot_global (4×3=12)`: Velocities of foot joints
   - `vel_traj (2)`: 2D trajectory velocity
   - Computed using temporal differences with acceleration extrapolation

6. **Label Channel (1 dim)**:
   - Binary indicator: 0=good frame, 1=corrupted frame
   - Used for detection training and inpainting mask construction

#### Preprocessing Pipeline

```python
# Key transformations in smpldata_to_alignglobsmplrifkefeats():
# 1. Ground alignment and coordinate normalization
joints[:, :, 2] -= ground_level  # Remove ground
trajectory = joints[:, 0, :2] - joints[0, 0, :2]  # Relative trajectory

# 2. Canonical facing direction
global_euler = matrix_to_euler_angles(global_orient, "ZYX")
rotZ_angle = rotZ_angle - rotZ_angle[0]  # Relative rotation

# 3. Coordinate frame transformation
joints_local = rotate_to_canonical_frame(joints)
trajectory = rotate_to_canonical_frame(trajectory)

# 4. Feature packing
features = group(root_grav_axis, trajectory, rotZ_2d, poses_local, 
                joints_local, foot_global, vel_foot_global, vel_traj)
```

#### Normalization Strategy

```python
class Normalizer:
    # Per-feature Z-score normalization
    normalized = (features - mean) / (std + eps)
    # Label channel gets special treatment: mean=0.5, std=0.5
```

### Adapting to Other Data Types (e.g., Keypoints)

To use StableMotion with keypoint data instead of SMPL, you would need to:

#### 1. Create New Feature Extraction Pipeline

**Replace**: `data_loaders/amasstools/globsmplrifke_feats.py`
**With**: Custom keypoint feature extractor

```python
def keypoints_to_features(keypoints_3d, labels=None):
    """
    Convert 3D keypoints to motion features.
    
    Args:
        keypoints_3d: (T, N_joints, 3) - 3D keypoint positions
        labels: (T,) - Optional quality labels
        
    Returns:
        features: (T, feature_dim) - Motion feature representation
    """
    # Required transformations:
    # 1. Root normalization (align to pelvis/hip)
    # 2. Coordinate canonicalization (facing direction)
    # 3. Velocity/acceleration computation
    # 4. Feature grouping and concatenation
```

**Key Design Decisions:**
- **Root Joint**: Choose primary joint for trajectory (e.g., pelvis, spine)
- **Coordinate Frame**: Define canonical facing direction method
- **Joint Subset**: Select relevant joints (exclude fingers, face, etc.)
- **Velocity Features**: Decide which joints need velocity/acceleration
- **Feature Dimensionality**: May differ from 232, requires model adjustment

#### 2. Modify Model Architecture

**Update**: `utils/model_util.py` and `model/stablemotion.py`

```python
def create_model_and_diffusion(args):
    model = StableMotionDiTModel( 
        in_channels=NEW_FEATURE_DIM + 1,  # Your features + label channel
        out_channels=NEW_FEATURE_DIM + 1,
        # ... other params unchanged
    )
```

#### 3. Create Data Loaders

**Replace**: `data_loaders/globsmpl_dataset.py`
**With**: Keypoint dataset loader

```python
class KeypointMotionLoader:
    def __call__(self, path):
        # Load keypoint data from your format (NPZ, JSON, etc.)
        keypoints = load_keypoint_file(path)
        
        # Extract features
        motion_features = keypoints_to_features(keypoints)
        
        # Add label channel and normalize
        motion_with_labels = add_quality_labels(motion_features)
        normalized_motion = self.normalizer(motion_with_labels)
        
        return {"x": normalized_motion, "length": len(normalized_motion)}
```

#### 4. Adapt Corruption Types

**Update**: `data_loaders/dataset_utils.py`

```python
def motion_artifacts_keypoints(keypoints, mode='train'):
    """
    Inject realistic keypoint-specific artifacts:
    - Joint tracking failures (sudden jumps)
    - Occlusion artifacts (missing/interpolated joints)
    - Temporal inconsistencies (jitter)
    - Depth estimation errors
    - Joint swapping/mislabeling
    """
```

#### 5. Evaluation Adaptations

**Considerations for keypoint evaluation**:
- **Metrics**: Joint position error, bone length consistency, temporal smoothness
- **Reconstruction**: Convert features back to 3D keypoints for visualization
- **Ground Truth**: May need different evaluation protocols than SMPL

#### 6. Normalization Statistics

**Compute new statistics**:
```python
# Update data_loaders/globsmpl_dataset.py __main__ section
# Compute mean/std for your keypoint features
keypoint_features = collect_all_features(keypoint_dataset)
mean_features = keypoint_features.mean(0)[:-1]  # Exclude label
std_features = keypoint_features.std(0)[:-1]
normalizer.save(mean_features, std_features)
```

### Critical Implementation Notes

#### Temporal Consistency
- **Velocity Computation**: Use `my_diff()` function for consistent temporal derivatives
- **Padding Strategy**: Handle sequence boundaries carefully
- **Frame Rate**: Ensure consistent FPS across all sequences

#### Coordinate Systems
- **Canonical Alignment**: Essential for rotation invariance
- **Ground Plane**: Consistent ground level normalization
- **Facing Direction**: Reproducible canonical orientation

#### Label Channel Integration
- **Binary Labels**: 0=good, 1=corrupted (consistent with training)
- **Label Normalization**: Special handling (mean=0.5, std=0.5)
- **Detection Logic**: Adapt thresholds for your data characteristics

#### Model Compatibility
- **Feature Dimension**: Update `in_channels`/`out_channels` in model config
- **Sequence Length**: Ensure compatibility with attention mechanisms
- **Batch Collation**: Adapt `collate_motion` for variable sequence lengths

### Example: Converting MediaPipe Keypoints

```python
def mediapipe_to_stablemotion_features(mediapipe_landmarks):
    """
    Convert MediaPipe pose landmarks to StableMotion-compatible features.
    
    MediaPipe provides 33 pose landmarks in normalized coordinates.
    """
    # 1. Extract key joints (exclude face/hands if not needed)
    key_joints = mediapipe_landmarks[:, RELEVANT_JOINT_INDICES]  # (T, N, 3)
    
    # 2. Convert to metric coordinates (if needed)
    key_joints_metric = normalize_coordinates_to_meters(key_joints)
    
    # 3. Apply StableMotion preprocessing
    features = keypoints_to_stablemotion_format(
        keypoints_3d=key_joints_metric,
        hip_joint_idx=HIP_INDEX,
        shoulder_indices=[LEFT_SHOULDER, RIGHT_SHOULDER]
    )
    
    return features  # (T, feature_dim)
```

This data representation system is the core of StableMotion's effectiveness - the careful feature engineering enables the diffusion model to learn meaningful motion priors and corruption patterns.

## DeepLabCut Integration

StableMotion now includes complete support for DeepLabCut (DLC) animal pose estimation data. This integration allows you to use StableMotion as a post-processing tool to clean up pose estimation artifacts and improve tracking quality.

### DLC Data Support Features

- **Multi-animal support**: Handles DLC's multi-animal tracking format
- **Likelihood-based corruption detection**: Frames with any keypoint below threshold are marked as corrupted
- **Keypoint-specific feature extraction**: Converts DLC coordinates to motion features suitable for diffusion processing
- **Temporal artifact correction**: Fixes tracking failures, jitter, occlusion artifacts, and coordinate drift
- **Bone length consistency**: Maintains anatomical constraints during cleanup

### Quick Start with DLC Data

#### 1. Setup Pipeline
```bash
# Setup the complete DLC pipeline
python setup_dlc_pipeline.py \
  --data_dir path/to/your/dlc/files \
  --output_dir dataset/dlc_motion \
  --likelihood_threshold 0.6 \
  --fps 30
```

#### 2. Train Model
```bash
# Train StableMotion on your DLC data
python -m train.train_stablemotion_dlc \
  --save_dir save/stablemotion_dlc \
  --data_dir path/to/your/dlc/files \
  --normalizer_dir dataset/dlc_motion/meta \
  --likelihood_threshold 0.6 \
  --batch_size 32 \
  --num_steps 500000
```

#### 3. Clean Your Data
```bash
# Clean corrupted DLC data
python -m sample.fix_dlc_keypoints \
  --model_path save/stablemotion_dlc/ema001000000.pt \
  --input_file your_corrupted_data.csv \
  --output_file your_cleaned_data.csv \
  --normalizer_dir dataset/dlc_motion/meta \
  --likelihood_threshold 0.6
```

### DLC Data Format Requirements

Your DLC files should follow the standard DeepLabCut multi-animal format:

```python
# Column structure (MultiIndex)
df.columns = [
    ('scorer', 'individuals', 'bodyparts', 'coords'),
    # Example: ('DLC_model', 'mouse1', 'snout', 'x'),
    #          ('DLC_model', 'mouse1', 'snout', 'y'),
    #          ('DLC_model', 'mouse1', 'snout', 'likelihood'),
    #          ('DLC_model', 'mouse1', 'leftear', 'x'), ...
]

# Data example
df.iloc[0]  # First frame
# mouse1_snout_x: 202.05
# mouse1_snout_y: 243.02  
# mouse1_snout_likelihood: 0.993
# ...
```

### Corruption Detection Logic

StableMotion identifies corrupted frames using the following criteria:

```python
def is_frame_corrupted(frame_data, likelihood_threshold=0.6):
    """
    A frame is considered corrupted if ANY keypoint has:
    - Likelihood below threshold, OR
    - NaN/missing coordinates
    """
    for animal in animals:
        for bodypart in bodyparts:
            likelihood = frame_data[animal][bodypart]['likelihood']
            if likelihood < likelihood_threshold or np.isnan(likelihood):
                return True
    return False
```

### Feature Engineering for DLC Data

DLC keypoints are converted to motion features through several steps:

1. **Coordinate Normalization**: Align to reference animal and bodypart
2. **Temporal Features**: Compute velocities and accelerations
3. **Anatomical Features**: Calculate bone lengths and inter-keypoint distances
4. **Quality Labels**: Binary indicators for frame-level corruption

```python
# Feature breakdown for DLC data:
features = [
    reference_trajectory,      # 2D trajectory of reference keypoint
    trajectory_velocity,       # Temporal derivatives
    normalized_coordinates,    # All keypoint positions relative to reference
    coordinate_velocities,     # Temporal motion patterns
    bone_lengths,             # Inter-keypoint distances
    bone_velocities,          # Temporal bone length changes
    quality_labels            # Corruption indicators
]
# Total: Variable dimension based on number of animals/bodyparts
```

### Advanced DLC Usage

#### Custom Bodypart Groups
```python
# Define which bodyparts to use for reference alignment
reference_bodyparts = ['shoulder', 'spine', 'hip']  # Priority order

# Define bone connections for consistency constraints
bone_pairs = [
    ('snout', 'leftear'), ('snout', 'rightear'),
    ('shoulder', 'tailbase'), ('tailbase', 'tail1')
]
```

#### Ensemble Cleanup
```python
# Use multiple cleanup attempts for robust results
python -m sample.fix_dlc_keypoints \
  --model_path model.pt \
  --input_file data.csv \
  --output_file cleaned.csv \
  --ensemble True \
  --num_ensemble 5 \
  --normalizer_dir meta/
```

#### Evaluation Metrics
```python
from eval.dlc_metrics import evaluate_dlc_cleanup, generate_evaluation_report

# Evaluate cleanup quality
metrics = evaluate_dlc_cleanup(original_df, cleaned_df, likelihood_threshold=0.6)
report = generate_evaluation_report(metrics)

print(report)
# Output:
# - Acceleration reduction: 45.2%
# - Trajectory correlation: 0.892
# - Mean displacement: 3.4 pixels
# - ✅ Good cleanup quality
```

### DLC-Specific Corruption Types

StableMotion addresses common DLC artifacts:

- **Tracking Failures**: Sudden jumps when tracking is lost and recovered
- **Likelihood Drops**: Periods of low confidence with unreliable poses
- **Temporal Jitter**: High-frequency noise in tracking
- **Occlusion Interpolation**: Poor linear interpolation during occlusions
- **Coordinate Drift**: Systematic bias accumulation over time
- **Joint Swapping**: Occasional identity switches in multi-animal tracking

### Performance Considerations

- **Memory Usage**: Feature dimension scales with number of animals × bodyparts
- **Training Time**: Depends on sequence length and corruption complexity
- **Inference Speed**: ~0.1-1 seconds per sequence depending on length and ensemble settings

### Troubleshooting DLC Integration

**Common Issues:**

1. **Column Format**: Ensure DLC files have proper MultiIndex columns
2. **Missing Data**: Handle NaN values consistently across animals/bodyparts
3. **Feature Dimension**: Let the system auto-detect or specify explicitly
4. **Likelihood Threshold**: Adjust based on your pose estimation quality

**Solutions:**
```python
# Check DLC file format
df = pd.read_csv('your_file.csv', header=[0,1,2,3], index_col=0)
print(df.columns.names)  # Should be ['scorer', 'individuals', 'bodyparts', 'coords']

# Verify feature extraction
from data_loaders.dlc_keypoint_feats import dlc_dataframe_to_features
result = dlc_dataframe_to_features(df, likelihood_threshold=0.6)
print(f"Feature shape: {result['features'].shape}")
print(f"Animals: {result['metadata']['animal_names']}")
```

## Configuration Options

### Model Parameters
- `--layers`: Number of transformer layers (default: 8)
- `--heads`: Number of attention heads (default: 8)
- `--zero_init`: Zero-initialize output layers for stable training

### Diffusion Parameters
- `--diffusion_steps`: Number of diffusion timesteps (default: 50)
- `--noise_schedule`: Noise schedule type ('cosine' or 'linear')
- `--predict_xstart`: Predict x0 (clean data) directly

### Training Parameters
- `--batch_size`: Training batch size
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Adam weight decay
- `--gradient_clip`: Gradient clipping threshold
- `--model_ema`: Enable exponential moving averages

### Inference Parameters
- `--ProbDetNum`: Number of detection passes for averaging
- `--ProbDetTh`: Detection threshold for quality labels
- `--ensemble`: Enable ensemble cleanup
- `--enable_sits`: Soft inpainting timestep scheduling

## Notes for LLM Assistant

### Common Issues & Solutions
1. **SMPL Dependencies**: Ensure SMPL models are properly installed and paths are correct
2. **CUDA Memory**: Use gradient checkpointing and adjust batch size for GPU memory constraints
3. **Data Path Issues**: All paths should be absolute; check normalizer directory structure
4. **Evaluation Dependencies**: TMR models required for motion-text retrieval evaluation

### Extension Points
- **New Motion Representations**: Modify feature extraction in `amasstools/`
- **Custom Corruption Types**: Extend `dataset_utils.py` artifact functions
- **Alternative Architectures**: Replace transformer blocks in `stablemotion.py`
- **Additional Metrics**: Extend evaluation pipeline in `eval/`

### Performance Optimization
- Enable FlashAttention with `torch.backends.cuda.enable_flash_sdp(True)`
- Use mixed precision training with built-in GradScaler
- Optimize data loading with appropriate `num_workers`
- Consider model parallelism for very large models

This project represents a novel approach to motion cleanup that learns from naturally corrupted data without requiring clean references, making it highly practical for real-world mocap processing scenarios.
