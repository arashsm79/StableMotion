import cv2
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Visualize motion reconstruction comparison')
    parser.add_argument('--res', required=True, help='Path to results.npy file')
    parser.add_argument('--output_dir', required=True, help='Output directory for videos')
    parser.add_argument('--height', type=int, default=800, help='Height of output video (default: 800)')
    
    args = parser.parse_args()
    
    height = args.height
    width = height * 2
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = np.load(args.res, allow_pickle=True).item()
    
    for i in range(len(results['motion'])):
        motion_data = results['motion'][i]
        motion_fix_data = results['motion_fix'][i]
        label_data = results['label'][i]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(args.output_dir, f'keypoints_comparison_{i}.mp4')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        num_frames = len(motion_data)

        scorer = motion_data.columns.get_level_values('scorer')[0]
        bodyparts = motion_data.columns.get_level_values('bodyparts').unique()

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
                (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                (192, 192, 192), (255, 165, 0), (255, 192, 203), (173, 216, 230), (144, 238, 144),
                (221, 160, 221), (255, 218, 185), (205, 133, 63), (72, 61, 139), (47, 79, 79)]

        bodypart_colors = {bodypart: colors[i % len(colors)] for i, bodypart in enumerate(bodyparts)}

        for frame_idx in range(num_frames):
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            motion_frame = motion_data.iloc[frame_idx]
            motion_fix_frame = motion_fix_data.iloc[frame_idx]
            
            for bodypart in bodyparts:
                motion_x = motion_frame[scorer, bodypart, 'x']
                motion_y = motion_frame[scorer, bodypart, 'y']
                motion_likelihood = motion_frame[scorer, bodypart, 'likelihood']
                
                motion_fix_x = motion_fix_frame[scorer, bodypart, 'x']
                motion_fix_y = motion_fix_frame[scorer, bodypart, 'y']
                motion_fix_likelihood = motion_fix_frame[scorer, bodypart, 'likelihood']
                
                color = bodypart_colors[bodypart]
                
                cv2.circle(frame, (int(motion_x), int(motion_y)), 5, color, -1)
                
                cv2.circle(frame, (int(motion_fix_x + height), int(motion_fix_y)), 5, color, -1)
            cv2.putText(frame, f'Label: {label_data[frame_idx]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.line(frame, (height, 0), (height, height), (0, 0, 0), 2)
            cv2.putText(frame, f'Frame: {frame_idx}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            out.write(frame)

        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()