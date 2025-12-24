# assumed input
# Demo directory
#   processed_{left/right}.mp4 (processed mp4, matching the campose in camera_trajectory.csv)
#   camera_trajectory.csv (campose from ARkit for GoPro)
#   coinFT data .csv (wired coinFT data with raw time step referenced to NTP)
#   TODO: incorpoate audio data

# output
# zarr file with the following structure
#  ├── data
#  │   ├── camera0_rgb (N, 224, 224, 3) uint8
#  │   ├── robot0_demo_end_pose (N, 6) float64
#  │   ├── robot0_demo_start_pose (N, 6) float64
#  │   ├── robot0_eef_pos (N, 3) float32 # eef translation
#  │   ├── robot0_eef_rot_axis_angle (N, 3) float32 # eef rotation
#  │   └── robot0_gripper_width (N, 1) float32 # width of the gripper
#  additional data
#  │   └── ft0 (N, 6) float32 # coinFT data (TODO)
#  │   └── time_ft (N, float32) # time step of coinFT data (TODO)
#  │   └── time_eef (N, float32) # time step of {eef, rgb, gripper} data (TODO)
#  │   └── time_audio (N, float32) # time step of audio data (TODO)

#  └── meta
#      └── episode_ends (5,) int64 

import argparse
import os
import numpy as np
import pandas

def load_csv_coinFT(file_path):
    camera_traj = pandas.read_csv(f"{args.input_path}/camera_trajectory.csv")
    cam_p = camera_traj[["x", "y", "z"]].to_numpy()
    cam_q = camera_traj[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coinft_input_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/data/data_wired_coinFT/coinFT-20241104', help='coinFT data directory')
    parser.add_argument('--coinft_data_file', type=str, default='result_241104_172718_first_multimodal.csv', help='specific coinFT data file (.csv)')
    parser.add_argument('--camera_trajectory_file', type=str, default='camera_trajectory.csv', help='camera trajectory file (.csv)')
    args = parser.parse_args()
    
    
    
 