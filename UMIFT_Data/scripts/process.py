# assumed input
# Demo directory
#   processed_{left/right}.mp4 (processed mp4, matching the campose in camera_trajectory.csv)
#   camera_trajectory.csv (campose from ARkit for GoPro)
#   coinFT data .csv (wired coinFT data with raw time step referenced to NTP)

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
    
    
    
 