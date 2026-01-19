import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../"))

import argparse, copy, zarr, os, pathlib, subprocess, pickle, platform, glob
from umift.processing.wired_time_sync import fetch_campose_gripper_time_json, fetch_ft_time_cvs, trim_ft_wrt_campose, fetch_image_time, fetch_ultrawide_image_time, fetch_depth_time, get_session_name
from umift.processing.zarr_dataset import  create_zarr_dataset_gripper_depth, slice_zarr_into_episodes_gripper_depth
from umift.processing.wired_time_sync import time_match_ft_campose

from umi_day.demonstration_processing.utils.generic_util import iterate_demonstrations
from umi_day.common.cv_util import get_image_transform_with_border

def is_headless():
    return not os.getenv("DISPLAY")  # If DISPLAY is not set, it's likely headless

if platform.system() == "Linux" and is_headless():
    # Set environment variables only for headless Linux systems
    print("[process_multimodal] Changing environment variables for headless systems.")
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["DISPLAY"] = ":0"

# Usage:
# (for ft, image, campose data)
# python scripts/process_multimodal.py 
# --ft_data_dir /Users/chuerpan/Documents/repo/umiFT/wired_collection/Python/Data/241119 
# --visual_data_dir /Users/chuerpan/Documents/repo/umiFT/data/umi_day_data/coinFT_20241119_meeting_room 
# --output_dir /Users/chuerpan/Documents/repo/umiFT/output/coinFT_20241119_meeting_room 
# --img_downscale 0.05 
# --plot   

# ASSUMPTIONS:
# - The input directories contain the data for a single demo session (with multiple demos, one gripper calibration video, one time sync with qr video).

def process_ft_img_campose_gripper(args):
    # read pose data from json file
    campose_dic, gripper_dic = fetch_campose_gripper_time_json(demo_data_dir = args.visual_data_dir, side=args.gripper_side, output_path=args.output_dir) # output json file

    # read ft data from csv file
    ft_dic = fetch_ft_time_cvs(args.ft_data_dir, output_path=args.output_dir) # output json file

    # make sure the ft data and campose data has the same length
    print("ft left data length: ", len(ft_dic['data']['left']))
    print("campose data length: ", len(campose_dic['data']))
    print("gripper data length: ", len(gripper_dic['data']))

    for ft_demo_idx in ft_dic['data']['left'].keys():
        # loop through each ft demo index since ft data is always saved
        ft_time_range = [ft_dic['data']['left'][ft_demo_idx]['ftTimeStamp'][0], ft_dic['data']['left'][ft_demo_idx]['ftTimeStamp'][-1]]
        campose_time_range = [campose_dic['data'][ft_demo_idx]['camPoseTimeStamp'][0], campose_dic['data'][ft_demo_idx]['camPoseTimeStamp'][-1]]

        if ft_time_range[0] > campose_time_range[1] or ft_time_range[1] < campose_time_range[0]:
            print(f"FT and Campose data do not match in time for demo {ft_demo_idx}. Please check the data.")
            print(f"FT time range: {ft_time_range}")
            print(f"FT left filename: {ft_dic['data']['left'][ft_demo_idx]['fileName']}")
            print(f"FT right filename: {ft_dic['data']['right'][ft_demo_idx]['fileName']}")
            print(f"Campose time range: {campose_time_range}")
            print(f"Campose filename: {campose_dic['data'][ft_demo_idx]['fileName']}")

            return None

    assert len(ft_dic['data']['left']) == len(campose_dic['data']), "FT and Campose data do not match in length. Please check the data."

    print(f'Fetching RGB images')
    image_dict_trimmed = fetch_image_time(image_data_dir=args.visual_data_dir, 
                                          campose_dic=campose_dic,
                                        output_path=args.output_dir, 
                                        end=-1, 
                                        every=1,  
                                        image_output_res=(args.output_rgb_w, args.output_rgb_h),  
                                        verbose=False,
                                        side = args.gripper_side)
    
    print(f'Fetching ultrawide images')
    uw_image_dict_trimmed = fetch_ultrawide_image_time(ultrawide_image_data_dir=args.visual_data_dir, 
                                          campose_dic=campose_dic,
                                        output_path=args.output_dir, 
                                        end=-1, 
                                        every=1,  
                                        image_output_res=(args.output_rgb_w, args.output_rgb_h),  
                                        verbose=False,
                                        side = args.gripper_side)
    
    print(f'Fetching depth images')
    depth_dict_trimmed = fetch_depth_time(depth_data_dir=args.visual_data_dir, 
                                          campose_dic=campose_dic,
                                        output_path=args.output_dir, 
                                        end=-1, 
                                        every=1,  
                                        depth_output_res=(args.output_rgb_w, args.output_rgb_h),  
                                        verbose=False,
                                        side = args.gripper_side)


    # aligned_data_dict = {'ft_dic_trimmed': ft_dic, 
    #              'campose_dic_trimmed': campose_dic, 
    #              'gripper_dic_trimmed': gripper_dic,
    #              'image_dict_trimmed': image_dict_trimmed}
    aligned_data_dict = {'ft_dic_trimmed': ft_dic, 
                 'campose_dic_trimmed': campose_dic, 
                 'gripper_dic_trimmed': gripper_dic,
                 'image_dict_trimmed': image_dict_trimmed,
                 'uw_image_dict_trimmed': uw_image_dict_trimmed,
                 'depth_dict_trimmed': depth_dict_trimmed}
 
    num_frames, expected_H, expected_W, chan = image_dict_trimmed['data'][0]['imgData'].shape
    
    return aligned_data_dict, expected_H, expected_W

def get_session_name(args):
    return args.visual_data_dir.split('/')[-3] # ['', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '0118-home-coinft-test-3', 'processed_data', 'gopro_iphone']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_data_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/wired_collection/Python/Data/241119', help='csv file name of coinFT raw data')
    parser.add_argument('--visual_data_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/data/umi_day_data/coinFT_20241119_meeting_room', help='coinFT data directory')
    parser.add_argument('--output_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/output/coinFT_20241119_meeting_room_gripper_test', help='camera trajectory file (.csv)')
    parser.add_argument('--output_rgb_w', type=int, default=224, help='image transform output width')
    parser.add_argument('--output_rgb_h', type=int, default=224, help='image transform output height')
    parser.add_argument('--calibration_dir', type=str, default='/store/real/hjchoi92/repo/UMI-FT/gripper_calibration', help='Path to calibration')
    parser.add_argument('--intermediate_umi_session_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/intermediate_umi_output/coinft_umi_folder', help='Path to calibration')
    parser.add_argument('--umi_submodule_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/submodules/universal_manipulation_interface', help='Path to calibration')
    parser.add_argument('--umi_day_submodule_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/submodules/umi_day', help='Path to calibration')
    parser.add_argument('--plot', action='store_true', help='plot the time sequences')
    parser.add_argument('--gripper_side', type=str, default='left', help='print debug info') 
    parser.add_argument('--plot_horizontal', action='store_true', help='plot the time sequences horizontally')
    args = parser.parse_args()
    
    ### process ft, image, campose data, and create time-aligned zarr dataset with data + timestamps
    pose_gripper_img_ft_dict, expected_H, expected_W = process_ft_img_campose_gripper(args)

    # image_data = pose_gripper_img_ft_dict['image_dict_trimmed']['data']
    # # policy_transform_im = get_image_transform_with_border(
    # #     in_res=(320, 240), out_res=(224, 224), bgr_to_rgb=True)
    # for key, value in image_data.items():
    #     print(f"Image {key} shape: {value['imgData'].shape}")
    #     img_pre = value['imgData'][0, ...]
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img_pre)
    #     ax[0].set_title('Original Image')
    #     plt.show()

    zarr_output_path = os.path.join(args.output_dir, 'replay_buffer_gripper.zarr')
    print(f'Creating zarr dataset, to: {zarr_output_path}')
    buffer = create_zarr_dataset_gripper_depth(pose_gripper_img_ft_dict, zarr_output_path, cam_id=0, gripper_id=0, expected_H=expected_H, expected_W=expected_W)

    # open and read zarr file
    zarr_data = zarr.open(zarr_output_path, mode='r')
    print('zarr_data.tree()')
    print(zarr_data.tree())
    
    acp_zarr_output_path = os.path.join(args.output_dir, 'acp_replay_buffer_gripper.zarr')
    slice_zarr_into_episodes_gripper_depth(input_zarr_path = zarr_output_path, output_zarr_path = acp_zarr_output_path)
    
    acp_zarr_data = zarr.open(acp_zarr_output_path, mode='r')
    print('acp_zarr_data.tree()')
    print(acp_zarr_data.tree())
    
    print('done')