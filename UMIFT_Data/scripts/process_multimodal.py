import sys
import os

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(SCRIPT_PATH, "../"))

import argparse, zarr, os, platform
from umift.processing.wired_time_sync import fetch_campose_gripper_time_json, fetch_ft_time_cvs, trim_ft_wrt_campose, fetch_image_time, fetch_ultrawide_image_time, fetch_depth_time, get_session_name
from umift.processing.zarr_dataset import  create_zarr_dataset_gripper_depth, slice_zarr_into_episodes_gripper_depth

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

    # Time-overlap matching: pair each FT demo with the first campose demo whose time range overlaps.
    # Both dicts are already sorted chronologically, so we use a single forward pointer for campose.
    ft_indices = sorted(ft_dic['data']['left'].keys())
    cam_indices = sorted(campose_dic['data'].keys())

    matched_pairs = []        # list of (ft_idx, cam_idx)
    unmatched_ft = []         # ft indices with no campose match
    unmatched_cam = []        # cam indices skipped before any match or left over

    cam_pointer = 0
    for ft_idx in ft_indices:
        ft_start = ft_dic['data']['left'][ft_idx]['ftTimeStamp'][0]
        ft_end   = ft_dic['data']['left'][ft_idx]['ftTimeStamp'][-1]
        matched = False

        while cam_pointer < len(cam_indices):
            cam_idx = cam_indices[cam_pointer]
            cam_start = campose_dic['data'][cam_idx]['camPoseTimeStamp'][0]
            cam_end   = campose_dic['data'][cam_idx]['camPoseTimeStamp'][-1]

            if cam_end < ft_start:
                # campose demo is entirely before this FT demo — skip it
                unmatched_cam.append(cam_idx)
                cam_pointer += 1
                continue

            if cam_start > ft_end:
                # campose demo is entirely after this FT demo — no match possible
                break

            # time ranges overlap
            matched_pairs.append((ft_idx, cam_idx))
            cam_pointer += 1
            matched = True
            break

        if not matched:
            unmatched_ft.append(ft_idx)

    # any remaining campose demos are unmatched
    for cam_idx in cam_indices[cam_pointer:]:
        unmatched_cam.append(cam_idx)

    # print matching summary
    print(f"\n=== Matching Summary ===")
    print(f"Matched: {len(matched_pairs)} demo pair(s)")
    if unmatched_ft:
        print(f"Unmatched FT demos ({len(unmatched_ft)}):")
        for ft_idx in unmatched_ft:
            ft_start = ft_dic['data']['left'][ft_idx]['ftTimeStamp'][0]
            ft_end   = ft_dic['data']['left'][ft_idx]['ftTimeStamp'][-1]
            print(f"  [ft {ft_idx}] time range: [{ft_start}, {ft_end}] | LF: {ft_dic['data']['left'][ft_idx]['fileName']} | RF: {ft_dic['data']['right'][ft_idx]['fileName']}")
    if unmatched_cam:
        print(f"Unmatched Campose demos ({len(unmatched_cam)}):")
        for cam_idx in unmatched_cam:
            cam_start = campose_dic['data'][cam_idx]['camPoseTimeStamp'][0]
            cam_end   = campose_dic['data'][cam_idx]['camPoseTimeStamp'][-1]
            print(f"  [cam {cam_idx}] time range: [{cam_start}, {cam_end}] | file: {campose_dic['data'][cam_idx]['fileName']}")
    print(f"========================\n")

    if not matched_pairs:
        print("No matched demo pairs found. Cannot proceed.")
        return None

    # rebuild dicts with contiguous indices using only matched pairs
    matched_ft_dic = {'meta': ft_dic['meta'], 'data': {'left': {}, 'right': {}}}
    matched_campose_dic = {'meta': campose_dic['meta'], 'data': {}}
    matched_gripper_dic = {'data': {}}
    for new_idx, (ft_idx, cam_idx) in enumerate(matched_pairs):
        matched_ft_dic['data']['left'][new_idx]  = ft_dic['data']['left'][ft_idx]
        matched_ft_dic['data']['right'][new_idx] = ft_dic['data']['right'][ft_idx]
        matched_campose_dic['data'][new_idx]     = campose_dic['data'][cam_idx]
        matched_gripper_dic['data'][new_idx]     = gripper_dic['data'][cam_idx]

    ft_dic       = matched_ft_dic
    campose_dic  = matched_campose_dic
    gripper_dic  = matched_gripper_dic

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



    # Reconcile: fetch functions may have skipped demos with corrupt/mismatched data.
    # Find the intersection of demo names across all modalities, then rebuild all dicts
    # with consistent sequential indices so downstream zarr creation stays aligned.
    def get_filenames(d, key='fileName'):
        return {v[key] for v in d['data'].values()}

    valid_names = (
        get_filenames(campose_dic, 'fileName')
        & get_filenames(image_dict_trimmed, 'fileName')
        & get_filenames(uw_image_dict_trimmed, 'fileName')
        & get_filenames(depth_dict_trimmed, 'fileName')
    )

    dropped = get_filenames(campose_dic, 'fileName') - valid_names
    if dropped:
        print(f"[WARNING] Dropping {len(dropped)} demo(s) missing from one or more modalities: {dropped}")

    def filter_and_reindex(d, name_key='fileName'):
        kept = sorted(
            [(k, v) for k, v in d['data'].items() if v[name_key] in valid_names],
            key=lambda x: x[0]
        )
        d['data'] = {new_idx: v for new_idx, (_, v) in enumerate(kept)}
        return d

    ft_dic['data']['left']  = {new_idx: v for new_idx, (_, v) in enumerate(
        sorted([(k, v) for k, v in ft_dic['data']['left'].items()
                if campose_dic['data'][k]['fileName'] in valid_names], key=lambda x: x[0]))}
    ft_dic['data']['right'] = {new_idx: v for new_idx, (_, v) in enumerate(
        sorted([(k, v) for k, v in ft_dic['data']['right'].items()
                if campose_dic['data'][k]['fileName'] in valid_names], key=lambda x: x[0]))}
    campose_dic  = filter_and_reindex(campose_dic,  'fileName')
    gripper_dic  = filter_and_reindex(gripper_dic,  'fileName')
    image_dict_trimmed       = filter_and_reindex(image_dict_trimmed,    'fileName')
    uw_image_dict_trimmed    = filter_and_reindex(uw_image_dict_trimmed, 'fileName')
    depth_dict_trimmed       = filter_and_reindex(depth_dict_trimmed,    'fileName')

    if not campose_dic['data']:
        print("No valid demos remain after modality reconciliation.")
        return None

    aligned_data_dict = {'ft_dic_trimmed': ft_dic,
                 'campose_dic_trimmed': campose_dic,
                 'gripper_dic_trimmed': gripper_dic,
                 'image_dict_trimmed': image_dict_trimmed,
                 'uw_image_dict_trimmed': uw_image_dict_trimmed,
                 'depth_dict_trimmed': depth_dict_trimmed}

    num_frames, expected_H, expected_W, chan = image_dict_trimmed['data'][0]['imgData'].shape

    return aligned_data_dict, expected_H, expected_W

def get_session_name(args):
    return args.visual_data_dir.split('/')[-3] # ['', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '250118-home-coinft-test-3', 'processed_data', 'gopro_iphone']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_data_dir', type=str, help='csv file name of coinFT raw data')
    parser.add_argument('--visual_data_dir', type=str, help='coinFT data directory')
    parser.add_argument('--output_dir', type=str, help='camera trajectory file (.csv)')
    parser.add_argument('--output_rgb_w', type=int, default=224, help='image transform output width')
    parser.add_argument('--output_rgb_h', type=int, default=224, help='image transform output height')
    parser.add_argument('--calibration_dir', type=str, help='Path to calibration')
    parser.add_argument('--intermediate_umi_session_dir', type=str, help='Path to calibration')
    parser.add_argument('--umi_submodule_dir', type=str, help='Path to calibration')
    parser.add_argument('--umi_day_submodule_dir', type=str, help='Path to calibration')
    parser.add_argument('--plot', action='store_true', help='plot the time sequences')
    parser.add_argument('--gripper_side', type=str, help='print debug info') 
    parser.add_argument('--plot_horizontal', action='store_true', help='plot the time sequences horizontally')
    args = parser.parse_args()
    
    # process ft, image, campose data, and create time-aligned zarr dataset with data + timestamps
    pose_gripper_img_ft_dict, expected_H, expected_W = process_ft_img_campose_gripper(args)
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