# Wired Time Sync Utilities
import numpy as np
import pandas as pd
import concurrent.futures
from umift.utils.time_utils import convert_timestamp_to_iso_processed, isostringformat_to_timestamp, array_isostringformat_to_timestamp, array_ntp_time_to_timestamp, convert_timestamp_to_iso_z_format
from umift.utils.print_utils import color_print, debug_print, info_print
from umift.utils.json_utils import save_ft_data, load_visual_data_json, get_demonstration_dirs, save_campose_data, save_json
from umift.utils.depth_utils import load_depth, get_depth_transform_with_border, stack_depth_channels
from umi_day.demonstration_processing.utils.gripper_util import get_demo_gripper_width, iphone_to_tcp_poses
from umi_day.common.cv_util import get_image_transform_with_border
from umi_day.common.timecode_util import datetime_fromisoformat
from umi.common.pose_util import pose_to_mat
from PyriteUtility.spatial_math import spatial_utilities as su


from pathlib import Path
import cv2, yaml, tqdm, os

def reorder_dict_helper(sorted_demos, data_dict):
    reorder_dict = {}
    for order_idx, demo in enumerate(sorted_demos):
        filename = demo['filename']
        reorder_dict[filename] = {
            'filename': filename,
            'start_time': demo['start_time'],
            'order_idx': order_idx
        }

    reorg_data_dict = {}
    for key, value in reorder_dict.items():
        reorg_data_dict[value['order_idx']] = data_dict[key]
        """
            key: demo index (int), 
            value: 
                'camPoseTimeStamp', N, str
                'camPoseData', N, 6, float64
                'fileName', str of file_name before .csv
        """
    return reorg_data_dict

def fetch_ft_time_cvs(ft_data_dir, output_path):
    """
    Args:   
        ft_data_dir: str, path to the directory containing csv files of coinFT raw data
        output_path: str, path to the directory where to save the JSON files
    Return:
        all_data: dict, a dictionary containing data from all CSV files where:
            meta: dict, a dictionary containing the session name
                'session_name': str of session name,
                    eg. '241119'
            data: dict, a dictionary containing data from all CSV files where:
                first level key: filename (without extension)
                second level keys: 
                    'ftTimeStamp': (N,), str, # processed iso format, eg. 2024-11-19T23:52:14.671083
                    'ftData' (N, 6), float64
                    'fileName': str of file_name before .csv. 
                        eg., 'result_241119_155152_20241119_meeting_room'

    """
    all_data = {}
    ft_data_dic = {}
    meta_dic = {}

    # Convert input path to Path object for easier handling
    data_path = Path(ft_data_dir)
    
    # Find all CSV files in the directory (recursively search subdirectories)
    csv_files = list(data_path.glob('**/*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {ft_data_dir}")

    # Get session name from path (handle both direct files and files in date subdirectories)
    # eg: coinft/260130/UMIFT_data_260130_120326_try2_LF.csv -> session is 2 levels up from coinft
    session_name = csv_files[0].parts[-4] if len(csv_files[0].parts) >= 4 else csv_files[0].parts[-3]
    color_print(f'csv_file: {csv_files}', color = 'cyan')

    for csv_file in csv_files:
        file_key = csv_file.stem
        ft_data_dic[file_key] = fetch_ft_time_cvs_file_level(csv_file)
    
    ft_data_dic = reorg_demo_data_ordered_by_time_ft(ft_data_dic, data_prefix='ft')
    
    meta_dic['session_name'] = session_name
    all_data['meta'] = meta_dic
    all_data['data'] = ft_data_dic
    save_ft_data(all_data, output_path, stage = 'pre_trim')
    return all_data
             
def fetch_ft_time_cvs_file_level(ft_data_file):
    """
    Args:   
    ft_data_dir: str, path to the csv file of coinFT raw data
    Return:
    dic: dict, a dictionary containing:
        key, values.shape
        ftTimeStamp, (N,), dtype: str
        ftData, (N,6), dtype: float64
    """
    file_key = ft_data_file.stem
    
    dic = {}
    df = pd.read_csv(ft_data_file)
    
    # Rename only the first 7 columns for clarity, keep the rest unchanged
    new_column_names = ['timestamp', 'force_x', 'force_y', 'force_z', 
                        'torque_x', 'torque_y', 'torque_z']
    num_columns_to_rename = min(len(new_column_names), len(df.columns))
    
    # Apply renaming
    df.columns = new_column_names[:num_columns_to_rename] + list(df.columns[num_columns_to_rename:])
    
    # Convert timestamps - first ensure they're floats, then convert to ISO format
    timestamps = df['timestamp'].apply(lambda x: convert_timestamp_to_iso_processed(float(x)))
    timeframe = timestamps.to_numpy()
    
    # Extract force-torque data in the specified sequence
    ft_columns = ['force_x', 'force_y', 'force_z', 
                 'torque_x', 'torque_y', 'torque_z']
    ft_data = df[ft_columns].to_numpy()
    
    color_print(f'-'*100, color = 'cyan')
    color_print(f'ft_data.shape: {ft_data.shape}, ft_data.dtype: {ft_data.dtype}')
    color_print(f'timeframe.shape: {timeframe.shape}, timeframe.dtype: {timeframe.dtype}')
    color_print(f'file_key: {file_key}')
    
    dic['ftTimeStamp'] = timeframe
    dic['ftData'] = ft_data
    dic['fileName'] = file_key
    
    # ft_data.shape: (10757, 6), ft_data.dtype: float64
    # timestamp.shape: (10757,), timeframe.dtype: object
    # file_key: result_241119_155152_20241119_meeting_room
    return dic

def get_start_time(timestamp_arr):
    """
    Args:
        timestamp_arr: np.array, array of timestamps
    Returns:
        start_time: float, start time of the array, timestamp format
    """
    return isostringformat_to_timestamp(timestamp_arr[0])

def reorg_demo_data_ordered_by_time(data_dict, data_prefix='camPose'):
    """
    Args:
        data_dict: dict, a dictionary containing data from all CSV files where:
            first level key: filename (without extension)
            second level keys: 
                '{data_prefix}TimeStamp', (N,), str
                '{data_prefix}Data', (N, 6), float64
                'fileName', str of file_name before .csv or .json 
        data_prefix: str, prefix of the data type. Default: 'camPose'
    """
    reorder_dict = {}
    
    temp_list = []
    for key, value in data_dict.items():
        if (data_prefix == 'ft' and key.endswith('_LF')) or data_prefix != 'ft': # TODO: remove this after debug
            temp_dict = {
                'filename': key,
                'start_time': get_start_time(value[f"{data_prefix}TimeStamp"])
            }
            temp_list.append(temp_dict)

  
    sorted_demos = sorted(temp_list, key = lambda x: x['start_time'])
    reorder_dict = {}
    
    for order_idx, demo in enumerate(sorted_demos):
        filename = demo['filename']
        reorder_dict[filename] = {
            'filename': filename,
            'start_time': demo['start_time'],
            'order_idx': order_idx
        }
    
    reorg_data_dict = {}
    for key, value in reorder_dict.items():
        reorg_data_dict[value['order_idx']] = data_dict[key]
        """
            key: demo index (int), 
            value: 
                'camPoseTimeStamp', N, str
                'camPoseData', N, 6, float64
                'fileName', str of file_name before .csv
        """
 
    return reorg_data_dict

def reorg_demo_data_ordered_by_time_ft(data_dict, data_prefix='ft'):
    
    """
    For two coinfts, *_LF.csv and *_RF.csv for left and right coinfts respectively.
    Args:
        data_dict: dict, a dictionary containing data from all CSV files where:
            first level key: filename (without extension)
            second level keys: 
                '{data_prefix}TimeStamp', (N,), str
                '{data_prefix}Data', (N, 6), float64
                'fileName', str of file_name before .csv or .json 
        data_prefix: str, prefix of the data type. Default: 'ft'
    """
    assert data_prefix == 'ft', f'Currently only support ft data, but got {data_prefix}'
    reorder_dict = {}
    
    temp_list_left = []
    temp_list_right = []
    
    LF_count = 0
    RF_count = 0
    for key, value in data_dict.items():
        # Hack for FT data: add Z to the end of the timestamp to make it UTC
        value[f"{data_prefix}TimeStamp"] += 'Z'  # add Z to the end of the timestamp

        if key.endswith('_LF'):
            temp_dict_left = {
                'filename': key,
                'start_time': get_start_time(value[f"{data_prefix}TimeStamp"])
            }
            temp_list_left.append(temp_dict_left)
            LF_count += 1 
        elif key.endswith('_RF'):
            temp_dict_right = {
                'filename': key,
                'start_time': get_start_time(value[f"{data_prefix}TimeStamp"])
            }
            temp_list_right.append(temp_dict_right)
            RF_count += 1  
    assert LF_count == RF_count, f'LF_count: {LF_count}, RF_count: {RF_count}'

  
    sorted_demos_left = sorted(temp_list_left, key = lambda x: x['start_time'])
    sorted_demos_right = sorted(temp_list_right, key = lambda x: x['start_time'])
    
    reorg_data_dict_left = reorder_dict_helper(sorted_demos_left, data_dict)
    reorg_data_dict_right = reorder_dict_helper(sorted_demos_right, data_dict)
    
   
    reorg_data_two_coinfts = {'left': reorg_data_dict_left, 'right': reorg_data_dict_right}
    return reorg_data_two_coinfts
    
def fetch_campose_gripper_time_json(demo_data_dir, output_path, side = 'right'):
    """
    Args:   
        image_data_dir: str, path to the directory containing the demo directories. eg. X, where X/*_demonstration/processed_*.json
    Return:
        campose_dic: dict, a dictionary containing:
            key, values.shape
            camPoseTimeStamp, (N,), dtype: str
            camPoseData, (N,6), dtype: float64
    """
    image_data_dir = Path(demo_data_dir)
    demo_dirs = get_demonstration_dirs(image_data_dir)
    session_name = image_data_dir.parts[-3] # eg. ('/', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '0118-home-coinft-test-3', 'processed_data', 'gopro_iphone') should get '0118-home-coinft-test-3'
    
    campose_dict = {}
    campose_data_dict = {}
    meta_data_dict = {}
    
    gripper_dict = {}
    gripper_data_dict = {}
    
    for demo_dir in demo_dirs:
        demo_campose_dict, demo_gripper_dict = fetch_campose_gripper_time_demo_level(demo_dir, side=side)
        campose_data_dict[demo_campose_dict['fileName']] = demo_campose_dict
        gripper_data_dict[demo_gripper_dict['fileName']] = demo_gripper_dict
    
    campose_data_dict = reorg_demo_data_ordered_by_time(campose_data_dict, data_prefix='camPose')
    gripper_data_dict = reorg_demo_data_ordered_by_time(gripper_data_dict, data_prefix='gripper')

    meta_data_dict['session_name'] = session_name
    campose_dict['meta'] = meta_data_dict
    campose_dict['data'] = campose_data_dict
    
    # gripper_dict['meta'] = meta_data_dict
    gripper_dict['data'] = gripper_data_dict
    
    # # reorg the data such that the key is 'index' and the value dictionary, while file name 
    # save_campose_data(campose_dict, output_path, stage = 'pre_trim')
    return campose_dict, gripper_dict

def fetch_campose_gripper_time_demo_level(demo_dir, side = 'right'):
    """
    Args:   
        demo_dir: str, path to the directory containing image files
    Return:
        campose_dic: dict, a dictionary containing:
            key, values.shape
            camPoseTimeStamp, (N,), dtype: str
            camPoseData, (N,6), dtype: float64
    """
    demo_dir = Path(demo_dir)
    demo_name = demo_dir.parts[-1]  # eg. '2025-01-19T06-01-27.409Z_85713_0118-home-coinft-test-3_demonstration'
    json_data, found = load_visual_data_json(demo_dir, side=side)
    gripper_widths = get_demo_gripper_width(demo_dir, side)
    
    # TODO: right now assume single-hand data, ie. side = {'right', 'left'}. Come back to this to extend to handle bimaual
    
    # JSON DATA FORMAT
    # { 
    #   times: [timestamp1, timestamp2, ...], (timestamp1: str) # processed iso format, eg. 2024-11-19T23:52:14.671083
    #   poseTransforms: [pose1, pose2, ...], (pose1: list of of list for 4x4 SE3 matrix)
    # }
    
    # convert to numpy array, and change the naming convention to match across modality,
    # '{dataType}TimeStamp' (eg. camPoseTimeStamp)
    # '{dataType}Data'  (eg. camPoseData)

    # SE3 matrix to convert from iphone frame to (imaginary) gopro frame
    # I_T_G = I_T_TCP @ inv(G_T_TCP)

    print("fetching campose data W_T_G; W: ARKit, G: GoPro.")
    campose_dic = {}
    campose_dic['camPoseTimeStamp']= np.array(json_data['poseTimes'])
    # campose_dic['camPoseData'] = np.array(json_data['poseTransforms'])
    campose_dic['camPoseData'] = convert_pose_iphone_to_gopro_wrt_arkit(np.array(json_data['poseTransforms']))
    campose_dic['fileName'] = demo_name
    
    gripper_dic = {}
    gripper_dic['gripperTimeStamp'] = np.array(json_data['poseTimes'])
    gripper_dic['gripperData'] = np.array(gripper_widths)
    gripper_dic['fileName'] = demo_name
 
    return campose_dic, gripper_dic

def fetch_image_time_demo_level(mp4_file_dir, end=-1, every=1, image_output_res=(224,224), max_workers = 16, verbose=False, side='right'):
    """
    Read the mp4 file and optionally downsample the video to images.

    Args:
        mp4_file_dir: str, path to the mp4 file
        output_dir: str, path to the directory where to save the images
        every: int, save every `every` frame for downsample. Default: 1
        end: int, end frame index. Default: -1, indicate the end of the video
        downscale: int, downscale factor for image. Default: 1
        verbose: bool, print out the downscaled image shape. Default: False
    Return:
        img_data_demo_dict: dict, a dictionary containing:
            key = fileName 
            imgData, (N, H, W, C), dtype: uint8
            fileName, str of file_name before .mp4
            processing_meta: dict, a dictionary containing:
                downscale: int, downscale factor for image
                every: int, save every `every` frame for downsample
                end: int, end frame index
                trimmed_start: int, start frame index
                trimmed_end: int, end
    """
    img_data_demo_dict = {}
 
    file_name = mp4_file_dir.parts[-1]
    # eg. ('/', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '0118-home-coinft-test-3', 'processed_data', 'gopro_iphone', '2025-01-19', '2025-01-19T06-01-27.409Z_85713_0118-home-coinft-test-3_demonstration')

    mp4_file = os.path.join(mp4_file_dir, f'{side}_rgb.mp4')
    assert os.path.exists(mp4_file), f'{mp4_file} does not exist'

    # grab the *.mp4 file under this dir that are right.mp4 or left.mp4
    info_print(f'mp4_file :{mp4_file}')
    
    cap = cv2.VideoCapture(mp4_file)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    info_print(f'Starting to load video from Frame 0 to Frame {frame_end} every {every} frame')
    # Downsample the image to a resolution close to the desired output image for speed.
    # For {side}_rgb.mp4, this would convert 1920 x 1440 to 358 x 268.
    resize_factor = 1.2
    downsample_factor = (image_output_res[0] * resize_factor) / min(raw_width, raw_height)
    temp_width = int(raw_width * downsample_factor)
    temp_height = int(raw_height * downsample_factor)
    
    mp4_frames = np.zeros((frame_end, temp_height, temp_width, 3), dtype=np.uint8)
    for i in tqdm.tqdm(range(0, frame_end)):
        ret = cap.grab()
        if not ret:
            break
        ret, frame = cap.retrieve()
        if ret:
            frame = cv2.resize(frame, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)
            mp4_frames[i, ...] = frame
    cap.release()
    
    transform_im = get_image_transform_with_border(
        in_res=(temp_width, temp_height), out_res=image_output_res, bgr_to_rgb=True)
    # processing
    processed_frames = np.zeros((len(mp4_frames), image_output_res[1], image_output_res[0], 3), dtype=np.uint8)
    
    def process_frame(frames_in, frames_out, i):
        frames_out[i] = transform_im(frames_in[i])
        return True
        
    print(f'Start to process video with {max_workers} threads: ')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(len(mp4_frames)):
            futures.add(
                executor.submit(
                    process_frame,
                    mp4_frames,
                    processed_frames,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to read image!")

    img_data_demo_dict = {}
    img_data_demo_dict['imgData'] = np.array(processed_frames)
    img_data_demo_dict['fileName'] = file_name

    return img_data_demo_dict


# def fetch_campose_gripper_uw_time_demo_level(demo_dir, side = 'right'):
#     """
#     Args:   
#         demo_dir: str, path to the directory containing image files
#     Return:
#         campose_dic: dict, a dictionary containing:
#             key, values.shape
#             camPoseTimeStamp, (N,), dtype: str
#             camPoseData, (N,6), dtype: float64
#     """
#     demo_dir = Path(demo_dir)
#     demo_name = demo_dir.parts[-1]  # eg. '2025-01-19T06-01-27.409Z_85713_0118-home-coinft-test-3_demonstration'
#     json_data, found = load_visual_data_json(demo_dir, side=side)
#     gripper_widths = get_demo_gripper_width(demo_dir, side)
    
#     # TODO: right now assume single-hand data, ie. side = {'right', 'left'}. Come back to this to extend to handle bimaual
    
#     # JSON DATA FORMAT
#     # { 
#     #   times: [timestamp1, timestamp2, ...], (timestamp1: str) # processed iso format, eg. 2024-11-19T23:52:14.671083
#     #   poseTransforms: [pose1, pose2, ...], (pose1: list of of list for 4x4 SE3 matrix)
#     # }
    
#     # convert to numpy array, and change the naming convention to match across modality,
#     # '{dataType}TimeStamp' (eg. camPoseTimeStamp)
#     # '{dataType}Data'  (eg. camPoseData)
    
#     campose_dic = {}
#     campose_dic['camPoseTimeStamp']= np.array(json_data['poseTimes'])
#     campose_dic['camPoseData'] = np.array(json_data['poseTransforms'])
#     campose_dic['fileName'] = demo_name
    
#     gripper_dic = {}
#     gripper_dic['gripperTimeStamp'] = np.array(json_data['poseTimes'])
#     gripper_dic['gripperData'] = np.array(gripper_widths)
#     gripper_dic['fileName'] = demo_name



 
#     return campose_dic, gripper_dic

def fetch_ultrawide_image_time_demo_level(mp4_file_dir, end=-1, every=1, image_output_res=(224,224), max_workers = 16, verbose=False, side='right'):
    """
    Read the mp4 file of rgb_ultrawide and optionally downsample the video to images.

    Args:
        mp4_file_dir: str, path to the mp4 file
        output_dir: str, path to the directory where to save the images
        every: int, save every `every` frame for downsample. Default: 1
        end: int, end frame index. Default: -1, indicate the end of the video
        downscale: int, downscale factor for image. Default: 1
        verbose: bool, print out the downscaled image shape. Default: False
    Return:
        img_data_demo_dict: dict, a dictionary containing:
            key = fileName 
            imgData, (N, H, W, C), dtype: uint8
            fileName, str of file_name before .mp4
            processing_meta: dict, a dictionary containing:
                downscale: int, downscale factor for image
                every: int, save every `every` frame for downsample
                end: int, end frame index
                trimmed_start: int, start frame index
                trimmed_end: int, end
    """
    uw_img_data_demo_dict = {}
 
    file_name = mp4_file_dir.parts[-1]
    # eg. ('/', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '0118-home-coinft-test-3', 'processed_data', 'gopro_iphone', '2025-01-19', '2025-01-19T06-01-27.409Z_85713_0118-home-coinft-test-3_demonstration')

    mp4_file = os.path.join(mp4_file_dir, f'{side}_ultrawidergb.mp4')
    assert os.path.exists(mp4_file), f'{mp4_file} does not exist'

    # grab the *.mp4 file under this dir that are right.mp4 or left.mp4
    info_print(f'mp4_file :{mp4_file}')
    
    cap = cv2.VideoCapture(mp4_file)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    info_print(f'Starting to load video from Frame 0 to Frame {frame_end} every {every} frame')
    # Downsample the image to a resolution close, but slightly larger to the desired output image for speed.
    # For {side}_rgb.mp4, this would convert 640 x 480 to 358 x 268.
    resize_factor = 1.2
    downsample_factor = (image_output_res[0] * resize_factor) / min(raw_width, raw_height)
    temp_width = int(raw_width * downsample_factor)
    temp_height = int(raw_height * downsample_factor)
    
    mp4_frames = np.zeros((frame_end, temp_height, temp_width, 3), dtype=np.uint8)
    for i in tqdm.tqdm(range(0, frame_end)):
        ret = cap.grab()
        if not ret:
            break
        ret, frame = cap.retrieve()
        if ret:
            frame = cv2.resize(frame, (temp_width, temp_height), interpolation=cv2.INTER_LINEAR)
            mp4_frames[i, ...] = frame
    cap.release()
    
    transform_im = get_image_transform_with_border(
        in_res=(temp_width, temp_height), out_res=image_output_res, bgr_to_rgb=True)
    # processing
    processed_frames = np.zeros((len(mp4_frames), image_output_res[1], image_output_res[0], 3), dtype=np.uint8)
    
    def process_frame(frames_in, frames_out, i):
        frames_out[i] = transform_im(frames_in[i])
        return True
        
    print(f'Start to process video with {max_workers} threads: ')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(len(mp4_frames)):
            futures.add(
                executor.submit(
                    process_frame,
                    mp4_frames,
                    processed_frames,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to read image!")

    uw_img_data_demo_dict = {}
    uw_img_data_demo_dict['uwImgData'] = np.array(processed_frames)
    uw_img_data_demo_dict['fileName'] = file_name

    return uw_img_data_demo_dict


def fetch_depth_time_map_idx_demo_level(raw_file_dir, end=-1, every=1, depth_output_res=(224,224), max_workers = 16, verbose=False, side='right'):
    """
    Read the raw file of depth, resize them to depth_output_res, and copy values across three channels.
    The depth data is downsampled to 30Hz (collected at 60Hz) because during inference depth is streamed at 30Hz.

    Args:
        raw_file_dir: str, path to the raw file
        output_dir: str, path to the directory where to save the images
        every: int, save every `every` frame for downsample. Default: 1
        end: int, end frame index. Default: -1, indicate the end of the video
        downscale: int, downscale factor for image. Default: 1
        verbose: bool, print out the downscaled image shape. Default: False
    Return:
        depth_data_demo_dict: dict, a dictionary containing:
            key = fileName 
            depthData, (N, H, W, C), dtype: float16
            fileName, str of file_name before .raw
            processing_meta: dict, a dictionary containing:
                downscale: int, downscale factor for image
                every: int, save every `every` frame for downsample
                end: int, end frame index
                trimmed_start: int, start frame index
                trimmed_end: int, end
    """
 
    file_name = raw_file_dir.parts[-1]
    # eg. ('/', 'Users', 'chuerpan', 'Documents', 'repo', 'umiFT', 'data', 'tmp_data', '0118-home-coinft-test-3', 'processed_data', 'gopro_iphone', '2025-01-19', '2025-01-19T06-01-27.409Z_85713_0118-home-coinft-test-3_demonstration')

    raw_file = os.path.join(raw_file_dir, f'{side}_depth.raw')
    if not os.path.exists(raw_file):
        print(f'Warning: depth file not found, skipping depth for: {raw_file}')
        return None

    info_print(f'raw_depth_file :{raw_file}')
    depth_array = load_depth(raw_file)
    depth_length = len(depth_array)
    H, W = depth_array[0].shape

    depth_transform = get_depth_transform_with_border(in_res=(H, W), out_res=depth_output_res)

    processed_depth_array = np.zeros((depth_length, depth_output_res[0], depth_output_res[1]), dtype=np.float16)

    def process_depth_frame(frames_in, frames_out, i):
        frames_out[i] = depth_transform(frames_in[i])
        return True
        
    print(f'Start to process depth values with {max_workers} threads: ')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        for i in range(depth_length):
            futures.add(
                executor.submit(
                    process_depth_frame,
                    depth_array,
                    processed_depth_array,
                    i,
                )
            )
        completed, futures = concurrent.futures.wait(futures)
        for f in completed:
            if not f.result():
                raise RuntimeError("Failed to read image!")
            
    processed_depth_frames = stack_depth_channels(processed_depth_array)

    # Create an index mapping variable to get the depth data streamed at 30Hz at deployment time when using indices based on 60Hz rgb data. 
    # The depth values are later downsampled to 30Hz. 
    map_to_depth_idx = np.arange(depth_length // 2).repeat(2)
    if len(map_to_depth_idx) < depth_length:
        map_to_depth_idx = np.append(map_to_depth_idx, map_to_depth_idx[-1] + 1)

    depth_data_demo_dict = {}
    depth_data_demo_dict['depthData'] = np.array(processed_depth_frames)
    depth_data_demo_dict['fileName'] = file_name
    depth_data_demo_dict['rgbToDepthIdx'] = map_to_depth_idx

    return depth_data_demo_dict


def find_key_and_timestamp_by_demo(demo_name, dic):
    for key, value in dic['data'].items():
        if value['fileName'] == demo_name:
            return key, value
    return None, None

def fetch_image_time(
    image_data_dir, 
    campose_dic, 
    # ft_dic, 
    # trimming_meta_dict, 
    output_path = None, 
    end = -1, 
    every = 1, 
    image_output_res = (224, 224), 
    verbose = False, 
    side = 'right'
):
    """
    Args:   
        image_data_dir: str, path to the directory containing mp4_file
        ...: other args
    Return:
        image_dict: dict, a dictionary containing:
            data
                demo_idx, int, eg. 0
                    imgTimeStamp, (N,), dtype: str
                    imgData, (N, H, W, C), dtype: uint8
                    fileName, str of file_name before .mp4
            meta
                'session_name', session_name, str, eg. '241119'
            
    """
    demo_dirs = get_demonstration_dirs(image_data_dir)
    session_name = Path(image_data_dir).parts[-3]
    
    image_dict = {}
    image_data_dict = {}
    meta_data_dict = {}

    for mp4_file_dir in demo_dirs:
        img_dict_demo = fetch_image_time_demo_level(mp4_file_dir = mp4_file_dir, 
                                                    end = end, 
                                                    every = every, 
                                                    image_output_res = image_output_res, 
                                                    verbose = False, 
                                                    side = side)
        image_data_dict[img_dict_demo['fileName']] = img_dict_demo

    image_dict['data'] = image_data_dict
    meta_data_dict['session_name'] = session_name
    image_dict['meta'] = meta_data_dict
    
    for demo_name, _ in image_dict['data'].items():
        print(f"demo name: {demo_name}")
        matched_demo_dict_key, matched_demo_dict = find_key_and_timestamp_by_demo(demo_name=demo_name, dic = campose_dic)
        
        if matched_demo_dict_key is None:
            print(f"Error: image demo name: {demo_name} not found in campose_dic")
            exit(0)
        assert image_data_dict[demo_name]['imgData'].shape[0] == campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp'].shape[0],\
            f'imgData.shape[0]: {image_data_dict[demo_name]["imgData"].shape[0]}, camPoseTimeStamp.shape[0]: {campose_dic["data"][matched_demo_dict_key]["camPoseTimeStamp"].shape[0]}'
        image_data_dict[demo_name]['imgTimeStamp'] = campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp']

    image_data_dict = reorg_demo_data_ordered_by_time(image_data_dict, data_prefix = 'img')
 
    image_dict['data'] = image_data_dict
    
    return image_dict

def fetch_ultrawide_image_time(
    ultrawide_image_data_dir, 
    campose_dic, 
    output_path = None, 
    end = -1, 
    every = 1, 
    image_output_res = (224, 224), 
    verbose = False, 
    side = 'right'
):
    """
    Args:   
        image_data_ultrawide_dir: str, path to the directory containing mp4_file of the ultrawide video
        ...: other args
    Return:
        image_dict: dict, a dictionary containing:
            data
                demo_idx, int, eg. 0 (in temporal order)
                    uwImgTimeStamp, (N_uwRGB,), dtype: str
                    imgData, (N_nwRGB, H, W, C), dtype: uint8
                    fileName, str of file_name before .mp4
                    mainToUltrawideIdx, (N_rgb,), dtype: uint16
            meta
                'session_name', session_name, str, eg. '241119'
            
    """
    demo_dirs = get_demonstration_dirs(ultrawide_image_data_dir)
    session_name = Path(ultrawide_image_data_dir).parts[-3]
    
    uw_image_dict = {}
    uw_image_data_dict = {}
    meta_data_dict = {}

    for mp4_file_dir in demo_dirs:
        uw_img_dict_demo = fetch_ultrawide_image_time_demo_level(mp4_file_dir = mp4_file_dir, 
                                                    end = end, 
                                                    every = every, 
                                                    image_output_res = image_output_res, 
                                                    verbose = False, 
                                                    side = side)
        uw_image_data_dict[uw_img_dict_demo['fileName']] = uw_img_dict_demo

        rgb_to_uw_index_demo, uw_img_timestamp_demo = get_rgb_to_uw_index_and_time_demo_level(mp4_file_dir = mp4_file_dir, side = side)
        uw_image_data_dict[uw_img_dict_demo['fileName']]['rgbToUWIdx'] = rgb_to_uw_index_demo # length should be same as campose
        uw_image_data_dict[uw_img_dict_demo['fileName']]['uwImgTimeStamp'] = uw_img_timestamp_demo # length should be same as campose. The entry will be the actual timestep at 10Hz and 0 otherwise. 

    uw_image_dict['data'] = uw_image_data_dict
    meta_data_dict['session_name'] = session_name
    uw_image_dict['meta'] = meta_data_dict
    
    for demo_name, _ in uw_image_dict['data'].items():
        print(f"demo name: {demo_name}")
    
        matched_demo_dict_key, matched_demo_dict = find_key_and_timestamp_by_demo(demo_name=demo_name, dic = campose_dic)

        if matched_demo_dict_key is None:
            print(f"Error: image demo name: {demo_name} not found in campose_dic")
            exit(0)
        assert uw_image_data_dict[demo_name]['rgbToUWIdx'].shape[0] == campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp'].shape[0],\
            f'rgbToUWIdx.shape[0]: {uw_image_data_dict[demo_name]["rgbToUWIdx"].shape[0]}, camPoseTimeStamp.shape[0]: {campose_dic["data"][matched_demo_dict_key]["camPoseTimeStamp"].shape[0]}'


    uw_image_data_dict = reorg_demo_data_ordered_by_time(uw_image_data_dict, data_prefix = 'uwImg')
 
    uw_image_dict['data'] = uw_image_data_dict
    
    return uw_image_dict


def fetch_depth_time(
    depth_data_dir, 
    campose_dic, 
    output_path = None, 
    end = -1, 
    every = 1, 
    depth_output_res = (224, 224), 
    verbose = False, 
    side = 'right'
):
    """
    Args:   
        depth_data_dir: str, path to the directory containing .raw of the depth video
        ...: other args
    Return:
        depth_dict: dict, a dictionary containing:
            data
                demo_idx, int, eg. 0 (in temporal order)
                    depthTimeStamp, (N_depth,), dtype: str
                    depthData, (N_depth, H, W, C), dtype: uint8
                    fileName, str of file_name before .raw
            meta
                'session_name', session_name, str, eg. '241119'
            
    """
    demo_dirs = get_demonstration_dirs(depth_data_dir)
    session_name = Path(depth_data_dir).parts[-3]
    
    depth_dict = {}
    depth_data_dict = {}
    meta_data_dict = {}

    for raw_file_dir in demo_dirs:
        depth_dict_demo = fetch_depth_time_map_idx_demo_level(raw_file_dir = raw_file_dir,
                                                    end = end,
                                                    every = every,
                                                    depth_output_res = depth_output_res,
                                                    verbose = False,
                                                    side = side)
        if depth_dict_demo is None:
            file_name = Path(raw_file_dir).parts[-1]
            depth_data_dict[file_name] = None
        else:
            depth_data_dict[depth_dict_demo['fileName']] = depth_dict_demo


    depth_dict['data'] = depth_data_dict
    meta_data_dict['session_name'] = session_name
    depth_dict['meta'] = meta_data_dict
    
    for demo_name, _ in depth_dict['data'].items():
        print(f"demo name: {demo_name}")
        matched_demo_dict_key, matched_demo_dict = find_key_and_timestamp_by_demo(demo_name=demo_name, dic = campose_dic)

        if matched_demo_dict_key is None:
            print(f"Error: image demo name: {demo_name} not found in campose_dic")
            exit(0)

        if depth_data_dict[demo_name] is None:
            cam_length = campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp'].shape[0]
            map_to_depth_idx = np.arange(cam_length // 2).repeat(2)
            if len(map_to_depth_idx) < cam_length:
                map_to_depth_idx = np.append(map_to_depth_idx, map_to_depth_idx[-1] + 1)
            depth_data_dict[demo_name] = {
                'depthData': np.zeros((cam_length, depth_output_res[0], depth_output_res[1], 3), dtype=np.float16),
                'fileName': demo_name,
                'rgbToDepthIdx': map_to_depth_idx.astype(np.uint32),
            }

        assert depth_data_dict[demo_name]['depthData'].shape[0] == campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp'].shape[0],\
            f'imgData.shape[0]: {depth_data_dict[demo_name]["depthData"].shape[0]}, camPoseTimeStamp.shape[0]: {campose_dic["data"][matched_demo_dict_key]["camPoseTimeStamp"].shape[0]}'
        depth_data_dict[demo_name]['depthTimeStamp'] = campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp']

        # downsample depth and depth_time to 30Hz
        depth_data_dict[demo_name]['depthData'] = depth_data_dict[demo_name]['depthData'][::2]
        depth_data_dict[demo_name]['depthTimeStamp'] = depth_data_dict[demo_name]['depthTimeStamp'][::2]

        # check: new depth length VS unique values of mapping index
        assert depth_data_dict[demo_name]['depthData'].shape[0] == len(np.unique(depth_data_dict[demo_name]['rgbToDepthIdx']))
        assert campose_dic['data'][matched_demo_dict_key]['camPoseTimeStamp'].shape[0] == depth_data_dict[demo_name]['rgbToDepthIdx'].shape[0],\
            f'camPoseTimeStamp.shape[0]: {campose_dic["data"][matched_demo_dict_key]["camPoseTimeStamp"].shape[0]}, rgbToDepthIdx.shape[0]: {depth_data_dict[demo_name]["rgbToDepthIdx"].shape[0]}'
        assert depth_data_dict[demo_name]['depthData'].dtype == np.float16, \
            f"Expected float16, got {depth_data_dict[demo_name]['depthData'].dtype}"

    depth_data_dict = reorg_demo_data_ordered_by_time(depth_data_dict, data_prefix = 'depth')
 
    depth_dict['data'] = depth_data_dict
    
    return depth_dict



def get_rgb_to_uw_index_and_time_demo_level(mp4_file_dir, side = 'right'):

    json_data, found = load_visual_data_json(mp4_file_dir, side=side)
    assert found, f'json file not found in {mp4_file_dir}'

    # construct the index
    T = len(json_data['poseTimes']) 
    rgb_idx_to_uw_idx = np.zeros(T, dtype=np.uint32) 
    ultrawide_timestamps = np.array([datetime_fromisoformat(x).timestamp() if x != "" else 0 for x in json_data['ultrawideRGBTimes']])
    latest_ultrawide_frame_index = 0
    found_first_ultrawide = False

    # The rgb indices points to the latest ultrawide frame index, except for the indices until the very first ultrawide frame, where there
    # is no ultrawide frame. In this case, the rgb index points to the first ultrawide which is in the future.
    for i in range(T):
        if ultrawide_timestamps[i] != 0:
            if found_first_ultrawide:
                latest_ultrawide_frame_index += 1
            else:
                found_first_ultrawide = True
        rgb_idx_to_uw_idx[i] = latest_ultrawide_frame_index

    ultrawide_vid_length = latest_ultrawide_frame_index + 1

    assert ultrawide_vid_length == sum(ultrawide_timestamps != 0)

    ultrawide_timestamps = ultrawide_timestamps[ultrawide_timestamps != 0]

    # the reorg_demo_data_ordered_by_time() later expects time in ISO format.
    ultrawide_timestamps_iso_str = [convert_timestamp_to_iso_z_format(ts) for ts in ultrawide_timestamps]

    return rgb_idx_to_uw_idx, ultrawide_timestamps_iso_str



def trim_to_timeframe(X, Y):
    """
    Trim arrays to their overlapping timeframe
    Args:
        X: First timestamp array in ISO format
        Y: Second timestamp array in ISO format
    Returns:
        X_trim: Trimmed version of X 
        Y_trim: Trimmed version of Y
    """
    # Convert to float timestamps
    X_timestamp = array_isostringformat_to_timestamp(X)
    Y_timestamp = array_isostringformat_to_timestamp(Y)
    
    start = max(X_timestamp[0], Y_timestamp[0])
    end = min(X_timestamp[-1], Y_timestamp[-1])
 
    X_start_idx = np.where(X_timestamp >= start)[0][0]
    X_end_idx = np.where(X_timestamp <= end)[0][-1]
    X_trim = X_timestamp[X_start_idx:X_end_idx+1]
    
    # Get indices for Y
    Y_start_idx = np.where(Y_timestamp >= start)[0][0]
    Y_end_idx = np.where(Y_timestamp <= end)[0][-1]
    Y_trim = Y_timestamp[Y_start_idx:Y_end_idx+1]
    
    return array_ntp_time_to_timestamp(X_trim), array_ntp_time_to_timestamp(Y_trim), (int(X_start_idx), int(X_end_idx)+1), (int(Y_start_idx), int(Y_end_idx)+1)

def find_overlapping_segments(array_a, array_b):
    """
    Find indices of overlapping time segments between two lists.

    Parameters:
        array_a (list of tuples): [(start_time_a, end_time_a), ...] of length n.
        array_b (list of tuples): [(start_time_b, end_time_b), ...] of length m.

    Returns:
        tuple: (indices_in_a, indices_in_b), where:
            - indices_in_a: List of indices in array_a with overlap in array_b.
            - indices_in_b: List of indices in array_b with overlap in array_a.
    """
    indices_a = set()
    indices_b = set()

    for i, (start_a, end_a) in enumerate(array_a):
        for j, (start_b, end_b) in enumerate(array_b):
            # Check for overlap condition
            if start_a < end_b and start_b < end_a:
                indices_a.add(i)
                indices_b.add(j)

    return list(indices_a), list(indices_b)

def get_ft_indices_matched(start_time_ft_epoch_time, start_time_campose_epoch_time):
    matched_A_indices = []
    used_A_indices = set()
    
    # Sort A and B while keeping track of original indices
    sorted_A_indices = np.argsort(start_time_ft_epoch_time)
    sorted_A = np.array(start_time_ft_epoch_time)[sorted_A_indices]
    
    sorted_B_indices = np.argsort(start_time_campose_epoch_time)
    sorted_B = np.array(start_time_campose_epoch_time)[sorted_B_indices]

    # Iterate over B and find the closest available match in A
    for b_time in sorted_B:
        closest_idx = None
        min_diff = float('inf')

        # Search for the closest available match in A
        for i in sorted_A_indices:
            if i in used_A_indices:  # Skip already used indices
                continue
            
            diff = abs(start_time_ft_epoch_time[i] - b_time)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            
            # Since A is sorted, if diff starts increasing, we can stop early
            if start_time_ft_epoch_time[i] > b_time and closest_idx is not None:
                break

        if closest_idx is not None:
            matched_A_indices.append(closest_idx)
            used_A_indices.add(closest_idx)

    # Restore original order of B
    final_mapping = [None] * len(start_time_campose_epoch_time)
    for i, b_original_idx in enumerate(sorted_B_indices):
        final_mapping[b_original_idx] = matched_A_indices[i]

    return final_mapping

def time_match_ft_campose(ft_dic, campose_dic):
    """
    Args:
        output_dir: str, path to the directory where to save the trimmed data
        ft_dic: dict, a dictionary containing ft data
        campose_dic: dict, a dictionary containing camPose data
    Returns:
        matched_ft_dic: dict, a dictionary containing ft data and camPose data that are trimmed to the same time frame
    """
    start_time_ft = []
    start_time_campose = []
    end_time_ft = []
    end_time_campose = []
    for ft_demo_idx in ft_dic['data']['left'].keys():
        start_time_ft.append(ft_dic['data']['left'][ft_demo_idx]['ftTimeStamp'][0])
        end_time_ft.append(ft_dic['data']['left'][ft_demo_idx]['ftTimeStamp'][-1])
    for campose_demo_idx in campose_dic['data'].keys():
        start_time_campose.append(campose_dic['data'][campose_demo_idx]['camPoseTimeStamp'][0])
        end_time_campose.append(campose_dic['data'][campose_demo_idx]['camPoseTimeStamp'][-1])
    
    start_time_ft_epoch_time = array_isostringformat_to_timestamp(start_time_ft)
    start_time_campose_epoch_time = array_isostringformat_to_timestamp(start_time_campose)
    end_time_campose_epoch_time = array_isostringformat_to_timestamp(end_time_campose)
    end_time_ft_epoch_time = array_isostringformat_to_timestamp(end_time_ft)
    
    # matched_ft_idx_list = get_ft_indices_matched(start_time_ft_epoch_time, start_time_campose_epoch_time)
    ft_time_segments = []
    campose_time_segments = []
    for idx in range(len(start_time_ft_epoch_time)):
        ft_time_segments.append((start_time_ft_epoch_time[idx], end_time_ft_epoch_time[idx]))
    for idx in range(len(start_time_campose_epoch_time)):
        campose_time_segments.append((start_time_campose_epoch_time[idx], end_time_campose_epoch_time[idx]))
        
    matched_ft_indices, matched_campose_indices = find_overlapping_segments(ft_time_segments, campose_time_segments)
    print(f'ft_time_segments: {ft_time_segments}')
    print(f'campose_time_segments: {campose_time_segments}')
    print(f'matched_ft_indices: {matched_ft_indices}, matched_campose_indices: {matched_campose_indices}')
    return matched_ft_indices, matched_campose_indices
 
    
    
def trim_ft_wrt_campose(output_dir, ft_dic, campose_dic):
    
    """
    Args:
        output_dir: str, path to the directory where to save the trimmed data
        ft_dic: dict, a dictionary containing ft data
        campose_dic: dict, a dictionary containing camPose data
    Returns:
        trim_ft_dic: dict, a dictionary containing ft data and camPose data that are trimmed to the same time frame
    """
    
    # load ft time frame
    trimming_meta_dict = {}
    
    for demo_idx in campose_dic['data'].keys(): 
        print(f'demo_idx: {demo_idx}')
        ft_time_frame_left = ft_dic['data']['left'][demo_idx]['ftTimeStamp'] # only need to retrive one of the left/right coinft, since the timestamp difference between them are well within the error margin of the alignment between camera pose and ft
        campose_time_frame = campose_dic['data'][demo_idx]['camPoseTimeStamp']
        
        ft_time_frame_trimmed, campose_time_frame_trimmed, (ft_trim_start_idx, ft_trim_end_idx), (campose_trim_start_idx, campose_trim_end_idx) = trim_to_timeframe(ft_time_frame_left, campose_time_frame)     
 
        trimming_meta_dict[demo_idx] = {}
        trimming_meta_dict[demo_idx]['ft_trim_start_idx'] = ft_trim_start_idx
        trimming_meta_dict[demo_idx]['ft_trim_end_idx'] = ft_trim_end_idx
        trimming_meta_dict[demo_idx]['campose_trim_start_idx'] = campose_trim_start_idx
        trimming_meta_dict[demo_idx]['campose_trim_end_idx'] = campose_trim_end_idx

        print(f"ft_trim_start_idx: {ft_trim_start_idx}, ft_trim_end_idx: {ft_trim_end_idx}/{len(ft_time_frame_left)}, campose_trim_start_idx: {campose_trim_start_idx}, campose_trim_end_idx: {campose_trim_end_idx}/{len(campose_time_frame)}")
  
    exit(0)
    save_json(output_dir=output_dir, data_dic=trimming_meta_dict, filename='trimming_meta.json')

    return trimming_meta_dict

def get_session_name(args):
    return args.visual_data_dir.split('/')[-1]

def convert_pose_iphone_to_gopro_wrt_arkit(pose_iphone):
    """
    Convert the pose from iPhone frame to GoPro frame, with respect to the iPhone world frame.
    Args:
        pose_iphone: (N, 4, 4) numpy array of SE3 matrix in iPhone frame, with respect to world frame

    Description:
        This function returns the transformation matrix from gopro to tcp.
        Tcp here is defined as the center of the two original UMI fingers, both in width and height.
        To visualize, it would be a floating point mid-air between the original UMI fingers.
    """
    W_T_I = pose_iphone
    W_T_TCP = iphone_to_tcp_poses(W_T_I)

    tcp_offset = 0.205 # [m]. Distancefrom gripper tip to mounting screw in the original UMI
    cam_to_center_height = 0.086 # y axis in camera frame # constant for UMI
    cam_to_mount_offset = 0.01465 # constant for GoPro Hero 9,10,11
    error_correction = 0.0016 # check Hojung's notes
    cam_to_tip_offset = cam_to_mount_offset + tcp_offset - error_correction

    # tcp pose w.r.t. gopro optical center 
    pose_gopro_tcp = np.array([0, cam_to_center_height, cam_to_tip_offset, 0,0,0])
    G_T_TCP = pose_to_mat(pose_gopro_tcp)
    TCP_T_G = su.SE3_inv(G_T_TCP)

    W_T_G = W_T_TCP @ TCP_T_G

    return W_T_G






