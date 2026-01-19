from umift.processing.zarr_replay_buffer import ReplayBuffer
from umift.processing.imagecodecs_numcodecs import JpegXl
from numcodecs import Blosc, VLenUTF8
import numpy as np
import os
import zarr 
from umift.utils.print_utils import debug_print, info_print
from umift.utils.time_utils import array_isostringformat_to_timestamp
from scipy.spatial.transform import Rotation
from umift.utils.rotation_utils import adjoint, wrench2fm, fm2wrench, transform_coinft_l2tcp, transform_coinft_r2cp
from umift.processing.imagecodecs_numcodecs import register_codecs
register_codecs()


def slice_zarr_into_episodes(input_zarr_path, output_zarr_path):
    """
    Processes a Zarr dataset into a new Zarr dataset with episode-wise structure.

    Args:
        input_zarr_path (str): Path to the input Zarr dataset.
        output_zarr_path (str): Path to save the new Zarr dataset.
    """
    # Open the input Zarr group
    input_zarr = zarr.open(input_zarr_path, mode="r")
    input_data_group = input_zarr["data"]
    input_meta_group = input_zarr["meta"]

    # Create a new Zarr structure for the output dataset
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    meta_group = root.create_group("meta")

    # Copy metadata from input to output
    for key, value in input_meta_group.items():
        meta_group.array(
            name=key,
            data=value[:],
            shape=value.shape,
            chunks=value.chunks,
            dtype=value.dtype,
            compressor=value.compressor,
        )

 
    # Get episode lengths
    episode_rgbcamera0_len = input_meta_group["episode_rgb0_len"][:]
    episode_robotcamera0_len = input_meta_group["episode_robot0_len"][:]
    episode_wrenchrobot0_len = input_meta_group["episode_wrench0_len"][:]

    # Process episodes
    for episode_idx, (rgb_end, robot_end, wrench_end) in enumerate(
        zip(episode_rgbcamera0_len, episode_robotcamera0_len, episode_wrenchrobot0_len)
    ):
        # Calculate start indices
        rgb_start = episode_rgbcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        robot_start = episode_robotcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        wrench_start = episode_wrenchrobot0_len[episode_idx - 1] if episode_idx > 0 else 0

        # Create a new group for the episode
        episode_group = root.create_group(f"data/episode_{episode_idx}")

        # Copy and slice datasets into episode group
        for dataset_name, dataset in input_data_group.items():
            if "rgb" in dataset_name:
                sliced_data = dataset[rgb_start:rgb_end]
            elif "robot_time_stamps" in dataset_name or "ts_pose_fb" in dataset_name:
                sliced_data = dataset[robot_start:robot_end]
            elif "wrench" in dataset_name:
                sliced_data = dataset[wrench_start:wrench_end]
            else:
                continue

            # Save the sliced data directly under the episode group
            episode_group.array(
                name=dataset_name,
                data=sliced_data,
                shape=sliced_data.shape,
                chunks=dataset.chunks,
                dtype=dataset.dtype,
                compressor=dataset.compressor,
            )

        print(
            f"Processed episode {episode_idx}: rgb ({rgb_start}:{rgb_end}), "
            f"robot ({robot_start}:{robot_end}), wrench ({wrench_start}:{wrench_end})"
        )

    print(f"New Zarr dataset created at {output_zarr_path}")

def slice_zarr_into_episodes_gripper(input_zarr_path, output_zarr_path):
    """
    Processes a Zarr dataset into a new Zarr dataset with episode-wise structure.

    Args:
        input_zarr_path (str): Path to the input Zarr dataset.
        output_zarr_path (str): Path to save the new Zarr dataset.
    """
    # Open the input Zarr group
    input_zarr = zarr.open(input_zarr_path, mode="r")
    input_data_group = input_zarr["data"]
    input_meta_group = input_zarr["meta"]

    # Create a new Zarr structure for the output dataset
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    meta_group = root.create_group("meta")

    # Copy metadata from input to output
    for key, value in input_meta_group.items():
        meta_group.array(
            name=key,
            data=value[:],
            shape=value.shape,
            chunks=value.chunks,
            dtype=value.dtype,
            compressor=value.compressor,
        )

 
    # Get episode lengths
    episode_rgbcamera0_len = input_meta_group["episode_rgb0_len"][:]
    episode_robotcamera0_len = input_meta_group["episode_robot0_len"][:]
    episode_wrenchrobot0_len = input_meta_group["episode_wrench0_len"][:]
 

    # Process episodes
    for episode_idx, (rgb_end, robot_end, wrench_end) in enumerate(
        zip(episode_rgbcamera0_len, episode_robotcamera0_len, episode_wrenchrobot0_len)
    ):
        # Calculate start indices
        rgb_start = episode_rgbcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        robot_start = episode_robotcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        wrench_start = episode_wrenchrobot0_len[episode_idx - 1] if episode_idx > 0 else 0
 
        # Create a new group for the episode
        episode_group = root.create_group(f"data/episode_{episode_idx}")

        # Copy and slice datasets into episode group
        for dataset_name, dataset in input_data_group.items():
            if "rgb" in dataset_name:
                sliced_data = dataset[rgb_start:rgb_end]
            elif "robot_time_stamps" in dataset_name or "ts_pose_fb" in dataset_name:
                sliced_data = dataset[robot_start:robot_end]
            elif "wrench" in dataset_name:
                sliced_data = dataset[wrench_start:wrench_end]
            elif "gripper" in dataset_name:
                sliced_data = dataset[robot_start:robot_end]
            else:
                continue

            # Save the sliced data directly under the episode group
            episode_group.array(
                name=dataset_name,
                data=sliced_data,
                shape=sliced_data.shape,
                chunks=dataset.chunks,
                dtype=dataset.dtype,
                compressor=dataset.compressor,
            )

        print(
            f"Processed episode {episode_idx}: rgb ({rgb_start}:{rgb_end}), "
            f"robot ({robot_start}:{robot_end}), wrench ({wrench_start}:{wrench_end})"
        )

    print(f"New Zarr dataset created at {output_zarr_path}")

def slice_zarr_into_episodes_gripper_depth(input_zarr_path, output_zarr_path):
    """
    Processes a Zarr dataset into a new Zarr dataset with episode-wise structure.

    Args:
        input_zarr_path (str): Path to the input Zarr dataset.
        output_zarr_path (str): Path to save the new Zarr dataset.
    """
    # Open the input Zarr group
    input_zarr = zarr.open(input_zarr_path, mode="r")
    input_data_group = input_zarr["data"]
    input_meta_group = input_zarr["meta"]

    # Create a new Zarr structure for the output dataset
    store = zarr.DirectoryStore(output_zarr_path)
    root = zarr.group(store=store, overwrite=True)
    meta_group = root.create_group("meta")

    # Copy metadata from input to output
    for key, value in input_meta_group.items():
        meta_group.array(
            name=key,
            data=value[:],
            shape=value.shape,
            chunks=value.chunks,
            dtype=value.dtype,
            compressor=value.compressor,
        )

 
    # Get episode lengths
    episode_rgbcamera0_len = input_meta_group["episode_rgb0_len"][:]
    episode_ultrawide0_len = input_meta_group["episode_ultrawide0_len"][:]
    episode_depth0_len = input_meta_group["episode_depth0_len"][:]
    episode_robotcamera0_len = input_meta_group["episode_robot0_len"][:]
    episode_wrenchrobot0_len = input_meta_group["episode_wrench0_len"][:]
 

    # Process episodes
    for episode_idx, (rgb_end, ultrawide_end, depth_end, robot_end, wrench_end) in enumerate(
        zip(episode_rgbcamera0_len, episode_ultrawide0_len, episode_depth0_len, episode_robotcamera0_len, episode_wrenchrobot0_len)
    ):
        # Calculate start indices
        rgb_start = episode_rgbcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        ultrawide_start = episode_ultrawide0_len[episode_idx - 1] if episode_idx > 0 else 0
        depth_start = episode_depth0_len[episode_idx - 1] if episode_idx > 0 else 0
        robot_start = episode_robotcamera0_len[episode_idx - 1] if episode_idx > 0 else 0
        wrench_start = episode_wrenchrobot0_len[episode_idx - 1] if episode_idx > 0 else 0
 
        # Create a new group for the episode
        episode_group = root.create_group(f"data/episode_{episode_idx}")

        # Copy and slice datasets into episode group
        for dataset_name, dataset in input_data_group.items():
            if "ultrawide" in dataset_name:
                sliced_data = dataset[ultrawide_start:ultrawide_end]
            elif "map_to_uw_idx" in dataset_name:
                sliced_data = dataset[rgb_start:rgb_end]    
            elif "map_to_d_idx" in dataset_name:
                sliced_data = dataset[rgb_start:rgb_end]
            elif "depth" in dataset_name:
                sliced_data = dataset[depth_start:depth_end]
            elif "rgb" in dataset_name: 
                sliced_data = dataset[rgb_start:rgb_end]
            elif "robot_time_stamps" in dataset_name or "ts_pose_fb" in dataset_name:
                sliced_data = dataset[robot_start:robot_end]
            elif "wrench" in dataset_name:
                sliced_data = dataset[wrench_start:wrench_end]
            elif "gripper" in dataset_name:
                sliced_data = dataset[robot_start:robot_end]
            else:
                continue

            # Save the sliced data directly under the episode group
            episode_group.array(
                name=dataset_name,
                data=sliced_data,
                shape=sliced_data.shape,
                chunks=dataset.chunks,
                dtype=dataset.dtype,
                compressor=dataset.compressor,
            )

        print(
            f"Processed episode {episode_idx}: rgb ({rgb_start}:{rgb_end}), "
            f"ultrawide ({ultrawide_start}:{ultrawide_end}), "
            f"depth ({depth_start}:{depth_end}), "
            f"robot ({robot_start}:{robot_end}), wrench ({wrench_start}:{wrench_end})"
        )

    print(f"New Zarr dataset created at {output_zarr_path}")

def convert_SE3_to_xyz_quat(SE3_matrices):
    """
    Converts an array of 4x4 SE(3) matrices to XYZ (translation) and quaternion representation.

    Args:
        SE3_matrices (np.ndarray): Array of shape (N, 4, 4) representing SE(3) matrices.

    Returns:
        np.ndarray: Array of shape (N, 7) where each row contains [x, y, z, qw, qx, qy, qz].
    """
    xyz_quat = []
    for mat in SE3_matrices:
        xyz = mat[:3, 3]
        rotation = Rotation.from_matrix(mat[:3, :3])
        quat = rotation.as_quat(scalar_first=True)  # scalar_first set to true for qw, qx, qy, qz
        xyz_quat.append(np.hstack((xyz, quat)))

    return np.array(xyz_quat)

def create_zarr_dataset(data_dict, output_path, cam_id=0, gripper_id=0, expected_H=101, expected_W=135):
    """
    campose, ft, image
    """
    # Configure compressors
    image_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    numeric_compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    string_compressor = VLenUTF8()
    
    cam_name = f'{cam_id}'
    robot_name = f'{gripper_id}'
    # cam_name = f'camera{cam_id}'
    # robot_name = f'robot{gripper_id}'
    
    # Create root group
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # Calculate total lengths
    total_cam_length = 0
    total_ft_length = 0
    ft_episode_ends = []
    cam_episode_ends = []
    

    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        ft_length_left = len(data_dict['ft_dic_trimmed']['data']['left'][demo_idx]['ftData'])
        ft_length_right = len(data_dict['ft_dic_trimmed']['data']['right'][demo_idx]['ftData'])
        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'

        cam_length = len(data_dict['image_dict_trimmed']['data'][demo_idx]['imgData'])
        
        total_ft_length += ft_length_left
        total_cam_length += cam_length
        
        ft_episode_ends.append(total_ft_length)
        cam_episode_ends.append(total_cam_length)
    
    # Create datasets
    data_group.create_dataset(
        f'rgb_{cam_name}',
        shape=(total_cam_length, expected_H, expected_W, 3),
        dtype=np.uint8,
        compressor=image_compressor
    )
    
    data_group.create_dataset(
         f'ts_pose_fb_{robot_name}',
        shape=(total_cam_length, 7),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_left_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_right_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    # Create timestamp datasets with object dtype
    data_group.create_dataset(
        f'rgb_time_stamps_{cam_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'robot_time_stamps_{robot_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_left_{robot_name}',
        shape=(total_ft_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_right_{robot_name}',
        shape=(total_ft_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    # Create episode ends datasets
    meta_group.create_dataset(
        f'episode_wrench{robot_name}_len',
        data=np.array(ft_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )
    
    meta_group.create_dataset(
        f'episode_rgb{cam_name}_len',
        data=np.array(cam_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )

    meta_group.create_dataset(
        f'episode_robot{cam_name}_len',   
        data=np.array(cam_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )
    
    # Fill data
    cam_current_idx = 0
    ft_current_idx = 0
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        ft_data_left = data_dict['ft_dic_trimmed']['data']['left'][demo_idx]
        ft_data_right = data_dict['ft_dic_trimmed']['data']['right'][demo_idx]
        campose_data = data_dict['campose_dic_trimmed']['data'][demo_idx]
        image_data = data_dict['image_dict_trimmed']['data'][demo_idx]

        ft_length_left = len(ft_data_left['ftData'])
        ft_length_right = len(ft_data_right['ftData'])
        
        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'
         
        
        cam_length = len(image_data['imgData'])
        
        # Fill camera, campera pose, force/torque data
        data_group[f'rgb_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(image_data['imgData'], dtype=np.uint8)
        cam_pose_SE3_mat = np.array(campose_data['camPoseData'], dtype=np.float64).reshape(-1, 4, 4)
        cam_pose_xyzqwxyz = convert_SE3_to_xyz_quat(cam_pose_SE3_mat) # N, 4, 4
        data_group[f'ts_pose_fb_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = cam_pose_xyzqwxyz # N, 7
        
        data_group[f'wrench_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length_left] = np.array(ft_data_left['ftData'], dtype=np.float64)
        data_group[f'wrench_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length_right] = np.array(ft_data_right['ftData'], dtype=np.float64)
        
        # Convert timestamps to strings
        data_group[f'rgb_time_stamps_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(image_data['imgTimeStamp']).reshape(-1, 1)
        data_group[f'robot_time_stamps_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(campose_data['camPoseTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length_left] = array_isostringformat_to_timestamp(ft_data_left['ftTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length_right] = array_isostringformat_to_timestamp(ft_data_right['ftTimeStamp']).reshape(-1, 1) 
        
        cam_current_idx += cam_length
        ft_current_idx += ft_length_left
    
    return root

def create_zarr_dataset_gripper(data_dict, output_path, cam_id=0, gripper_id=0, expected_H=101, expected_W=135):
    # Configure compressors
    image_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    numeric_compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    string_compressor = VLenUTF8()
    
    cam_name = f'{cam_id}'
    robot_name = f'{gripper_id}'
    
    # Create root group
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # Calculate total lengths
    total_cam_length = 0
    total_ft_length = 0
    ft_episode_ends = []
    cam_episode_ends = []
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        # gripper_length = len(data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData']) 
        gripper_widths = data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData']
        data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData'] = gripper_widths.reshape(-1, 1) # convert from (N,) to (N, 1)
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        ft_length_left = len(data_dict['ft_dic_trimmed']['data']['left'][demo_idx]['ftData'])
        ft_length_right = len(data_dict['ft_dic_trimmed']['data']['right'][demo_idx]['ftData'])
        cam_length = len(data_dict['image_dict_trimmed']['data'][demo_idx]['imgData'])
        
        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'
        ft_length = ft_length_left
        
        total_ft_length += ft_length
        total_cam_length += cam_length
        
        ft_episode_ends.append(total_ft_length)
        cam_episode_ends.append(total_cam_length)
    
    # Create datasets
    data_group.create_dataset(
        f'rgb_{cam_name}',
        shape=(total_cam_length, expected_H, expected_W, 3),
        dtype=np.uint8,
        compressor=image_compressor
    )
    
    data_group.create_dataset(
        f'gripper_{cam_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
         f'ts_pose_fb_{robot_name}',
        shape=(total_cam_length, 7),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_left_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_right_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_left_coinft_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'wrench_concat_{robot_name}', # left-right order
        shape=(total_ft_length, 12),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    

    data_group.create_dataset(
        f'wrench_right_coinft_{robot_name}',
        shape=(total_ft_length, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_concat_coinft_{robot_name}', # always left-right
        shape=(total_ft_length, 12),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    # Create timestamp datasets with object dtype
    data_group.create_dataset(
        f'rgb_time_stamps_{cam_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'robot_time_stamps_{robot_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_left_{robot_name}',
        shape=(total_ft_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_right_{robot_name}',
        shape=(total_ft_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'gripper_time_stamps_{robot_name}',
        shape=(total_cam_length, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    # Create episode ends datasets
    meta_group.create_dataset(
        f'episode_wrench{robot_name}_len',
        data=np.array(ft_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )
    
    meta_group.create_dataset(
        f'episode_rgb{cam_name}_len',
        data=np.array(cam_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )

    meta_group.create_dataset(
        f'episode_robot{cam_name}_len',   
        data=np.array(cam_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )
    meta_group.create_dataset(
        f'episode_gripper{cam_name}_len',   
        data=np.array(cam_episode_ends, dtype=np.int64),
        compressor=numeric_compressor
    )
    
    # Fill data
    cam_current_idx = 0
    ft_current_idx = 0
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        ft_data_left = data_dict['ft_dic_trimmed']['data']['left'][demo_idx]
        ft_data_right = data_dict['ft_dic_trimmed']['data']['right'][demo_idx]
        campose_data = data_dict['campose_dic_trimmed']['data'][demo_idx]
        image_data = data_dict['image_dict_trimmed']['data'][demo_idx]
        gripper_width_data = data_dict['gripper_dic_trimmed']['data'][demo_idx]

        ft_length_left = len(ft_data_left['ftData'])
        ft_length_right = len(ft_data_right['ftData'])
        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'
        ft_length = ft_length_left
        cam_length = len(image_data['imgData'])
        
        # Fill camera, campera pose, force/torque data
        data_group[f'rgb_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(image_data['imgData'], dtype=np.uint8)
        cam_pose_SE3_mat = np.array(campose_data['camPoseData'], dtype=np.float64).reshape(-1, 4, 4)
        cam_pose_xyzqwxyz = convert_SE3_to_xyz_quat(cam_pose_SE3_mat) # N, 4, 4
        data_group[f'ts_pose_fb_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = cam_pose_xyzqwxyz # N, 7
        data_group[f'wrench_left_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = np.array(ft_data_left['ftData'], dtype=np.float64)
        data_group[f'wrench_right_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = np.array(ft_data_right['ftData'], dtype=np.float64)
        data_group[f'wrench_concat_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = concat_left_right(ft_data_left, ft_data_right) # (N, 12)
        
        ft_tcp_left, ft_tcp_right, ft_tcp_conca = coinft2tcp(gripper_width_data, ft_data_left, ft_data_right, campose_data)
        
        # # plot for debugging
        # import matplotlib.pyplot as plt
        # plt.figure()
        
        # plt.plot(ft_tcp_left[:, 0], label='ft_tcp_left')
        data_group[f'wrench_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_right
        data_group[f'wrench_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_left
        data_group[f'wrench_concat_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_conca
        data_group[f'gripper_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(gripper_width_data['gripperData'], dtype=np.float64)
        
        # Convert timestamps to strings
        data_group[f'rgb_time_stamps_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(image_data['imgTimeStamp']).reshape(-1, 1)
        data_group[f'robot_time_stamps_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(campose_data['camPoseTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = array_isostringformat_to_timestamp(ft_data_left['ftTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = array_isostringformat_to_timestamp(ft_data_right['ftTimeStamp']).reshape(-1, 1) 
        data_group[f'gripper_time_stamps_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(gripper_width_data['gripperTimeStamp']).reshape(-1, 1) 

        cam_current_idx += cam_length
        ft_current_idx += ft_length
    
    return root


def create_zarr_dataset_gripper_depth(data_dict, output_path, cam_id=0, gripper_id=0, expected_H=101, expected_W=135, compression_level=99):
    # Configure compressors
    # aiming for ~1MB
    img_chunk_size = 8
    depth_chunk_size = 8
    wrench_chunk_size = 4096
    pose_chunk_size = 4096
    gripper_chunk_size = 16384
    time_stamp_chunk_size = 16384

    image_compressor = JpegXl(level=compression_level, numthreads=16)
    # image_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    numeric_compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    depth_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    string_compressor = VLenUTF8()
    
    cam_name = f'{cam_id}'
    robot_name = f'{gripper_id}'
    
    # Create root group
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')
    
    # Calculate total lengths
    total_cam_length = 0
    total_uw_cam_legnth = 0
    total_depth_length = 0
    total_ft_length = 0
    ft_episode_ends = []
    cam_episode_ends = []
    uw_cam_episode_ends = []
    depth_episode_ends = []
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        # gripper_length = len(data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData']) 
        gripper_widths = data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData']
        data_dict['gripper_dic_trimmed']['data'][demo_idx]['gripperData'] = gripper_widths.reshape(-1, 1) # convert from (N,) to (N, 1)
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        ft_length_left = len(data_dict['ft_dic_trimmed']['data']['left'][demo_idx]['ftData'])
        ft_length_right = len(data_dict['ft_dic_trimmed']['data']['right'][demo_idx]['ftData'])
        cam_length = len(data_dict['image_dict_trimmed']['data'][demo_idx]['imgData'])
        uw_cam_length = len(data_dict['uw_image_dict_trimmed']['data'][demo_idx]['uwImgData'])
        depth_length = len(data_dict['depth_dict_trimmed']['data'][demo_idx]['depthData'])

        assert cam_length == len(data_dict['uw_image_dict_trimmed']['data'][demo_idx]['rgbToUWIdx']), f'cam_length: {cam_length}, rgb_indx_conversion: {len(data_dict["uw_image_dict_trimmed"]["data"][demo_idx]["rgbToUWIdx"])}'
        assert uw_cam_length == len(data_dict['uw_image_dict_trimmed']['data'][demo_idx]['uwImgTimeStamp']), f'uw_cam_length: {uw_cam_length}, uw_rgb_timestep_len: {len(data_dict["uw_image_dict_trimmed"]["data"][demo_idx]["uwImgTimeStamp"])}'
        assert cam_length == len(data_dict['depth_dict_trimmed']['data'][demo_idx]['rgbToDepthIdx']), f'cam_length: {cam_length}, rgb_to_depth_idx_conversion_len: {len(data_dict["depth_dict_trimmed"]["data"][demo_idx]["rgbToDepthIdx"])}'

        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'
        ft_length = ft_length_left
        
        total_ft_length += ft_length
        total_cam_length += cam_length
        total_uw_cam_legnth += uw_cam_length
        total_depth_length += depth_length
        
        ft_episode_ends.append(total_ft_length)
        cam_episode_ends.append(total_cam_length)
        uw_cam_episode_ends.append(total_uw_cam_legnth)
        depth_episode_ends.append(total_depth_length)
    
    # Create datasets
    data_group.create_dataset(
        f'rgb_{cam_name}',
        shape=(total_cam_length, expected_H, expected_W, 3),
        chunks=(img_chunk_size, expected_H, expected_W, 3),
        dtype=np.uint8,
        compressor=image_compressor
    )

    data_group.create_dataset(
        f'ultrawide_{cam_name}',
        shape=(total_uw_cam_legnth, expected_H, expected_W, 3),
        chunks=(img_chunk_size, expected_H, expected_W, 3),
        dtype=np.uint8,
        compressor=image_compressor
    )

    data_group.create_dataset(
        f'depth_{cam_name}',
        shape=(total_depth_length, expected_H, expected_W, 3),
        chunks=(depth_chunk_size, expected_H, expected_W, 3), 
        dtype=np.float16,
        compressor=depth_compressor
    )
    
    data_group.create_dataset(
        f'gripper_{cam_name}',
        shape=(total_cam_length, 1),
        chunks=(gripper_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
         f'ts_pose_fb_{robot_name}',
        shape=(total_cam_length, 7),
        chunks=(pose_chunk_size, 7),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_left_{robot_name}',
        shape=(total_ft_length, 6),
        chunks=(wrench_chunk_size, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_right_{robot_name}',
        shape=(total_ft_length, 6),
        chunks=(wrench_chunk_size, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_left_coinft_{robot_name}',
        shape=(total_ft_length, 6),
        chunks=(wrench_chunk_size, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'wrench_concat_{robot_name}', # left-right order
        shape=(total_ft_length, 12),
        chunks=(wrench_chunk_size, 12),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_right_coinft_{robot_name}',
        shape=(total_ft_length, 6),
        chunks=(wrench_chunk_size, 6),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_concat_coinft_{robot_name}', # always left-right
        shape=(total_ft_length, 12),
        chunks=(wrench_chunk_size, 12),
        dtype=np.float64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'map_to_uw_idx_{cam_name}',
        shape=(total_cam_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.int64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'map_to_d_idx_{cam_name}',
        shape=(total_cam_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.int64,
        compressor=numeric_compressor
    )
    
    # Create timestamp datasets with object dtype
    data_group.create_dataset(
        f'rgb_time_stamps_{cam_name}',
        shape=(total_cam_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'ultrawide_time_stamps_{cam_name}',
        shape=(total_uw_cam_legnth, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )

    data_group.create_dataset(
        f'depth_time_stamps_{cam_name}',
        shape=(total_depth_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'robot_time_stamps_{robot_name}',
        shape=(total_cam_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_left_{robot_name}',    
        shape=(total_ft_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'wrench_time_stamps_right_{robot_name}',
        shape=(total_ft_length, 1),
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    data_group.create_dataset(
        f'gripper_time_stamps_{robot_name}',
        shape=(total_cam_length, 1),    
        chunks=(time_stamp_chunk_size, 1),
        dtype=np.float64,
        compressor=numeric_compressor
    )
    
    # Create episode ends datasets
    meta_group.create_dataset(
        f'episode_wrench{robot_name}_len',
        data=np.array(ft_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(ft_episode_ends)),),
        compressor=numeric_compressor
    )
    
    meta_group.create_dataset(
        f'episode_rgb{cam_name}_len',
        data=np.array(cam_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(cam_episode_ends)),),
        compressor=numeric_compressor
    )

    meta_group.create_dataset(
        f'episode_ultrawide{cam_name}_len',
        data=np.array(uw_cam_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(uw_cam_episode_ends)),),
        compressor=numeric_compressor
    )

    meta_group.create_dataset(
        f'episode_depth{cam_name}_len',
        data=np.array(depth_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(depth_episode_ends)),),
        compressor=numeric_compressor
    )

    meta_group.create_dataset(
        f'episode_robot{cam_name}_len',   
        data=np.array(cam_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(cam_episode_ends)),),
        compressor=numeric_compressor
    )
    meta_group.create_dataset(
        f'episode_gripper{cam_name}_len',   
        data=np.array(cam_episode_ends, dtype=np.int64),
        chunks=(min(1024, len(cam_episode_ends)),),
        compressor=numeric_compressor
    )
    
    # Fill data
    cam_current_idx = 0
    uw_cam_current_idx = 0
    depth_current_idx = 0
    ft_current_idx = 0
    
    for demo_idx in data_dict['campose_dic_trimmed']['data'].keys():
        print(f'[create_zarr_dataset_gripper_depth] filling data for demo_idx: {demo_idx}')

        ft_data_left = data_dict['ft_dic_trimmed']['data']['left'][demo_idx]
        ft_data_right = data_dict['ft_dic_trimmed']['data']['right'][demo_idx]
        campose_data = data_dict['campose_dic_trimmed']['data'][demo_idx]
        image_data = data_dict['image_dict_trimmed']['data'][demo_idx]
        uw_image_data = data_dict['uw_image_dict_trimmed']['data'][demo_idx]
        depth_data = data_dict['depth_dict_trimmed']['data'][demo_idx]
        gripper_width_data = data_dict['gripper_dic_trimmed']['data'][demo_idx]

        ft_length_left = len(ft_data_left['ftData'])
        ft_length_right = len(ft_data_right['ftData'])
        assert ft_length_left == ft_length_right, f'ft_length_left: {ft_length_left}, ft_length_right: {ft_length_right}'
        ft_length = ft_length_left
        cam_length = len(image_data['imgData'])
        uw_cam_length = len(uw_image_data['uwImgData'])
        depth_length = len(depth_data['depthData'])
        
        # Fill camera, campera pose, force/torque data
        data_group[f'rgb_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(image_data['imgData'], dtype=np.uint8)
        data_group[f'ultrawide_{cam_name}'][uw_cam_current_idx:uw_cam_current_idx+uw_cam_length] = np.array(uw_image_data['uwImgData'], dtype=np.uint8)
        data_group[f'depth_{cam_name}'][depth_current_idx:depth_current_idx+depth_length] = np.array(depth_data['depthData'], dtype=np.float16)
        data_group[f'map_to_uw_idx_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(uw_image_data['rgbToUWIdx'], dtype=np.uint32).reshape(-1, 1)
        data_group[f'map_to_d_idx_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(depth_data['rgbToDepthIdx'], dtype=np.uint32).reshape(-1, 1)
        cam_pose_SE3_mat = np.array(campose_data['camPoseData'], dtype=np.float64).reshape(-1, 4, 4)
        cam_pose_xyzqwxyz = convert_SE3_to_xyz_quat(cam_pose_SE3_mat) # N, 4, 4
        data_group[f'ts_pose_fb_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = cam_pose_xyzqwxyz # N, 7
        data_group[f'wrench_left_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = np.array(ft_data_left['ftData'], dtype=np.float64)
        data_group[f'wrench_right_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = np.array(ft_data_right['ftData'], dtype=np.float64)
        data_group[f'wrench_concat_coinft_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = concat_left_right(ft_data_left, ft_data_right) # (N, 12)
        
        ft_tcp_left, ft_tcp_right, ft_tcp_conca = coinft2tcp(gripper_width_data, ft_data_left, ft_data_right, campose_data)
        
        data_group[f'wrench_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_right
        data_group[f'wrench_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_left
        data_group[f'wrench_concat_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = ft_tcp_conca
        data_group[f'gripper_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = np.array(gripper_width_data['gripperData'], dtype=np.float64)
        
        # Convert timestamps to strings
        data_group[f'rgb_time_stamps_{cam_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(image_data['imgTimeStamp']).reshape(-1, 1)
        data_group[f'ultrawide_time_stamps_{cam_name}'][uw_cam_current_idx:uw_cam_current_idx+uw_cam_length] = array_isostringformat_to_timestamp(uw_image_data['uwImgTimeStamp']).reshape(-1, 1)
        data_group[f'depth_time_stamps_{cam_name}'][depth_current_idx:depth_current_idx+depth_length] = array_isostringformat_to_timestamp(depth_data['depthTimeStamp']).reshape(-1, 1)
        data_group[f'robot_time_stamps_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(campose_data['camPoseTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_left_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = array_isostringformat_to_timestamp(ft_data_left['ftTimeStamp']).reshape(-1, 1) 
        data_group[f'wrench_time_stamps_right_{robot_name}'][ft_current_idx:ft_current_idx+ft_length] = array_isostringformat_to_timestamp(ft_data_right['ftTimeStamp']).reshape(-1, 1) 
        data_group[f'gripper_time_stamps_{robot_name}'][cam_current_idx:cam_current_idx+cam_length] = array_isostringformat_to_timestamp(gripper_width_data['gripperTimeStamp']).reshape(-1, 1) 

        cam_current_idx += cam_length
        uw_cam_current_idx += uw_cam_length
        depth_current_idx += depth_length
        ft_current_idx += ft_length
    
    return root

def concat_left_right(ft_data_left, ft_data_right):
    """
    Args:
        ft_data_left, dict: {'ftData': (N, 6), 'ftTimeStamp': (N,)}
        ft_data_right, dict: {'ftData': (N, 6), 'ftTimeStamp': (N,)}
    Returns:
        concat (N, 12)
    """
    left_wrench = np.array(ft_data_left['ftData'], dtype=np.float64) # (N, 6)
    right_wrench = np.array(ft_data_right['ftData'], dtype=np.float64) # (N, 6)
    return np.hstack((left_wrench, right_wrench))

def coinft2tcp(gripper_width_data, ft_data_left, ft_data_right, campose_data):
    """
    Args:
        gripper_width_data, dict: {'gripperData': (N, 1), 'gripperTimeStamp': (N,)}
        ft_data_left, dict: {'ftData': (N, 6), 'ftTimeStamp': (N,)}
        ft_data_right, dict: {'ftData': (N, 6), 'ftTimeStamp': (N,)}
    Returns:
        fm_tcp (N, 6)
    """
    
    left_fm = np.array(ft_data_left['ftData'], dtype=np.float64) # (N, 6)
    right_fm = np.array(ft_data_right['ftData'], dtype=np.float64) # 

    ft_left_timestamp = array_isostringformat_to_timestamp(ft_data_left['ftTimeStamp'])
    ft_right_timestamp = array_isostringformat_to_timestamp(ft_data_right['ftTimeStamp'])
    gripper_timestamp = array_isostringformat_to_timestamp(campose_data['camPoseTimeStamp'])
    
    gripper_width = np.array(gripper_width_data['gripperData'], dtype=np.float64)
    gripper_interpolated_at_ft = np.interp(ft_left_timestamp, gripper_timestamp, np.squeeze(gripper_width)) # N,
 
    left_fm_tcp_list = []
    right_fm_tcp_list = []
 
    for i in range(gripper_interpolated_at_ft.shape[0]):
        gripper_width_i = gripper_interpolated_at_ft[i]
 
        T_coinft_left2tcp = transform_coinft_l2tcp(gripper_width_i/2)
        T_coinft_right2tcp = transform_coinft_r2cp(gripper_width_i/2)
        
        wrench_left_tcp = adjoint(T_coinft_left2tcp).T @ fm2wrench(left_fm[i])
        fm_left_tcp = wrench2fm(wrench_left_tcp)
        
        wrench_right_tcp = adjoint(T_coinft_right2tcp).T @ fm2wrench(right_fm[i])
        fm_right_tcp = wrench2fm(wrench_right_tcp)
        
        left_fm_tcp_list.append(fm_left_tcp)
        right_fm_tcp_list.append(fm_right_tcp)
    ft_tcp_left = np.array(left_fm_tcp_list, dtype=np.float64)
    ft_tcp_right = np.array(right_fm_tcp_list, dtype=np.float64)
    ft_tcp_concat = np.hstack((ft_tcp_left, ft_tcp_right)) # N, 12
    return ft_tcp_left, ft_tcp_right, ft_tcp_concat
        
        
 
    
    
