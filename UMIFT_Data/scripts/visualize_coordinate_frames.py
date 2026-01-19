# # Load a short clip and compute both trajectories

# import sys
# import os

# sys.path.append(os.path.join(os.getcwd(), '..', 'submodules', 'umi_day'))
# sys.path.append(os.path.join(os.getcwd(), '..', '..', 'PyriteUtility'))

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (needed for 3-D plots)
# from pathlib import Path
# from scipy.spatial.transform import Rotation
# from umift.processing.wired_time_sync import convert_pose_iphone_to_gopro_wrt_arkit
# from umift.utils.time_utils import convert_timestamp_to_iso_processed, isostringformat_to_timestamp, array_isostringformat_to_timestamp, array_ntp_time_to_timestamp
# from umift.utils.print_utils import color_print, debug_print, info_print
# from umift.utils.json_utils import save_ft_data, load_visual_data_json, get_demonstration_dirs, save_campose_data, save_json
# from umi_day.demonstration_processing.utils.gripper_util import get_demo_gripper_width, iphone_to_tcp_poses
# from umi_day.common.cv_util import get_image_transform_with_border
# from umi_day.common.timecode_util import datetime_fromisoformat
# from umi.common.pose_util import pose_to_mat
# from PyriteUtility.spatial_math import spatial_utilities as su

# # ------------------------------------------------------------------
# demo_dir = '/store/real/hjchoi92/data/real/umift/WBW-iph-b0/processed_data/iphone/2025-04-17/2025-04-17T23-37-13.567Z_61579_WBW-iph-b0_demonstration'
# json_data, found = load_visual_data_json(demo_dir, side='left')
# # 1) grab 200 consecutive SE3s from your JSON (or from Zarr)
# W_T_I = np.array(json_data['poseTransforms'])      # (N,4,4)

# # 2) convert each one → GoPro using the same routine you use downstream
# W_T_G= convert_pose_iphone_to_gopro_wrt_arkit(W_T_I)              # (N,4,4)

# # 3) turn both into world-frame positions so we can plot them
# pos_I = W_T_I[..., :3, 3]                                # (N,3)
# pos_G = W_T_G[..., :3, 3]                      # (N,3)

# # pos_W_T_TCP = W_T_TCP[..., :3, 3]                  # (N,3)
# # pos_W_T_G = W_T_G[..., :3, 3]            # (N,3)

# delta = pos_I - pos_G
# print(delta[100:106, :])


# # fig = plt.figure(figsize=(6,5))
# # ax  = fig.add_subplot(111, projection='3d')
# # ax.plot(pos_I[:,0], pos_I[:,1], pos_I[:,2], '.', label='iPhone')
# # ax.plot(pos_G[:,0], pos_G[:,1], pos_G[:,2], '.', label='GoPro')
# # ax.plot(pos_W_T_TCP[:,0], pos_W_T_TCP[:,1], pos_W_T_TCP[:,2], '.', label='W_T_TCP')
# # # ax.plot(pos_W_T_G_temp[:,0], pos_W_T_G_temp[:,1], pos_W_T_G_temp[:,2], '.', label='W_T_G_temp')
# # ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
# # ax.legend(); ax.set_title('World–frame positions')
# # plt.tight_layout()
# # plt.show()


# def draw_axes(ax, T, length=0.05, lw=1.5):
#     o   = T[:3, 3]
#     R   = T[:3, :3]
#     X,Y,Z = R[:,0], R[:,1], R[:,2]
#     ax.quiver(*o, *X, length=length, color='r', linewidth=lw)
#     ax.quiver(*o, *Y, length=length, color='g', linewidth=lw)
#     ax.quiver(*o, *Z, length=length, color='b', linewidth=lw)

# fig = plt.figure(figsize=(6,5)); ax = fig.add_subplot(111, projection='3d')
# for k in range(0, W_T_I.shape[0], 25):        # every 20th frame
#     draw_axes(ax, W_T_I[k], length=0.03)
#     draw_axes(ax, W_T_G[k], length=0.03, lw=1)

# ax.set_title('Red/Green/Blue = iPhone axes, faint = GoPro axes')
# plt.show()


# visualize_coordinate_frames.py

import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..', 'submodules', 'umi_day'))
sys.path.append(os.path.join(os.getcwd(), '..', '..', 'PyriteUtility'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy.spatial.transform import Rotation

from umift.processing.wired_time_sync import convert_pose_iphone_to_gopro_wrt_arkit
from umift.utils.json_utils import load_visual_data_json
from umi.common.pose_util import pose_to_mat

# Define coordinate axis drawer
def draw_axes(ax, T, length=0.05, lw=1.5):
    o   = T[:3, 3]
    R   = T[:3, :3]
    X, Y, Z = R[:,0], R[:,1], R[:,2]
    ax.quiver(*o, *X, length=length, color='r', linewidth=lw)
    ax.quiver(*o, *Y, length=length, color='g', linewidth=lw)
    ax.quiver(*o, *Z, length=length, color='b', linewidth=lw)

# Main logic
if __name__ == "__main__":
    demo_dir = '/store/real/hjchoi92/data/real/umift/WBW-iph-b0/processed_data/iphone/2025-04-17/2025-04-17T23-37-13.567Z_61579_WBW-iph-b0_demonstration'
    json_data, found = load_visual_data_json(demo_dir, side='left')
    
    # 1) grab SE3s from your JSON (or from Zarr). iphone frame poses w.r.t arkit
    W_T_I = np.array(json_data['poseTransforms'])        # (N, 4, 4)

    # 2) Convert each pose to GoPro frame, w.r.t arkit
    W_T_G = convert_pose_iphone_to_gopro_wrt_arkit(W_T_I)  # (N, 4, 4)

    # 3) Extract translation components
    pos_I = W_T_I[..., :3, 3]
    pos_G = W_T_G[..., :3, 3]

    # Print difference between the two trajectories
    delta = pos_I - pos_G
    dists = np.linalg.norm(delta, axis=1)
    print(f"Mean distance: {np.mean(dists):.6f} m")
    print(f"Std deviation: {np.std(dists):.6f} m")
    print(f"Max distance: {np.max(dists):.6f} m")
    print(f"Min distance: {np.min(dists):.6f} m")

    # Plot coordinate axes
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    for k in range(0, W_T_I.shape[0], 25):  # every 25th frame
        draw_axes(ax, W_T_I[k], length=0.03)        # iPhone
        draw_axes(ax, W_T_G[k], length=0.03, lw=1)  # GoPro

    ax.set_title('iPhone and GoPro frame movements')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    # ax.legend(['iPhone', 'GoPro'])
    plt.tight_layout()
    plt.show()







