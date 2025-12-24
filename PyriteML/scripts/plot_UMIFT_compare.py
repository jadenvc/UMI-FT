import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PyriteUtility.spatial_math import spatial_utilities as su

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

dataset_path_ur = dataset_folder_path + "/umi-ft/umift_debug"
dataset_path_umi = dataset_folder_path + "/umi-ft/umift_debug_umi"

print("Loading dataset from: ", dataset_path_ur)
print("Loading umi dataset from: ", dataset_path_umi)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer_ur = zarr.open(dataset_path_ur, mode="r+")
buffer_umi = zarr.open(dataset_path_umi, mode="r+")

# for ep, ep_data in buffer_ur["data"].items():
#     print(ep)
# print("-------------------")
# for ep, ep_data in buffer_umi["data"].items():
#     print(ep)

# episode_1742848069
# episode_1742848131
# episode_1742848182
# episode_1742848250
# episode_1742848311
# -------------------
# episode_0
# episode_1
# episode_2
# episode_3


# id_ur = "episode_1742848311"
# id_umi = "episode_3"
# time_offset = 0.3

# id_ur = "episode_1742848250"
# id_umi = "episode_2"
# time_offset = 0.5

# id_ur = "episode_1742848182"
# id_umi = "episode_1"
# time_offset = 5.9

id_ur = "episode_1742848069"
id_umi = "episode_0"
time_offset = 4.9

# Initialize figure and axes
fig, axes = plt.subplots(7, 1, figsize=(10, 18), gridspec_kw={"hspace": 0.5})

# Set plot title
fig.suptitle(f"Plotting for episode {id_ur} and {id_umi}")


## read data from umi

# preprocess pose data into relative pose
pose_IG = np.array(buffer_umi["data"][id_umi]["ts_pose_fb_0"])
pose_IG0 = pose_IG[0]
SE3_IG0 = su.pose7_to_SE3(pose_IG0)
SE3_G0I = su.SE3_inv(SE3_IG0)
SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
pose_G0G = np.zeros((pose_IG.shape[0], 7))
for i in range(pose_IG.shape[0]):
    SE3_IGi = su.pose7_to_SE3(pose_IG[i])
    SE3_G0G[i] = SE3_G0I @ SE3_IGi
    pose_G0G[i] = su.SE3_to_pose7(SE3_G0G[i])

# compute net wrench
wrench_left = np.array(buffer_umi["data"][id_umi]["wrench_left_0"][..., :3])
wrench_right = np.array(buffer_umi["data"][id_umi]["wrench_right_0"][..., :3])
wrench_diff = wrench_left - wrench_right

gripper = np.array(buffer_umi["data"][id_umi]["gripper_0"])

# images = np.array(buffer_umi["data"][id_umi]["rgb_0"])
time_data1 = buffer_umi["data"][id_umi]["robot_time_stamps_0"]
time_data2 = buffer_umi["data"][id_umi]["gripper_time_stamps_0"]
time_data3 = buffer_umi["data"][id_umi]["wrench_time_stamps_left_0"]

umi_robot = copy.copy(pose_G0G)
umi_robot_time = copy.copy(time_data1)


## read data from ur

# preprocess pose data into relative pose
pose_IG = np.array(buffer_ur["data"][id_ur]["ts_pose_fb_0"])
pose_IG0 = pose_IG[0]
SE3_IG0 = su.pose7_to_SE3(pose_IG0)
SE3_G0I = su.SE3_inv(SE3_IG0)
SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
pose_G0G = np.zeros((pose_IG.shape[0], 7))
for i in range(pose_IG.shape[0]):
    SE3_IGi = su.pose7_to_SE3(pose_IG[i])
    SE3_G0G[i] = SE3_G0I @ SE3_IGi
    pose_G0G[i] = su.SE3_to_pose7(SE3_G0G[i])

time_data1 = buffer_ur["data"][id_ur]["robot_time_stamps_0"]

ur_robot = copy.copy(pose_G0G)
ur_robot_time = copy.copy(time_data1)

ur_robot_time = np.array(ur_robot_time) / 1000.0 + time_offset


# plot
axes[0].plot(ur_robot_time, ur_robot[:, 0], "-", label="ur")
axes[1].plot(ur_robot_time, ur_robot[:, 1], "-", label="ur")
axes[2].plot(ur_robot_time, ur_robot[:, 2], "-", label="ur")
axes[3].plot(ur_robot_time, ur_robot[:, 3], "-", label="ur")
axes[4].plot(ur_robot_time, ur_robot[:, 4], "-", label="ur")
axes[5].plot(ur_robot_time, ur_robot[:, 5], "-", label="ur")
axes[6].plot(ur_robot_time, ur_robot[:, 6], "-", label="ur")

axes[0].plot(umi_robot_time, umi_robot[:, 0], "-", label="umi")
axes[1].plot(umi_robot_time, umi_robot[:, 1], "-", label="umi")
axes[2].plot(umi_robot_time, umi_robot[:, 2], "-", label="umi")
axes[3].plot(umi_robot_time, umi_robot[:, 3], "-", label="umi")
axes[4].plot(umi_robot_time, umi_robot[:, 4], "-", label="umi")
axes[5].plot(umi_robot_time, umi_robot[:, 5], "-", label="umi")
axes[6].plot(umi_robot_time, umi_robot[:, 6], "-", label="umi")

# Set titles
axes[0].set_title("Pose x (m)")
axes[1].set_title("Pose y (m)")
axes[2].set_title("Pose z (m)")
axes[3].set_title("Quat w")
axes[4].set_title("Quat x")
axes[5].set_title("Quat y")
axes[6].set_title("Quat z")

# legend
axes[0].legend()

plt.show()
