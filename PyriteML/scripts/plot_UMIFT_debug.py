import os

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PyriteUtility.spatial_math import spatial_utilities as su

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

dataset_path = dataset_folder_path + "/umi-ft/WBW90/acp_replay_buffer_gripper.zarr"

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

# for ep, ep_data in buffer["data"].items():
#     print(ep)


start_id = 0
end_id = 200


# Initialize figure and axes
fig, axes = plt.subplots(7, 1, figsize=(10, 18), gridspec_kw={"hspace": 0.5})

# Set plot title
fig.suptitle(f"Plotting for episode {start_id} to {end_id}")

# pose x,y,z
# gripper
# wrench x y z

for i in range(start_id, end_id):
    ep = "episode_" + str(i)
    # preprocess pose data into relative pose
    pose_IG = np.array(buffer["data"][ep]["ts_pose_fb_0"])
    pose_IG0 = pose_IG[0]
    SE3_IG0 = su.pose7_to_SE3(pose_IG0)
    SE3_G0I = su.SE3_inv(SE3_IG0)
    SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
    for i in range(pose_IG.shape[0]):
        SE3_IGi = su.pose7_to_SE3(pose_IG[i])
        SE3_G0G[i] = SE3_G0I @ SE3_IGi

    # compute net wrench
    wrench_left = np.array(buffer["data"][ep]["wrench_left_0"][..., :3])
    wrench_right = np.array(buffer["data"][ep]["wrench_right_0"][..., :3])
    wrench_diff = wrench_left - wrench_right

    gripper = np.array(buffer["data"][ep]["gripper_0"])

    # images = np.array(buffer["data"][ep]["rgb_0"])
    time_data1 = buffer["data"][ep]["robot_time_stamps_0"]
    time_data2 = buffer["data"][ep]["gripper_time_stamps_0"]
    time_data3 = buffer["data"][ep]["wrench_time_stamps_left_0"]

    axes[0].plot(time_data1, SE3_G0G[:, 0, 3], "-")
    axes[1].plot(time_data1, SE3_G0G[:, 1, 3], "-")
    axes[2].plot(time_data1, SE3_G0G[:, 2, 3], "-")
    axes[3].plot(time_data2, gripper, "-")
    axes[4].plot(time_data3, wrench_diff[:, 0], "-")
    axes[5].plot(time_data3, wrench_diff[:, 1], "-")
    axes[6].plot(time_data3, wrench_diff[:, 2], "-")


# Set titles
axes[0].set_title("Pose x")
axes[1].set_title("Pose y")
axes[2].set_title("Pose z")
axes[3].set_title("Gripper")
axes[4].set_title("Wrench x")
axes[5].set_title("Wrench y")
axes[6].set_title("Wrench z")

plt.show()
