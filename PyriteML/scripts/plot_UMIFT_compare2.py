import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
from PyriteUtility.spatial_math import spatial_utilities as su

register_codecs()

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

dataset_path_ur = dataset_folder_path + "/umi-ft/umift_debug_playback"
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
# exit(0)

# episode_1742940546
# episode_1742940641
# episode_1742940683
# episode_1742940746
# -------------------
# episode_0
# episode_1
# episode_2
# episode_3


# id_ur = "episode_1742946956"
# id_umi = "episode_0"
# time_offset = 2.9

# id_ur = "episode_1742947075"
# id_umi = "episode_1"
# time_offset = 2.62

# id_ur = "episode_1742947118"
# id_umi = "episode_2"
# time_offset = -3.2

id_ur = "episode_1742947160"
id_umi = "episode_3"
time_offset = -1.1


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

umi_rgb = buffer_umi["data"][id_umi]["rgb_0"]
umi_rgb_time = buffer_umi["data"][id_umi]["rgb_time_stamps_0"]
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


ur_rgb = buffer_ur["data"][id_ur]["rgb_0"]
ur_rgb_time = buffer_ur["data"][id_ur]["rgb_time_stamps_0"]
ur_robot = copy.copy(pose_G0G)
ur_robot_time = copy.copy(time_data1)


ur_rgb_time = np.array(ur_rgb_time) / 1000.0 + time_offset
ur_robot_time = np.array(ur_robot_time) / 1000.0 + time_offset


titles = [
    "RGB Image UMI",
    "RGB Image UR",
    "Pose X",
    "Pose Y",
    "Pose Z",
    "Pose Qw",
    "Pose Qx",
    "Pose Qy",
    "Pose Qz",
]

print("Finished loading data")

id_umi = "episode_0"

time_images1 = umi_rgb_time
time_images2 = ur_rgb_time
images1 = umi_rgb
images2 = ur_rgb

# Initialize figure and axes
fig, axes = plt.subplots(
    9,
    1,
    figsize=(10, 18),
    gridspec_kw={"height_ratios": [3, 3, 1, 1, 1, 1, 1, 1, 1], "hspace": 0.5},
)
axes[0].axis("off")  # Hide image axis
axes[1].axis("off")  # Hide image axis

# Set plot title
fig.suptitle(f"Plotting for {id_umi}")

# Plot data
for i in range(7):
    axes[i + 2].plot(umi_robot_time, umi_robot[:, i], ".-")
    axes[i + 2].plot(ur_robot_time, ur_robot[:, i], ".-")

# Set titles
for i in range(9):
    axes[i].set_title(titles[i])

# set legends for plot 1 and 3
axes[2].legend(["UMI", "UR"])

# Shared vertical cursor
cursor_time = np.min([np.min(time_images1), np.min(time_images2)])
# Initialize cursor at first timestamp
cursor_line = [
    axes[2].axvline(cursor_time, color="r", linestyle="--"),
    axes[3].axvline(cursor_time, color="r", linestyle="--"),
    axes[4].axvline(cursor_time, color="r", linestyle="--"),
    axes[5].axvline(cursor_time, color="r", linestyle="--"),
    axes[6].axvline(cursor_time, color="r", linestyle="--"),
    axes[7].axvline(cursor_time, color="r", linestyle="--"),
    axes[8].axvline(cursor_time, color="r", linestyle="--"),
]


# Function to update the displayed image
def update_image(time):
    closest_idx = np.argmin(np.abs(time_images1 - time))
    axes[0].clear()
    axes[0].imshow(images1[closest_idx])
    axes[0].axis("off")
    closest_idx = np.argmin(np.abs(time_images2 - time))
    axes[1].clear()
    axes[1].imshow(images2[closest_idx])
    axes[1].axis("off")
    fig.canvas.draw_idle()


update_image(cursor_time)


# Keyboard event handler
def on_key(event):
    global cursor_time
    step = 0.1  # Time step
    if event.key == "right":
        cursor_time += step
    elif event.key == "left":
        cursor_time -= step
    else:
        return

    print("Showing time: ", cursor_time)
    print("Image1 shape:", images1.shape)
    print("Image2 shape:", images2.shape)
    for line in cursor_line:
        line.set_xdata([cursor_time])
    update_image(cursor_time)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
