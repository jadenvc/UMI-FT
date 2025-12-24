import os
import time

import matplotlib.pyplot as plt
import numpy as np
import PyriteUtility.spatial_math.spatial_utilities as su
import zarr
from einops import rearrange
from hardware_interfaces.workcell.table_top_manip.python import (
    manip_server_pybind as ms,
)
from PyriteUtility.common import GracefulKiller
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform
from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs
from PyriteUtility.plotting.matplotlib_helpers import set_axes_equal
from PyriteUtility.umi_utils.usb_util import reset_all_elgato_devices

register_codecs()

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")


### PLOTTING FUNCTIONS ###
# TODO (Chuer): reorg them in PyriteUtility
def plot_camera_frame(ax, pose, axis_length=0.01):
    # Extract translation
    translation = pose[0:3, 3]

    # Define the local axes in the camera frame
    axes = np.array(
        [
            [1, 0, 0],  # X-axis
            [0, 1, 0],  # Y-axis
            [0, 0, 1],
        ]
    )  # Z-axis

    # Draw each axis
    for i, color in zip(range(3), ["r", "g", "b"]):
        # Calculate the end point of the axis
        end_point = translation + axis_length * (pose[0:3, 0:3] @ axes[i])
        # Plot the axis line
        ax.plot(
            [translation[0], end_point[0]],
            [translation[1], end_point[1]],
            [translation[2], end_point[2]],
            color=color,
            linewidth=2,
        )


def plot_images(img_A, img_B, opacity_A=0.6):
    """
    Plots three panels: img_A, img_B, and an overlay of both with
    img_A at 70% opacity and img_B at 30% opacity.

    Args:
        img_A: (1, H, W, 3) NumPy array (observation)
        img_B: (1, H, W, 3) NumPy array (data)
        opacity_A: float, opacity for image A
    """
    # Remove the batch dimension (1, H, W, 3) -> (H, W, 3)
    img_A = img_A.squeeze(0).astype(np.uint8)
    img_B = img_B.squeeze(0)

    H, W = img_A.shape[:2]
    wi, hi = img_B.shape[:2]

    # img tranform data img to obs img shape and crop
    tf = get_image_transform(input_res=(hi, wi), output_res=(W, H), bgr_to_rgb=True)
    img_B = tf(img_B)

    # Ensure images are the same shape
    assert (
        img_A.shape == img_B.shape
    ), f"Images must have the same dimensions, img_A.shape: {img_A.shape}, img_B.shape: {img_B.shape}"

    overlay = opacity_A * img_A + (1 - opacity_A) * img_B
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_A)
    axes[0].set_title("Obs")
    axes[0].axis("off")

    axes[1].imshow(img_B)
    axes[1].set_title("Data")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Initial Pose for Obs with Data")
    axes[2].axis("off")

    plt.show()


dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")
# (CHUER): TBCHANGED
dataset_path = (
    dataset_folder_path + "/umift/WBW_250217_batch0/acp_replay_buffer_gripper.zarr"
)
id_list = [0]
print("dataset_path:", dataset_path)
buffer = zarr.open(dataset_path, mode="r")

input_res = (224, 224)
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
config_folders = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")

reset_all_elgato_devices()
hardware_config_path = config_folders + "/right_arm_coinft.yaml"

Tr = np.eye(6)
n_af = 0
default_stiffness = [5000, 5000, 5000, 100, 100, 100]

# manipulation server
server = ms.ManipServer()

if not server.initialize(hardware_config_path):
    raise RuntimeError("Failed to initialize hardware server.")
server.set_high_level_maintain_position()
if server.is_bimanual():
    id_list = [0, 1]
else:
    id_list = [0]

default_stiffness = np.diag(default_stiffness)

rgb_row_combined_buffer = []
rgb_buffer = []
output_rgb_buffer = []
for id in id_list:
    server.set_force_controlled_axis(Tr, n_af, id)
    server.set_stiffness_matrix(default_stiffness, id)

print("[Main] Waiting for hardware server to be ready...")
while not server.is_ready():
    time.sleep(0.1)
print("[Main] Hardware server is ready. Now reading data...")

# read data
ep = "episode_0"
data = buffer["data"][ep]

# get initial rgb frame and align it with initial frame of data episode being played back
query_size_sparse_rgb = 11
obs_rgb_raw = server.get_camera_rgb(query_size_sparse_rgb, 0)
obs_rgb_reshape = rearrange(
    obs_rgb_raw, "(c h) (n w)->n h w c", c=3, n=query_size_sparse_rgb
)
plot_images(img_A=obs_rgb_reshape[0:1], img_B=data["rgb_0"][0:1])
input("Press ENTER to continue ... ")

for key in data.keys():
    print("data keys: ", key)

print(data["ts_pose_fb_0"].shape)

# get current pose
pose_W0 = np.squeeze(server.get_pose(1, 0))
SE3_W0 = su.pose7_to_SE3(pose_W0)

# read the first pose wp
pose_X0 = data["ts_pose_fb_0"][0]
SE3_0X = su.SE3_inv(su.pose7_to_SE3(pose_X0))

# read and convert all data to world
N = data["ts_pose_fb_0"].shape[0]
pose_WT = np.zeros((7, N))
pose_SE3_WT = np.zeros((N, 4, 4))

for i in range(N):
    pose_XT = data["ts_pose_fb_0"][i]
    SE3_XT = su.pose7_to_SE3(pose_XT)
    SE3_0T = SE3_0X @ SE3_XT
    SE3_WT = SE3_W0 @ SE3_0T
    pose_WT[:, i] = su.SE3_to_pose7(SE3_WT)
    pose_SE3_WT[i] = SE3_WT

# read all Gripper data
Ng = data["gripper_0"].shape[0]
assert Ng == N
gripper = data["gripper_0"]
print("[Main] Finished reading data. Now playing back...")

# PLOTTING poses to play back
plt.ion()
fig = plt.figure()
ax = plt.axes(projection="3d")
x = np.linspace(-0.02, 0.2, 20)
y = np.linspace(-0.1, 0.1, 20)
z = np.linspace(-0.1, 0.1, 20)
ax.set_title("Playback EE Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

for pose in pose_SE3_WT:
    plot_camera_frame(ax, pose)

plt.show()
input("Press Enter to continue...")
plt.close()

# plot gripper
plt.ion()
plt.plot(gripper, marker="o", linestyle="-")
plt.show()
plt.title("Gripper")
input("Press Enter to continue...")
plt.close()

# Create Plot for Short Horizon Play Back
plt.ion()
fig = plt.figure()
ax = plt.axes(projection="3d")
x = np.linspace(-0.02, 0.2, 20)
y = np.linspace(-0.1, 0.1, 20)
z = np.linspace(-0.1, 0.1, 20)
ax.set_title("Playback EE Trajectory: Current Horizon")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# start playing back
dN = 200  # number of waypoints to send at a time
i = 0

while i < N:
    killer = GracefulKiller()
    end_id = min(i + dN, N)
    pose_cmd = pose_WT[:, i:end_id]  # 7 x dN
    gp_pos_cmd = gripper[i:end_id] * 1000
    gp_f_cmd = np.zeros((len(gp_pos_cmd), 1))
    gp_cmd = np.concatenate((gp_pos_cmd, gp_f_cmd), axis=1).transpose()  # 2 x dN

    # Plotting current window trajectory
    ax.cla()
    for pose in pose_SE3_WT[i:end_id]:
        plot_camera_frame(ax, pose)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)
    plt.draw()

    i += dN
    pose_fb = np.squeeze(server.get_pose(1, 0))
    print("gp_cmd:", gp_cmd)

    input("Press Enter to execute...")
    timenow_ms = server.get_timestamp_now_ms()
    time_cmd = timenow_ms + 100 + np.arange(dN) * 30
    server.schedule_waypoints(pose_cmd, time_cmd, 0)
    server.schedule_eoat_waypoints(gp_cmd, time_cmd, 0)
    time.sleep(0.1)
