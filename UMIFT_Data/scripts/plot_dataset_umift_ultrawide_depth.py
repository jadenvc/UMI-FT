import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

import sys
import os
from typing import Dict, Callable, Tuple, List

SCRIPT_PATH = "/local/real/hjchoi92/repo/UMI-FT/scripts"
sys.path.append(os.path.join(SCRIPT_PATH, '../'))
sys.path.append(os.path.join(SCRIPT_PATH, '../../PyriteUtility'))

from PyriteUtility.spatial_math import spatial_utilities as su


import numpy as np
import zarr

if "PYRITE_DATASET_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_DATASET_FOLDERS")
dataset_folder_path = os.environ.get("PYRITE_DATASET_FOLDERS")

dataset_path = "/store/real/hjchoi92/data/real_processed/umift/lightbulb-soft/acp_replay_buffer_gripper.zarr"

print("Loading dataset from: ", dataset_path)
# load the zarr dataset from the path
# ‘r’ means read only (must exist); ‘r+’ means read/write (must exist); ‘a’ means read/write (create if doesn’t exist); ‘w’ means create (overwrite if exists); ‘w-’ means create (fail if exists).
buffer = zarr.open(dataset_path, mode="r+")

# select which episode to visualize    
ep = "episode_4"
print("Loading episode: ", ep)


print(buffer.tree())

# preprocess pose data into relative pose
pose_IG = np.array(buffer["data"][ep]["ts_pose_fb_0"]) # This is the pose of iPhone w.r.t to the global frame
pose_IG0 = pose_IG[0] # This is the initial pose of iPhone w.r.t the global frame
SE3_IG0 = su.pose7_to_SE3(pose_IG0) # T matrix of initial pose
SE3_G0I = su.SE3_inv(SE3_IG0) 
SE3_G0G = np.zeros((pose_IG.shape[0], 4, 4))
for i in range(pose_IG.shape[0]):
    SE3_IGi = su.pose7_to_SE3(pose_IG[i])
    SE3_G0G[i] = SE3_G0I @ SE3_IGi


print("finished processing pose data")

# compute net wrench
wrench_left = np.array(buffer["data"][ep]["wrench_left_0"][..., :3])
wrench_right = np.array(buffer["data"][ep]["wrench_right_0"][..., :3])
wrench_TCP = wrench_left + wrench_right
grasp_force = (-wrench_left[:, 0] + wrench_right[:, 0]) /2

print("finished processing wrench data")

images = np.array(buffer["data"][ep]["rgb_0"])
ultrawide = np.array(buffer["data"][ep]["ultrawide_0"])
depth = np.array(buffer["data"][ep]["depth_0"])
print("got images")
data1 = np.array(SE3_G0G[..., :3, 3])
print("got data1")
data2 = np.array(buffer["data"][ep]["gripper_0"])
print("got data2")

print("got data3")
data4 = np.array(buffer["data"][ep]["stiffness_0"])
# change data4 from (N,) to (N, 1)
data4 = data4.reshape(-1, 1)

print("finish distributing data for plotting")

time_images = buffer["data"][ep]["rgb_time_stamps_0"]
time_ultrawide = buffer["data"][ep]["ultrawide_time_stamps_0"]
time_depth = buffer["data"][ep]["depth_time_stamps_0"]

time_data1 = buffer["data"][ep]["robot_time_stamps_0"]
time_data2 = buffer["data"][ep]["gripper_time_stamps_0"]
time_data3 = buffer["data"][ep]["wrench_time_stamps_left_0"]
time_data4 = buffer["data"][ep]["robot_time_stamps_0"]

wrench_time = np.array(buffer["data"][ep]["wrench_time_stamps_left_0"])
rgb_time = np.array(buffer["data"][ep]["rgb_time_stamps_0"])


start_time = np.min(rgb_time)
end_time = np.max(rgb_time)
map_to_uw_idx = np.array(buffer["data"][ep]["map_to_uw_idx_0"])
map_to_d_idx = np.array(buffer["data"][ep]["map_to_d_idx_0"])
# Mask: keep only wrench data within RGB time range
mask = np.squeeze((wrench_time >= start_time) & (wrench_time <= end_time))
# Apply trim
print(mask.shape)
# data3 = wrench_TCP[mask]
# data3 = grasp_force[mask]
data3 = wrench_TCP[mask, 0:3]
time_data3 = wrench_time[mask]

titles = ["Pose", "Gripper Width", "Tool Center Point Force", "Stiffness"]


print("Finished loading data")


# Initialize figure and axes
fig = plt.figure(figsize=(12, 18))
gs = fig.add_gridspec(5, 3, height_ratios=[3, 1, 1, 1, 1], hspace=0.8, wspace=0.3)

ax_rgb = fig.add_subplot(gs[0, 0])
ax_uw = fig.add_subplot(gs[0, 1])
ax_depth = fig.add_subplot(gs[0, 2])
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :])
ax3 = fig.add_subplot(gs[3, :])
ax4 = fig.add_subplot(gs[4, :])


# ax_rgb, ax_uw, ax_depth, ax1, ax2, ax3, ax4 = axes
ax_rgb.axis("off")
ax_uw.axis("off")
ax_depth.axis("off")


# Set plot title
fig.suptitle(f"Plotting for {ep}. (Use arrowkey to toggle different timestamps)")

# Plot data
ax1.plot(time_data1, data1, '.-')
ax2.plot(time_data2, data2, '.-')
ax3.plot(time_data3, data3, '.-')
ax4.plot(time_data4, data4, '.-')

# Set titles
ax1.set_title(titles[0])
ax2.set_title(titles[1])
ax3.set_title(titles[2])
ax4.set_title(titles[3])

# set legends for plot 1 and 3
ax1.legend(["x", "y", "z"])
ax3.legend(["x", "y", "z"])

# Shared vertical cursor
cursor_time = np.min(time_images)  # Initialize cursor at first timestamp
cursor_line = [ax1.axvline(cursor_time, color='r', linestyle='--'),
               ax2.axvline(cursor_time, color='r', linestyle='--'),
               ax3.axvline(cursor_time, color='r', linestyle='--'),
               ax4.axvline(cursor_time, color='r', linestyle='--')]

# Function to update the displayed image
def update_image(time):
    closest_idx = np.argmin(np.abs(time_images - time))
    closest_grasp_force_idx = np.argmin(np.abs(time_data3 - time))

    print("Checking timestamps")
    print("time_images: ", time_images[closest_idx])
    print('time ultrawide: ', time_ultrawide[map_to_uw_idx[closest_idx]])
    print('time depth: ', time_depth[map_to_d_idx[closest_idx]])

    print("Grasp Force: ", data3[closest_grasp_force_idx])
    
    # RGB
    ax_rgb.clear()
    ax_rgb.imshow(images[closest_idx])
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")

    # Ultrawide
    ax_uw.clear()
    uw_idx = map_to_uw_idx[closest_idx]
    uw_frame = ultrawide[uw_idx]
    if uw_frame.ndim == 4 and uw_frame.shape[0] == 1:
        uw_frame = uw_frame.squeeze(0)
    ax_uw.imshow(uw_frame)
    ax_uw.set_title("Ultrawide Image")
    ax_uw.axis("off")

    # Depth
    ax_depth.clear()
    d_idx = map_to_d_idx[closest_idx]
    d_frame = depth[d_idx]
    if d_frame.ndim == 4 and d_frame.shape[-1] == 3:
        d_frame = d_frame[..., 0]  # Take one channel for grayscale
        d_frame = d_frame.squeeze(0)
    ax_depth.imshow(d_frame, cmap='gray', vmin=0.0, vmax=0.5)
    ax_depth.set_title("Depth Image")
    ax_depth.axis("off")

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
    print("Image shape:", images.shape)
    for line in cursor_line:
        line.set_xdata([cursor_time])
    update_image(cursor_time)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
