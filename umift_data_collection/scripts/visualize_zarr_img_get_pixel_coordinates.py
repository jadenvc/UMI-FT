import zarr
import matplotlib.pyplot as plt
import numpy as np

# ============================
# CONFIGURATION
# ============================
ZARR_PATH = "/store/real/hjchoi92/data/real/test/WBW-iph-b0/processed_data/all/acp_replay_buffer_gripper.zarr"
EPISODE = "episode_2"  # change to your desired episode
ZARR_KEY = f"data/{EPISODE}/ultrawide_0"

# ============================
# LOAD RGB IMAGES FROM ZARR
# ============================
z = zarr.open(ZARR_PATH, mode="r")
rgb_frames = np.array(z[ZARR_KEY])  # shape: (T, 3, 244, 244)
# print(rgb_frames.shape)

# Convert from (T, C, H, W) to (T, H, W, C)
# rgb_frames = np.transpose(rgb_frames, (0, 2, 3, 1))

# Validate shape
assert rgb_frames.shape[1:] == (224, 224, 3), f"Unexpected shape: {rgb_frames.shape}"

# Convert to uint8 if needed
if rgb_frames.dtype != np.uint8:
    rgb_frames = (rgb_frames * 255).astype(np.uint8)

# ============================
# DISPLAY & INTERACT
# ============================
fig, ax = plt.subplots(figsize=(6, 6))
current_frame_idx = 0
img_display = ax.imshow(rgb_frames[current_frame_idx])
plt.title(f"Frame {current_frame_idx + 1}/{len(rgb_frames)}")

red_pixels = []  # Store as list to track order

def update_frame():
    frame_copy = rgb_frames[current_frame_idx].copy()
    for x, y in red_pixels:
        if 0 <= x < 224 and 0 <= y < 224:
            frame_copy[y, x] = [255, 0, 0]
    img_display.set_data(frame_copy)
    plt.title(f"Frame {current_frame_idx + 1}/{len(rgb_frames)}")
    fig.canvas.draw_idle()

def print_red_pixels():
    print(f"Current red pixels ({len(red_pixels)} total): {red_pixels}")

def on_key(event):
    global current_frame_idx
    if event.key == "right":
        current_frame_idx = min(current_frame_idx + 1, len(rgb_frames) - 1)
        update_frame()
    elif event.key == "left":
        current_frame_idx = max(current_frame_idx - 1, 0)
        update_frame()
    elif event.key == "down":
        if red_pixels:
            removed = red_pixels.pop()
            print(f"Removed last red dot: {removed}")
            print_red_pixels()
            update_frame()

def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        red_pixels.append((x, y))
        print(f"Added red dot: (X={x}, Y={y})")
        print_red_pixels()
        update_frame()

fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()