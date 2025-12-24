import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================
# CONFIGURATION
# ============================
VIDEO_PATH = "/store/real/hjchoi92/data/real/umift_data/WBW90-250305-real-HC-batch0/DCIM/100GOPRO/GX013720.MP4"  # Change to your actual MP4 file path
TARGET_WIDTH, TARGET_HEIGHT = 270, 202  # Downscaled frame size

# ============================
# LOAD VIDEO & CONVERT TO FRAMES
# ============================
cap = cv2.VideoCapture(VIDEO_PATH)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no more frames

    # Resize frame from [2704, 2028] → [270, 202]
    frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR (OpenCV format) → RGB (Matplotlib format)
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    frames.append(frame_resized)

cap.release()  # Release video capture
print(f"Loaded {len(frames)} frames from {VIDEO_PATH}")

# Create a copy of frames to modify (preserving original)
edited_frames = [frame.copy() for frame in frames]

# ============================
# DISPLAY & INTERACT
# ============================
fig, ax = plt.subplots(figsize=(6, 5))
current_frame_idx = 0  # Start at first frame
img_display = ax.imshow(edited_frames[current_frame_idx])  # Show first frame
plt.title(f"Frame {current_frame_idx + 1}/{len(frames)}")

# Dictionary to store red pixel positions (shared across all frames)
red_pixels = set()  # Using a set to prevent duplicates

# Function to update the displayed frame
def update_frame():
    updated_frame = edited_frames[current_frame_idx]  # Reference the modified frame
    for x, y in red_pixels:  # Draw all stored red pixels
        updated_frame[y, x] = [255, 0, 0]  # Red color in RGB
    img_display.set_data(updated_frame)
    plt.title(f"Frame {current_frame_idx + 1}/{len(frames)}")
    fig.canvas.draw_idle()

# Keyboard navigation handler
def on_key(event):
    global current_frame_idx
    if event.key == "right":  # Next frame
        current_frame_idx = min(current_frame_idx + 1, len(frames) - 1)
    elif event.key == "left":  # Previous frame
        current_frame_idx = max(current_frame_idx - 1, 0)
    update_frame()

# Mouse click handler to get pixel coordinates & mark red (persists across frames)
def on_click(event):
    if event.inaxes:  # Ensure click was inside the image
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked Coordinates: (X={x}, Y={y}) in {TARGET_WIDTH}x{TARGET_HEIGHT} space")
        red_pixels.add((x, y))  # Store clicked pixel globally
        for i in range(len(edited_frames)):  # Apply red pixel to all frames
            edited_frames[i][y, x] = [255, 0, 0]
        update_frame()

# Connect event handlers
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
