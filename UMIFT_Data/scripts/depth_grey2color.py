import cv2
import os

input_path = "/Users/hojungchoi/Desktop/code_hub/umift_overall/UMI-FT/data/UMI_iPhone/2025-03-28/2025-03-28T17-23-43.659Z_64187_WBW300-b8_demonstration_left/2025-03-28T17-23-43.659Z_64187_WBW300-b8_demonstration_left_depth.mp4"


def colorize_depth_video(input_path, colormap_name='TURBO'):
    # Map string to OpenCV colormap constants
    colormap_dict = {
        'TURBO': cv2.COLORMAP_TURBO,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'JET': cv2.COLORMAP_JET,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'HOT': cv2.COLORMAP_HOT
    }

    if colormap_name not in colormap_dict:
        raise ValueError(f"Invalid colormap name. Choose from: {list(colormap_dict.keys())}")

    # Build output path by replacing _left_depth with _left_depth_color
    dir_path, file_name = os.path.split(input_path)
    base_name, ext = os.path.splitext(file_name)
    if "_left_depth" in base_name:
        new_base_name = base_name.replace("_left_depth", "_left_depth_color")
    else:
        raise ValueError("Expected '_left_depth' in the filename.")
    output_path = os.path.join(dir_path, new_base_name + ext)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            print("Input frame is in color, converting to grayscale for depth processing.")
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame

        norm_frame = cv2.normalize(frame_gray, None, 0, 255, cv2.NORM_MINMAX)
        color_frame = cv2.applyColorMap(norm_frame.astype('uint8'), colormap_dict[colormap_name])
        out.write(color_frame)

    cap.release()
    out.release()

    print(f"Saved colorized video to: {output_path}")
    return output_path

# Example usage
colorize_depth_video(input_path, colormap_name='TURBO')  # You can switch to 'TURBO', 'VIRIDIS', etc.
