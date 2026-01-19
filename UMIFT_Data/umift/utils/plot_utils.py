import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from umift.utils.time_utils import array_isostringformat_to_timestamp
from umift.utils.print_utils import info_print, debug_print

def plot_time_aligned_trajectories(time_aligned_trajectories, colors=None, axis_size=0.04, include_base_frame=True):
    # time_aligned_trajectories is a list of matrices with shape (T, 4, 4), where T is the number of time steps. Returns list of Open3D geometry objects that can be visualized.
    T = len(time_aligned_trajectories[0])  # Number of time steps

    if colors is None:
        colors = [[1,0,0], [0,1,0], [0,0,1], [0.6,0,0.6]]  # Default colors

    assert len(colors) >= len(time_aligned_trajectories)

    geometries = []

    for i, trajectory in enumerate(time_aligned_trajectories):
        assert trajectory.shape == (T, 4, 4)

        trajectory_1 = trajectory[:,:3,3]

        # Create LineSet for trajectory 1
        lines_1 = [[i, i + 1] for i in range(T - 1)]
        colors_1 = [colors[i] for j in range(T - 1)]

        line_set_1 = o3d.geometry.LineSet()
        line_set_1.points = o3d.utility.Vector3dVector(trajectory_1)
        line_set_1.lines = o3d.utility.Vector2iVector(lines_1)
        line_set_1.colors = o3d.utility.Vector3dVector(colors_1)

        def plot_trajectory_with_rgb_axes(poses, K=1):
            """
            Plots the trajectory with RGB axes using Open3D.

            Parameters:
            - poses: numpy array of shape (T, 4, 4), where T is the number of trajectories.
            - K: Subsampling factor, use every K-th frame in the trajectory.
            """
            # Create a list to hold the frames for visualization
            frames = []
            
            # Subsample the poses
            subsampled_poses = poses[::K]
            
            for i, pose in enumerate(subsampled_poses):
                # Extract the translation vector
                translation = pose[:3, 3]
                
                # Extract the rotation matrix
                rotation = pose[:3, :3]
                
                # Create the coordinate frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=translation)
                
                # Apply the rotation to the frame
                frame.rotate(rotation, center=translation)
                
                # Append the frame to the list
                frames.append(frame)
            
            # Create an Open3D visualization object
            return frames

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2*axis_size)
        frames = plot_trajectory_with_rgb_axes(trajectory, K=30)

        if len(trajectory_1) > 1:
            geometries.append(line_set_1)
        if include_base_frame:
            geometries.append(base_frame)
        geometries.extend(frames)
    
    return geometries

def interpolate_ft_data(ft_data, ft_timestamps, frame_timestamps):
    """Interpolate F/T data across each dimension to match frame timestamps."""
    interpolated_ft = np.zeros((len(frame_timestamps), 6))
    for i in range(6):
        interpolated_ft[:, i] = np.interp(frame_timestamps, ft_timestamps, ft_data[:, i])
    return interpolated_ft

def create_ft_plot(ft_data, current_idx, window_size=100):
    """Create force/torque plot for the current frame."""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    # Create two subplots for forces and torques
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Calculate plot range
    start_idx = max(0, current_idx - window_size)
    end_idx = current_idx + 1
    
    # Plot forces
    ax1.plot(ft_data[start_idx:end_idx, 0], label='Fx', color='r')
    ax1.plot(ft_data[start_idx:end_idx, 1], label='Fy', color='g')
    ax1.plot(ft_data[start_idx:end_idx, 2], label='Fz', color='b')
    ax1.set_ylabel('Force (N)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot torques
    ax2.plot(ft_data[start_idx:end_idx, 3], label='Tx', color='r')
    ax2.plot(ft_data[start_idx:end_idx, 4], label='Ty', color='g')
    ax2.plot(ft_data[start_idx:end_idx, 5], label='Tz', color='b')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time steps')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to image
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    return plot_image

def create_ft_plot_fixed_scale(ft_data, current_idx):
    """Create force/torque plot for the current frame with fixed timeline."""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    # Create two subplots for forces and torques
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Plot forces
    time_steps = np.arange(len(ft_data))
    ax1.plot(time_steps, ft_data[:, 0], label='Fx', color='r')
    ax1.plot(time_steps, ft_data[:, 1], label='Fy', color='g')
    ax1.plot(time_steps, ft_data[:, 2], label='Fz', color='b')
    ax1.axvline(current_idx, color='k', linestyle='--', label='Current Frame')  # Marker for the current frame
    ax1.set_ylabel('Force (N)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot torques
    ax2.plot(time_steps, ft_data[:, 3], label='Tx', color='r')
    ax2.plot(time_steps, ft_data[:, 4], label='Ty', color='g')
    ax2.plot(time_steps, ft_data[:, 5], label='Tz', color='b')
    ax2.axvline(current_idx, color='k', linestyle='--', label='Current Frame')  # Marker for the current frame
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time steps')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to image
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    return plot_image

def create_ft_plot_up_to_current(ft_data, current_idx):
    """Create force/torque plot up to the current frame."""
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    # Create two subplots for forces and torques
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Plot forces up to the current index
    time_steps = np.arange(current_idx + 1)  # Only up to the current index
    ax1.plot(time_steps, ft_data[:current_idx + 1, 0], label='Fx', color='r')
    ax1.plot(time_steps, ft_data[:current_idx + 1, 1], label='Fy', color='g')
    ax1.plot(time_steps, ft_data[:current_idx + 1, 2], label='Fz', color='b')
    ax1.axvline(current_idx, color='k', linestyle='--')  # Marker for the current frame
    ax1.set_ylabel('Force (N)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot torques up to the current index
    ax2.plot(time_steps, ft_data[:current_idx + 1, 3], label='Tx', color='r')
    ax2.plot(time_steps, ft_data[:current_idx + 1, 4], label='Ty', color='g')
    ax2.plot(time_steps, ft_data[:current_idx + 1, 5], label='Tz', color='b')
    ax2.axvline(current_idx, color='k', linestyle='--')  # Marker for the current frame
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time steps')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to image
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    return plot_image

def create_ft_plot_up_to_current_with_fixed_scale(ft_data, current_idx):
    """Create force/torque plot up to the current frame, with fixed timeline and y-axis scales."""
    # Initialize the figure and canvas
    fig = Figure(figsize=(6, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    
    # Create subplots for forces and torques
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Define the full range of the timeline for consistent axis scaling
    time_steps = np.arange(len(ft_data))
    current_time_steps = np.arange(current_idx + 1)  # Truncated up to current index

    # Compute y-axis limits with margins
    force_min, force_max = ft_data[:, :3].min(), ft_data[:, :3].max()
    torque_min, torque_max = ft_data[:, 3:].min(), ft_data[:, 3:].max()
    force_margin = 0.1 * (force_max - force_min)
    torque_margin = 0.1 * (torque_max - torque_min)
    force_ymin, force_ymax = force_min - force_margin, force_max + force_margin
    torque_ymin, torque_ymax = torque_min - torque_margin, torque_max + torque_margin

    # Plot forces up to the current index
    ax1.plot(current_time_steps, ft_data[:current_idx + 1, 0], label='Fx', color='r')
    ax1.plot(current_time_steps, ft_data[:current_idx + 1, 1], label='Fy', color='g')
    ax1.plot(current_time_steps, ft_data[:current_idx + 1, 2], label='Fz', color='b')
    ax1.axvline(current_idx, color='k', linestyle='--')  # Highlight the current frame
    ax1.set_xlim(0, len(ft_data) - 1)  # Fixed x-axis range for the entire timeline
    ax1.set_ylim(force_ymin, force_ymax)  # Fixed y-axis range for forces
    ax1.set_ylabel('Force (N)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot torques up to the current index
    ax2.plot(current_time_steps, ft_data[:current_idx + 1, 3], label='Tx', color='r')
    ax2.plot(current_time_steps, ft_data[:current_idx + 1, 4], label='Ty', color='g')
    ax2.plot(current_time_steps, ft_data[:current_idx + 1, 5], label='Tz', color='b')
    ax2.axvline(current_idx, color='k', linestyle='--')  # Highlight the current frame
    ax2.set_xlim(0, len(ft_data) - 1)  # Fixed x-axis range for the entire timeline
    ax2.set_ylim(torque_ymin, torque_ymax)  # Fixed y-axis range for torques
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time steps')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Adjust layout for aesthetics
    fig.tight_layout()
    
    # Convert the plot to an image for integration into video
    canvas.draw()
    plot_image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    return plot_image

def create_ft_plot_up_to_current_with_fixed_scale_plotly(ft_data, current_idx):
    """Create force/torque plot up to the current frame, with fixed timeline and y-axis scales using Plotly."""
    # Define the full range of the timeline for consistent axis scaling
    time_steps = np.arange(len(ft_data))
    current_time_steps = np.arange(current_idx + 1)  # Truncated up to current index

    # Compute y-axis limits with margins
    force_min, force_max = ft_data[:, :3].min(), ft_data[:, :3].max()
    torque_min, torque_max = ft_data[:, 3:].min(), ft_data[:, 3:].max()
    force_margin = 0.1 * (force_max - force_min)
    torque_margin = 0.1 * (torque_max - torque_min)
    force_ymin, force_ymax = force_min - force_margin, force_max + force_margin
    torque_ymin, torque_ymax = torque_min - torque_margin, torque_max + torque_margin

    # Create a subplot figure with two rows (forces and torques)
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        # subplot_titles=("Force (N)", "Torque (Nm)")
                        )

    # Add force plots to the first subplot
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 0], mode='lines', name='Fx', line=dict(color='tomato')), row=1, col=1)
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 1], mode='lines', name='Fy', line=dict(color='forestgreen')), row=1, col=1)
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 2], mode='lines', name='Fz', line=dict(color='steelblue')), row=1, col=1)
    fig.add_vline(x=current_idx, line=dict(color='palevioletred', dash='dash'), row=1, col=1)
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 3], mode='lines', name='Tx', line=dict(color='lightcoral')), row=2, col=1)
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 4], mode='lines', name='Ty', line=dict(color='lightgreen')), row=2, col=1)
    fig.add_trace(go.Scatter(x=current_time_steps, y=ft_data[:current_idx + 1, 5], mode='lines', name='Tz', line=dict(color='lightskyblue')), row=2, col=1)
    fig.add_vline(x=current_idx, line=dict(color='palevioletred', dash='dash'), row=2, col=1)
    fig.update_yaxes(range=[force_ymin, force_ymax], title_text='Force (N)', row=1, col=1)
    fig.update_yaxes(range=[torque_ymin, torque_ymax], title_text='Torque (Nm)', row=2, col=1)
    fig.update_xaxes(range=[0, len(ft_data) - 1], title_text='Time', row=2, col=1)

    # Update layout for aesthetics
    fig.update_layout(title_text='F/T',
                      height=600, width=800,
                      legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
                      margin=dict(l=40, r=40, t=40, b=40))

    return fig

def create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(ft_data, timestamps, current_idx, coinft_side='left'):
    """
    Create force/torque plot up to the current frame, with fixed elapsed time-based X-axis and y-axis scales using Plotly.

    Args:
        ft_data: numpy array of shape (N, 6) containing force (Fx, Fy, Fz) and torque (Tx, Ty, Tz) data.
        timestamps: numpy array of shape (N,) containing timestamps as floats in datetime.timestamp() format.
        current_idx: integer index of the current frame.
        coinft_side: string indicating the side of the coinft sensor ('left' or 'right').

    Returns:
        Plotly figure object.
    """
    # Ensure inputs are valid
    assert len(ft_data) == len(timestamps), "F/T data and timestamps must have the same length."
    
    # Normalize timestamps to start at 0 seconds
    elapsed_time = timestamps - timestamps[0]
    current_elapsed_time = elapsed_time[:current_idx + 1]  # Elapsed time up to the current frame

    # Compute y-axis limits with margins
    force_min, force_max = ft_data[:, :3].min(), ft_data[:, :3].max()
    torque_min, torque_max = ft_data[:, 3:].min(), ft_data[:, 3:].max()
    force_margin = 0.1 * (force_max - force_min) if force_max > force_min else 0.1
    torque_margin = 0.1 * (torque_max - torque_min) if torque_max > torque_min else 0.1
    force_ymin, force_ymax = force_min - force_margin, force_max + force_margin
    torque_ymin, torque_ymax = torque_min - torque_margin, torque_max + torque_margin

    # Create a subplot figure with two rows (forces and torques)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
    )

    # Add force plots to the first subplot
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 0],
        mode='lines', name='Fx', line=dict(color='tomato')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 1],
        mode='lines', name='Fy', line=dict(color='forestgreen')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 2],
        mode='lines', name='Fz', line=dict(color='steelblue')
    ), row=1, col=1)

    # Add torque plots to the second subplot
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 3],
        mode='lines', name='Tx', line=dict(color='lightcoral')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 4],
        mode='lines', name='Ty', line=dict(color='lightgreen')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=current_elapsed_time, y=ft_data[:current_idx + 1, 5],
        mode='lines', name='Tz', line=dict(color='lightskyblue')
    ), row=2, col=1)

    # Add a vertical line at the current elapsed time
    fig.add_vline(
        x=elapsed_time[current_idx], line=dict(color='palevioletred', dash='dash'), row=1, col=1
    )
    fig.add_vline(
        x=elapsed_time[current_idx], line=dict(color='palevioletred', dash='dash'), row=2, col=1
    )

    # Update y-axes with fixed ranges
    fig.update_yaxes(range=[force_ymin, force_ymax], title_text='Force (N)', row=1, col=1)
    fig.update_yaxes(range=[torque_ymin, torque_ymax], title_text='Torque (Nm)', row=2, col=1)

    # Update x-axis with elapsed time range
    fig.update_xaxes(
        range=[0, elapsed_time[-1]],  # Elapsed time from 0 to the end
        title_text='Elapsed Time (s)',
        row=2, col=1
    )

    # Update layout for aesthetics
    fig.update_layout(
        title_text='Force/Torque Over Elapsed Time {} Coinft'.format(coinft_side.capitalize()),
        height=600, width=800,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=-0.2),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def create_gripper_width_plot_up_to_current(gripper_widths, frame_index):
    """
    Create a plot for gripper widths up to the current frame, with fixed Y-axis range and total frame-based X-axis.

    Args:
        gripper_widths: numpy array of shape (N) containing gripper widths
        frame_index: current frame index

    Returns:
        Plotly figure object
    """
    total_frames = len(gripper_widths)
    current_time_steps = np.arange(frame_index + 1)  # X-axis up to the current frame

    # Compute fixed Y-axis range with margins
    gripper_min, gripper_max = gripper_widths.min(), gripper_widths.max()
    margin = 0.1 * (gripper_max - gripper_min) if gripper_max > gripper_min else 0.1
    y_min = gripper_min - margin
    y_max = gripper_max + margin

    # Create the figure
    fig = go.Figure()

    # Add the gripper width line plot
    fig.add_trace(go.Scatter(
        x=current_time_steps,
        y=gripper_widths[:frame_index + 1],
        mode='lines',
        name='Gripper Width',
        line=dict(color='gold')
    ))

    # Add a vertical line indicating the current frame
    fig.add_vline(x=frame_index, line=dict(color='palevioletred', dash='dash'))

    # Update layout with fixed axis ranges and labels
    fig.update_layout(
        title="Gripper Width",
        xaxis_title="Time Step",
        yaxis_title="Gripper Width (m)",
        xaxis=dict(range=[0, total_frames - 1]),
        yaxis=dict(range=[y_min, y_max]),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig

def create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, timestamps, frame_index):
    """
    Create a plot for gripper widths up to the current frame, with fixed Y-axis range and elapsed time-based X-axis.

    Args:
        gripper_widths: numpy array of shape (N) containing gripper widths.
        timestamps: numpy array of shape (N,) containing timestamps as floats in datetime.timestamp() format.
        frame_index: Current frame index.

    Returns:
        Plotly figure object.
    """
    # Ensure inputs are valid
    assert len(gripper_widths) == len(timestamps), "Gripper widths and timestamps must have the same length."

    # Normalize timestamps to start from 0 ms
    elapsed_time = (timestamps - timestamps[0]) * 1000  # Convert to milliseconds
    current_elapsed_time = elapsed_time[:frame_index + 1]  # Up to current frame

    # Compute fixed Y-axis range with margins
    gripper_min, gripper_max = gripper_widths.min(), gripper_widths.max()
    margin = 0.1 * (gripper_max - gripper_min) if gripper_max > gripper_min else 0.1
    y_min = gripper_min - margin
    y_max = gripper_max + margin

    # Create the figure
    fig = go.Figure()

    # Add the gripper width line plot
    fig.add_trace(go.Scatter(
        x=current_elapsed_time,
        y=gripper_widths[:frame_index + 1],
        mode='lines',
        name='Gripper Width',
        line=dict(color='gold')
    ))

    # Add a vertical line indicating the current elapsed time
    fig.add_vline(x=elapsed_time[frame_index], line=dict(color='palevioletred', dash='dash'))

    # Update layout with fixed axis ranges and labels
    fig.update_layout(
        title="Gripper Width Over Time",
        xaxis_title="Elapsed Time (ms)",
        yaxis_title="Gripper Width (m)",
        xaxis=dict(
            range=[0, elapsed_time[-1]],  # Elapsed time from 0 ms to the end
        ),
        yaxis=dict(range=[y_min, y_max]),
        height=300,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    return fig


import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_video_aligned_trajectory_offscreen(frames, poses, ft_data_left, ft_data_right, ft_timestamps_left, ft_timestamps_right, 
                                  frame_timestamps, gripper_widths,
                                  video_output_path, fps=60, max_frames=-1, 
                                  skip_start_frames=0, skip_end_frames=0):
    """
    Given poses and F/T data synchronized with video frames, plot everything together.
    
    Args:
        frames: numpy array of shape (N, H, W, 3) containing video frames in BGR format
        poses: list of poses synchronized with the frames
        ft_data_left: numpy array of shape (M, 6) containing force/torque measurements for left coinft
        ft_data_right: numpy array of shape (M, 6) containing force/torque measurements for right coinft
        ft_timestamps_left: numpy array of shape (M,) containing F/T timestamps for left
        ft_timestamps_right: numpy array of shape (M,) containing F/T timestamps for right
        frame_timestamps: numpy array of shape (N,) containing frame timestamps
        video_output_path: output path for the video
        fps: frames per second for output video
        max_frames: maximum number of frames to process (-1 for all)
        skip_start_frames: number of frames to skip at start
        skip_end_frames: number of frames to skip at end
    """
    # Get video dimensions from frames array
    T, H, W, _ = frames.shape
    assert T - skip_start_frames - skip_end_frames == len(poses)
    assert len(frame_timestamps) == T
 
    # Interpolate F/T data to match frame timestamps
    interpolated_ft_left = interpolate_ft_data(ft_data_left, ft_timestamps_left, frame_timestamps)
    interpolated_ft_right = interpolate_ft_data(ft_data_right, ft_timestamps_right, frame_timestamps)

    # Define rendering resolution
    scale = 360 / H
    out_H = int(H * scale)
    out_W = int(W * scale)

    # Create Offscreen Renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
    scene = renderer.scene

    if gripper_widths is not None:
        ft_plot_height = out_H * 2 * 2  # For left and right F/T plots
        gripper_plot_height = int(out_H * 4 / 5)  # Smaller plot for gripper width
        combined_width = out_W * 2
        combined_height = ft_plot_height + gripper_plot_height + out_H
    else:
        ft_plot_height = out_H * 2 * 2  # For left and right F/T plots
        combined_width = out_W * 2
        combined_height = ft_plot_height + out_H
        
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'avc1'), 
                                   fps, (combined_width, combined_height))

    num_frames = T if max_frames == -1 else min(max_frames + skip_start_frames, T)
    
    for i in tqdm(range(num_frames)):
        if i < skip_start_frames:
            continue
        if i == num_frames - skip_end_frames:
            break
            
        # Clear previous geometries
        scene.clear_geometry()
        
        # Add the trajectory points up to the current frame
        current_trajectory = poses[:i+1]
        geometries = plot_time_aligned_trajectories([current_trajectory], include_base_frame=False, colors=[[0,0,0]])
        
        for j, geometry in enumerate(geometries):
            scene.add_geometry(f"trajectory_{j}", geometry, o3d.visualization.rendering.MaterialRecord())

        # Set up camera
        scene.camera.look_at([1, 1, 1], [0, 0, 0], [0, 1, 0])
        scene.camera.set_projection(60, out_W / out_H, 0.1, 10.0)

        # Render trajectory image
        image = renderer.render_to_image()
        trajectory_image = np.asarray(image).astype(np.uint8)

        # Resize trajectory image
        trajectory_image_resized = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)

        # Get the video frame and resize
        video_frame = frames[i]
        video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)

        # Create F/T plots
        ft_plot_left = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_left, frame_timestamps, i, coinft_side='left')
        ft_plot_right = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_right, frame_timestamps, i, coinft_side='right')

        # Convert the Plotly figure to an image
        ft_plot_image_left = ft_plot_left.to_image(format="png")
        ft_plot_left = cv2.imdecode(np.frombuffer(ft_plot_image_left, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_left = cv2.resize(ft_plot_left, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        ft_plot_image_right = ft_plot_right.to_image(format="png")
        ft_plot_right = cv2.imdecode(np.frombuffer(ft_plot_image_right, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_right = cv2.resize(ft_plot_right, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        if gripper_widths is not None:
            # Create gripper width plot
            gripper_plot = create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
            gripper_plot_image = gripper_plot.to_image(format="png")
            gripper_plot = cv2.imdecode(np.frombuffer(gripper_plot_image, np.uint8), cv2.IMREAD_COLOR)
            gripper_plot = cv2.resize(gripper_plot, (combined_width, gripper_plot_height), interpolation=cv2.INTER_AREA)

        # Combine video and trajectory side by side
        upper_half = np.hstack((video_frame, trajectory_image_resized[:, :, ::-1]))  # Convert RGB to BGR
        
        if gripper_widths is not None:
            # Combine upper half and F/T plot vertically
            combined_image = np.vstack((upper_half, ft_plot_left, ft_plot_right, gripper_plot))
        else:
            # Combine upper half and F/T plot vertically
            combined_image = np.vstack((upper_half, ft_plot_left, ft_plot_right))

        video_writer.write(combined_image)

    # Release resources
    cv2.destroyAllWindows()
    video_writer.release()
    renderer = None  # Explicitly remove renderer to free memory

def plot_video_aligned_trajectory(frames, poses, ft_data_left, ft_data_right, ft_timestamps_left, ft_timestamps_right, 
                                  frame_timestamps, gripper_widths,
                                video_output_path, fps=60, max_frames=-1, 
                                skip_start_frames=0, skip_end_frames=0):
    """
    Given poses and F/T data synchronized with video frames, plot everything together.
    
    Args:
        frames: numpy array of shape (N, H, W, 3) containing video frames in BGR format
        poses: list of poses synchronized with the frames
        ft_data_left: numpy array of shape (M, 6) containing force/torque measurements for left coinft
        ft_data_right: numpy array of shape (M, 6) containing force/torque measurements for right coinft
        ft_timestamps: numpy array of shape (M,) containing F/T timestamps
        frame_timestamps: numpy array of shape (N,) containing frame timestamps
        video_output_path: output path for the video
        fps: frames per second for output video
        max_frames: maximum number of frames to process (-1 for all)
        skip_start_frames: number of frames to skip at start
        skip_end_frames: number of frames to skip at end
    """
    # Get video dimensions from frames array
    T, H, W, _ = frames.shape
    assert T - skip_start_frames - skip_end_frames == len(poses)
    assert len(frame_timestamps) == T
 
    # TODO: remove this tmp solution: padding
    
    if gripper_widths is not None:
        if len(gripper_widths) < T:
            prepad = np.array([gripper_widths[0]] * (T - len(gripper_widths))) # pad with first value
            gripper_widths = np.hstack((prepad, gripper_widths))

        assert len(gripper_widths) == T 
    
    # Interpolate F/T data to match frame timestamps
    interpolated_ft_left = interpolate_ft_data(ft_data_left, ft_timestamps_left, frame_timestamps)
    interpolated_ft_right = interpolate_ft_data(ft_data_right, ft_timestamps_right, frame_timestamps)
    
    scale = 360 / H
    out_H = int(H * scale)
    out_W = int(W * scale)
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=out_W, height=out_H)
    
    if gripper_widths is not None:
        # Calculate combined width: video frame + trajectory + F/T plot
        ft_plot_height = out_H * 2 * 2 # for left and right coinft
        gripper_plot_height = int(out_H*4/5)  # Smaller plot for gripper width
        combined_width = out_W * 2
        combined_height = ft_plot_height + gripper_plot_height + out_H
    else:
        # Calculate combined width: video frame + trajectory + F/T plot
        ft_plot_height = out_H * 2 * 2 # for left and right coinft
        combined_width = out_W * 2
        combined_height = ft_plot_height + out_H
        
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('a','v','c','1'), 
                                 fps, (combined_width, combined_height))
    
    num_frames = T if max_frames == -1 else min(max_frames + skip_start_frames, T)
    for i in tqdm(range(num_frames)):
        if i < skip_start_frames:
            continue
        if i == num_frames - skip_end_frames:
            break
            
        # Clear previous geometries
        vis.clear_geometries()
        
        # Add the trajectory points up to the current frame
        current_trajectory = poses[:i+1]
        geometries = plot_time_aligned_trajectories([current_trajectory], include_base_frame=False, colors=[[0,0,0]])
        for geometry in geometries:
            vis.add_geometry(geometry)
            
        # Update the visualizer with the current frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture the screen image
        trajectory_image = vis.capture_screen_float_buffer(do_render=False)
        trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
        
        # Get the video frame and resize
        video_frame = frames[i]
        video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)
        trajectory_image_resized = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)
        # Create F/T plot
        ft_plot_left = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_left, frame_timestamps, i, coinft_side='left')
        ft_plot_right = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_right, frame_timestamps, i, coinft_side='right')
        
        # Convert the Plotly figure to an image in PNG format
        ft_plot_image_left = ft_plot_left.to_image(format="png")

        # Decode the PNG image into an OpenCV-compatible format
        ft_plot_left = cv2.imdecode(np.frombuffer(ft_plot_image_left, np.uint8), cv2.IMREAD_COLOR)

        # Resize the plot image to the desired dimensions
        ft_plot_left = cv2.resize(ft_plot_left, (combined_width,  out_H * 2), interpolation=cv2.INTER_AREA)
        
        ft_plot_image_right = ft_plot_right.to_image(format="png")
        ft_plot_right = cv2.imdecode(np.frombuffer(ft_plot_image_right, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_right = cv2.resize(ft_plot_right, (combined_width,  out_H * 2), interpolation=cv2.INTER_AREA)

        if gripper_widths is not None:
            # Create gripper width plot
            gripper_plot = create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
            gripper_plot_image = gripper_plot.to_image(format="png")
            gripper_plot = cv2.imdecode(np.frombuffer(gripper_plot_image, np.uint8), cv2.IMREAD_COLOR)
            gripper_plot = cv2.resize(gripper_plot, (combined_width, gripper_plot_height), interpolation=cv2.INTER_AREA)

        # Combine video and trajectory side by side
        upper_half = np.hstack((video_frame, trajectory_image_resized[:, :, ::-1]))  # Convert RGB to BGR
        
        if gripper_widths is not None:
            # Combine upper half and F/T plot vertically
            combined_image = np.vstack((upper_half, ft_plot_left, ft_plot_right, gripper_plot))
        else:
            # Combine upper half and F/T plot vertically
            combined_image = np.vstack((upper_half, ft_plot_left, ft_plot_right))

        video_writer.write(combined_image)
    
    cv2.destroyAllWindows()
    video_writer.release()
    vis.destroy_window()
    
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_video_aligned_trajectory_horizontal_plot_offscreen(frames, poses, ft_data_left, ft_data_right, ft_timestamps_left, ft_timestamps_right, 
                                                  frame_timestamps, gripper_widths,
                                                  video_output_path, fps=60, max_frames=-1, 
                                                  skip_start_frames=0, skip_end_frames=0):
    """
    Given poses and F/T data synchronized with video frames, plot everything together.
    
    Args:
        frames: numpy array of shape (N, H, W, 3) containing video frames in BGR format
        poses: list of poses synchronized with the frames
        ft_data_left: numpy array of shape (M, 6) containing force/torque measurements for left coinft
        ft_data_right: numpy array of shape (M, 6) containing force/torque measurements for right coinft
        ft_timestamps_left: numpy array of shape (M,) containing F/T timestamps for left
        ft_timestamps_right: numpy array of shape (M,) containing F/T timestamps for right
        frame_timestamps: numpy array of shape (N,) containing frame timestamps
        video_output_path: output path for the video
        fps: frames per second for output video
        max_frames: maximum number of frames to process (-1 for all)
        skip_start_frames: number of frames to skip at start
        skip_end_frames: number of frames to skip at end
    """

    # Get video dimensions from frames array
    T, H, W, _ = frames.shape
    assert T - skip_start_frames - skip_end_frames == len(poses)
    assert len(frame_timestamps) == T

    # Interpolate F/T data to match frame timestamps
    interpolated_ft_left = interpolate_ft_data(ft_data_left, ft_timestamps_left, frame_timestamps)
    interpolated_ft_right = interpolate_ft_data(ft_data_right, ft_timestamps_right, frame_timestamps)

    scale = 360 / H
    out_H = int(H * scale)
    out_W = int(W * scale)

    # Create Offscreen Renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(out_W, out_H)
    scene = renderer.scene

    if gripper_widths is not None:
        gripper_plot_height = out_H  # Smaller plot for gripper width
        combined_width = out_W * 2
    else:
        combined_width = out_W * 2

    # Calculate the widths and heights for the combined image
    combined_width = out_W * 2  # Left column + Right column
    left_height = out_H * 3
    right_height = out_H + (out_H * 2 if gripper_widths is not None else 0)
    total_combined_height = max(left_height, right_height)
    total_combined_width = out_W * 2 * 2  # Expand horizontally for a wider format

    # Initialize video writer
    video_writer = cv2.VideoWriter(
        video_output_path,
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (total_combined_width, total_combined_height)
    )

    num_frames = T if max_frames == -1 else min(max_frames + skip_start_frames, T)

    for i in tqdm(range(num_frames)):
        if i < skip_start_frames:
            continue
        if i == num_frames - skip_end_frames:
            break

        # Clear previous geometries
        scene.clear_geometry()

        # Add the trajectory points up to the current frame
        current_trajectory = poses[:i+1]
        geometries = plot_time_aligned_trajectories([current_trajectory], include_base_frame=False, colors=[[0, 0, 0]])
        
        for j, geometry in enumerate(geometries):
            scene.add_geometry(f"trajectory_{j}", geometry, o3d.visualization.rendering.MaterialRecord())

        # Set up camera
        scene.camera.look_at([1, 1, 1], [0, 0, 0], [0, 1, 0])
        scene.camera.set_projection(60, out_W / out_H, 0.1, 10.0)

        # Render trajectory image
        image = renderer.render_to_image()
        trajectory_image = np.asarray(image).astype(np.uint8)

        # Resize trajectory image
        trajectory_image_resized = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)

        # Get the video frame and resize
        video_frame = frames[i]
        video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)

        # Create F/T plots
        ft_plot_left = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_left, frame_timestamps, i, coinft_side='left')
        ft_plot_right = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_right, frame_timestamps, i, coinft_side='right')

        # Convert the Plotly figure to an image
        ft_plot_image_left = ft_plot_left.to_image(format="png")
        ft_plot_left = cv2.imdecode(np.frombuffer(ft_plot_image_left, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_left = cv2.resize(ft_plot_left, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        ft_plot_image_right = ft_plot_right.to_image(format="png")
        ft_plot_right = cv2.imdecode(np.frombuffer(ft_plot_image_right, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_right = cv2.resize(ft_plot_right, (combined_width, out_H * 2), interpolation=cv2.INTER_AREA)

        if gripper_widths is not None:
            # Create gripper width plot
            gripper_plot = create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
            gripper_plot_image = gripper_plot.to_image(format="png")
            gripper_plot = cv2.imdecode(np.frombuffer(gripper_plot_image, np.uint8), cv2.IMREAD_COLOR)
            gripper_plot = cv2.resize(gripper_plot, (total_combined_width // 2, gripper_plot_height), interpolation=cv2.INTER_AREA)

        # Combine video and trajectory side by side
        upper_half = np.hstack((video_frame, trajectory_image_resized[:, :, ::-1]))  # Convert RGB to BGR

        # Assemble the left and right columns
        left_column = np.vstack((upper_half, ft_plot_left)) if gripper_widths is not None else upper_half
        right_column = np.vstack((gripper_plot, ft_plot_right))

        # Pad the left and right columns to match height
        left_height = left_column.shape[0]
        right_height = right_column.shape[0]
        total_height = max(left_height, right_height)

        if left_height < total_height:
            left_column = np.pad(left_column, ((0, total_height - left_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if right_height < total_height:
            right_column = np.pad(right_column, ((0, total_height - right_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Combine the columns side by side
        combined_image = np.hstack((left_column, right_column))
        video_writer.write(combined_image)

    # Release resources
    cv2.destroyAllWindows()
    video_writer.release()
    renderer = None  # Free memory

    
def plot_video_aligned_trajectory_horizontal_plot(frames, poses, ft_data_left, ft_data_right, ft_timestamps_left, ft_timestamps_right, 
                                  frame_timestamps, gripper_widths,
                                video_output_path, fps=60, max_frames=-1, 
                                skip_start_frames=0, skip_end_frames=0):
    """
    Given poses and F/T data synchronized with video frames, plot everything together.
    
    Args:
        frames: numpy array of shape (N, H, W, 3) containing video frames in BGR format
        poses: list of poses synchronized with the frames
        ft_data_left: numpy array of shape (M, 6) containing force/torque measurements for left coinft
        ft_data_right: numpy array of shape (M, 6) containing force/torque measurements for right coinft
        ft_timestamps: numpy array of shape (M,) containing F/T timestamps
        frame_timestamps: numpy array of shape (N,) containing frame timestamps
        video_output_path: output path for the video
        fps: frames per second for output video
        max_frames: maximum number of frames to process (-1 for all)
        skip_start_frames: number of frames to skip at start
        skip_end_frames: number of frames to skip at end
    """
    
    # Get video dimensions from frames array
    T, H, W, _ = frames.shape
    assert T - skip_start_frames - skip_end_frames == len(poses)
    assert len(frame_timestamps) == T
 
    # TODO: remove this tmp solution: padding
    if gripper_widths is not None:
        if len(gripper_widths) < T:
            prepad = np.array([gripper_widths[0]] * (T - len(gripper_widths))) # pad with first value
            gripper_widths = np.hstack((prepad, gripper_widths))

        assert len(gripper_widths) == T 
    
    # Interpolate F/T data to match frame timestamps
    interpolated_ft_left = interpolate_ft_data(ft_data_left, ft_timestamps_left, frame_timestamps)
    interpolated_ft_right = interpolate_ft_data(ft_data_right, ft_timestamps_right, frame_timestamps)
    
    scale = 360 / H
    out_H = int(H * scale)
    out_W = int(W * scale)
    
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=out_W, height=out_H)
    
    if gripper_widths is not None:
        # Calculate combined width: video frame + trajectory + F/T plot
        gripper_plot_height = out_H  # Smaller plot for gripper width
        combined_width = out_W * 2
    else:
        # Calculate combined width: video frame + trajectory + F/T plot
        combined_width = out_W * 2
 
    # Calculate the widths and heights for the combined image
    combined_width = out_W * 2  # Left column + Right column
    left_height = out_H * 3
    right_height = out_H + (out_H * 2 if gripper_widths is not None else 0)
    
    total_combined_height = max(left_height, right_height)
    total_combined_width = out_W * 2 *2 

    # Initialize the video writer with the correct dimensions
    video_writer = cv2.VideoWriter(
        video_output_path,
        cv2.VideoWriter_fourcc('a', 'v', 'c', '1'),
        fps,
        (total_combined_width, total_combined_height)
    )
    
    num_frames = T if max_frames == -1 else min(max_frames + skip_start_frames, T)
    for i in tqdm(range(num_frames)):
        if i < skip_start_frames:
            continue
        if i == num_frames - skip_end_frames:
            break
            
        # Clear previous geometries
        vis.clear_geometries()
        
        # Add the trajectory points up to the current frame
        current_trajectory = poses[:i+1]
        geometries = plot_time_aligned_trajectories([current_trajectory], include_base_frame=False, colors=[[0,0,0]])
        for geometry in geometries:
            vis.add_geometry(geometry)
            
        # Update the visualizer with the current frame
        vis.poll_events()
        vis.update_renderer()
        
        # Capture the screen image
        trajectory_image = vis.capture_screen_float_buffer(do_render=False)
        trajectory_image = (255 * np.asarray(trajectory_image)).astype(np.uint8)
        trajectory_image = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)
        
        # Get the video frame and resize
        video_frame = frames[i]
        video_frame = cv2.resize(video_frame, dsize=(out_W, out_H), interpolation=cv2.INTER_CUBIC)
        trajectory_image_resized = cv2.resize(trajectory_image, (out_W, out_H), interpolation=cv2.INTER_AREA)
        # Create F/T plot
        ft_plot_left = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_left, frame_timestamps, i, coinft_side='left')
        ft_plot_right = create_ft_plot_up_to_current_with_fixed_scale_plotly_timestamp_x(interpolated_ft_right, frame_timestamps, i, coinft_side='right')
        
        # Convert the Plotly figure to an image in PNG format
        ft_plot_image_left = ft_plot_left.to_image(format="png")

        # Decode the PNG image into an OpenCV-compatible format
        ft_plot_left = cv2.imdecode(np.frombuffer(ft_plot_image_left, np.uint8), cv2.IMREAD_COLOR)

        # Resize the plot image to the desired dimensions
        ft_plot_left = cv2.resize(ft_plot_left, (combined_width,  out_H * 2), interpolation=cv2.INTER_AREA)
        
        ft_plot_image_right = ft_plot_right.to_image(format="png")
        ft_plot_right = cv2.imdecode(np.frombuffer(ft_plot_image_right, np.uint8), cv2.IMREAD_COLOR)
        ft_plot_right = cv2.resize(ft_plot_right, (combined_width,  out_H * 2), interpolation=cv2.INTER_AREA)

        if gripper_widths is not None:
            # Create gripper width plot
            gripper_plot = create_gripper_width_plot_up_to_current_timestamp_x(gripper_widths, frame_timestamps, i)
            gripper_plot_image = gripper_plot.to_image(format="png")
            gripper_plot = cv2.imdecode(np.frombuffer(gripper_plot_image, np.uint8), cv2.IMREAD_COLOR)
            gripper_plot = cv2.resize(gripper_plot, (total_combined_width//2, gripper_plot_height), interpolation=cv2.INTER_AREA)
        upper_half = np.hstack((video_frame, trajectory_image_resized[:, :, ::-1]))  # Convert RGB to BGR
        
        info_print(f'upper_half shape: {upper_half.shape}')
        info_print(f'gripper_plot shape: {gripper_plot.shape}')
        info_print(f'ft_plot_left shape: {ft_plot_left.shape}')
        info_print(f'ft_plot_right shape: {ft_plot_right.shape}')
        
 
        # Assemble the left and right columns
        left_column = np.vstack((upper_half, ft_plot_left)) if gripper_widths is not None else upper_half
        right_column = np.vstack((gripper_plot, ft_plot_right))
        
        # Pad the left and right columns to match height
        left_height = left_column.shape[0]
        right_height = right_column.shape[0]
        total_height = max(left_height, right_height)

        if left_height < total_height:
            left_column = np.pad(left_column, ((0, total_height - left_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        if right_height < total_height:
            right_column = np.pad(right_column, ((0, total_height - right_height), (0, 0), (0, 0)), mode='constant', constant_values=0)

        # Combine the columns side by side
        combined_image = np.hstack((left_column, right_column))
        video_writer.write(combined_image)

    
    cv2.destroyAllWindows()
    video_writer.release()
    vis.destroy_window()

def plot_video_aligned_trajectory_from_multimodal(campose_dic, image_dict, ft_dic, video_output_path, gripper_dic=None,
                                                demo_idx=0, fps=60, max_frames=-1, 
                                                skip_start_frames=0, skip_end_frames=0,
                                                plot_horizontal = False):
    
    frames = image_dict['data'][demo_idx]['imgData']
    poses = campose_dic['data'][demo_idx]['camPoseData']
    ft_data_left = ft_dic['data']['left'][demo_idx]['ftData']
    ft_data_right = ft_dic['data']['right'][demo_idx]['ftData']
    
    if gripper_dic is not None:
        gripper_data = gripper_dic['data'][demo_idx]['gripperData']
    else:
        gripper_data = None
        
    ft_timestamps_left = ft_dic['data']['left'][demo_idx]['ftTimeStamp'] # iso format object
    ft_timestamps_right = ft_dic['data']['right'][demo_idx]['ftTimeStamp'] # iso format object
    frame_timestamps = image_dict['data'][demo_idx]['imgTimeStamp'] # iso format object
    ft_timestamps_left = array_isostringformat_to_timestamp(ft_timestamps_left) # convert to timestamp
    ft_timestamps_right = array_isostringformat_to_timestamp(ft_timestamps_right) # convert to timestamp
    frame_timestamps = array_isostringformat_to_timestamp(frame_timestamps) # convert to timestamp

    if plot_horizontal:
        plot_video_aligned_trajectory_horizontal_plot(frames = frames, 
            poses = poses, 
            ft_data_left=ft_data_left, 
            ft_data_right=ft_data_right,
            ft_timestamps_left = ft_timestamps_left, 
            ft_timestamps_right = ft_timestamps_right,
            frame_timestamps = frame_timestamps, 
            gripper_widths=gripper_data,
            video_output_path = video_output_path, 
            fps=fps, 
            max_frames=max_frames,
            skip_start_frames=skip_start_frames, 
            skip_end_frames=skip_end_frames)
    else: 
        plot_video_aligned_trajectory(frames = frames, 
            poses = poses, 
            ft_data_left=ft_data_left, 
            ft_data_right=ft_data_right,
            ft_timestamps_left = ft_timestamps_left, 
            ft_timestamps_right = ft_timestamps_right,
            frame_timestamps = frame_timestamps, 
            gripper_widths=gripper_data,
            video_output_path = video_output_path, 
            fps=fps, 
            max_frames=max_frames,
            skip_start_frames=skip_start_frames, 
            skip_end_frames=skip_end_frames)