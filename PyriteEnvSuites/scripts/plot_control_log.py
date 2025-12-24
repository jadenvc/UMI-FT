import os

import matplotlib.pyplot as plt
import numpy as np

# from mpl_toolkits import mplot3d
import zarr


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
        raise ValueError(
            "Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS"
        )

    control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")
    control_log_path = control_log_folder_path + "/feb24_nogripper_nowrench_rollout/"

    # robot_id = 0  # robot front is +Y
    robot_id = 1  # robot front is -Y

    # view_angle = [0, 90, 0] # viewing from +Y to -Y
    view_angle = [0, -90, 0]  # viewing from -Y to +Y

    # umift
    horizon = 16
    exe_horizon = 12
    downsample = 2
    scale = 0.6

    # # wiping
    # horizon = 32
    # exe_horizon = 24
    # downsample = 4
    # scale = 1.2

    save_figure_no_show = False

    # log
    print(f"Loading control log from {control_log_path}")
    log_store = zarr.DirectoryStore(path=control_log_path)
    log = zarr.open(store=log_store, mode="r")  # r: read only, must exist

    num_horizons = len(log.keys())

    episode_ts_virtual_targets = []
    episode_ts_nominal_targets = []
    episode_ts_stiffnesses = []

    plt.ion()  # to run GUI event loop
    _fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(elev=view_angle[0], azim=view_angle[1], roll=view_angle[2])
    x = np.linspace(-0.02, 0.2, 20)
    y = np.linspace(-0.1, 0.8, 20)
    z = np.linspace(-0.1, 0.1, 20)
    ax.plot3D(x, y, z, color="blue", marker="o", markersize=3)
    ax.plot3D(x, y, z, color="red", marker="o", markersize=3)
    ax.set_title("Target and virtual target")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
    ax.cla()

    for horizon_count in range(num_horizons):
        # read the log
        print(f"Horizon {horizon_count}")
        horizon_log = log[f"horizon_{horizon_count}"]
        action_log = horizon_log["action"]
        obs_log = horizon_log["obs"]

        # plot obs
        # obs_log_rgb = obs_log["rgb_0"]
        # obs_log_eef_pos = obs_log["robot0_eef_pos"]
        # obs_log_eef_rot_axis_angle = obs_log["robot0_eef_rot_axis_angle"]
        # obs_log_gripper = obs_log["robot0_gripper"]
        # obs_log_eef_wrench_left = obs_log["robot0_eef_wrench_left"]
        for key, value in obs_log.items():
            print(f"key: {key}, value: {value.shape}")
            if len(value.shape) == 2:
                print(value[:])
        exit()

        # # plot rgb
        # for i in range(len(obs_log_rgb)):
        #     rgb = obs_log_rgb[i]
        #     rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        #     plt.imshow(rgb)
        #     plt.show()
        #     input("Press Enter to continue...")

        # plot action
        for key, value in action_log.items():
            print(f"key: {key}, value: {value.shape}")
            # if len(value.shape) == 2:
            #     print(value[:])
        exit()

        ts_virtual_targets = action_log[f"ts_virtual_targets_{robot_id}"][:]  # 16x7
        ts_nominal_targets = action_log[f"ts_nominal_targets_{robot_id}"][:]  # 16x7
        ts_stiffnesses = action_log[f"ts_stiffnesses_{robot_id}"][:]  # 6x96
        timestamps = action_log["timestamps_s"][:]
        timestamps = timestamps - timestamps[0]

        # SE3_virtual_targets = su.pose7_to_SE3(ts_virtual_targets)  # 16x4x4
        # SE3_nominal_targets = su.pose7_to_SE3(ts_nominal_targets)  # 16x4x4

        episode_ts_virtual_targets.append(ts_virtual_targets)
        episode_ts_nominal_targets.append(ts_nominal_targets)
        episode_ts_stiffnesses.append(ts_stiffnesses)

        # plot a horizon

        # plot the whole horizon
        ax.plot3D(
            ts_nominal_targets[::downsample, 0],
            ts_nominal_targets[::downsample, 1],
            ts_nominal_targets[::downsample, 2],
            color="gold",
            marker="o",
            markersize=8 * scale,
            linewidth=3 * scale,
        )

        # # plot the executed part
        # ax.plot3D(
        #     ts_virtual_targets[:exe_horizon, 0],
        #     ts_virtual_targets[:exe_horizon, 1],
        #     ts_virtual_targets[:exe_horizon, 2],
        #     color="royalblue",
        #     marker="o",
        #     markersize=8,
        #     linewidth=5,
        # )
        # ax.plot3D(
        #     ts_nominal_targets[:exe_horizon, 0],
        #     ts_nominal_targets[:exe_horizon, 1],
        #     ts_nominal_targets[:exe_horizon, 2],
        #     color="dimgrey",
        #     marker="o",
        #     markersize=8,
        #     linewidth=5,
        # )
        # # plot the unexecuted part
        # ax.plot3D(
        #     ts_virtual_targets[exe_horizon - 1 :, 0],
        #     ts_virtual_targets[exe_horizon - 1 :, 1],
        #     ts_virtual_targets[exe_horizon - 1 :, 2],
        #     color="teal",
        #     marker="o",
        #     markersize=8,
        #     linewidth=5,
        # )
        # ax.plot3D(
        #     ts_nominal_targets[exe_horizon - 1 :, 0],
        #     ts_nominal_targets[exe_horizon - 1 :, 1],
        #     ts_nominal_targets[exe_horizon - 1 :, 2],
        #     color="sienna",
        #     marker="o",
        #     markersize=8,
        #     linewidth=5,
        # )

        # fin
        for i in np.arange(0, exe_horizon, downsample):
            ax.plot3D(
                [ts_nominal_targets[i][0], ts_virtual_targets[i][0]],
                [ts_nominal_targets[i][1], ts_virtual_targets[i][1]],
                [ts_nominal_targets[i][2], ts_virtual_targets[i][2]],
                color="gold",
                marker="o",
                markersize=1 * scale,
                linewidth=2 * scale,
            )
        for i in np.arange(exe_horizon, horizon, downsample):
            ax.plot3D(
                [ts_nominal_targets[i][0], ts_virtual_targets[i][0]],
                [ts_nominal_targets[i][1], ts_virtual_targets[i][1]],
                [ts_nominal_targets[i][2], ts_virtual_targets[i][2]],
                color="gold",
                # color="sienna",
                marker="o",
                markersize=1 * scale,
                linewidth=2 * scale,
            )

        # virtual target
        ax.plot3D(
            ts_virtual_targets[::downsample, 0],
            ts_virtual_targets[::downsample, 1],
            ts_virtual_targets[::downsample, 2],
            color="orangered",
            marker="o",
            markersize=8 * scale,
            linewidth=3 * scale,
        )

        # initial point
        ax.plot3D(
            ts_nominal_targets[0, 0],
            ts_nominal_targets[0, 1],
            ts_nominal_targets[0, 2],
            color="gold",
            marker="o",
            markersize=8 * scale,
        )

        ## Plot for a horizon
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        if robot_id == 1:
            ax.axes.set_xlim3d(left=0.15, right=0.4)
            ax.axes.set_ylim3d(bottom=-0.55, top=-0.3)
            ax.axes.set_zlim3d(bottom=0.4, top=0.65)
        else:
            ax.axes.set_xlim3d(left=0, right=0.3)
            ax.axes.set_ylim3d(bottom=0.3, top=0.6)
            ax.axes.set_zlim3d(bottom=0.2, top=0.5)

        set_axes_equal(ax)
        ax.grid(False)
        if not save_figure_no_show:
            plt.draw()
            input("Done. Press Enter to quit...")
        ax.set_axis_off()
        plt.savefig(
            f"/home/yifanhou/git/PyriteEnvSuites/scripts/outputs/demo_r{robot_id}_{horizon_count}.png",
            transparent=True,
        )
        ax.cla()


if __name__ == "__main__":
    main()
