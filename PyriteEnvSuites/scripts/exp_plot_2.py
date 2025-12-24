import matplotlib.pyplot as plt
import numpy as np
import zarr
from PyriteUtility.spatial_math import spatial_utilities as su


def main():
    control_log_path = (
        # "/home/yifanhou/data/control_log/wiping_recorded_12_1"
        "/home/yifanhou/data/control_log/wiping_recorded_24_1"
    )
    # horizon = 16
    # exe_horizon = 12

    horizon = 32
    exe_horizon = 24

    plot_horizon = 5

    # log
    log_store = zarr.DirectoryStore(path=control_log_path)
    log = zarr.open(store=log_store, mode="r")  # r: read only, must exist

    num_horizons = len(log.keys())
    num_horizons = min(num_horizons, plot_horizon)

    episode_ts_virtual_targets = []
    episode_ts_nominal_targets = []
    episode_ts_stiffnesses = []
    episode_ws_stiffnesses = []

    episode_plot_data_x = []
    episode_plot_data_y = []
    episode_plot_data_z = []
    episode_labels = []

    for horizon_count in range(num_horizons - 1):
        # read the log
        print(f"Horizon {horizon_count}")
        horizon_log = log[f"horizon_{horizon_count}"]
        ts_virtual_targets = horizon_log["ts_virtual_targets_1"][:]  # 16x7
        ts_nominal_targets = horizon_log["ts_nominal_targets_1"][:]  # 16x7
        ts_stiffnesses = horizon_log["ts_stiffnesses_1"][:]  # 6x96
        timestamps = horizon_log["timestamps_s"][:]
        timestamps = timestamps - timestamps[0]

        # convert stiffness matrix to world frame
        ws_stiffnesses = np.zeros((3, 3, horizon))
        for i in range(horizon):
            ts_stiffness_3x3 = ts_stiffnesses[:3, i * 6 : i * 6 + 3]
            R_WT = su.pose7_to_SE3(ts_nominal_targets[i])[:3, :3]
            ws_stiffnesses[:, :, i] = R_WT @ ts_stiffness_3x3 @ R_WT.T

        episode_ts_virtual_targets.append(ts_virtual_targets)
        episode_ts_nominal_targets.append(ts_nominal_targets)
        episode_ts_stiffnesses.append(ts_stiffnesses)
        episode_ws_stiffnesses.append(ws_stiffnesses)

        episode_labels.append(f"Horizon {horizon_count}")

        episode_plot_data_x.append(
            (
                np.linspace(
                    horizon_count * exe_horizon,
                    horizon_count * exe_horizon + horizon,
                    horizon,
                ),
                ws_stiffnesses[0, 0, :],
                # ts_stiffnesses[0, 0 : ts_stiffnesses.shape[1] : 6],
            )
        )
        episode_plot_data_y.append(
            (
                np.linspace(
                    horizon_count * exe_horizon,
                    horizon_count * exe_horizon + horizon,
                    horizon,
                ),
                ws_stiffnesses[1, 1, :],
                # ts_stiffnesses[1, 1 : ts_stiffnesses.shape[1] : 6],
            )
        )
        episode_plot_data_z.append(
            (
                np.linspace(
                    horizon_count * exe_horizon,
                    horizon_count * exe_horizon + horizon,
                    horizon,
                ),
                ws_stiffnesses[2, 2, :],
                # ts_stiffnesses[2, 2 : ts_stiffnesses.shape[1] : 6],
            )
        )

    plt.figure(figsize=(10, 6))

    for idx, (x, y) in enumerate(episode_plot_data_x):
        plt.plot(
            x[:exe_horizon],
            y[:exe_horizon],
            color="red",
            linestyle="solid",
            linewidth=2,
        )
        plt.plot(
            x[exe_horizon - 1 :],
            y[exe_horizon - 1 :],
            color="red",
            linestyle="dotted",
            linewidth=2,
        )
    for idx, (x, y) in enumerate(episode_plot_data_y):
        plt.plot(
            x[:exe_horizon],
            y[:exe_horizon],
            color="green",
            linestyle="solid",
            linewidth=2,
        )
        plt.plot(
            x[exe_horizon - 1 :],
            y[exe_horizon - 1 :],
            color="green",
            linestyle="dotted",
            linewidth=2,
        )
    for idx, (x, y) in enumerate(episode_plot_data_z):
        plt.plot(
            x[:exe_horizon],
            y[:exe_horizon],
            color="blue",
            linestyle="solid",
            linewidth=2,
        )
        plt.plot(
            x[exe_horizon - 1 :],
            y[exe_horizon - 1 :],
            color="blue",
            linestyle="dotted",
            linewidth=2,
        )

        # Mark the start of the curve on the X-axis
        plt.scatter(
            x[0], 0, color="black"
        )  # Mark the start on the X-axis with a red dot
        plt.text(
            x[0],
            200,
            f"Horizon {idx + 1}",
            ha="center",
            fontsize=15,
            color="black",
        )

    h_starts = [t[0] for (t, v) in episode_plot_data_z]
    ax = plt.gca()
    ax.set_xticks(h_starts, minor=False)
    # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
    ax.xaxis.grid(True, which="major")
    # ax.xaxis.grid(True, which="minor")

    ax.yaxis.grid(True, which="major")

    plt.axhline(0, color="black", linewidth=3)  # X-axis line
    # plt.xlabel("X-axis")
    # plt.ylabel("Stiffness Value")
    plt.ylim([0, 5100])
    # plt.title("Stiffness Value in the World Frame")
    # plt.legend()
    # plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
