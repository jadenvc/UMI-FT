import os
import shutil

# Set PYTHONPATH for subprocesses to find PyriteEnvSuites package
os.environ["PYTHONPATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)

import env_runners.flip_up_env_runner_heuristic as env_runner

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_memory_usage_threshold"] = "0.9"

import ray

if __name__ == "__main__":
    output_folder = "/home/yifanhou/data/easy_flip_up_500"
    # step 0: clean up data output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # step 1: initialize ray with the number of threads you want to use
    ray.init()
    # step 2: create a ray.remote variant of your function
    data_generation_job = ray.remote(env_runner.simulate_one_episode)
    # step 3: launch tasks with .remote(), passing in its arguments
    rand_para = {
        "bookend_yaw_deg": 20,
        "bookend_distance_m": 0.2,
        "bookend_side_distance_m": 0.2,
        "bookend_z_m": 0.2,
        "book_yaw_deg": 0,
        "book_distance_to_bookend_m": 0.00,
        "book_side_m": 0.02,
    }
    plan_para = {
        "pre_x_gap_m": 0.05,
        "book_nominal_length_m": 0.159,  # 0.016
        "rotation_points_num": 20,
        "pause_time_engage_s": 0.5,
        "pause_time_finish_s": 1.5,
    }
    traj_gen_para = {
        "joint_vel_limit_rad_s": 0.2,
        "joint_acc_limit_rad_s2": 3,
        "trans_vel_limit_m_s": 0.1,
        "trans_acc_limit_m_s2": 5,
        "rot_vel_limit_rad_s": 0.1,
        "rot_acc_limit_rad_s2": 5,
        "approach_limit_multiplier": 3,
        "engage_limit_multiplier": 0.5,
    }
    pipeline_para = {
        "show_mujoco_viewer": False,
        "show_cv_window": False,
        "plot_eposide": False,
        "auto_save": True,
        "fc_axis_update_down_sample": 250,
        "fc_axis_update_start_timestep": 1990,
        "termination_timestep": 20000,
        "refresh_cv_window_every_N_frame": 100,
        "save_visual_every_N_frame": 5,
        "save_low_dim_every_N_frame": 5,
        "cam_ids_render": [0, 1, 2],  # render all; shape_meta will select from them
        "cam_id_to_show": 2,
        "saving_img_hw": [224, 224],  # needs to match ground.xml
        "output_folder": output_folder,
    }
    force_filtering_para = {
        "sampling_freq": 100,
        "cutoff_freq": 5,
        "order": 5,
    }
    verbose = 0

    tasks = [
        data_generation_job.remote(
            rand_seed=episode_id,
            episode_id=episode_id,
            rand_para=rand_para,
            plan_para=plan_para,
            traj_gen_para=traj_gen_para,
            pipeline_para=pipeline_para,
            force_filtering_para=force_filtering_para,
            verbose=verbose,
        )
        for episode_id in range(600)
    ]

    # step 4: synchronize with tasks
    ray.get(tasks)

    # step 5: generate meta data
    print("Generating meta data")
    import zarr
    from PyriteUtility.computer_vision.imagecodecs_numcodecs import register_codecs

    register_codecs()
    buffer = zarr.open(output_folder)
    meta = buffer.create_group("meta", overwrite=True)
    episode_low_dim_len = []
    episode_rgb_len = []

    count = 0
    for key in buffer["data"].keys():
        episode = key
        ep_data = buffer["data"][episode]
        episode_low_dim_len.append(ep_data["ts_pose_command"].shape[0])
        episode_rgb_len.append(ep_data["camera0_rgb"].shape[0])
        print(
            f"Number {count}: {episode}: low dim len: {episode_low_dim_len[-1]}, rgb len: {episode_rgb_len[-1]}"
        )
        count += 1

    meta["episode_low_dim_len"] = zarr.array(episode_low_dim_len)
    meta["episode_rgb_len"] = zarr.array(episode_rgb_len)

    print(f"All done! Generated {count} episodes in {output_folder}")
