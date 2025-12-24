import os
import shutil
import sys

import env_runners.flip_up_env_runner_hybrid_closed_loop as env_runner

SCRIPT_PATH = os.path.abspath(os.path.dirname(__file__))
print("script path: ", SCRIPT_PATH)
print("PATH: ", sys.path)

# Set PYTHONPATH for subprocesses
os.environ["PYTHONPATH"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../")
)

if __name__ == "__main__":
    output_folder = "/home/yifanhou/data/flip_up"
    # step 0: clean up data output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    # step 3: launch tasks with .remote(), passing in its arguments
    rand_para = {
        "bookend_yaw_deg": 20,
        "bookend_distance_m": 0.2,
        "bookend_side_distance_m": 0.2,
        "bookend_z_m": 0.2,
        "book_yaw_deg": 0,
        "book_distance_to_bookend_m": 0.03,
        "book_side_m": 0.02,
    }
    control_para = {
        "policy_time_step_s": 0.1,
        "execution_horizon": 6,
        "dense_execution_horizon": 2,
        "max_duration_s": 10,
        "test_sparse_action": False,
        "device": "cuda",
    }
    pipeline_para = {
        "show_mujoco_viewer": False,
        "cam_ids_render": [0, 1, 2],  # render all; shape_meta will select later
        "saving_img_hw": [224, 224],  # needs to match ground.xml
        "save_low_dim_every_N_frame": 5,
        "save_visual_every_N_frame": 5,
        "update_control_every_N_frame": 5,
        "output_folder": output_folder,
        "ckpt_path": "2024.05.22_19.56.54_flip_up_linear_dense_action_no_obs_encoder_for_dense",
    }
    verbose = 0

    for episode_id in range(501, 550):
        print("[Evaluation] Running episode_id: ", episode_id)
        env_runner.simulate_one_episode(
            episode_id, episode_id, rand_para, pipeline_para, control_para, verbose
        )

    # check success of results
    import zarr

    from utils.flip_up_env_utils import is_flip_up_success

    dataset_path = output_folder
    buffer = zarr.open(dataset_path)

    success_id = []
    failure_id = []

    angle_thresh = 15
    for episode, ep_data in buffer["data"].items():
        obj_poses = ep_data["low_dim_state"]

        N, D = obj_poses.shape
        assert D == 7
        success, angle_deg = is_flip_up_success(obj_poses[-1], angle_thresh)

        if success:
            success_id.append(episode)
        else:
            failure_id.append(episode)

        print(f"{episode}, \tangle: {angle_deg:.2f}, \tSuccess: {success}")

    print(f"Success: {len(success_id)}, Failure: {len(failure_id)}")
    print(f"Success rate: {len(success_id) / (len(success_id) + len(failure_id)):.2f}")
