import os

import genesis as gs
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.common import GracefulKiller
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerHybrid
from PyriteUtility.pytorch_utils.model_io import load_policy

from PyriteEnvSuites.utils.env_utils import js_to_traj

if "PYRITE_CHECKPOINT_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CHECKPOINT_FOLDERS")
if "PYRITE_HARDWARE_CONFIG_FOLDERS" not in os.environ:
    raise ValueError(
        "Please set the environment variable PYRITE_HARDWARE_CONFIG_FOLDERS"
    )
if "PYRITE_CONTROL_LOG_FOLDERS" not in os.environ:
    raise ValueError("Please set the environment variable PYRITE_CONTROL_LOG_FOLDERS")

checkpoint_folder_path = os.environ.get("PYRITE_CHECKPOINT_FOLDERS")
hardware_config_folder_path = os.environ.get("PYRITE_HARDWARE_CONFIG_FOLDERS")
control_log_folder_path = os.environ.get("PYRITE_CONTROL_LOG_FOLDERS")


def main():
    control_para = {
        "raw_time_step_s": 0.1,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        "slow_down_factor": 1,
        "sparse_execution_horizon": 24,
        "max_duration_s": 3500,
        "robot_initial_dofs": np.array(
            [0.01, -0.2, -0.35, 0, 0, 0, 0.055, 0.055]
        ),  # initial joint positions
        "pausing_mode": False,
        "device": "cuda",
    }
    pipeline_para = {
        # "ckpt_path": "/2025.10.09_01.09.47_state_based_bin_extraction",
        "ckpt_path": "/2025.11.17_14.13.09_state_based_bin1__bin_extraction",
        # "ckpt_path": "/2025.11.17_17.32.35_vt_bin1__bin_extraction",
        # "ckpt_path": "/2025.11.17_21.04.57_vt_bin_new_3__bin_extraction",
        "env_config_path": "../../PyriteGenesis/PyriteGenesis/configs",
        "env_config_name": "bin1",
        # "env_config_name": "bin_new_3",
    }
    episode_id = 0

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    js_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_js"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_js"]["down_sample_steps"] + 1
    item_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["item_poses0_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["item_poses0_pos"][
        "down_sample_steps"
    ] + 1
    query_sizes = {
        "js_fb": js_pose_query_size,
        "item": item_query_size,
    }

    # initialize the simulation environment
    with hydra.initialize(
        version_base=None, config_path=pipeline_para["env_config_path"]
    ):
        cfg = hydra.compose(config_name=pipeline_para["env_config_name"])
    OmegaConf.resolve(cfg)

    np.random.seed(cfg.rand_seed)

    if cfg.backend == "gpu":
        gs.init(backend=gs.gpu)
    else:
        gs.init(backend=gs.cpu)

    # # clean and create output folders (TODO: proper logging for env runner)
    # log_folder_path = cfg.log_folder
    # if os.path.exists(log_folder_path):
    #     shutil.rmtree(log_folder_path)
    # os.makedirs(log_folder_path, exist_ok=True)

    # Instantiate the environment
    cfg.env.num_envs = 1
    cfg.env.show_viewer = True
    env = hydra.utils.instantiate(cfg.env)
    env.build_env()

    # # plot goal item poses
    # goal_items_pose7 = np.array(cfg.rrt["goal_items_pose7"])
    # # draw the goal items
    # for item_id in range(len(goal_items_pose7)):
    #     print("goal item", item_id, "pose7:", goal_items_pose7[item_id])
    #     geom = env.items[item_id].geoms[0] # assuming there is only one geom for the item
    #     mesh = geom.get_trimesh()
    #     SE3_item = su.pose7_to_SE3(goal_items_pose7[item_id])
    #     env.scene.draw_debug_mesh(mesh, None, SE3_item)
    # env.scene.visualizer.update()

    p_timestep_s = control_para["raw_time_step_s"]

    action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    execution_horizon_s = (
        control_para["sparse_execution_horizon"]
        * action_down_sample_steps
        * p_timestep_s
    )
    action_timesteps_s = (
        np.arange(0, action_horizon)
        * action_down_sample_steps
        * p_timestep_s
        * control_para["slow_down_factor"]
    )

    action_type = "pose7"
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 7:
        action_type = "pose7"
    elif shape_meta["action"]["shape"][0] == 14:
        action_type = "pose7"
        id_list = [0, 1]
    else:
        raise RuntimeError("unsupported")

    if action_type == "pose7":
        action_to_trajectory = js_to_traj
    else:
        raise RuntimeError("unsupported")

    print("Creating MPC.")
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
    )
    controller.set_time_offset(env.current_time_s)

    episode_initial_time_s = env.current_time_s
    print("Starting main loop.")

    # # log
    # log_store = zarr.DirectoryStore(path=pipeline_para["control_log_path"])
    # log = zarr.open(store=log_store, mode="w")  # w: overrite if exists

    horizon_count = 0

    #########################################
    # main loop starts
    #########################################
    while True:
        print("New episode. Episode ID: ", episode_id)
        # reset and simulate for 1s to populate the observation buffer
        env.reset(duration_s=1.0, robot_initial_dofs=control_para["robot_initial_dofs"])
        input("Press Enter to start the episode.")
        killer = GracefulKiller()
        while not killer.kill_now:
            horizon_start_time_s = env.current_time_s
            print("Starting new horizon at ", horizon_start_time_s)

            obs_task = dict()
            obs_raw = env.get_sparse_observation_from_buffer(query_sizes)
            raw_to_obs(obs_raw, obs_task, shape_meta)

            # Run inference
            controller.set_observation(obs_task["obs"])
            actions = controller.compute_sparse_control(device)

            action_traj = js_to_traj(actions, action_timesteps_s + horizon_start_time_s)

            print(
                "Executing action from time ",
                horizon_start_time_s,
                " to ",
                horizon_start_time_s + execution_horizon_s,
            )
            env.control_and_simulate(
                action_traj=action_traj,
                execution_end_time_s=execution_horizon_s + horizon_start_time_s,
            )

            horizon_count += 1

            if control_para["pausing_mode"]:
                c = input("Press Enter to start the next horizon. q to quit.")
                if c == "q":
                    break

            time_s = env.current_time_s

            if not control_para["pausing_mode"]:
                # only check duration when not in pausing mode
                if time_s - episode_initial_time_s > control_para["max_duration_s"]:
                    break

        print("End of episode.")
        episode_id += 1

        print("Options:")
        print("     c: continue to next episode.")
        print("     others: quit the program.")
        c = input("Please select an option: ")
        if c == "c":
            pass
        else:
            print(f"Getting {c}. Quitting the program.")
            break

        print("Continuing to execution.")


if __name__ == "__main__":
    main()
