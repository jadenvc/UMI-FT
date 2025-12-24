import copy
import os
import time
from typing import Tuple

import torch
import torch.multiprocessing
from einops import rearrange
from PyriteConfig.tasks.common import common_type_conversions as task
from PyriteConfig.tasks.common.common_type_conversions import raw_to_obs
from PyriteUtility.common import dict_apply
from PyriteUtility.planning_control.mpc import ModelPredictiveControllerHybrid
from PyriteUtility.pytorch_utils.model_io import load_policy

from PyriteEnvSuites.envs.task.manip_server_env import ManipServerEnv

# from PyriteEnvSuites.envs.wrapper.record import RecordWrapper
from PyriteEnvSuites.utils.env_utils import pose9pose9s1_to_traj

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


# Run inference
def run_inference(obs_sample_np, policy, id, device):
    time0 = time.time()
    count = 0
    with torch.no_grad():
        while True:
            # convert to torch tensor
            obs_sample = dict_apply(
                obs_sample_np, lambda x: torch.from_numpy(x).to(device)
            )

            result = policy.predict_action(obs_sample)
            _raw_action = result["sparse"][0].detach().to("cpu").numpy()
            time1 = time.time()
            average_time = (time1 - time0) / (count + 1)
            print("Average time for id ", id, ": ", average_time * 1000, "ms")
            count += 1


def main():
    control_para = {
        "raw_time_step_s": 0.002,  # dt of raw data collection. Used to compute time step from time_s such that the downsampling according to shape_meta works.
        "slow_down_factor": 1.5,  # 3 for flipup, 1.5 for wiping
        "sparse_execution_horizon": 12,  # 12 for flipup, 8/24 for wiping
        "dense_execution_horizon": 2,
        "dense_execution_offset": 0.000,  # hack < 0.02
        "max_duration_s": 3500,
        "test_sparse_action": True,
        "test_nominal_target": False,
        "test_nominal_target_stiffness": -1,  # -1,
        "fix_orientation": False,
        "pausing_mode": True,
        "device": "cuda",
    }
    pipeline_para = {
        "plot_eposide": True,
        "save_low_dim_every_N_frame": 1,
        "save_visual_every_N_frame": 1,
        "update_control_every_N_frame": 1,
        "ckpt_path": "/public/flip_up_conv",  # wiping checkpoint
        "hardware_config_path": hardware_config_folder_path
        + "/single_arm_evaluation.yaml",
        "control_log_path": control_log_folder_path + "/temp/",
    }
    force_filtering_para = {
        "sampling_freq": 100,
        "cutoff_freq": 5,
        "order": 5,
    }
    verbose = 1

    episode_id = 0

    def get_real_obs_resolution(shape_meta: dict) -> Tuple[int, int]:
        out_res = None
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            shape = attr.get("shape")
            if type == "rgb":
                co, ho, wo = shape
                if out_res is None:
                    out_res = (wo, ho)
                assert out_res == (wo, ho)
        return out_res

    def printOrNot(verbose, *args):
        if verbose >= 0:
            print(f"[Episode {episode_id}] ", *args)

    # vbs_h1 = verbose + 1  # verbosity for header
    vbs_h2 = verbose  # verbosity for sub-header
    # vbs_p = verbose - 1  # verbosity for paragraph

    # load policy
    print("Loading policy: ", checkpoint_folder_path + pipeline_para["ckpt_path"])
    device = torch.device(control_para["device"])
    policy, shape_meta = load_policy(
        checkpoint_folder_path + pipeline_para["ckpt_path"], device
    )

    # image size
    (image_width, image_height) = get_real_obs_resolution(shape_meta)

    # zarr img buffer size estimation: 20s, 1000hz step rate
    # img_buffer_size_estimated = int(
    #     20 * 1000 / pipeline_para["save_visual_every_N_frame"]
    # )
    # rgb_buffer_shape_nhwc = (
    #     img_buffer_size_estimated,
    #     image_height,
    #     image_width,
    #     3,
    # )

    rgb_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["rgb_0"]["down_sample_steps"] + 1
    ts_pose_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_pos"]["down_sample_steps"] + 1
    wrench_query_size = (
        shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["horizon"] - 1
    ) * shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"][
        "down_sample_steps"
    ] + 1
    query_sizes = {
        "rgb": rgb_query_size,
        "ts_pose_fb": ts_pose_query_size,
        "wrench": wrench_query_size,
    }

    if (
        control_para["test_nominal_target"]
        and control_para["test_nominal_target_stiffness"] < 0
    ):
        n_af = 0
    else:
        n_af = 3

    env = ManipServerEnv(
        camera_res_hw=(image_height, image_width),
        hardware_config_path=pipeline_para["hardware_config_path"],
        filter_params=force_filtering_para,
        query_sizes=query_sizes,
        compliant_dimensionality=n_af,
    )
    env.reset()

    # p_timestep_s = control_para["raw_time_step_s"]

    sparse_action_down_sample_steps = shape_meta["sample"]["action"]["sparse"][
        "down_sample_steps"
    ]
    # sparse_action_horizon = shape_meta["sample"]["action"]["sparse"]["horizon"]
    # sparse_action_horizon_s = (
    #     sparse_action_horizon * sparse_action_down_sample_steps * p_timestep_s
    # )
    sparse_execution_horizon = (
        sparse_action_down_sample_steps * control_para["sparse_execution_horizon"]
    )
    # sparse_action_timesteps_s = (
    #     np.arange(0, sparse_action_horizon)
    #     * sparse_action_down_sample_steps
    #     * p_timestep_s
    #     * control_para["slow_down_factor"]
    # )

    flag_has_dense = "dense" in shape_meta["sample"]["action"]
    dense_execution_horizon = None
    if flag_has_dense:
        dense_action_down_sample_steps = shape_meta["sample"]["action"]["dense"][
            "down_sample_steps"
        ]
        dense_execution_horizon = (
            dense_action_down_sample_steps * control_para["dense_execution_horizon"]
        )

    action_type = "pose9"  # "pose9" or "pose9pose9s1"
    id_list = [0]
    if shape_meta["action"]["shape"][0] == 9:
        action_type = "pose9"
    elif shape_meta["action"]["shape"][0] == 19:
        action_type = "pose9pose9s1"
    elif shape_meta["action"]["shape"][0] == 38:
        action_type = "pose9pose9s1"
        id_list = [0, 1]
    else:
        raise RuntimeError("unsupported")

    if action_type == "pose9pose9s1":
        action_to_trajectory = pose9pose9s1_to_traj
    else:
        raise RuntimeError("unsupported")

    printOrNot(vbs_h2, "Creating MPC.")
    controller = ModelPredictiveControllerHybrid(
        shape_meta=shape_meta,
        id_list=id_list,
        policy=policy,
        action_to_trajectory=action_to_trajectory,
        sparse_execution_horizon=sparse_execution_horizon,
        dense_execution_horizon=dense_execution_horizon,
        test_sparse_action=control_para["test_sparse_action"],
        fix_orientation=control_para["fix_orientation"],
        dense_execution_offset=control_para["dense_execution_offset"],
    )
    controller.set_time_offset(env.current_hardware_time_s)

    #########################################
    # main loop starts
    #########################################
    obs_raw = env.get_observation_from_buffer()
    obs_task = dict()
    raw_to_obs(obs_raw, obs_task, shape_meta)
    sparse_obs_data = {}
    for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
        data = obs_task["obs"][key]
        horizon = attr["horizon"]
        down_sample_steps = attr["down_sample_steps"]
        # sample 'horizon' number of latest obs from the queue
        assert len(data) >= (horizon - 1) * down_sample_steps + 1
        sparse_obs_data[key] = data[
            -(horizon - 1) * down_sample_steps - 1 :: down_sample_steps
        ]
    obs_sample_np = {}
    obs_sample_np["sparse"], _SE3_WBase = task.sparse_obs_to_obs_sample(
        obs_sparse=sparse_obs_data,
        shape_meta=shape_meta,
        reshape_mode="reshape",
        id_list=id_list,
        ignore_rgb=False,
    )
    # add batch dimension
    obs_sample_np = dict_apply(obs_sample_np, lambda x: rearrange(x, "... -> 1 ..."))

    # make a copy
    obs_sample_np2 = copy.deepcopy(obs_sample_np)
    policy2 = copy.deepcopy(policy)

    # run inference
    # s1 = torch.cuda.Stream()
    # s2 = torch.cuda.Stream()
    process2 = torch.multiprocessing.Process(
        target=run_inference, args=(obs_sample_np2, policy2, 2, device)
    )
    process2.start()

    run_inference(obs_sample_np, policy, 1, device)

    process2.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
