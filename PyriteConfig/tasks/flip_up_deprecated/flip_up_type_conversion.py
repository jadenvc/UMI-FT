from typing import Dict, Union

import numpy as np
import PyriteUtility.spatial_math.spatial_utilities as su
import zarr
from PyriteUtility.computer_vision.computer_vision_utility import get_image_transform

##
## raw: keys used in the dataset. Each key contains data for a whole episode
## obs: keys used in inference. Needs some pre-processing before sending to the policy NN.
## obs_preprocessed: obs with normalized rgb keys. len = whole episode
## obs_sample: len = obs horizon, pose computed relative to current pose (id = -1)
## action: pose command in world frame. len = whole episode
## action_sample: len = action horizon, pose computed relative to current pose (id = 0)


def raw_to_obs(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
    shape_meta: dict,
):
    """convert shape_meta.raw data to shape_meta.obs.

    This function keeps image data as compressed zarr array in memory, while loads and decompresses
    low dim data.

    Args:
      raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
      episode_data: output dictionary that matches shape_meta.obs
    """
    episode_data["obs"] = {}
    # obs.rgb: keep entry, keep as compressed zarr array in memory
    for key, attr in shape_meta["raw"].items():
        type = attr.get("type", "low_dim")
        if type == "rgb":
            # obs.rgb: keep as compressed zarr array in memory
            episode_data["obs"][key] = raw_data[key]

    # obs.low_dim: load entry, convert to obs.low_dim
    pose7_fb = raw_data["ts_pose_fb"]
    pose9_fb = su.SE3_to_pose9(su.pose7_to_SE3(pose7_fb))

    episode_data["obs"]["robot0_eef_pos"] = pose9_fb[..., :3]
    episode_data["obs"]["robot0_eef_rot_axis_angle"] = pose9_fb[..., 3:]
    episode_data["obs"]["robot0_eef_wrench"] = raw_data["wrench"][:]

    if "low_dim_time_stamps" in raw_data.keys():
        # time stamp format used in training
        episode_data["obs"]["low_dim_time_stamps"] = raw_data["low_dim_time_stamps"][:]
        episode_data["obs"]["visual_time_stamps"] = raw_data["visual_time_stamps"][:]

    if "ts_pose_fb_timestamp_s" in raw_data.keys():
        # time stamp format used in real experiments
        episode_data["obs"]["camera0_rgb_timestamp_s"] = raw_data[
            "camera0_rgb_timestamp_s"
        ][:]
        episode_data["obs"]["robot0_eef_pos_timestamp_s"] = raw_data[
            "ts_pose_fb_timestamp_s"
        ][:]
        episode_data["obs"]["robot0_eef_rot_axis_angle_timestamp_s"] = raw_data[
            "ts_pose_fb_timestamp_s"
        ][:]
        episode_data["obs"]["robot0_eef_wrench_timestamp_s"] = raw_data[
            "wrench_timestamp_s"
        ][:]


def raw_to_action9(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    # action: assemble from low_dim
    ts_pose7_command = raw_data["ts_pose_command"]
    ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
    episode_data["action"] = ts_pose9_command[:]
    assert episode_data["action"].shape[1] == 9
    episode_data["action_time_stamps"] = raw_data["low_dim_time_stamps"]


def raw_to_action19(
    raw_data: Union[zarr.Group, Dict[str, np.ndarray]],
    episode_data: dict,
):
    """Convert shape_meta.raw data to shape_meta.action.
    Note: if relative action is used, the relative pose still needs to be computed every time a sample
    is made. This function only converts the whole episode, and does not know what pose to be relative to.

    Args:
        raw_data: input, has keys from shape_meta.raw, each value is an ndarray of shape (t, ...)
        episode_data: output dictionary that has an 'action' field that matches shape_meta.action
    """
    # action: assemble from low_dim
    ts_pose7_command = raw_data["ts_pose_command"][:]
    ts_pose9_command = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_command))
    ts_pose7_virtual_target = raw_data["ts_pose_virtual_target"][:]
    ts_pose9_virtual_target = su.SE3_to_pose9(su.pose7_to_SE3(ts_pose7_virtual_target))
    stiffness = raw_data["stiffness"][:][:, np.newaxis]
    episode_data["action"] = np.concatenate(
        [ts_pose9_command, ts_pose9_virtual_target, stiffness], axis=-1
    )
    assert episode_data["action"].shape[1] == 19
    episode_data["action_time_stamps"] = raw_data["low_dim_time_stamps"]


def obs_rgb_preprocess(
    obs: dict,
    obs_output: dict,
    reshape_mode: str,
    shape_meta: dict,
):
    """Pre-process the rgb data in the obs dictionary as inputs to policy network.

    This function does the following to the rgb keys in the obs dictionary:
    * Unpack/unzip it, if the rgb data is still stored as a compressed zarr array (not recommended)
    * Reshape the rgb image, or just check its shape.
    * Convert uint8 (0~255) to float32 (0~1)
    * Move its axes from THWC to TCHW.
    Since this function unpacks the whole key, it should only be used for online inference.
    If used in training, so the data length is the obs horizon instead of the whole episode len.

    Args:
        obs: dict with keys from shape_meta.obs
        obs_output: dict with the same keys but processed images
        reshape_mode: One of 'reshape', 'check', or 'none'.
        shape_meta: the shape_meta from task.yaml
    """
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = obs[key]
            t, hi, wi, ci = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi):
                if reshape_mode == "reshape":
                    tf = get_image_transform(
                        input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                    )
                    out_imgs = np.stack([tf(x) for x in this_imgs_in])
                elif reshape_mode == "check":
                    print(
                        f"[obs_rgb_preprocess] shape check failed! Require {ho}x{wo}, get {hi}x{wi}"
                    )
                    assert False
            if this_imgs_in.dtype == np.uint8:
                out_imgs = out_imgs.astype(np.float32) / 255

            # THWC to TCHW
            obs_output[key] = np.moveaxis(out_imgs, -1, 1)


def sparse_obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    shape_meta: dict,
    reshape_mode: str,
    ignore_rgb: bool = False,
):
    """Prepare a sample of sparse obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.sparse, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    return:
        sparse_obs_processed: dict with keys from shape_meta['sample']['obs']['sparse']
        base_SE3: the initial pose used for relative pose calculation
    """
    sparse_obs_processed = {}
    assert len(obs_sparse) > 0
    if not ignore_rgb:
        obs_rgb_preprocess(obs_sparse, sparse_obs_processed, reshape_mode, shape_meta)

    for key, attr in shape_meta["obs"].items():
        type = attr.get("type", "low_dim")
        if type == "low_dim":
            sparse_obs_processed[key] = obs_sparse[key].astype(
                np.float32
            )  # astype() makes a copy

    # generate relative pose
    # convert pose to mat
    SE3_WT = su.pose9_to_SE3(
        np.concatenate(
            [
                sparse_obs_processed["robot0_eef_pos"],
                sparse_obs_processed["robot0_eef_rot_axis_angle"],
            ],
            axis=-1,
        )
    )

    # solve relative obs
    base_SE3_WT = SE3_WT[-1]
    SE3_base_i = su.SE3_inv(base_SE3_WT) @ SE3_WT

    pose9_relative = su.SE3_to_pose9(SE3_base_i)
    sparse_obs_processed["robot0_eef_pos"] = pose9_relative[..., :3]
    sparse_obs_processed["robot0_eef_rot_axis_angle"] = pose9_relative[..., 3:]

    # solve relative wrench
    SE3_i_base = su.SE3_inv(SE3_base_i)
    wrench_0 = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
        obs_sparse["robot0_eef_wrench"], -1
    )
    sparse_obs_processed["robot0_eef_wrench"] = np.squeeze(wrench_0)

    # double check the shape
    for key, attr in shape_meta["sample"]["obs"]["sparse"].items():
        sparse_obs_horizon = attr["horizon"]
        if shape_meta["obs"][key]["type"] == "low_dim":
            assert len(sparse_obs_processed[key].shape) == 2  # (T, D)
            assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon
        else:
            if not ignore_rgb:
                assert len(sparse_obs_processed[key].shape) == 4  # (T, C, H, W)
                assert sparse_obs_processed[key].shape[0] == sparse_obs_horizon

    return sparse_obs_processed, base_SE3_WT


def dense_obs_to_obs_sample(
    obs_dense: dict,  # each key: (H, T, D)
    shape_meta: dict,
    base_SE3: np.ndarray,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.sample.obs.dense, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    Since dense obs only contains low dim data, it only does the low dim part:
        low dim: convert pose to relative pose about the initial pose of the SPARSE horizon

    Args:
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
    """
    dense_obs_processed = {}
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        dense_obs_processed[key] = obs_dense[key].astype(
            np.float32
        )  # astype() makes a copy
    # get the length of the first key in the dictionary obs_dense
    H = next(iter(obs_dense.values())).shape[0]

    # convert each dense horizon to the same relative pose
    for step in range(H):
        # generate relative pose. Everything is (T, D)
        # convert pose to mat
        SE3 = su.pose9_to_SE3(
            np.concatenate(
                [
                    obs_dense["robot0_eef_pos"][step],
                    obs_dense["robot0_eef_rot_axis_angle"][step],
                ],
                axis=-1,
            )
        )

        # solve relative obs
        SE3_base_i = np.linalg.inv(base_SE3) @ SE3

        pose9_relative = su.SE3_to_pose9(SE3_base_i)
        dense_obs_processed["robot0_eef_pos"][step] = pose9_relative[..., :3]
        dense_obs_processed["robot0_eef_rot_axis_angle"][step] = pose9_relative[..., 3:]

        SE3_i_base = su.SE3_inv(SE3_base_i)
        wrench_0 = su.transpose(su.SE3_to_adj(SE3_i_base)) @ np.expand_dims(
            obs_dense["robot0_eef_wrench"][step], -1
        )
        dense_obs_processed["robot0_eef_wrench"][step] = np.squeeze(wrench_0)

    # double check the shape
    for key in shape_meta["sample"]["obs"]["dense"].keys():
        assert dense_obs_processed[key].shape[0] == H
        assert len(dense_obs_processed[key].shape) == 3  # (H, T, D)

    return dense_obs_processed


def obs_to_obs_sample(
    obs_sparse: dict,  # each key: (T, D)
    obs_dense: dict,  # each key: (H, T, D)
    shape_meta: dict,
    reshape_mode: str,
    ignore_rgb: bool = False,
):
    """Prepare a sample of obs as inputs to policy network.

    After packing an obs dictionary with keys according to shape_meta.obs, with
    length corresponding to the correct horizons, pass it to this function to get it ready
    for the policy network.

    It does two things:
        1. RGB: unpack, reshape, normalize, turn into float
        2. low dim: convert pose to relative pose, turn into float
    For sparse obs, it does both. For dense obs, it only does the low dim part, and all poses are
    computed relative to the same current pose (id = 0).

    Args:
        obs_sparse: dict with keys from shape_meta['sample']['obs']['sparse']
        obs_dense: dict with keys from shape_meta['sample']['obs']['dense']
        shape_meta: the shape_meta from task.yaml
        reshape_mode: One of 'reshape', 'check', or 'none'.
        ignore_rgb: if True, skip the rgb keys. Used when computing normalizers.
    """
    obs_processed = {"sparse": {}, "dense": {}}
    obs_processed["sparse"], base_pose_mat = sparse_obs_to_obs_sample(
        obs_sparse, shape_meta, reshape_mode, ignore_rgb
    )
    if len(obs_dense) > 0:
        obs_processed["dense"] = dense_obs_to_obs_sample(
            obs_dense, shape_meta, base_pose_mat
        )

    return obs_processed


def action9_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 9
    action_dense: np.ndarray,  # (H, T, D), D = 9
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    assert D == 9

    # generate relative pose
    # convert pose to mat
    pose9 = action_sparse
    SE3 = su.pose9_to_SE3(pose9)

    # solve relative obs
    base_SE3 = SE3[0]
    SE3_relative = su.SE3_inv(base_SE3) @ SE3

    pose9_relative = su.SE3_to_pose9(SE3_relative)
    action_processed["sparse"] = pose9_relative

    if len(action_dense) > 0:
        action_processed["dense"] = np.zeros_like(action_dense)
        H = action_dense.shape[0]
        for step in range(H):
            # generate relative pose
            # convert pose to mat
            pose9 = action_dense[step]
            SE3 = su.pose9_to_SE3(pose9)

            # solve relative obs
            SE3_relative = su.SE3_inv(base_SE3) @ SE3

            pose9_relative = su.SE3_to_pose9(SE3_relative)
            action_processed["dense"][step] = pose9_relative

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action19_to_action_sample(
    action_sparse: np.ndarray,  # (T, D), D = 19
    action_dense: np.ndarray,  # (H, T, D) not used
):
    """Prepare a sample of actions as labels for the policy network.

    This function is used in training. It takes a sample of actions (len = action_horizon)
    and convert the poses in it to be relative to the current pose (id = 0).

    """
    action_processed = {"sparse": {}, "dense": {}}
    T, D = action_sparse.shape
    assert D == 19

    # generate relative pose
    # convert pose to mat
    pose9 = action_sparse[:, 0:9]
    pose9_vt = action_sparse[:, 9:18]
    stiffness = action_sparse[:, 18:19]
    SE3 = su.pose9_to_SE3(pose9)
    SE3_vt = su.pose9_to_SE3(pose9_vt)

    # solve relative obs
    SE3_WBase_inv = su.SE3_inv(SE3[0])
    SE3_relative = SE3_WBase_inv @ SE3
    SE3_vt_relative = SE3_WBase_inv @ SE3_vt

    pose9_relative = su.SE3_to_pose9(SE3_relative)
    pose9_vt_relative = su.SE3_to_pose9(SE3_vt_relative)
    action_processed["sparse"] = np.concatenate(
        [pose9_relative, pose9_vt_relative, stiffness], axis=-1
    )

    if len(action_dense) > 0:
        action_processed["dense"] = np.zeros_like(action_dense)
        H = action_dense.shape[0]
        for step in range(H):
            # generate relative pose
            # convert pose to mat
            pose9 = action_dense[step]
            SE3 = su.pose9_to_SE3(pose9)

            # solve relative obs
            SE3_relative = SE3_WBase_inv @ SE3

            pose9_relative = su.SE3_to_pose9(SE3_relative)
            action_processed["dense"][step] = pose9_relative

    # double check the shape
    assert action_processed["sparse"].shape == (T, D)
    if len(action_dense) > 0:
        assert action_processed["dense"].shape == action_dense.shape

    return action_processed


def action9_postprocess(
    action: np.ndarray,
    env_obs: Dict[str, np.ndarray],
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    # convert poses to mat
    current_SE3 = su.pose9_to_SE3(
        np.concatenate(
            [env_obs["robot0_eef_pos"][-1], env_obs["robot0_eef_rot_axis_angle"][-1]],
            axis=-1,
        )
    )

    action_pose9 = action[..., 0:9]
    action_SE3 = su.pose9_to_SE3(action_pose9)

    action_SE3_absolute = current_SE3 @ action_SE3

    # return pose matrices
    return action_SE3_absolute


def action19_postprocess(
    action: np.ndarray, current_SE3: np.ndarray, fix_orientation=False
):
    """Convert policy outputs from relative pose to world frame pose
    Used in online inference
    """
    action_pose9 = action[..., 0:9]
    action_pose9_vt = action[..., 9:18]
    stiffness = action[..., 18]
    action_SE3 = su.pose9_to_SE3(action_pose9)
    action_SE3_vt = su.pose9_to_SE3(action_pose9_vt)

    action_SE3_absolute = current_SE3 @ action_SE3
    action_SE3_vt_absolute = current_SE3 @ action_SE3_vt

    # print(f"fix_orientation: {fix_orientation}")
    if fix_orientation:
        action_SE3_absolute[:, :3, :3] = current_SE3[:3, :3]
        action_SE3_vt_absolute[:, :3, :3] = current_SE3[:3, :3]

    # return pose matrices
    return action_SE3_absolute, action_SE3_vt_absolute, stiffness
