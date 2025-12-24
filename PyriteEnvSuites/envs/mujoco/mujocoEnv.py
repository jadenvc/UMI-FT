import gc
import os
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

# import defaultdict
import mujoco
import numpy as np
import scipy.spatial.transform as st
from dm_control import mjcf
from dm_control.mujoco.engine import Physics
from PyriteUtility.data_pipeline.episode_data_buffer import VideoData
from PyriteUtility.planning_control.filtering import LiveLPFilter

from PyriteEnvSuites.envs.mujoco.ur5 import Ur5, Ur5WSG50
from PyriteEnvSuites.envs.utlity.env_utility import (
    JointState,
    LinkState,
    Velocity,
    get_part_path,
    parse_contact_data,
)
from PyriteEnvSuites.envs.utlity.robot_utlity import (
    JSControlCmd,
    Pose,
    TSControlCmd,
    euler2quat,
)

# Get package root path (PyriteEnvSuites/) for loading assets
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class SegmentMapping(Enum):
    # backgroun_seg = 0
    wrist_seg = 1
    shoulder_seg = 2
    upperarm_seg = 3
    top_seg = 4
    forearm_seg = 5
    leftfinger_seg = 6
    rightfinger_seg = 7
    wsg_seg = 8
    object_seg = 9
    # bin_seg = 10


class MujocoEnv:
    def __init__(
        self, robot_cls, prefix, verbose, in_place_ik=True, use_gui=False
    ) -> None:
        self.verbose = verbose
        self.in_place_ik = in_place_ik
        self.mj_physics = self.setup()
        self.robot = robot_cls(self.mj_physics, None, None, prefix)
        print("mj_physics.timestep: ", self.mj_physics.timestep())
        self.controll_frequency = 1 / self.mj_physics.timestep()
        self.use_gui = use_gui
        self.gui_handle = None

        # TODO: this should be a parameter read from a yaml
        robot_name = "ur5e"
        axis_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ]
        joint_names = []
        actuator_names = []
        for name in axis_names:
            actuator_names.append(robot_name + "/" + name)
            joint_names.append(robot_name + "/" + name + "_joint")

        self.dof_ids = np.array(
            [self.mj_physics.model.joint(name).id for name in joint_names]
        )
        self.actuator_ids = np.array(
            [self.mj_physics.model.actuator(name).id for name in actuator_names]
        )
        print("dof_ids: ", self.dof_ids)
        print("actuator_ids: ", self.actuator_ids)

        ## Parameters for JS mode
        self.js_pos_gain = np.array([8000, 8000, 8000, 8000, 8000, 8000])
        self.js_vel_gain = np.array([400, 400, 400, 100, 100, 100])

        ## parameters for TS mode
        self.ts_Kp = 2 * np.diag([8000, 8000, 8000, 2000, 2000, 2000])
        self.ts_Kd = 8 * np.array([8, 8, 8, 2, 2, 2])

        # pre-allocate memory for TS mode
        self.jac = np.zeros((6, self.mj_physics.model.nv))
        self.twist = np.zeros(6)
        self.site_quat = np.zeros(4)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_inv = np.zeros((self.mj_physics.model.nv, self.mj_physics.model.nv))

        # TODO: additional setups
        #  model.opt.timestep = dt

    def reset(self, **kwargs):
        del self.mj_physics
        gc.collect()
        mj_physics = self.setup()
        self.update_mj_physics(mj_physics)
        gc.collect()
        self.mj_physics.forward()
        self._time_counter = 0

    def load_mj_physics(self, mj_physics):
        self.mj_physics = mj_physics
        self.mj_physics.forward()

    def load_from_vector(self, qpos, qvel):
        self.mj_physics.data.qpos[:] = qpos
        self.mj_physics.data.qvel[:] = qvel
        # crutial step
        self.mj_physics.forward()

    @property
    def dt(self):
        return 1 / self.controll_frequency

    @property
    def current_time(self):
        return self.dt * self._time_counter

    def update_mj_physics(self, mj_physics: Physics):
        self.robot.mj_physics = mj_physics
        self.mj_physics = mj_physics
        from mujoco import viewer

        if self.gui_handle is not None:
            self.gui_handle.close()
            self.gui_handle = None
        if self.use_gui:
            self.gui_handle = viewer.launch_passive(
                model=self.mj_physics.model.ptr, data=self.mj_physics.data.ptr
            )
            self.gui_handle.user_scn.ngeom = 0

    def cleanup(self):
        if self.gui_handle is not None:
            self.gui_handle.close()
            self.gui_handle = None

    @abstractmethod
    def setup_model(self):
        raise NotImplementedError()

    @abstractmethod
    def setup_objs(self, world_model):
        pass

    def setup(self) -> Tuple[Physics, int]:
        world_model = self.setup_model()
        self.setup_objs(world_model=world_model)
        mj_physics = mjcf.Physics.from_mjcf_model(world_model)

        return mj_physics

    def update_stiffness_arrows(self, sizes, pos, mat0, mat1, mat2):
        self.gui_handle.user_scn.ngeom = 0

        mujoco.mjv_initGeom(
            self.gui_handle.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_ARROW2,
            size=np.array([0.01, 0.01, 0.2]) * sizes[0],
            pos=pos,
            mat=mat0.flatten(),
            rgba=np.array([100, 0, 0, 2]),
        )
        mujoco.mjv_initGeom(
            self.gui_handle.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_ARROW2,
            size=np.array([0.01, 0.01, 0.2]) * sizes[1],
            pos=pos,
            mat=mat1.flatten(),
            rgba=np.array([0, 100, 0, 2]),
        )
        mujoco.mjv_initGeom(
            self.gui_handle.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_ARROW2,
            size=np.array([0.01, 0.01, 0.2]) * sizes[2],
            pos=pos,
            mat=mat2.flatten(),
            rgba=np.array([0, 0, 100, 2]),
        )
        self.gui_handle.user_scn.ngeom = 3

    def update_one_arrow(
        self,
        size,
        pos,
        mat,
        rgba=np.array([100, 0, 0, 2]),
        size_limit=10,
        ngeom_before=0,
    ):
        """Update one arrow in the scene.
        Arrow head is aligned with the last column of mat (z axis)
        """
        if size > size_limit:
            size = size_limit
        self.gui_handle.user_scn.ngeom = ngeom_before

        mujoco.mjv_initGeom(
            self.gui_handle.user_scn.geoms[ngeom_before],
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([0.05, 0.05, 1]) * size,
            pos=pos,
            mat=mat.flatten(),
            rgba=rgba,
        )
        self.gui_handle.user_scn.ngeom = ngeom_before + 1
        return ngeom_before + 1

    def update_trajs(self, pos, rgba, ngeom_before=0):
        self.gui_handle.user_scn.ngeom = ngeom_before
        N = pos.shape[0]
        for i in range(N):
            mujoco.mjv_initGeom(
                self.gui_handle.user_scn.geoms[ngeom_before + i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.01, 0.0, 0.0]),
                pos=pos[i, :],
                mat=np.eye(3).flatten(),
                rgba=rgba,
            )
        self.gui_handle.user_scn.ngeom = ngeom_before + N
        return ngeom_before + N

    ##
    ## step forward a timestep. Must be called at the Mujoco step size
    ##
    def step_js(self, ctrl_cmd: np.array):
        """Step the simulation forward by one timestep given JS command.
        Motivated by:
        https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
        """
        q_target = ctrl_cmd
        tau = (
            self.js_pos_gain * (q_target - self.mj_physics.data.qpos[self.dof_ids])
            - self.js_vel_gain * self.mj_physics.data.qvel[self.dof_ids]
        )

        # # Add gravity compensation.
        tau += self.mj_physics.data.qfrc_bias[self.dof_ids]

        # Set the control signal and step the simulation.
        np.clip(tau, *self.mj_physics.model.actuator_forcerange.T, out=tau)

        self.mj_physics.data.ctrl[self.actuator_ids] = tau[self.actuator_ids]

        mujoco.mj_step(self.mj_physics.model.ptr, self.mj_physics.data.ptr)

        # self.mj_physics.data.ctrl[:] = np.append(
        #     ctrl_cmd.jpos, ctrl_cmd.gripper_ctrl
        # )
        # self.mj_physics.step()

        if self.gui_handle is not None:
            self.gui_handle.sync()
        self._time_counter += 1

    def step_ts(self, pose_cmd: np.ndarray, ts_Kp: np.ndarray = None):
        """
        Step the simulation forward by one timestep given TS command.
        Modified from:
        https://github.com/kevinzakka/mjctrl/blob/main/opspace.py

        :param pose_cmd: (7,) np array, desired end-effector pose
        """
        # mujoco variables
        data = self.mj_physics.data
        model = self.mj_physics.model
        site_name = self.robot.end_effector_site_name
        site_id = model.site(site_name).id
        # command
        cmd_pos = pose_cmd[:3]
        cmd_quat = pose_cmd[3:]
        # Spatial error
        dx = cmd_pos - data.site(site_id).xpos
        self.twist[:3] = dx
        mujoco.mju_mat2Quat(self.site_quat, data.site(site_id).xmat)
        mujoco.mju_negQuat(self.site_quat_conj, self.site_quat)
        mujoco.mju_mulQuat(self.error_quat, cmd_quat, self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)

        # Jacobian.
        mujoco.mj_jacSite(model.ptr, data.ptr, self.jac[:3], self.jac[3:], site_id)

        # Compute generalized forces.
        if ts_Kp is None:
            ts_Kp = self.ts_Kp
        tau = (
            self.jac.T @ (ts_Kp @ self.twist)
            - np.concatenate([self.ts_Kd, np.zeros(6)]) * data.qvel
        )

        # Add gravity compensation.
        tau += data.qfrc_bias

        # Set the control signal and step the simulation.
        np.clip(
            tau[self.actuator_ids],
            *self.mj_physics.model.actuator_forcerange.T,
            out=tau[self.actuator_ids],
        )
        data.ctrl[self.actuator_ids] = tau[self.actuator_ids]
        mujoco.mj_step(model.ptr, data.ptr)

        if self.gui_handle is not None:
            self.gui_handle.sync()
        self._time_counter += 1

    ###@profile
    def render_rgb(self, camera_ids: List[int], height=480, width=600):
        rgbs = {}
        for camera_id in camera_ids:
            rgbs[camera_id] = self.mj_physics.render(
                camera_id=camera_id,
                height=height,
                width=width,
            )
        return rgbs

    def render_depth(self, camera_ids: List[int], height=480, width=600):
        depths = {}
        for camera_id in camera_ids:
            depths[camera_id] = self.mj_physics.render(
                camera_id=camera_id, height=height, width=width, depth=True
            )
        return depths

    def render_segmentation(self, camera_ids: List[int], height=480, width=600):
        segmentations = {}
        for camera_id in camera_ids:
            segmentations[camera_id] = self.mj_physics.render(
                camera_id=camera_id, height=height, width=width, segmentation=True
            )
        return segmentations

    def render_all(
        self, camera_ids: List[int] = [], height=480, width=600, view="camera"
    ):
        if len(camera_ids) == 0:
            camera_ids = np.arange(self.mj_physics.model.ncam)
        rgbs = self.render_rgb(camera_ids=camera_ids, height=height, width=width)
        depths = self.render_depth(camera_ids=camera_ids, height=height, width=width)
        segmentations = self.render_segmentation(
            camera_ids=camera_ids, height=height, width=width
        )
        if view == "type":
            return rgbs, depths, segmentations
        elif view == "camera":
            return {
                camera_id: VideoData(
                    rgb=rgbs[camera_id],
                    depth=depths[camera_id],
                    segmentation=segmentations[camera_id],
                    camera_id=camera_id,
                )
                for camera_id in camera_ids
            }
        else:
            raise ValueError("view should be type or camera")

    def render_rgb_only(
        self, camera_ids: List[int] = [], height=480, width=600, view="camera"
    ):
        if len(camera_ids) == 0:
            camera_ids = np.arange(self.mj_physics.model.ncam)
        rgbs = self.render_rgb(camera_ids=camera_ids, height=height, width=width)
        if view == "type":
            return rgbs
        elif view == "camera":
            return {
                camera_id: VideoData(
                    rgb=rgbs[camera_id],
                    camera_id=camera_id,
                )
                for camera_id in camera_ids
            }
        else:
            raise ValueError("view should be type or camera")

    def render_camera(self, camera_id, height=480, width=600):
        img_arr = self.mj_physics.render(
            camera_id=camera_id,
            height=height,
            width=width,
        )
        return img_arr

    def get_state(self):
        obj_link_states: Dict[str, Dict[str, LinkState]] = {}
        obj_joint_states: Dict[str, Dict[str, JointState]] = {}
        model = self.mj_physics.model
        data = self.mj_physics.data

        obj_link_contacts = parse_contact_data(physics=self.mj_physics)
        for bodyid in range(model.nbody):
            body_model = model.body(bodyid)
            body_data = data.body(bodyid)  # type: ignore
            pose = Pose(
                position=body_data.xpos.copy(),
                orientation=body_data.xquat.copy(),
            )
            root_name = model.body(body_model.rootid).name
            if root_name not in obj_link_states:
                obj_link_states[root_name] = {}
            if root_name not in obj_joint_states:
                obj_joint_states[root_name] = {}
            part_path = get_part_path(self.mj_physics.model, body_model)
            obj_link_states[root_name][part_path] = LinkState(
                link_path=part_path,
                obj_name=root_name,
                pose=pose,
                velocity=Velocity(
                    linear_velocity=body_data.cvel[3:].copy(),
                    angular_velocity=body_data.cvel[:3].copy(),
                ),
                contacts=(
                    obj_link_contacts[root_name][part_path]
                    if (
                        root_name in obj_link_contacts
                        and part_path in obj_link_contacts[root_name]
                    )
                    else set()
                ),
            )
        return obj_joint_states, obj_link_states, obj_link_contacts


class MujocoUR5WSG50FinrayEnv(MujocoEnv):
    def __init__(
        self,
        prefix="ur5e",
        random_init_robot=False,
        **kwargs,
    ) -> None:
        self.random_init_robot = random_init_robot
        self.mimic_real = kwargs.get("mimic_real", False)
        super().__init__(robot_cls=Ur5WSG50, prefix=prefix, **kwargs)

    def setup_model(self):
        if self.mimic_real:
            world_model = mjcf.from_path(PACKAGE_PATH + "assets/ground2.xml")
        else:
            world_model = mjcf.from_path(PACKAGE_PATH + "assets/ground.xml")
        robot_model = mjcf.from_path(
            PACKAGE_PATH + "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        attachment_site = robot_model.find("site", "attachment_site")
        assert attachment_site is not None
        gripper = mjcf.from_path(PACKAGE_PATH + "assets/wsg50/wsg50_finray_milibar.xml")
        attachment_site.attach(gripper)

        cam_mount_site = gripper.find("site", "cam_mount")
        cam = mjcf.from_path(
            PACKAGE_PATH + "assets/mujoco_menagerie/realsense_d435i/d435i_with_cam.xml"
        )
        cam_mount_site.attach(cam)

        robit_site = world_model.worldbody.add(
            "site", name="robot_site", pos=(-0.3, 0.0, 0.05)
        )
        robit_site.attach(robot_model)

        return world_model

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(**kwargs)
        if self.random_init_robot:
            self.robot.reset_random_robot_home()
        else:
            self.robot.reset_robot_home()


class MujocoUR5BigProbeEnv(MujocoEnv):
    def __init__(
        self,
        prefix="ur5e",
        filter_params=None,
        **kwargs,
    ) -> None:
        super().__init__(robot_cls=Ur5, prefix=prefix, **kwargs)
        self.robot.set_end_effector_site_name("big_probe/end_effector_site")
        self.robot.set_ft_sensor_site_name("big_probe/ft_sensor_site")
        if filter_params is not None:
            self.ft_filter = LiveLPFilter(
                fs=filter_params["sampling_freq"],
                cutoff=filter_params["cutoff_freq"],
                order=filter_params["order"],
                dim=6,
            )
        else:
            self.ft_filter = None

    def setup_model(self):
        world_model = mjcf.from_path(PACKAGE_PATH + "assets/ground.xml")
        robot_model = mjcf.from_path(
            PACKAGE_PATH + "assets/mujoco_menagerie/universal_robots_ur5e/ur5e.xml"
        )
        del robot_model.keyframe
        robot_model.worldbody.light.clear()
        attachment_site = robot_model.find("site", "attachment_site")
        assert attachment_site is not None
        eoat = mjcf.from_path(PACKAGE_PATH + "assets/custom/big_probe/big_probe.xml")
        attachment_site.attach(eoat)

        cam_mount_site = eoat.find("site", "cam_mount")
        cam = mjcf.from_path(
            PACKAGE_PATH + "assets/mujoco_menagerie/realsense_d435i/d435i_with_cam.xml"
        )
        cam_mount_site.attach(cam)

        robit_site = world_model.worldbody.add(
            "site", name="robot_site", pos=(-0.3, 0.0, 0.05)
        )
        robit_site.attach(robot_model)

        return world_model

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(**kwargs)
        self.robot.reset_robot_home()

    def get_ft_sensor_reading(self):
        force = self.mj_physics.data.sensor("ur5e/big_probe/wrist_force_sensor").data
        torque = self.mj_physics.data.sensor("ur5e/big_probe/wrist_torque_sensor").data
        return np.concatenate([force, torque])

    def get_ft_sensor_reading_filtered(self):
        assert self.ft_filter is not None
        return self.ft_filter(self.get_ft_sensor_reading())


class TableTopUR5WSG50FinrayEnv(MujocoUR5WSG50FinrayEnv):
    def __init__(
        self,
        camera_ids: List[int] = [0, 2],
        camera_res: List[int] = [480, 600],
        euler_gripper_rotation=True,
        delta_space=True,
        delta_orientation_limit=0.05,
        delta_position_limit=0.05,
        TabelTopConstraints: List[float] = [
            -0.1,
            1.2,
            -0.7,
            0.7,
            0.06,
            0.5,
        ],
        **kwargs,
    ) -> None:
        self.camera_ids = camera_ids
        self.camera_res = camera_res
        super().__init__(**kwargs)
        self.euler_gripper_rotation = euler_gripper_rotation
        self.delta_space = delta_space
        self.delta_orientation_limit = delta_orientation_limit
        self.delta_position_limit = delta_position_limit
        (
            self.x_lower_limit,
            self.x_upper_limit,
            self.y_lower_limit,
            self.y_upper_limit,
            self.z_lower_limit,
            self.z_upper_limit,
        ) = TabelTopConstraints

    @property
    def euler_observation_scaler(self):
        return 1 / (2 * np.pi)

    def setup_model(self):
        world_model = super().setup_model()
        if self.mimic_real:
            table_model = mjcf.from_file(PACKAGE_PATH + "assets/custom/table2.xml")
        else:
            table_model = mjcf.from_file(PACKAGE_PATH + "assets/custom/table/table.xml")
        table_attachment_site = world_model.worldbody.add(
            "site", name="table_attachment_site", pos=(0.5, 0, 0)
        )
        table_attachment_site.attach(table_model)
        return world_model

    # @profile
    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed)
        return self.get_observation(), {}

    # @profile
    def stepWP(self, action):
        # process gripper
        if self.euler_gripper_rotation:
            if self.delta_space:
                current_position = self.robot.get_end_effector_position()
                current_orientation = self.robot.get_end_effector_rotation()
                delta_position = action[:3]
                delta_rot = st.Rotation.from_euler("xyz", action[3:6])
                target_rotation_in_quat = (delta_rot * current_orientation).as_quat()[
                    [3, 0, 1, 2]
                ]
                target_position = current_position + delta_position
                control_action = np.concatenate(
                    [target_position, target_rotation_in_quat, [action[-1]]]
                )
            else:
                euler_rotation = action[3:6]
                target_rotation_in_quat = euler2quat(euler_rotation, degrees=False)
                control_action = np.concatenate(
                    [action[:3], target_rotation_in_quat, [action[-1]]]
                )
        else:
            control_action = action.copy()

        control_action = self.clip_control_action(control_action)

        ctrl_cmd = TSControlCmd.from_flattened(control_action)
        super().stepWP(ctrl_cmd)
        return None, 0, False, False, {}

    def clip_control_action(self, control_action):
        control_action[0] = np.clip(
            control_action[0], self.x_lower_limit, self.x_upper_limit
        )
        control_action[1] = np.clip(
            control_action[1], self.y_lower_limit, self.y_upper_limit
        )
        control_action[2] = np.clip(
            control_action[2], self.z_lower_limit, self.z_upper_limit
        )
        return control_action

    # @profile
    @abstractmethod
    def get_observation(self):
        pass

    @abstractmethod
    def get_low_dim_observation(self):
        pass

    # @profile
    def get_visual_observation(self):
        visual_observation = self.render_rgb(
            camera_ids=self.camera_ids,
            height=self.camera_res[0],
            width=self.camera_res[1],
        )
        visual_observation = {
            "camera_{}".format(k): v for k, v in visual_observation.items()
        }
        return visual_observation

    # @profile
    def get_ts_pose_fb(self):
        # TODO: fix range
        gripper_position = self.robot.curr_gripper_qpos
        if self.euler_gripper_rotation:
            ee_position = self.robot.get_end_effector_position()
            euler_rotation = self.robot.get_end_effector_rotation().as_euler(
                "xyz", degrees=False
            )
            normalized_euler_rotation = euler_rotation * self.euler_observation_scaler
            return np.concatenate(
                [ee_position, normalized_euler_rotation, [gripper_position]]
            ).astype(np.float32)
        else:
            end_effector_pose = self.robot.get_end_effector_pose().flattened
            return np.concatenate([end_effector_pose, [gripper_position]]).astype(
                np.float32
            )


class TableTopUR5BigProbeEnv(MujocoUR5BigProbeEnv):
    def __init__(
        self,
        camera_ids: List[int] = [0, 2],
        camera_res_hw: List[int] = [480, 600],
        **kwargs,
    ) -> None:
        self.camera_ids = camera_ids
        self.camera_res_hw = camera_res_hw
        super().__init__(**kwargs)

    def setup_model(self):
        world_model = super().setup_model()
        table_model = mjcf.from_file(PACKAGE_PATH + "assets/custom/table/table.xml")
        table_attachment_site = world_model.worldbody.add(
            "site", name="table_attachment_site", pos=(0.5, 0, 0)
        )
        table_attachment_site.attach(table_model)

        return world_model

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().reset(seed)
        return self.get_observation(), {}

    def step(self, jpos):
        ctrl_cmd = JSControlCmd.from_flattened(jpos, False)
        super().step(ctrl_cmd)

    @abstractmethod
    def get_low_dim_observation(self):
        pass

    def get_visual_observation(self):
        """
        return: visual_observation: list of rgb images, one for each camera
        """
        visual_observation = self.render_rgb(
            camera_ids=self.camera_ids,
            height=self.camera_res_hw[0],
            width=self.camera_res_hw[1],
        )
        return visual_observation

    def get_ts_pose_fb(self):
        end_effector_pose = self.robot.get_end_effector_pose().flattened
        return end_effector_pose.astype(np.float32)

    def get_js_fb(self):
        return self.robot.get_joint_positions()

    def get_ft_sensor_pose_fb(self):
        ft_sensor_pose = self.robot.get_ft_sensor_pose().flattened
        return ft_sensor_pose.astype(np.float32)

    def get_observation(self):
        visual_observation = self.get_visual_observation()
        ts_pose_fb = self.get_ts_pose_fb()
        ft_wrench = self.get_ft_sensor_reading()
        if self.ft_filter is not None:
            ft_wrench_filtered = self.get_ft_sensor_reading_filtered()

        # episode_js_force = self.robot.get_actuator_force()
        # episode_qpos = self.mj_physics.data.qpos.copy()
        # episode_qvel = self.mj_physics.data.qvel.copy()

        result = {
            "rgb": visual_observation,
            "ts_pose_fb": ts_pose_fb,
            "ft_wrench": ft_wrench,
        }
        if self.ft_filter is not None:
            ft_wrench_filtered = self.get_ft_sensor_reading_filtered()
            result["ft_wrench_filtered"] = ft_wrench_filtered
        return result
