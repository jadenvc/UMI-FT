import os

import numpy as np
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from scipy.spatial.transform import Rotation as R

from PyriteEnvSuites.envs.utlity.robot_utlity import Pose


class MujocoRobot:
    def __init__(self, mj_physics, init_joints, bodyid, prefix) -> None:
        self.mj_physics = mj_physics
        self.mj_physics_copy = mj_physics.copy(share_model=True)
        self.bodyid = bodyid
        self.prefix = prefix
        self.end_effector_site_name_postfix = "end_effector"
        self.ft_sensor_site_name_postfix = "ft_sensor"

    def joint_names(self):
        raise NotImplementedError()

    @property
    def joints(self):
        return [
            self.mj_physics.model.joint(joint_name) for joint_name in self.joint_names
        ]

    @property
    def joint_qpos_indices(self):
        return [joint.qposadr[0] for joint in self.joints]

    @property
    def joint_qvel_indices(self):
        return [joint.dofadr[0] for joint in self.joints]

    def get_joint_velocities(self):
        joint_velocities = self.mj_physics.data.qvel[self.joint_qvel_indices].copy()
        return joint_velocities

    def get_joint_positions(self):
        joint_positions = self.mj_physics.data.qpos[self.joint_qpos_indices].copy()
        return joint_positions

    @property
    def end_effector_site_name(self):
        return os.path.join(self.prefix, self.end_effector_site_name_postfix)

    def set_end_effector_site_name(self, name_postfix):
        self.end_effector_site_name_postfix = name_postfix

    @property
    def ft_sensor_site_name(self):
        return os.path.join(self.prefix, self.ft_sensor_site_name_postfix)

    def set_ft_sensor_site_name(self, name_postfix):
        self.ft_sensor_site_name_postfix = name_postfix

    def get_end_effector_pose(self, use_copied_physics=False):
        return self.get_site_pose(self.end_effector_site_name, use_copied_physics)

    def get_ft_sensor_pose(self, use_copied_physics=False):
        return self.get_site_pose(self.ft_sensor_site_name, use_copied_physics)

    def get_site_pose(self, site_name, use_copied_physics=False):
        # SCIPY using x, y, z, w
        quat_xyzw = self.get_site_rotation(site_name, use_copied_physics).as_quat()
        # MUJOCO, TRANSFORM3D using w, x, y, z
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return Pose(
            position=self.get_site_position(site_name, use_copied_physics),
            orientation=quat_wxyz,
        )

    def get_site_position(self, site_name, use_copied_physics=False):
        if use_copied_physics:
            physics = self.mj_physics_copy
        else:
            physics = self.mj_physics

        return physics.data.site(site_name).xpos.copy()

    def get_site_rotation(self, site_name, use_copied_physics=False):
        if use_copied_physics:
            physics = self.mj_physics_copy
        else:
            physics = self.mj_physics

        ee_xmt = physics.data.site(site_name).xmat.copy().reshape((3, 3))
        r = R.from_matrix(ee_xmt)
        return r

    def get_actuator_force(self):
        return self.mj_physics.data.actuator_force.copy()

    def forward_kinematics(self, qpos):
        self.mj_physics_copy.data.qpos[:] = self.mj_physics.data.qpos[:].copy()

        self.mj_physics_copy.data.qpos[: len(qpos)] = qpos
        mjbindings.mjlib.mj_fwdPosition(
            self.mj_physics_copy.model.ptr, self.mj_physics_copy.data.ptr
        )
        return self.get_end_effector_pose(True)

    # TODO: this function uses the legacy Pose class. Should delete
    # # @profile
    # def inverse_kinematics(self, pose: Pose, inplace=False):
    #     pose = Pose(pose.position, pose.orientation)
    #     if inplace:
    #         self.mj_physics_copy.data.qpos[:] = self.mj_physics.data.qpos[:].copy(
    #         )
    #         result = qpos_from_site_pose(
    #             physics=self.mj_physics_copy,
    #             site_name=self.end_effector_site_name,
    #             joint_names=self.joint_names,
    #             target_pos=pose.position,
    #             target_quat=pose.orientation,
    #             tol=1e-7,
    #             max_steps=100,
    #             inplace=inplace,
    #         )
    #     else:
    #         result = qpos_from_site_pose(
    #             physics=self.mj_physics,
    #             site_name=self.end_effector_site_name,
    #             target_pos=pose.position,
    #             target_quat=pose.orientation,
    #             joint_names=self.joint_names,
    #             tol=1e-7,
    #             max_steps=100,
    #             inplace=inplace,
    #         )
    #     if not result.success:
    #         return None
    #     return result.qpos[self.joint_qpos_indices].copy()

    def inverse_kinematics(self, pose_WT: np.array, inplace=False):
        pos = pose_WT[:3]
        quat = pose_WT[3:]
        if inplace:
            self.mj_physics_copy.data.qpos[:] = self.mj_physics.data.qpos[:].copy()
            result = qpos_from_site_pose(
                physics=self.mj_physics_copy,
                site_name=self.end_effector_site_name,
                joint_names=self.joint_names,
                target_pos=pos,
                target_quat=quat,
                tol=1e-7,
                max_steps=100,
                inplace=inplace,
            )
        else:
            result = qpos_from_site_pose(
                physics=self.mj_physics,
                site_name=self.end_effector_site_name,
                target_pos=pos,
                target_quat=quat,
                joint_names=self.joint_names,
                tol=1e-7,
                max_steps=100,
                inplace=inplace,
            )
        if not result.success:
            return None
        return result.qpos[self.joint_qpos_indices].copy()
