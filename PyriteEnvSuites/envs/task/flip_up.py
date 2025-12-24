import os

import numpy as np
from dm_control import mjcf
from scipy.spatial.transform import Rotation as R

# import defaultdict
from PyriteEnvSuites.envs.mujoco.mujocoEnv import TableTopUR5BigProbeEnv
from PyriteEnvSuites.envs.utlity.robot_utlity import (
    quat2euler,
)

# Get package root path (PyriteEnvSuites/) for loading assets
PACKAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


class FlipUp(TableTopUR5BigProbeEnv):
    def __init__(
        self,
        SE3_W_bookend,
        SE3_W_book,
        SE3_W_keyframe=[],
        **kwargs,
    ) -> None:
        self.SE3_W_book = SE3_W_book
        self.SE3_W_bookend = SE3_W_bookend
        self.SE3_W_keyframe = SE3_W_keyframe
        super().__init__(**kwargs)
        self.task_description = "Flip up flat item from bin floor"

    def update_task_description(self, task_description):
        self.task_description = task_description

    def setup_model(self):
        world_model = super().setup_model()
        return world_model

    def setup_objs(self, world_model):
        path_to_model = PACKAGE_PATH + "assets/custom"

        bookend2_blender = "bookend2_blender"
        object_model = mjcf.from_path(
            os.path.join(path_to_model, bookend2_blender, "bookend2_blender.xml")
        )
        object_model.model = "bookend2_blender"
        self.add_obj_from_model(
            world_model,
            object_model,
            self.SE3_W_bookend.t,
            quat=R.from_matrix(self.SE3_W_bookend.R).as_quat()[[3, 0, 1, 2]],
            add_freejoint=False,
        )

        book2_blend = "book2_blend"
        object_model = mjcf.from_path(
            os.path.join(path_to_model, book2_blend, "book2_blend.xml")
        )
        object_model.model = "book2_blend"
        self.add_obj_from_model(
            world_model,
            object_model,
            self.SE3_W_book.t,
            quat=R.from_matrix(self.SE3_W_book.R).as_quat()[[3, 0, 1, 2]],
            add_freejoint=True,
        )

        # need to enable group 4 to see it
        frame = "frame"
        object_model = mjcf.from_path(os.path.join(path_to_model, frame, "frame.xml"))
        object_model.model = "frame"
        self.add_obj_from_model(
            world_model,
            object_model,
            np.array([0.5, 1, 0.5]),
            quat=np.array([1, 0, 0, 0]),
            add_freejoint=False,
        )

        # Nk = len(self.SE3_W_keyframe)
        # print(f'    [flip_up_env] adding keyframe. Total: {Nk}')
        # if  Nk > 0:
        #     for iframe in range(Nk):
        #         keyframe_model = "ball_r_1cm"
        #         # keyframe_model = "frame"
        #         object_model = mjcf.from_path(os.path.join(path_to_model, keyframe_model, f"{keyframe_model}.xml"))
        #         object_model.model = keyframe_model
        #         self.add_obj_from_model(
        #             world_model,
        #             object_model,
        #             self.SE3_W_keyframe[iframe].t,
        #             quat=R.from_matrix(self.SE3_W_keyframe[iframe].R).as_quat()[[3, 0, 1, 2]],
        #             add_freejoint=False,
        #             name_prefix=f'keyframe_{iframe}'
        #         )

        self.selected_object = "book2_blend"

    def add_obj_from_model(
        self,
        world_model,
        obj_model,
        position,
        quat=None,
        add_freejoint=False,
        name_prefix=None,
    ):
        name = obj_model.model
        if name_prefix is not None:
            name = name_prefix
        object_site = world_model.worldbody.add(
            "site",
            name=f"{name}_site",
            pos=position,
            quat=quat,
            group=3,
        )
        if add_freejoint:
            object_site.attach(obj_model).add("freejoint")
        else:
            object_site.attach(obj_model)

    def get_obj_pose_observation(self, obj_link_states=None, obj_link_contacts=None):
        if obj_link_contacts is None or obj_link_contacts is None:
            _, obj_link_states, obj_link_contacts = self.get_state()
        return np.concatenate(
            [
                self.get_object_qpos(self.selected_object, obj_link_states),
                self.get_object_quat(self.selected_object, obj_link_states),
            ]
        ).astype(np.float32)

    # @profile
    def get_object_qpos(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_qpos = []
        for k, v in object_link_state.items():
            object_links_qpos.append(v.pose.position)
        return object_links_qpos[-1].astype(np.float32)

    def get_object_quat(self, object_name, obj_link_states):
        """w x y z"""
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return object_links_quat[-1].astype(np.float32)

    def get_object_euler(self, object_name, obj_link_states):
        object_link_state = obj_link_states[f"{object_name}/"]
        object_links_quat = []
        for k, v in object_link_state.items():
            object_links_quat.append(v.pose.orientation)
        return quat2euler(object_links_quat[-1].astype(np.float32))
