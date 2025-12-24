import os

import gymnasium as gym

# from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type
from PyriteUtility.data_pipeline.episode_data_buffer import (
    EpisodeDataBuffer,
    EpisodeDataIncreImageBuffer,
    VideoData,
)
from PyriteUtility.data_pipeline.file import save_data_as_pickle


##
## Low dim data:
##      saved in buffers with save_low_dim_observation() and flushed out once
##      for all with flush().
## Image data:
##      If always_flush = False, they are saved in buffer and flushed the same way.
##      If always_flush = True, they are saved to harddrive immediately without buffer.
## save_video: save a video using image data.
##      If always_flush = False, the video is created in the end during flush().
##      If always_flush = True, the video is created at the begining, with new frames
##          inserted during save_visual_observation()
class RecordWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        episode_id,
        rgb_shape_nhwc,  # (n, h, w, c). n: estimated buffer size.
        save_video,
        save_video_fps,
        store_path,
        save_mj_physics=False,
        always_flush=False,
    ) -> None:
        super().__init__(env)
        self.save_video_fps = save_video_fps
        self.store_path = store_path
        self.always_flush = always_flush
        self.rgb_shape_nhwc = rgb_shape_nhwc

        if always_flush:
            self.recoder_buffer = EpisodeDataIncreImageBuffer(
                rgb_shape_nhwc=rgb_shape_nhwc,
                episode_id=episode_id,
                store_path=store_path,
                camera_ids=self.env.camera_ids,
                save_video=save_video,
                save_video_fps=save_video_fps,
            )
        else:
            self.recoder_buffer = EpisodeDataBuffer(
                episode_id=episode_id,
                store_path=store_path,
                camera_ids=self.env.camera_ids,
                save_video=save_video,
                save_video_fps=save_video_fps,
            )
        self.save_mj_physics = save_mj_physics

    def reset(self, **kwargs):
        self.clear_buffer()
        return self.env.reset(**kwargs)

    def clear_buffer(self):
        self.episode_video_buffer = {camera_id: [] for camera_id in self.env.camera_ids}
        self.episode_js_command_buffer = []
        self.episode_js_fb_buffer = []
        self.episode_ts_pose_command_buffer = []
        self.episode_ts_pose_fb_buffer = []
        self.episode_ft_sensor_pose_fb_buffer = []
        self.episode_low_dim_state_buffer = []
        self.episode_qpos = []
        self.episode_qvel = []
        self.episode_info = []
        self.episode_js_force = []
        self.episode_ft_wrench = []
        self.episode_ft_wrench_filtered = []
        self.episode_low_dim_time_stamp_buffer = []
        self.episode_visual_time_stamp_buffer = []

    def delete_all_files(self):
        self.recoder_buffer.delete_episode_data()

    def save_video(self):
        if self.always_flush:
            self.recoder_buffer.save_video_to_file()

    def add_visual_observation(self, time_stamp):
        # Switched from render_all to render_rgb_only
        # depth costs too much space
        step_visual_obervations = self.env.get_visual_observation()
        if self.always_flush:
            self.recoder_buffer.save_one_img_frame(
                visual_observation=step_visual_obervations, visual_time_stamp=time_stamp
            )
        else:
            for camera_id in self.env.camera_ids:
                self.episode_video_buffer[camera_id].append(
                    step_visual_obervations[camera_id]
                )
            self.episode_visual_time_stamp_buffer.append(time_stamp)

    def add_low_dim_observation(self, js_command, ts_pose_command, time_stamp, info=""):
        # save commands
        # self.episode_js_command_buffer.append(js_command)
        self.episode_ts_pose_command_buffer.append(ts_pose_command)

        # save feedback
        # self.episode_js_force.append(self.env.robot.get_actuator_force())
        self.episode_ft_wrench.append(self.env.get_ft_sensor_reading())
        self.episode_ft_wrench_filtered.append(
            self.env.get_ft_sensor_reading_filtered()
        )

        # self.episode_js_fb_buffer.append(self.env.get_js_fb())
        self.episode_ts_pose_fb_buffer.append(self.env.get_ts_pose_fb())
        self.episode_ft_sensor_pose_fb_buffer.append(self.env.get_ft_sensor_pose_fb())
        # self.episode_qpos.append(self.env.mj_physics.data.qpos.copy())
        # self.episode_qvel.append(self.env.mj_physics.data.qvel.copy())

        # self.episode_low_dim_state_buffer.append(self.env.get_obj_pose_observation())

        self.episode_info.append(info)
        self.episode_low_dim_time_stamp_buffer.append(time_stamp)

    def save_whole_episode(self):
        if not self.always_flush:
            flushed_visual_obervations_buffer = {}
            for camera_id in self.env.camera_ids:
                episode_video_data = VideoData.stack(
                    self.episode_video_buffer[camera_id]
                )
                flushed_visual_obervations_buffer[camera_id] = episode_video_data

            self.recoder_buffer.create_zarr_groups_for_episode(self.rgb_shape_nhwc)
            self.recoder_buffer.save_video_for_episode(
                visual_observations=flushed_visual_obervations_buffer,
                visual_time_stamps=self.episode_visual_time_stamp_buffer,
            )
            self.recoder_buffer.save_low_dim_for_episode(
                js_command=self.episode_js_command_buffer,
                js_fb=self.episode_js_fb_buffer,
                ts_pose_command=self.episode_ts_pose_command_buffer,
                ts_pose_fb=self.episode_ts_pose_fb_buffer,
                ft_sensor_pose_fb=self.episode_ft_sensor_pose_fb_buffer,
                low_dim_state=self.episode_low_dim_state_buffer,
                qpos=self.episode_qpos,
                qvel=self.episode_qvel,
                js_force=self.episode_js_force,
                wrench=self.episode_ft_wrench,
                wrench_filtered=self.episode_ft_wrench_filtered,
                low_dim_time_stamps=self.episode_low_dim_time_stamp_buffer,
                info=self.episode_info,
            )
        else:
            self.recoder_buffer.save_low_dim_for_episode(
                js_command=self.episode_js_command_buffer,
                js_fb=self.episode_js_fb_buffer,
                ts_pose_command=self.episode_ts_pose_command_buffer,
                ts_pose_fb=self.episode_ts_pose_fb_buffer,
                ft_sensor_pose_fb=self.episode_ft_sensor_pose_fb_buffer,
                low_dim_state=self.episode_low_dim_state_buffer,
                qpos=self.episode_qpos,
                qvel=self.episode_qvel,
                js_force=self.episode_js_force,
                wrench=self.episode_ft_wrench,
                wrench_filtered=self.episode_ft_wrench_filtered,
                low_dim_time_stamps=self.episode_low_dim_time_stamp_buffer,
                info=self.episode_info,
            )
            self.recoder_buffer.save_video_to_file()

        if self.save_mj_physics:
            save_data_as_pickle(
                self.env.mj_physics,
                os.path.join(
                    self.store_path,
                    f"eps_{self.recoder_buffer.curr_eps}_mj_physics.pkl",
                ),
            )
        self.clear_buffer()

    def plot_episode(self):
        self.recoder_buffer.plot_low_dim()
