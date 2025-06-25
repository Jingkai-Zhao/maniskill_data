from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.ground import build_ground
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("Collect-v1", max_episode_steps=50)
class CollectEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
    capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100]
    cube_half_size = 0.02
    goal_thresh = 0.025
    cube_spawn_half_size = 0.1
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[-0.3, 0.5, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[1, 0, 0]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        # temporarily turn off the logging as there will be big red warnings
        # about the cabinets having oblong meshes which we ignore for now.
        sapien.set_log_level("off")
        self._load_assets()
        sapien.set_log_level("warn")
        
        ASSET_COLLISION_BIT = 29

        self.ground.set_collision_group_bit(
            group=2, bit_idx=ASSET_COLLISION_BIT, bit=1
        )
        
    def _load_assets(self):
        self.assets = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="assets",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
    # def _after_reconfigure(self, options):
    #     self.assets_zs = []
    #     for asset in self._assets:
    #         collision_mesh = asset.get_first_collision_mesh()
    #         self.assets_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
    #     self.assets_zs = common.to_tensor(self.assets_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            # xy[:, 2] = self.assets_zs[env_idx]
            xy[:, 2] = 0.02
            self.assets.set_pose(Pose.create_from_pq(p=xy))
            
            if self.robot_uids == "panda":
                qpos = np.array(
                    [
                        0.0,
                        np.pi / 8,
                        0,
                        -np.pi * 5 / 8,
                        0,
                        np.pi * 3 / 4,
                        np.pi / 4,
                        0.04,
                        0.04,
                    ]
                )
                if self._enhanced_determinism:
                    qpos = (
                        self._batched_episode_rng[env_idx].normal(
                            0, self.robot_init_qpos_noise, len(qpos)
                        )
                        + qpos
                    )
                else:
                    qpos = (
                        self._episode_rng.normal(
                            0, self.robot_init_qpos_noise, (b, len(qpos))
                        )
                        + qpos
                    )
                qpos[:, -2:] = 0.04
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose([-0.5, 0, 0]))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        # if "state" in self.obs_mode:
        #     obs.update(
        #         obj_pose=self.assets.pose.raw_pose,
        #         tcp_to_obj_pos=self.assets.pose.p - self.agent.tcp_pose.p,
        #         obj_to_goal_pos=self.goal_site.pose.p - self.assets.pose.p,
        #     )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.assets)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_robot_static,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = 1

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5


# @register_env("PickCubeSO100-v1", max_episode_steps=50)
# class PickCubeSO100Env(PickCubeEnv):

#     _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, robot_uids="so100", **kwargs)
