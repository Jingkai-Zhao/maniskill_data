import sapien
import torch
import numpy as np
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent

@register_agent()
class FreeTCP(BaseAgent):
    uid = "free_tcp"

    def __init__(self, scene, init_pose=None, **kwargs):
        super().__init__(scene, **kwargs)
        # 创建一个可视化的末端夹爪（比如小球）
        builder = scene.create_actor_builder()
        builder.add_sphere_collision(0.02)
        builder.add_sphere_visual(0.02, color=[0, 0, 1, 1])
        self.tcp = builder.build_kinematic(name="free_tcp")
        if init_pose is None:
            init_pose = sapien.Pose([0, 0, 0.1])
        self.tcp.set_pose(init_pose)
        self._pose = self.tcp.pose

    @property
    def tcp_pose(self):
        return self.tcp.pose

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    def set_pose(self, pose):
        self.tcp.set_pose(pose)
        self._pose = pose

    def reset(self, pose=None):
        if pose is None:
            pose = sapien.Pose([0, 0, 0.1])
        self.set_pose(pose)

    def is_static(self, threshold=0.2):
        # 没有动力学，始终静止
        return True

    def is_grasping(self, obj):
        # 没有夹爪功能，始终为False
        return False

    def get_qvel(self):
        # 没有关节速度
        return torch.zeros(1)

    @property
    def robot(self):
        # 兼容环境代码
        return self