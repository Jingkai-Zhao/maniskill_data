"""Augment ManiSkill trajectories for StackCube-v1 task

This script takes a trajectory file and augments it by:
1. Loading the original trajectory
2. Randomizing the target cube pose following the same rules as StackCube-v1
3. Using motion planning to generate new trajectories
"""

import os
import json
import h5py
import numpy as np
import gymnasium as gym
import sapien
from tqdm import tqdm
import argparse
import time
from mani_skill.examples.motionplanning.panda.solutions import solveStackCube
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import io_utils
from mani_skill.envs.utils import randomization

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", type=str, required=True, help="输入轨迹文件的路径")
    parser.add_argument("--output-dir", type=str, default="augmented_demos", help="输出轨迹的保存目录")
    parser.add_argument("--num-augment", type=int, default=5, help="每个原始轨迹要扩充的数量")
    parser.add_argument("--xy-range", type=float, default=0.1, help="XY平面上的随机化范围")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--save-video", action="store_true", help="是否保存视频")
    parser.add_argument("--sim-backend", type=str, default="physx_cpu", help="仿真后端")
    parser.add_argument("--vis", action="store_true", help="是否可视化运动规划过程")
    return parser.parse_args()

def load_trajectory(traj_path):
    """加载轨迹文件"""
    with h5py.File(traj_path, 'r') as f:
        env_states = f['env_states'][:]
        actions = f['actions'][:]
        rewards = f['rewards'][:]
        dones = f['dones'][:]
        infos = f['infos'][:]
    return env_states, actions, rewards, dones, infos

def randomize_cube_pose(env, xy_range=0.1):
    """按照StackCube-v1环境的规则随机化立方体位姿"""
    # 获取当前立方体位姿
    current_pose = env.target.pose
    
    # 生成新的XY位置
    xy = np.random.uniform(-xy_range, xy_range, 2)
    
    # 保持Z轴高度不变
    new_pos = np.array([xy[0], xy[1], current_pose.p[2]])
    
    # 仅绕Z轴随机旋转
    qs = randomization.random_quaternions(
        1,
        lock_x=True,
        lock_y=True,
        lock_z=False,
    )[0]
    
    return sapien.Pose(p=new_pos, q=qs)

def augment_trajectory(args):
    # 从轨迹文件中读取环境配置
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    episodes = json_data["episodes"]
    metadata = io_utils.load_json(os.path.join(os.path.dirname(args.traj_path), "metadata.json"))

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = env_info["env_kwargs"].copy()

    # 设置sim_backend
    if args.sim_backend is None:
        if "sim_backend" not in env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = env_kwargs["sim_backend"]
    env_kwargs["sim_backend"] = args.sim_backend

    # 创建环境
    env = gym.make(env_id, **env_kwargs)
    env = RecordEpisode(
        env,
        output_dir=os.path.dirname(args.traj_path),
        trajectory_name="augmented_trajectory",
        save_trajectory=True,
        save_video=False,
    )
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 对每个原始轨迹进行扩充
    total_augmentations = len(episodes) * args.num_augment
    pbar = tqdm(total=total_augmentations, desc="Generating augmented trajectories")
    
    for episode_idx, episode in enumerate(episodes):
        # 获取当前episode的轨迹数据
        traj_id = f"traj_{episode['episode_id']}"
        with h5py.File(args.input_trajectory, 'r') as f:
            traj_data = f[traj_id]
            env_states = traj_data['env_states'][:]
            actions = traj_data['actions'][:]
        
        # 对每个原始轨迹生成多个扩充版本
        for aug_idx in range(args.num_augment):
            # 重置环境
            env.reset()
            
            # 设置初始状态
            initial_state = env_states[0].copy()
            env.set_state(initial_state)
            
            # 随机化目标立方体位姿
            new_target_pose = randomize_cube_pose(env, args.xy_range)
            env.target.set_pose(new_target_pose)
            
            # 创建新的轨迹记录器
            new_traj_name = f"augmented_ep{episode_idx}_aug{aug_idx}_{int(time.time())}"
            record_env = RecordEpisode(
                env,
                output_dir=os.path.join(args.output_dir, "StackCube-v1", "motionplanning"),
                trajectory_name=new_traj_name,
                save_video=args.save_video,
                source_type="motionplanning",
                source_desc="augmented trajectory",
                video_fps=30
            )
            
            # 使用motion planning生成新轨迹
            try:
                res = solveStackCube(record_env, seed=np.random.randint(0, 10000), debug=False, vis=args.vis)
                if res != -1 and res[-1]["success"].item():
                    print(f"Successfully generated augmented trajectory {aug_idx+1}/{args.num_augment} for episode {episode_idx+1}/{len(episodes)}")
                else:
                    print(f"Failed to generate augmented trajectory {aug_idx+1}/{args.num_augment} for episode {episode_idx+1}/{len(episodes)}")
            except Exception as e:
                print(f"Error generating augmented trajectory {aug_idx+1}/{args.num_augment} for episode {episode_idx+1}/{len(episodes)}: {e}")
            
            record_env.flush_trajectory()
            if args.save_video:
                record_env.flush_video()
            
            pbar.update(1)
    
    pbar.close()
    env.close()

if __name__ == "__main__":
    args = parse_args()
    augment_trajectory(args) 