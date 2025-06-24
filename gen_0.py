"""Augment ManiSkill trajectories by randomizing target object poses

This script takes a trajectory file and augments it by:
1. Splitting the trajectory into segments based on keypoints
2. For each segment, generating new trajectories by:
   - Randomizing the target object pose
   - Computing new end poses based on the delta pose
   - Using motion planning to generate new trajectories
"""

import copy
import os
from dataclasses import dataclass
from typing import Annotated, Optional, List, Dict

import gymnasium as gym
import h5py
import numpy as np
import sapien
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to augment"""
    num_augmentations: int = 10
    """Number of augmented trajectories to generate for each segment"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'"""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory"""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to"""
    save_video: bool = False
    """Whether to save videos"""
    vis: bool = False
    """Whether to visualize the trajectory augmentation via the GUI"""
    output_dir: str = "augmented_trajectories"
    """Directory to save augmented trajectories"""
    shader: Optional[str] = None
    """Path to the shader directory"""
    reward_mode: str = "none"
    """Reward mode for the environment"""
    render_mode: str = "rgb_array"
    """Render mode for the environment"""
    video_fps: Optional[int] = None
    """Frames per second for video recording"""
    use_env_states: bool = False
    """Whether to use environment states"""
    delta_range: float = 0.1
    """Range for randomizing target object pose"""
    max_attempts: int = 10
    """Maximum number of attempts for motion planning"""
    max_time: float = 5.0
    """Maximum time for motion planning"""


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def load_trajectory(traj_path: str) -> tuple:
    """Load trajectory data from h5 file and json file"""
    h5_file = h5py.File(traj_path, "r")
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    return h5_file, json_data


def load_metadata(metadata_path: str) -> dict:
    """Load metadata containing keypoint information"""
    return io_utils.load_json(metadata_path)


def get_trajectory_segments(traj_data: dict, keypoints: List[int]) -> List[Dict]:
    """Split trajectory into segments based on keypoints"""
    segments = []
    for i in range(len(keypoints) - 1):
        start_idx = keypoints[i]
        end_idx = keypoints[i + 1]
        segment = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "actions": traj_data["actions"][start_idx:end_idx],
            "env_states": traj_data["env_states"][start_idx:end_idx],
        }
        segments.append(segment)
    return segments


def randomize_delta_pose(env, delta_range: float = 0.1) -> sapien.Pose:
    """Randomize target object pose within a range"""
    # Generate random delta pose
    delta_pos = np.random.uniform(-delta_range, delta_range, 3)
    delta_pos[2] = 0    # TODO
    delta_rot_z = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
    delta_rot = sapien.utils.euler2quat([0, 0, delta_rot_z])
    
    # Create delta pose
    delta_pose = sapien.Pose(p=delta_pos, q=delta_rot)
    
    return delta_pose


def augment_segment(env, segment: Dict, num_augmentations: int, args: Args) -> List[Dict]:
    """Generate augmented trajectories for a segment"""
    augmented_segments = []
    
    # Create motion planner
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=args.vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=args.vis,
        print_env_info=False,
    )
    
    for i in range(num_augmentations):
        # Randomize target pose
        new_target_pose, delta_pose = randomize_target_pose(env)
        env.target.set_pose(new_target_pose)
        
        # Get start and end poses from original segment
        start_state = segment["env_states"][0]
        end_state = segment["env_states"][-1]
        
        # Compute new end pose based on delta pose
        new_end_pose = end_state["target_pose"] * delta_pose
        
        # Plan motion to new end pose
        result = planner.move_to_pose_with_screw(new_end_pose)
        if result == -1:
            logger.warning(f"Failed to plan motion for augmentation {i}")
            continue
            
        # Record new trajectory
        new_segment = {
            "start_idx": segment["start_idx"],
            "end_idx": segment["end_idx"],
            "actions": result["position"],
            "env_states": [env.get_state_dict() for _ in range(len(result["position"]))],
            "delta_pose": delta_pose,
        }
        augmented_segments.append(new_segment)
    
    return augmented_segments


def save_augmented_trajectories(
    augmented_segments: List[Dict],
    original_traj_path: str,
    output_dir: str,
    args: Args
):
    """Save augmented trajectories to h5 and json files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create new h5 file
    output_h5_path = os.path.join(output_dir, os.path.basename(original_traj_path))
    h5_file = h5py.File(output_h5_path, "w")
    
    # Create new json data
    json_data = {
        "env_info": {
            "env_id": args.env_id,
            "env_kwargs": {
                "obs_mode": args.obs_mode,
                "control_mode": args.target_control_mode,
                "sim_backend": args.sim_backend,
            }
        },
        "episodes": []
    }
    
    # Save each augmented segment
    for i, segment in enumerate(augmented_segments):
        traj_id = f"traj_{i}"
        h5_file.create_dataset(f"{traj_id}/actions", data=segment["actions"])
        h5_file.create_dataset(f"{traj_id}/env_states", data=segment["env_states"])
        
        episode_info = {
            "episode_id": i,
            "start_idx": segment["start_idx"],
            "end_idx": segment["end_idx"],
            "delta_pose": {
                "position": segment["delta_pose"].p,
                "quaternion": segment["delta_pose"].q,
            }
        }
        json_data["episodes"].append(episode_info)
    
    # Save json file
    output_json_path = output_h5_path.replace(".h5", ".json")
    io_utils.dump_json(output_json_path, json_data)
    
    h5_file.close()


def main(args: Args):
    # Load trajectory metadata json
    json_path = args.traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
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
    env = wrappers.RecordEpisode(
        env,
        output_dir=os.path.dirname(args.traj_path),
        trajectory_name="augmented_trajectory",
        save_trajectory=True,
        save_video=False,
    )

    # 加载轨迹数据
    with h5py.File(args.traj_path, "r") as f:
        trajectories = f
        episodes = json_data["episodes"]
        
        # 处理每个episode
        for episode in episodes:
            episode_id = episode["episode_id"]
            traj_id = f"traj_{episode_id}"
            if traj_id not in trajectories:
                logger.warning(f"{traj_id} does not exist in {args.traj_path}")
                continue

            # 获取轨迹数据
            traj = trajectories[traj_id]
            actions = traj["actions"][:]
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            
            # 获取关键点
            keypoints = metadata["completed_trajectories"][episode_id]["steps"]
            
            # 分段处理轨迹
            segments = []
            for i in range(len(keypoints) - 1):
                start_idx = keypoints[i]
                end_idx = keypoints[i + 1]
                segment = {
                    "actions": actions[start_idx:end_idx],
                    "env_states": env_states[start_idx:end_idx],
                    "start_state": env_states[start_idx],
                    "end_state": env_states[end_idx-1]
                }
                segments.append(segment)
            
            # 对每个段进行扩充
            for segment in segments:
                # 重置环境到段的起始状态
                env.reset(**episode["reset_kwargs"])
                env.base_env.set_state_dict(segment["start_state"])
                
                # 随机化目标物体位姿
                delta_pose = randomize_delta_pose(env, args.delta_range)
                
                # 使用运动规划生成新的轨迹
                for _ in range(args.num_augmentations):
                    # 重置环境
                    env.reset(**episode["reset_kwargs"])
                    env.base_env.set_state_dict(segment["start_state"])
                    
                    # 设置新的目标位姿
                    env.target.pose = env.target.pose * delta_pose
                    
                    # 使用运动规划生成新的轨迹
                    success = generate_new_trajectory(env, segment["end_state"], args)
                    
                    if success:
                        # 保存新生成的轨迹
                        env.flush_trajectory()
                    else:
                        logger.warning(f"Failed to generate new trajectory for segment in {traj_id}")

    env.close()
    print(f"Successfully augmented trajectory {args.traj_path}")


def generate_new_trajectory(env, target_state, args):
    """使用运动规划生成新的轨迹"""
    # 创建运动规划器
    planner = PandaArmMotionPlanningSolver(env)
    
    # 获取目标状态中的机器人状态
    target_robot_state = target_state["robot"]
    
    # 使用运动规划移动到目标位置
    success = planner.move_to_pose_with_RRTConnect(
        target_robot_state["qpos"],
        max_attempts=args.max_attempts,
        max_time=args.max_time
    )
    
    return success


if __name__ == "__main__":
    main(parse_args()) 