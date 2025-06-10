import os
import json
import h5py
import numpy as np
from glob import glob
from pathlib import Path
import copy

def merge_trajectory_files(input_dir, output_file_prefix="merged_trajectory"):
    """
    合并多个轨迹文件（.h5 和 .json）为一个文件
    
    Args:
        input_dir: 包含轨迹文件的目录
        output_file_prefix: 输出文件的前缀
    """
    # 查找所有轨迹文件
    h5_files = sorted(glob(os.path.join(input_dir, "trajectory_*.h5")))
    json_files = sorted(glob(os.path.join(input_dir, "trajectory_*.json")))
    
    if not h5_files or not json_files:
        raise ValueError(f"在 {input_dir} 中未找到轨迹文件")
    
    # 确保 .h5 和 .json 文件数量匹配
    assert len(h5_files) == len(json_files), "H5 文件和 JSON 文件数量不匹配"
    
    # 创建输出文件
    output_h5_path = os.path.join(input_dir, f"{output_file_prefix}.h5")
    output_json_path = os.path.join(input_dir, f"{output_file_prefix}.json")
    
    with h5py.File(output_h5_path, 'w') as output_h5:
        # 合并 JSON 元数据
        merged_json = {}
        all_episodes = []
        next_episode_id = 0
        
        for i, (h5_path, json_path) in enumerate(zip(h5_files, json_files)):
            print(f"处理文件 {i+1}/{len(h5_files)}: {h5_path}")
            
            # 加载 JSON 文件
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # 初始化合并后的 JSON 数据结构
            if i == 0:
                merged_json = copy.deepcopy(json_data)
                # 移除原始的 episodes 列表
                merged_json.pop("episodes", None)
            else:
                # 确保环境信息一致
                assert json_data["env_info"] == merged_json["env_info"], "环境配置不一致"
            
            # 处理每个轨迹
            with h5py.File(h5_path, 'r') as h5_file:
                for traj_key in sorted(h5_file.keys()):
                    # 获取原始轨迹 ID
                    original_ep_id = int(traj_key.split('_')[1])
                    
                    # 为合并后的文件创建新的轨迹 ID
                    new_ep_id = next_episode_id
                    new_traj_key = f"traj_{new_ep_id}"
                    
                    # 复制轨迹数据到新文件
                    h5_file.copy(traj_key, output_h5, new_traj_key)
                    
                    # 更新 JSON 中的 episode 信息
                    for ep in json_data["episodes"]:
                        if ep["episode_id"] == original_ep_id:
                            # 创建新的 episode 记录，更新 ID
                            new_ep = ep.copy()
                            new_ep["episode_id"] = new_ep_id
                            all_episodes.append(new_ep)
                            break
                    
                    # 更新下一个轨迹 ID
                    next_episode_id += 1
        
        # 添加合并后的 episodes 到 JSON 数据
        merged_json["episodes"] = all_episodes
        
        # 保存合并后的 JSON 文件
        with open(output_json_path, 'w') as f:
            json.dump(merged_json, f, indent=2)
    
    print(f"合并完成！输出文件: {output_h5_path} 和 {output_json_path}")
    print(f"共合并 {len(all_episodes)} 个轨迹")
    
    # 可选：清理原始文件
    # for h5_path, json_path in zip(h5_files, json_files):
    #     os.remove(h5_path)
    #     os.remove(json_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="合并多个轨迹文件")
    parser.add_argument("--input-dir", required=True, help="包含轨迹文件的目录")
    parser.add_argument("--output-prefix", default="merged_trajectory", help="输出文件的前缀")
    args = parser.parse_args()
    
    merge_trajectory_files(args.input_dir, args.output_prefix)