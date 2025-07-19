import pandas as pd

# 读取CSV文件
df = pd.read_csv("operation_data.csv", encoding='utf-8-sig')

# 假设所有行的“指令”都相同
instruction = df.iloc[0]['指令']

# 构建嵌套字典
operation_data = {
    "instruction": instruction,
    "subtasks": []
}

# 按子任务分组
for task, group in df.groupby('子任务'):
    subtask = {
        "task": task,
        "atomic_actions": []
    }
    for _, row in group.iterrows():
        atomic_action = {
            "action": row['原子动作'],
            "target_object": row['目标物体'],
            # 如果你有目标位姿、初始位姿、结束位姿，可以在这里加上
            # "target_pose": row['目标位姿'],
            # "init_pose": row['初始位姿'],
            # "end_pose": row['结束位姿'],
        }
        subtask["atomic_actions"].append(atomic_action)
    operation_data["subtasks"].append(subtask)

print("还原后的字典：")
print(operation_data)