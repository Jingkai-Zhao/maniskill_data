#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
import requests
import json
import ast
import pandas as pd
fixed_prompt = """现在 你是一个机械臂专家 你的任务是生成一组机械臂操作的数据 具体而言：你需要先提出一个指令，如"将手机放在抽屉里" 并且将指令分解为一系列基础任务 如"开抽屉"->"放手机"->"关抽屉" 每个基础任务的形式为动词+名词 最终 每个基础任务还会被分解为原子任务 原子任务是最底层的动作 在"抓、放、插、倒、点击、滑、推、拉、扭、按、开、关"这个原子任务库中选择 每个原子任务的数据包括一个动作 也就是原子任务库的动作 以及这个原子动作的目标物体 以及一个目标点位姿 以及这个原子任务的初始位姿 和结束位姿  你只需要给出动作 并留好目标点位姿  初始位姿 以及结束位姿的变量位置来，最终，将这一组数据用python字典的形式输出出来，后续我会将字典中的变量导出到表格中。请严格按照我的要求与输出示例，防止文本提取有误，一个完整的输出示例如下：
operation_data = {
    "instruction": "Put the cup from the table into the microwave",
    "subtasks": [
        {
            "task": "grab the cup",
            "atomic_actions": [
                {
                    "action": "grab",
                    "target_object": "cup",
                    "target_pose": "cup_handle_pose",         # 杯把手的位姿
                    "init_pose": "cup_on_table_pose",         # 杯子在桌上的位姿
                    "end_pose": "cup_grasped_pose"           # 杯子被夹爪抓住的位姿
                }
            ]
        },
        {
            "task": "open the microwave door",
            "atomic_actions": [
                {
                    "action": "pull",
                    "target_object": "microwave",
                    "target_pose": "microwave_handle_pose",    # 微波炉门把手的位姿
                    "init_pose": "microwave_closed_pose",      # 微波炉门关闭时的位姿
                    "end_pose": "microwave_open_pose"         # 微波炉门打开时的位姿
                }
            ]
        },
        {
            "task": "put the cup",
            "atomic_actions": [
                {
                    "action": "put",
                    "target_object": "cup",
                    "target_pose": "microwave_inner_pose",     # 微波炉内部放置点的位姿
                    "init_pose": "cup_grasped_pose",           # 杯子被抓住的位姿
                    "end_pose": "cup_in_microwave_pose"       # 杯子在微波炉内的位姿
                }
            ]
        },
        {
            "task": "close the microwave door",
            "atomic_actions": [
                {
                    "action": "push",
                    "target_object": "microwave",
                    "target_pose": "microwave_handle_pose",    # 微波炉门把手的位姿
                    "init_pose": "microwave_open_pose",        # 微波炉门打开时的位姿
                    "end_pose": "microwave_closed_pose"       # 微波炉门关闭时的位姿
                }
            ]
        }
    ]
}请仿照这个示例 并按照我的要求 再输出一个python的字典示例"""

messages = [
    {"role": "system", "content": fixed_prompt},
    {"role": "user", "content": "请给我一个示例"}
]

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions", 
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    },
    data=json.dumps({
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": messages
    })
)
data = response.json()
response_text = data['choices'][0]['message']['content']

# 提取字典部分的文本
dict_text = response_text.split("operation_data = ")[-1].strip()
if dict_text.endswith("```"):
    dict_text = dict_text[:-3]

# 将文本转换为Python字典
operation_data = ast.literal_eval(dict_text)
print("\n转换后的Python字典:", operation_data)
print("\n")

# 创建一个空列表来存储数据
table_data = []

# 遍历字典提取数据
instruction = operation_data["instruction"]
for subtask in operation_data["subtasks"]:
    task = subtask["task"]
    for atomic_action in subtask["atomic_actions"]:
        table_data.append({
            "指令": instruction,
            "子任务": task,
            "原子动作": atomic_action["action"],
            "目标物体": atomic_action["target_object"],
            # "目标位姿": atomic_action["target_pose"],
            # "初始位姿": atomic_action["init_pose"],
            # "结束位姿": atomic_action["end_pose"]
        })

# 创建DataFrame并保存为CSV文件
df = pd.DataFrame(table_data)
df.to_csv("operation_data.csv", index=False, encoding='utf-8-sig')