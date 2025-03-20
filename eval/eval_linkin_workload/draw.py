import os
import re
import json
import matplotlib.pyplot as plt

# 配置文件夹路径
folder = '/root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_result_json'  # 这里改成你的文件夹路径
pattern = re.compile(r'result_test_data_(\d+)\.json')

# 你要提取和绘制的变量
target_keys = ['mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms']
data = {key: [] for key in target_keys}
x_axis = []

# 遍历文件夹，按数字排序
file_info = []
for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        file_num = int(match.group(1))
        file_info.append((file_num, filename))
file_info.sort()

# 读取JSON数据
for file_num, filename in file_info:
    with open(os.path.join(folder, filename), 'r') as f:
        json_data = json.load(f)
    x_axis.append(file_num)
    for key in target_keys:
        # 如果某个key可能不存在，安全取值
        data[key].append(json_data.get(key, None))

# 开始画图
plt.figure(figsize=(10, 6))
for key in target_keys:
    plt.plot(x_axis, data[key], marker='o', label=key)

plt.xlabel('Test Data Index (From Filename)', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)  # 加了单位 ms
plt.title('TTFT Metrics Across Test Data', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 存储图片而不是显示
output_file = 'output.png'
plt.savefig(output_file)
print(f"图已保存为 {output_file}")