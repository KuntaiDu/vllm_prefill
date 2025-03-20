import os
import re
import json
import matplotlib.pyplot as plt

# 配置文件夹路径
n_users = 2  # 可变，方便后续扩展
folder = f'/root/Eamin/vllm_prefill/eval/eval_linkin_workload/qps/{n_users}'
pattern = re.compile(r'result_qps_(\d+)_data_\d+\.json')

# 目标指标
target_keys = ['mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms']
data = {key: [] for key in target_keys}
x_axis = []

# 收集 (qps, filename)
file_info = []
for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        qps = int(match.group(1))
        file_info.append((qps, filename))

# 按 QPS 排序
file_info.sort()

# 读取数据
for qps, filename in file_info:
    with open(os.path.join(folder, filename), 'r') as f:
        json_data = json.load(f)
    x_axis.append(qps)
    for key in target_keys:
        data[key].append(json_data.get(key, None))

# 绘图
plt.figure(figsize=(10, 6))
for key in target_keys:
    plt.plot(x_axis, data[key], marker='o', label=key)

plt.xlabel('QPS', fontsize=12)
plt.ylabel('Time (ms)', fontsize=12)
plt.title(f'TTFT vs QPS (n_users={n_users})', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 坐标从0开始
plt.xlim(left=0)
plt.ylim(bottom=0)

# 保存图像
output_file = f'ttft_vs_qps_n_users_{n_users}.png'
plt.savefig(output_file)
print(f"图已保存为 {output_file}")