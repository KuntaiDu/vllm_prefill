#!/bin/bash

# 手动指定你要用的数据编号
data_num=30

# Define the directory containing the JSON datasets
DATA_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_json"
# 直接拼接 json_file
json_file="$DATA_DIR/test_data_${data_num}.json"

# 手动指定结果目录
RESULT_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/qps/${data_num}"
mkdir -p "$RESULT_DIR"

# 固定 num_prompts
num_prompts=$((data_num * 50))

# QPS 从 1 到 100 循环
for qps in $(seq 0.1 0.1 2.0 ); do
    # 生成结果文件名
    result_filename="result_qps_${qps}_data_${data_num}.json"

    echo "Running benchmark with QPS=$qps on data_$data_num"
    echo "Result will be saved as $result_filename"

    python benchmark_serving_linkin.py \
        --backend vllm \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name sharegpt \
        --dataset-path "$json_file" \
        --request-rate "$qps" \
        --num-prompts "$num_prompts" \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$result_filename"
done
