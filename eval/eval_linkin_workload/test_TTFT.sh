#!/bin/bash

# Define the directory containing the JSON datasets
DATA_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_json"
# Define the directory to save the benchmark results
RESULT_DIR="/root/Eamin/vllm_prefill/eval/eval_linkin_workload/link_result_json"

# Loop through each matching JSON file in the dataset directory
for json_file in "$DATA_DIR"/test_data_*.json; do
    # Extract the base filename (e.g., test_data_2.json)
    json_filename=$(basename "$json_file")

    # Extract the numeric part from the filename (e.g., 2 from test_data_2.json)
    num=$(echo "$json_filename" | grep -oP '(?<=test_data_)\d+(?=\.json)')

    # If number extraction fails, skip this file
    if [[ -z "$num" ]]; then
        echo "Skipping $json_filename (failed to extract number)"
        continue
    fi

    # Compute num-prompts as num * 50
    num_prompts=$((num * 50))

    # Generate the result filename
    result_filename="result_${json_filename}"

    # Print the info for debugging
    echo "Running benchmark for $json_filename"
    echo "num-prompts: $num_prompts"
    echo "Result will be saved as $result_filename"

    # Run the benchmark_serving.py with specified arguments
    python benchmark_serving.py \
        --backend vllm \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name sharegpt \
        --dataset-path "$json_file" \
        --request-rate 5 \
        --num-prompts "$num_prompts" \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$result_filename"
done