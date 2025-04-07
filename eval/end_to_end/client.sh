#!/bin/bash

# This is the name for the experiment. Different baselines need different experiment names.
experiment_name="LMPrefill"

# Tune the hyper-parameters here to change the workload.
num_users=40
num_documents=5
user_history_mean=15000
user_history_std=3000
user_history_min=10
user_history_max=33000
document_length=150


serialize_to_filename() {
  local prefix="$1"; shift
  local parts=()
  for var in "$@"; do
    local val="${!var}"
    val_escaped=$(printf '%q' "$val")         # safely escape
    parts+=("${var}__${val_escaped}")
  done
  local filename="${prefix}__$(IFS='__'; echo "${parts[*]}")"
  echo "$filename"
}


results_dir=$(realpath "results/$experiment_name")

echo "Results directory: $results_dir"

mkdir -p $results_dir

echo "Generating dataset..."
rm linkedin_datsaet_simulated.json
python generate_dataset_linkedin.py \
    --output-json linkedin_datsaet_simulated.json \
    --user-history-mean $user_history_mean \
    --user-history-std $user_history_std \
    --user-history-min $user_history_min \
    --user-history-max $user_history_max \
    --num-users $num_users \
    --document-length $document_length \
    --num-documents $num_documents

echo "Benchmarking vLLM"

for qps in 0.5 1 2 4; do

    # usage:
    # the first variable is the experiment name, give it whatever the name 
    # you like
    # then, put all variables you used sequenatially after that, so that you 
    # can resume these variables just from filename
    filename=$(serialize_to_filename "round1" qps num_users num_documents user_history_mean user_history_std user_history_min user_history_max document_length)

    if [ -f results/$filename.done ]; then
        echo "Skipping $filename because it already exists"
        continue
    fi

    echo "Running $filename"

    python benchmark_serving.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset-name sharegpt \
        --dataset-path linkedin_datsaet_simulated.json \
        --request-rate $qps \
        --sharegpt-output-len 1 \
        --backend vllm \
        --result-dir $results_dir \
        --result-filename $filename.json

    if [ $? -ne 0 ]; then
        echo "Failed to run $filename, skip creating the done file"
        continue
    fi

    touch $results_dir/$filename.done

done