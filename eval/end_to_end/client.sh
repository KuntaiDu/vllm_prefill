#!/bin/bash

# Get the name of the method we are running at the server side.
# currently, it can be "vanilla", "chunked", "tp", "pp", or "prefill"
method_name=$(curl -s http://localhost:5000/name)

if [ -z "$method_name" ]; then
    echo "Method name is empty, please wait for the server to start"
    exit 1
fi

echo "vLLM server is running $method_name"

# Tune the hyper-parameters here to change the workload.
num_users=6
num_documents=20
user_history_mean=40000
user_history_std=1
user_history_min=10000
user_history_max=55000
document_mean=300
document_std=150
document_min=10
document_max=500


serialize_to_filename() {
  local parts=()
  for var in "$@"; do
    local val="${!var}"
    val_escaped=$(printf '%q' "$val")         # safely escape
    parts+=("${var}__${val_escaped}")
  done
  filename="${parts[0]}"
  for part in "${parts[@]:1}"; do
    filename+="__${part}"
  done
  echo "$filename"
  sleep 5
}


results_dir=$(realpath "results/$method_name")

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
    --document-mean $document_mean \
    --document-std $document_std \
    --document-min $document_min \
    --document-max $document_max \
    --num-documents $num_documents

echo "Benchmarking vLLM"

for qps in "inf"; do

    # usage:
    # put all variables you used sequenatially, and it will be dumped into the filename.
    filename=$(serialize_to_filename qps num_users num_documents user_history_mean user_history_std user_history_min user_history_max document_mean document_std document_min document_max)

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
        --save-result \
        --result-dir $results_dir \
        --result-filename $filename.json

    if [ $? -ne 0 ]; then
        echo "Failed to run $filename, skip creating the done file"
        continue
    fi

    touch $results_dir/$filename.done

done