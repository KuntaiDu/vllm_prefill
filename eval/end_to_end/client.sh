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

# Workload 1: (showcase shortest remaining job first scheduling)
#   assume request length is k in maximum
#   make sure that prefill-only can cache the KV cache of k tokens
#   make sure that vanilla vLLM cannot handle request length k
#   make sure that vLLM + chunked prefill can handle request length k
#   num_user is even
#   num_documents reasonalby large.
#   make sure that the randomness of request length is high.

#   baselines:
#     - vLLM + chunked prefill
#     - vLLM + tp 2
#     - vLLM + pp 2

# Workload 2: (show hybrid prefilling mainly)
#   assume request length is k in maximum
#   make sure that k is close to max length that vLLM + prefill can handle
#   make sure num_documents = 1
#   make sure that the randomness of request length is high (for us to be better than pp)

#   baselines:
#     - vLLM + tp 2
#     - vLLM + pp 2

num_users=4
num_documents=50
user_history_mean=40000
user_history_std=1
user_history_min=10000
user_history_max=50000
document_mean=150
document_std=1
document_min=10
document_max=500
intra_delay=0.05 # 50ms




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

for qps in inf; do

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
        --result-filename $filename.json \
        --intra-delay $intra_delay

    if [ $? -ne 0 ]; then
        echo "Failed to run $filename, skip creating the done file"
        continue
    fi

    touch $results_dir/$filename.done

done