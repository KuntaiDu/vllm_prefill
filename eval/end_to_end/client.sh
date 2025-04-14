#!/bin/bash

# Get the name of the method we are running at the server side.
# currently, it can be "vanilla", "chunked", "tp", "pp", or "prefill"
method_name=$(curl -s http://localhost:5000/name)

if [ -z "$method_name" ]; then
    echo "Method name is empty, please wait for the server to start"
    exit 1
fi

echo "vLLM server is running $method_name"

# RESULTS_PATH='results-pp/pp-huge-chunk'
mkdir -p $RESULTS_PATH

# Tune the hyper-parameters here to change the workload.

if [ "$WORKLOAD" == "1" ]; then
  # Workload 1 (0412)
  num_users=10 # (一开始先用10，等 qps vs latency 的图我们的方法一直不差的时候，再去跑100）->100
  num_documents=50 # -> 5
  user_history_mean=14000
  user_history_std=3000
  user_history_min=11000
  user_history_max=17000
  document_mean=150
  document_std=1
  document_min=149
  document_max=151
  intra_delay=0.05
elif [ "$WORKLOAD" == "2" ]; then
  # Workload 2 (0412)
  num_users=10 #（同理，一开始先用10）->300
  num_documents=1
  user_history_mean=40000
  user_history_std=10000
  user_history_min=10000
  user_history_max=60000
  document_mean=2
  document_std=1
  document_min=1
  document_max=3
  intra_delay=0.05
else
  echo "Invalid workload. Please set WORKLOAD to 1 or 2."
  exit 1
fi

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

# num_users=4
# num_documents=50
# user_history_mean=35000
# user_history_std=1
# user_history_min=10000
# user_history_max=50000
# document_mean=150
# document_std=1
# document_min=10
# document_max=500
# intra_delay=0.05 # 50ms

# Workload 2
# num_users=200
# num_documents=1
# user_history_mean=40000
# user_history_std=1
# user_history_min=10000
# user_history_max=50000
# document_mean=300
# document_std=150
# document_min=10
# document_max=500


# Model length probe
# num_users=5
# num_documents=3
# user_history_mean=220850
# user_history_std=1
# user_history_min=220800
# user_history_max=220900
# document_mean=100
# document_std=1
# document_min=100
# document_max=100


# Workload 1 (Updated)
# num_users=6
# num_documents=20
# user_history_mean=9400
# user_history_std=1
# user_history_min=9000
# user_history_max=9500
# document_mean=300
# document_std=150
# document_min=10
# document_max=500

# num_users=50
# num_documents=1
# user_history_mean=40000
# user_history_std=1
# user_history_min=39000
# user_history_max=41000
# document_mean=2
# document_std=1
# document_min=1
# document_max=3

# Workload 1 (new)
# num_users=6
# num_documents=50
# user_history_mean=16000
# user_history_std=1
# user_history_min=15900
# user_history_max=16100
# document_mean=150
# document_std=1
# document_min=149
# document_max=151

# Workload 2 (new)
# num_users=50 # 300
# num_documents=1
# user_history_mean=40000
# user_history_std=3000 # -> 50000
# user_history_min=30000 # -> 10000
# user_history_max=50000 
# document_mean=2
# document_std=1
# document_min=1
# document_max=3

# Workload 2 (pp experimental)
# num_users=10
# num_documents=1
# user_history_mean=40000
# user_history_std=50000
# user_history_min=10000
# user_history_max=50000 
# document_mean=2
# document_std=1
# document_min=1
# document_max=3




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


results_dir=$(realpath "$RESULTS_PATH/$method_name")

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

# for qps in "inf"; do
# for qps in 1.9525 3.905 7.81 15.62 31.24; do

    # usage:
    # put all variables you used sequenatially, and it will be dumped into the filename.
    filename=$(serialize_to_filename qps num_users num_documents user_history_mean user_history_std user_history_min user_history_max document_mean document_std document_min document_max)

    if [ -f $results_dir/$filename.done ]; then
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
        # continue
        exit 1
    fi

    touch $results_dir/$filename.done

# done