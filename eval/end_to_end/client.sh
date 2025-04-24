#!/bin/bash

# Get the name of the method we are running at the server side.
# currently, it can be "vanilla", "chunked", "tp", "pp", or "prefill"
method_name=$(curl -s http://localhost:5000/name)

if [ -z "$method_name" ]; then
    echo "Method name is empty, please wait for the server to start"
    exit 1
fi

echo "vLLM server is running $method_name"

mkdir -p $RESULTS_PATH


if [ "$WORKLOAD" == "1" ]; then
  num_users=20
  num_documents=50
  user_history_mean=14000
  user_history_std=3000
  user_history_min=11000
  user_history_max=17000
  document_mean=150
  document_std=1
  document_min=149
  document_max=151
  intra_delay=0.1
elif [ "$WORKLOAD" == "2" ]; then
  num_users=60
  num_documents=1
  user_history_mean=40000
  user_history_std=10000
  user_history_min=10000
  user_history_max=60000
  document_mean=2
  document_std=1
  document_min=1
  document_max=3
  intra_delay=0.1
else
  echo "Invalid workload. Please set WORKLOAD to 1 or 2."
  exit 1
fi


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

###
### Generating dataset
###

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

###
### Benchmarking
###

echo "Benchmarking vLLM"

filename=$(serialize_to_filename qps num_users num_documents user_history_mean user_history_std user_history_min user_history_max document_mean document_std document_min document_max)

if [ -f $results_dir/$filename.done ]; then
    echo "Skipping $filename because it already exists"
    exit 0
fi

echo "Running $filename"

python benchmark_serving.py \
    --model $EVALUATION_MODEL_NAME \
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
