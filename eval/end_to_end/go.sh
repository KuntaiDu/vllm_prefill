
PIDS=()

mkdir -p logs

cleanup() {
    echo "Cleaning up in run-flat..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill -USR1 $pid
        fi
    done
    for pid in "${PIDS[@]}"; do
        # Waiting for the process to finish
        echo "Waiting $pid"
        wait "$pid"
    done
    # exit 130
}

cleanup_and_exit() {
    cleanup
    exit 130
}

get_gpu_type() {
  if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found"
    return 1
  fi

  gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

  if [[ $gpu_name == *"A100"* ]]; then
    echo "A100"
  elif [[ $gpu_name == *"H100"* ]]; then
    echo "H100"
  elif [[ $gpu_name == *"V100"* ]]; then
    echo "V100"
  elif [[ $gpu_name == *"L40"* ]]; then
    echo "L40"
  else
    echo "Unknown GPU name: $gpu_name"
    echo "Please add it to the get_gpu_type function"
    exit 1
  fi
}

trap cleanup_and_exit INT

# Utility function to wait for a server to start
wait_for_server() {
  local port=$1
  local timeout_seconds=120
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "http://localhost:${port}/name" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server, cleanup"
      cleanup

      return 1
    fi

    sleep 5
  done
}

go() {
    local setting=$1
    local workload=$2
    local qps=$3

    # Clear pid list
    PIDS=()

    export WORKLOAD=$workload
    export qps=$qps
    echo "Running with setting: $setting, workload: $workload, qps: $qps"

    bash server.sh $setting >logs/${RESULTS_PATH}-server_${setting}_${workload}_${qps}.log 2>&1 &
    PIDS+=($!)
    
    # Wait for the server to start
    if ! wait_for_server 5000; then
        echo "Server failed to start."
        return 0
    fi

    echo "Server started with PID ${PIDS[-1]}"
    
    # Run the client script
    bash client.sh | tee logs/${RESULTS_PATH}-client_${setting}_${workload}_${qps}.log

    # Done! Stop the server
    cleanup

    echo "Done: $setting, workload: $workload, qps: $qps\n"
}



### Main evaluation configurations


gpu_type=$(get_gpu_type)

export RESULTS_PATH="results-0424-$gpu_type"

get_qps() {
    # $1: gpu_type, $2: workload
    if [ "$1" = "H100" ]; then
        if [ "$2" = "1" ]; then
            echo 12.0
        elif [ "$2" = "2" ]; then
            echo 0.16
        fi
    fi
}

get_model_name() {
    if [ "$gpu_type" = "H100" ]; then
        echo "Infermatic/Llama-3.3-70B-Instruct-FP8-Dynamic"
    elif [ "$gpu_type" = "A100" ]; then
        echo "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic"
    elif [ "$gpu_type" = "L40" ]; then
        echo "meta-llama/Llama-3.1-8B-Instruct"
    else
        echo "Unknown GPU type: $gpu_type"
        exit 1
    fi

    # @ Bowen @ Yiming add the model name for other GPUs here
}

export EVALUATION_MODEL_NAME=$(get_model_name)


### Main evaluation loop

for workload in 1 2; do
    for setting in chunked prefill_csjf tp_nvlink pp_nvlink; do
        throughput=$(get_qps $gpu_type $workload)
        echo "The selected QPS for hardware $gpu_type, workload $workload is $throughput"
        # for qps in 0.25*$throughput 0.5*$throughput $throughput 2*$throughput 3*$throughput 4*$throughput; do
        for qps in $throughput; do
            echo "Running with qps $qps"
            go $setting $workload $qps
        done
    done
done
