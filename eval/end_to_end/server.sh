#!/bin/bash

set -e

# Allow context length longer than max model length
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1


if [[ $WORKLOAD == "1" ]]; then
    MAX_MODEL_LEN=17251
elif [[ $WORKLOAD == "2" ]]; then
    MAX_MODEL_LEN=60103
else
    echo "Invalid workload. Please set WORKLOAD to 1 or 2."
    exit 1
fi



# check the GPU type
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

get_gpu_util() {
    gpu_type=$(get_gpu_type)
    if [ "$gpu_type" = "H100" ]; then
        # GPU utilizations for H100 GPU
        if [ "$1" = "tp" ]; then
            echo 0.90
        elif [ "$1" = "tp_nvlink" ]; then
            echo 0.90
        elif [ "$1" = "pp" ]; then
            echo 0.90
        elif [ "$1" = "pp_nvlink" ]; then
            echo 0.90
        elif [ "$1" = "vanilla" ]; then
            echo 0.95
        elif [ "$1" = "chunked" ]; then
            echo 0.95
        elif [ "$1" = "prefill_csjf" ]; then
            echo 0.95
        elif [ "$1" = "prefill_sjf" ]; then
            echo 0.95
        else
            echo "Invalid argument. Use 'tp', 'tp_nvlink', 'pp', 'pp_nvlink', 'vanilla', 'chunked', 'prefill_csjf', or 'prefill_sjf'"
            exit 1
        fi
    elif [ "$gpu_type" = "A100" ]; then
        # GPU utilizations for L4 GPU
        if [ "$1" = "tp" ]; then
            echo 0.98
        elif [ "$1" = "pp" ]; then
            echo 0.93
        elif [ "$1" = "vanilla" ]; then
            echo 0.98
        elif [ "$1" = "chunked" ]; then
            echo 0.98
        elif [ "$1" = "prefill_csjf" ]; then
            echo 0.98
        elif [ "$1" = "prefill_sjf" ]; then
            echo 0.98
        else
            echo "Invalid argument. Use 'tp', 'pp', 'vanilla', 'chunked', 'prefill_csjf', or 'prefill_sjf'"
            exit 1
    fi
    elif [ "$gpu_type" = "L4" ]; then
        # GPU utilizations for L4 GPU
        if [ "$1" = "tp" ]; then
            echo 0.90
        elif [ "$1" = "pp" ]; then
            echo 0.95
        elif [ "$1" = "vanilla" ]; then
            echo 0.95
        elif [ "$1" = "chunked" ]; then
            echo 0.95
        elif [ "$1" = "prefill_csjf" ]; then
            echo 0.95
        elif [ "$1" = "prefill_sjf" ]; then
            echo 0.95
        else
            echo "Invalid argument. Use 'tp', 'pp', 'vanilla', 'chunked', 'prefill_csjf', or 'prefill_sjf'"
            exit 1
    fi
}

echo "The chosen GPU utilization is $(get_gpu_util $1) for $1"

# Store PIDs of background processes
PIDS=()

cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid"
            kill -INT "$pid"
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing PID $pid again."
            kill -9 "$pid"
        fi
    done
    wait
    exit 130
}

trap cleanup INT
trap cleanup USR1

# Utility function to wait for a server to start
wait_for_server() {
  local port=$1
  local timeout_seconds=1200
  local start_time=$(date +%s)

  echo "Waiting for server on port $port..."

  while true; do
    if curl -s "localhost:${port}/v1/completions" > /dev/null; then
      return 0
    fi

    local now=$(date +%s)
    if (( now - start_time >= timeout_seconds )); then
      echo "Timeout waiting for server"
      return 1
    fi

    sleep 1
  done
}


if [ $# -ne 1 ]; then
    echo "Usage: $0 [vanilla|chunked|tp|pp|prefill_csjf|prefill_sjf]"
    echo "vanilla: vanilla vLLM v1, w/o chunked prefill"
    echo "chunked: vLLM v1, with chunked prefill"
    echo "tp: vLLM v1, with tensor parallel = 2"
    echo "pp: vLLM v0 (v1 does not support pp), with pipeline parallel = 2"
    echo "prefill_csjf: vLLM v1, with continuous shortest job first scheduling for prefill. This is the default scheduling algorithm for prefill in vLLM v1."
    echo "prefill_sjf: vLLM v1, with SJF scheduling for prefill, should be only used for ablation study"
    exit 1
fi


if ! command -v vllm &> /dev/null; then
    echo "vllm could not be found, please make sure you setup conda environment correctly"
    exit 1
fi

if [ "$1" = "vanilla" ]; then
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "vanilla" &
    PIDS+=($!)

elif [ "$1" = "chunked" ]; then
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --max-num-batched-tokens 512 \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --max-num-batched-tokens 512 \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "chunked" &
    PIDS+=($!)

elif [ "$1" = "tp" ]; then
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens 512 \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --tensor-parallel-size 2 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "tp" &
    PIDS+=($!)

elif [ "$1" = "tp_nvlink" ]; then
    CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens 512 \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --tensor-parallel-size 2 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "tp_nvlink" &
    PIDS+=($!)

elif [ "$1" = "pp" ]; then
    # fall back to v0 for pipeline parallel
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --max-num-batched-tokens 30000 \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --pipeline-parallel-size 2 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "pp" &
    PIDS+=($!)

elif [ "$1" = "pp_nvlink" ]; then
    # fall back to v0 for pipeline parallel
    CUDA_VISIBLE_DEVICES=0,1 vllm serve $EVALUATION_MODEL_NAME \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --max-num-batched-tokens 30000 \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --pipeline-parallel-size 2 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "pp_nvlink" &
    PIDS+=($!)


elif [ "$1" = "prefill_csjf" ]; then
    FAIRNESS=500 CUDA_VISIBLE_DEVICES=0 SCHEDULING_ALGORITHM=CSJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve $EVALUATION_MODEL_NAME \
        -O 3 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8100 &
    PIDS+=($!)

    FAIRNESS=500 CUDA_VISIBLE_DEVICES=1 SCHEDULING_ALGORITHM=CSJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve $EVALUATION_MODEL_NAME \
        -O 3 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "prefill_csjf" &
    PIDS+=($!)

elif [ "$1" = "prefill_sjf" ]; then

    echo "WARNING: THIS IS FOR ABLATION STUDY ONLY"
    echo "Please use prefill_csjf for end-to-end eval"
    sleep 3
    CUDA_VISIBLE_DEVICES=0 SCHEDULING_ALGORITHM=SJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve $EVALUATION_MODEL_NAME \
        -O 3 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 SCHEDULING_ALGORITHM=SJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve $EVALUATION_MODEL_NAME \
        -O 3 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "prefill_sjf" &
    PIDS+=($!)

else
    echo "Invalid argument. Use 'vanilla', 'chunked', 'tp', 'tp_nvlink', 'pp', 'pp_nvlink', 'prefill_csjf', or 'prefill_sjf'"
    exit 1
fi

# Wait for all background processes
wait
