#!/bin/bash

# Turn off NVLink
export NCCL_P2P_LEVEL=SYS

# make sure MAX_MODEL_LEN is longer than user_history_max + document_length + 100
MAX_MODEL_LEN=35000
GPU_UTIL=0.31

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
    wait
    exit 130
}

trap cleanup INT

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
    echo "Usage: $0 [vanilla|chunked|tp|pp|prefill]"
    exit 1
fi

if [ "$1" = "vanilla" ]; then
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "vanilla" &
    PIDS+=($!)

elif [ "$1" = "chunked" ]; then
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "chunked" &
    PIDS+=($!)

elif [ "$1" = "tp" ]; then
    CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --tensor-parallel-size 2 \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "tp" &
    PIDS+=($!)

elif [ "$1" = "pp" ]; then
    # fall back to v0 for pipeline parallel
    CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --pipeline-parallel-size 2 \
        --port 8000 &
    PIDS+=($!)
    wait_for_server 8000

    python name_server.py --name "pp" &
    PIDS+=($!)

elif [ "$1" = "prefill" ]; then
    CUDA_VISIBLE_DEVICES=0 PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --enforce-eager \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --enforce-eager \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_UTIL \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --max-num-seqs 1 \
        --port 8200 &
    PIDS+=($!)

    wait_for_server 8100
    wait_for_server 8200

    python sessionid_routing.py &
    PIDS+=($!)

    python name_server.py --name "prefill" &
    PIDS+=($!)

else
    echo "Invalid argument. Use 'vanilla', 'chunked', 'tp', 'pp', or 'prefill'"
    exit 1
fi

# Wait for all background processes
wait
