#!/bin/bash

set -e

# Turn off NVLink
export NCCL_P2P_LEVEL=SYS

# make sure MAX_MODEL_LEN is longer than user_history_max + document_length + 100
MAX_MODEL_LEN=41000

get_gpu_util() {
    if [ "$1" = "tp" ]; then
        echo 0.31
    elif [ "$1" = "pp" ]; then
        echo 0.31
    elif [ "$1" = "vanilla" ]; then
        echo 0.31
    elif [ "$1" = "chunked" ]; then
        echo 0.31
    elif [ "$1" = "prefill_csjf" ]; then
        echo 0.31
    elif [ "$1" = "prefill_sjf" ]; then
        echo 0.31
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
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
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

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
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
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
        --enable-prefix-caching \
        --max-num-seqs 1 \
        --disable-log-stats \
        --disable-log-requests \
        --port 8100 &
    PIDS+=($!)

    CUDA_VISIBLE_DEVICES=1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
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

    python name_server.py --name "chunked" &
    PIDS+=($!)

elif [ "$1" = "tp" ]; then
    CUDA_VISIBLE_DEVICES=0,1 VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_MODEL_LEN \
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

elif [ "$1" = "pp" ]; then
    # fall back to v0 for pipeline parallel
    CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $(get_gpu_util $1) \
        --enforce-eager \
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

elif [ "$1" = "prefill_csjf" ]; then
    CUDA_VISIBLE_DEVICES=0 SCHEDULING_ALGORITHM=CSJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
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

    CUDA_VISIBLE_DEVICES=1 SCHEDULING_ALGORITHM=CSJF PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
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
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
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
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
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
    echo "Invalid argument. Use 'vanilla', 'chunked', 'tp', 'pp', 'prefill_csjf', or 'prefill_sjf'"
    exit 1
fi

# Wait for all background processes
wait
