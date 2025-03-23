#!/bin/bash

MAX_MODEL_LEN=35000

if [ $# -ne 1 ]; then
    echo "Usage: $0 [vanilla|prefill]"
    exit 1
fi

if [ "$1" = "vanilla" ]; then
    # add `--max-num-batched-tokens $MAX_MODEL_LEN` to disable chunked prefill.
    VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.31 \
        --enforce-eager \
        --max-num-seqs 1 
elif [ "$1" = "prefill" ]; then
    PREFILL_ONLY=1 PREFILL_ONLY_CHUNK_SIZE=4096 VLLM_USE_V1=1 \
    vllm serve meta-llama/Llama-3.1-8B-Instruct \
        --enforce-eager \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization 0.31 \
        --enable-prefix-caching \
        --max-num-batched-tokens $MAX_MODEL_LEN \
        --enable-chunked-prefill=false \
        --max-num-seqs 1
else
    echo "Invalid argument. Use 'vanilla' or 'prefill'"
    exit 1
fi
    