#!/bin/bash

# num_users=8
# num_documents=5
# user_history_length=20000
# document_length=150

num_users=10
num_documents=1
user_history_length=20000
document_length=150

echo "Generating dataset..."
rm linkedin_datsaet_simulated.json
python generate_dataset_linkedin.py \
    --output-json linkedin_datsaet_simulated.json \
    --user-history-length $user_history_length \
    --num-users $num_users \
    --document-length $document_length \
    --num-documents $num_documents

echo "Benchmarking vLLM"
python benchmark_serving.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path linkedin_datsaet_simulated.json \
    --request-rate 0.2 \
    --sharegpt-output-len 1 \
    --backend vllm
