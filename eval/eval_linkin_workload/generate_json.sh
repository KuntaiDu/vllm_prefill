#!/bin/bash

# save as: generate_dataset.sh
mkdir -p link_json

for num_users in $(seq 2 100); do
  python clean_gen.py \
    --prefill-only \
    --shuffle-seed 10 \
    --output-len 10 \
    --num-documents 8 \
    --num-users $num_users \
    --user-history-length 20000 \
    --output-json link_json/test_data_${num_users}.json
done