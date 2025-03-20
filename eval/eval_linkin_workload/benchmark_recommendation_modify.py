"""
Offline benchmark to test the long document QA throughput.

Example usage:
    # This command run the vllm with 50GB CPU memory for offloading
    # The workload samples 8 different prompts with a default input
    # length of 20000 tokens, then replicates each prompt 2 times 
    # in random order.
    python benchmark_long_document_qa_throughput.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-documents 8 \
        --repeat-count 2 

Commandline arguments:
    --num-documents: The number of documents to sample prompts from.

    --document-length: The length of each document in tokens. 
                       (Optional, default: 20000)

    --output-len: The number of tokens to generate for each prompt.
                  (Optional, default: 10)

    --repeat-count: The number of times to repeat each prompt.
                    (Optional, default: 2)

    --repeat-mode: The mode to repeat prompts. The supported modes are:
        - 'random': shuffle the prompts randomly. (Default)
        - 'tile': the entire prompt list is repeated in sequence. (Potentially
                  lowest cache hit)
        - 'interleave': each prompt is repeated consecutively before 
                        moving to the next element. (Highest cache hit)
    
    --shuffle-seed: Random seed when the repeat mode is "random".
                    (Optional, default: 0)

In the meantime, it also supports all the vLLM engine args to initialize the 
LLM engine. You can refer to the `vllm.engine.arg_utils.EngineArgs` for more
details.
python benchmark_recommendation_modify.py  --prefill-only --shuffle-seed 10 --output-len 10 --num-documents  8 --num-users 5 --user-history-length 2

python benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /root/Eamin/vllm_prefill/eval/eval_linkin_workload/ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 5 \
    --num-prompts 100
"""

import dataclasses
import random
import time
import string
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser
import json


def test_long_document_qa(llm=None, sampling_params=None, prompts=None):
    """
    Test long document QA with the given prompts and sampling parameters.
    Print the time spent in processing all the prompts.

    Args:
        llm: The language model used for generating responses.
        sampling_params: Sampling parameter used to generate the response.
        prompts: A list of prompt strings to be processed by the LLM.
    """
    start_time = time.time()
    llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    print(f"Time to execute all requests: {end_time - start_time:.4f} secs")
    
    return end_time - start_time


def repeat_prompts(prompts, repeat_count, mode: str):
    """
    Repeat each prompt in the list for a specified number of times.
    The order of prompts in the output list depends on the mode.

    Args:
        prompts: A list of prompts to be repeated.
        repeat_count: The number of times each prompt is repeated.
        mode: The mode of repetition. Supported modes are:
            - 'random': Shuffle the prompts randomly after repetition.
            - 'tile': Repeat the entire prompt list in sequence.
              Example: [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
            - 'interleave': Repeat each prompt consecutively before moving to 
              the next. Example: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].

    Returns:
        A list of repeated prompts in the specified order.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    print("Repeat mode: ", mode)
    if mode == 'random':
        repeated_prompts = prompts * repeat_count
        random.shuffle(repeated_prompts)
        return repeated_prompts
    elif mode == 'tile':
        return prompts * repeat_count
    elif mode == 'interleave':
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * repeat_count)
        return repeated_prompts
    else:
        raise ValueError(f"Invalid mode: {mode}, only support "
                         "'random', 'tile', 'interleave'")


def main(args):
    random.seed(args.shuffle_seed)

    # Prepare the prompts:
    # we append the document id at the beginning to avoid any of the document
    # being the prefix of other documents
# Generate unique random 2-letter alpha suffix for each user
    user_alphas = [
        ''.join(random.choices(string.ascii_lowercase, k=2))
        for _ in range(args.num_users)
    ]
    doc_alphas = [
        ''.join(random.choices(string.ascii_lowercase, k=2))
        for _ in range(args.num_documents)
    ]

    # Build user strings:
    # - Each user gets a unique alpha
    # - User section is "user:{i}:\n" + (for j in 1..n) "j + alpha" repeated 20000 times
    users = [
        f'user:{str(i)}:\n' + ' '.join(
            [f'{j+1}{user_alphas[i]}' for j in range(args.user_history_length) for _ in range(2)]
        )
        for i in range(args.num_users)
    ]



    # Build document strings:
    # - Document section is "document:{i}:\n" + (for j in n+1..n+50) "j + last_alpha" repeated 150 times
    documents = [
        f'document:{str(i)}:\n' + ' '.join(
            [f'{j + args.user_history_length + 1}{doc_alphas[i]}'for j in range(5)for _ in range(1)] 
        )
        for i in range(args.num_documents)
    ]

    # Combine user and document pairs by Cartesian product
    prompts = [i + j for i in users for j in documents]
    # warmup = [i+j for i in users for j in documents[:2]]
    with open("test_data_modify.txt", "w") as f:
            for prompt in prompts:
                f.write(prompt + "\n\n")
    random.shuffle(prompts)
    records = []
    for prompt in prompts:
        unique_id = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        record = {
            "id": unique_id,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": ""}
            ]
        }
        records.append(record)

    with open("test_data_modify.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        
    import os
    if args.prefill_only:
        os.environ['PREFILL_ONLY'] = '1'
        os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
        os.environ['CHUNK_SIZE'] = "2048"

    # Create the LLM engine
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
               enforce_eager=True,
               max_model_len= args.user_history_length + args.document_length + 300,
               gpu_memory_utilization=0.59,
               block_size=16,
               enable_chunked_prefill=not args.prefill_only,
               enable_prefix_caching=True,
               max_num_batched_tokens=54000 if args.prefill_only else None)
    
    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)
    
    # llm.generate(warmup, sampling_params=sampling_params)

    print("------start generating------")
    benchmark_time = test_long_document_qa(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )
    
    with open("results_modify.yaml", "a") as f:
        import yaml
        f.write(yaml.dump([{
            "user_history_length": args.user_history_length,
            "document_length": args.document_length,
            "benchmark_time": benchmark_time,
            "num_users": args.num_users,
            "num_documents": args.num_documents,
            "prefill_only": args.prefill_only,
        }]))


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        'Benchmark the performance with or without automatic prefix caching.')

    parser.add_argument(
        '--user-history-length',
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=9000,
        help='Range of input lengths for sampling prompts,'
        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--num-users',
                        type=int,
                        default=6,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument(
        '--document-length',
        type=int,
        # Roughly the number of tokens for a system paper,
        # excluding images
        default=10000,
        help='Range of input lengths for sampling prompts,'
        'specified as "min:max" (e.g., "128:256").')
    parser.add_argument('--num-documents',
                        type=int,
                        default=8,
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')

    parser.add_argument('--output-len', type=int, default=1)
    
    parser.add_argument('--shuffle-seed', type=int, default=0)
    
    parser.add_argument('--prefill-only', action='store_true')
    

    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)