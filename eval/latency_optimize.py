import os
import cProfile
import pstats
import random
import time
# os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
# os.environ['CHUNK_SIZE'] = "2048"
os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
# os.environ['BYPASS_SAMPLER'] = '1'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import vllm

# torch.cuda.set_per_process_memory_fraction(0.59, device=None)

######################################
# GPU UTILIZATION (pynvml) SETUP
######################################
from pynvml import (
    nvmlInit, nvmlShutdown,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates
)
from threading import Thread

nvmlInit()  # Initialize NVML
gpu_handle = [nvmlDeviceGetHandleByIndex(i) for i in range(8)]  # Use GPU index 0 (change if needed)

def get_gpu_utilization_pct():
    """Returns current GPU utilization percentage."""
    util = [nvmlDeviceGetUtilizationRates(gpu_handle[i]) for i in range(8)]
    return sum([util[i].gpu for i in range(8)]) / 8

MLEN = 30000
# MLEN = 50000

samp = vllm.SamplingParams(
    # logprobs=0,
    # prompt_logprobs=0,
    max_tokens=1
)

llm = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True,
    max_model_len=MLEN + 100,
    gpu_memory_utilization=0.8,
    block_size=16,
    enable_prefix_caching=True,
    max_num_batched_tokens=MLEN + 100,
    enable_chunked_prefill=False,
    tensor_parallel_size=8,
)

def tokenid_execution(token_ids):
    torch.cuda.synchronize()
    st = time.time()

    header = str(torch.rand(1).item())
    for idx, tokens in enumerate(token_ids):
        assert isinstance(tokens, list)
        llm.llm_engine.add_request(
            request_id=f"{header}-{idx}",
            prompt={'prompt_token_ids': tokens},
            params=samp
        )

    ########################################################
    # BENCHMARK GPU UTILIZATION DURING llm.llm_engine.step()
    ########################################################
    gpu_util_samples = []
    stop_sampling = False

    def sample_gpu():
        """Collect GPU utilization in a background thread."""
        while not stop_sampling:
            gpu_util_samples.append(get_gpu_utilization_pct())
            time.sleep(0.02)  # sample every 20ms (tune as desired)

    # Start sampling in the background
    sampler_thread = Thread(target=sample_gpu)
    sampler_thread.start()

    # Run the step function
    torch.cuda.synchronize()
    step_start = time.time()
    llm.llm_engine.step()
    torch.cuda.synchronize()
    step_end = time.time()

    # Stop sampling and wait for the sampler thread to exit
    stop_sampling = True
    sampler_thread.join()

    et = time.time()
    print(f"Time taken (full request cycle): {et - st} seconds")
    print(f"Time taken by llm.llm_engine.step(): {step_end - step_start} seconds")

    # Calculate average GPU utilization during step
    if gpu_util_samples:
        idx = 0
        # breakpoint()
        # while idx < len(gpu_util_samples) and gpu_util_samples[idx] < 5:
        #     idx += 1

        # print("Idle time: ", idx * 0.02, "seconds")
        avg_gpu = sum(gpu_util_samples[idx:]) / (len(gpu_util_samples) - idx)
        print(f"Average GPU utilization (during step): {avg_gpu:.2f}%")

    # assert llm.llm_engine.scheduler[0].get_num_unfinished_seq_groups() == 0


prompt_len = 20000
prefix = torch.randint(llm.llm_engine.model_config.get_vocab_size(), size=(prompt_len, )).tolist()

suffixs = [torch.randint(llm.llm_engine.model_config.get_vocab_size(), size=(150, )).tolist() for _ in range(50)]

new_prefix = prefix.copy()
new_prefix.extend([token for suffix in suffixs for token in suffix])
# [A B1 B2 .. B50]

# warmup
tokenid_execution([[i+1 for i in prefix]])

tokenid_execution([prefix])

import time

tokenid_execution([new_prefix])

# tokenid_execution([prefix + suffix for suffix in suffixs])

# Clean up NVML before exiting
nvmlShutdown()