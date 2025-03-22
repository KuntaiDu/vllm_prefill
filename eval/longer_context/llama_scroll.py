import vllm
import torch
import os

os.environ['PREFILL_ONLY'] = '1'
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
os.environ['PREFILL_ONLY_CHUNK_SIZE'] = "4096"
os.environ['VLLM_USE_V1'] = '1'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.cuda.set_per_process_memory_fraction(0.32, device=None)

# MLEN = 211000
MLEN = 1000

samp = vllm.SamplingParams(max_tokens=1)

llm = vllm.LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enforce_eager=True,
    max_model_len=MLEN + 100,
    gpu_memory_utilization=0.31,
    enable_prefix_caching=True,
    max_num_batched_tokens=MLEN + 100,
    enable_chunked_prefill=False,
    max_num_seqs=100,
    # tensor_parallel_size=8,
)

output = llm.generate("Hi" * MLEN, samp)[0]

print(output.outputs)

output = llm.generate("Hi" * MLEN, samp)[0]

print(output.outputs)
