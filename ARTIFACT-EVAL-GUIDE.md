# Artifact Evaluation Guide

This guide is for the artifact evaluation of paper
PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications
at SoSP 2025.

## Hardware Requirements

All evaluations are conducted on the following hardware:  

- 2 × NVIDIA L4 GPUs on **Google Kubernetes Engine (GKE)**

## Installation

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment. Please refer to the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for setup instructions.  

Once `uv` is installed, inside the `vllm_prefill` folder, run:

```bash
bash install.sh
```

This command creates a virtual environment and installs all required packages. You may need to restart your terminal to activate the environment, or adjust the activation command if you are not using bash.

This project requires the Python development headers to build extension modules. If you are using Debian or Ubuntu, please run the following command to install them:

```bash
sudo apt-get update && sudo apt-get install python3-dev
```

Otherwise, please refer to the documentation of your operating system to install the equivalent Python development package.

## Evaluation

To reproduce our evaluation, run:

```bash
export HF_TOKEN=<your HF token>
cd eval/end_to_end
bash go.sh
```

The evaluation above requires `HF_TOKEN` with the access to the model `meta-llama/Llama-3.1-8B-Instruct`. Please check [this link](https://huggingface.co/docs/hub/en/security-tokens) on how to get the huggingface access token (the `HF_TOKEN` above) and then head to [this page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to apply for the access of `meta-llama/Llama-3.1-8B-Instruct`.

This script executes two workloads:  

- **Post recommendation**  
- **Credit verification**  

under the following system configurations:  

- LMPrefill  
- PagedAttention  
- Chunked Prefill  
- Pipeline Parallel  
- Tensor Parallel  

at varying QPS scales, as described in the paper. The results are saved in the `eval/end_to_end/results-0801-L4-WL-{1,2}` directory and correspond to **Section 7 (Evaluation)**.

## Plotting

To generate the plots, run:

```bash
cd eval/end_to_end/visualize
bash plot_all.sh
```

The generated figures are saved in the `eval/end_to_end/eval_figures` directory and correspond to **Figure 6** and **Figure 7** in the paper.  
