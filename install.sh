uv venv
source .venv/bin/activate

# Install vLLM
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v

# Dev dependencies
uv pip install ipython matplotlib setuptools

# For simple proxy
uv pip install flask quart

# For datasets
uv pip install datasets
