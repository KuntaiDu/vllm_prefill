# this is used to install correct flash attention wheel.
# Copy-paste these commands, and execute them in the root directory of vllm.

export VLLM_COMMIT=241ad7b301facac0728e2b3312d71fe47acc8c9e
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .