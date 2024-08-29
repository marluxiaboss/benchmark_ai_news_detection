conda activate big_gen_bench

# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.5.1
export PYTHON_VERSION=311
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118
pip install vllm 
pip install prometheus-eval 
pip install -r requirements.txt

conda deactivate