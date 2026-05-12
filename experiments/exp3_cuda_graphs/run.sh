#!/bin/bash
cd "$(dirname "$0")"

mkdir -p ../../profiles

nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --output=../../profiles/exp3_cuda_graphs \
  --force-overwrite true \
  python3 - << 'EOF'
import sys
sys.path.insert(0, '..')
from prompts import CONVERSATIONS
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
    quantization="fp8",
    enforce_eager=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.92,
    max_model_len=512,
)

sampling_params = SamplingParams(
    temperature=0.3,
    max_tokens=300,
)

outputs = llm.chat(CONVERSATIONS, sampling_params)

for i, output in enumerate(outputs):
    print(f"Request {i+1}: {output.outputs[0].text[:80]}...")
EOF
