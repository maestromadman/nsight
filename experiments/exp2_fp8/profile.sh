#!/bin/bash
set -euo pipefail

# VLLM_ENABLE_V1_MULTIPROCESSING=0 forces the V1 engine to run the EngineCore
# in-process rather than as a subprocess. Without this, nsys only sees the
# parent coordinator process and misses all GPU kernel activity.
VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=../../profiles/exp2_fp8 \
  --force-overwrite true \
  python run.py
