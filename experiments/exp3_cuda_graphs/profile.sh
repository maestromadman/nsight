#!/bin/bash
set -euo pipefail

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=../../profiles/exp3_cuda_graphs \
  --force-overwrite true \
  python run.py
