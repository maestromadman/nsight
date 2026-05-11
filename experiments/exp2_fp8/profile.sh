#!/bin/bash
set -euo pipefail

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=../../profiles/exp2_fp8 \
  --force-overwrite true \
  python run.py
