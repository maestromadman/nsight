#!/bin/bash
set -euo pipefail

nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=../../profiles/exp1_baseline \
  --force-overwrite true \
  python run.py
