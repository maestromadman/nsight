#!/bin/bash
set -euo pipefail

PROFILES_DIR="$(dirname "$0")/../profiles"
ANALYSIS_DIR="$(dirname "$0")/../analysis"

nsys stats "${PROFILES_DIR}/exp1_baseline.nsys-rep" > "${ANALYSIS_DIR}/exp1_stats.txt"
nsys stats "${PROFILES_DIR}/exp2_fp8.nsys-rep"      > "${ANALYSIS_DIR}/exp2_stats.txt"
nsys stats "${PROFILES_DIR}/exp3_cuda_graphs.nsys-rep" > "${ANALYSIS_DIR}/exp3_stats.txt"

echo "Stats written to ${ANALYSIS_DIR}/"
