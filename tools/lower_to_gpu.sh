#!/bin/bash
# W9: Linalg → GPU Lowering Pipeline
#
# Usage:
#   bash tools/lower_to_gpu.sh <input.mlir> [output.mlir]
#
# Pipeline:
#   bufferize → parallel-loops → gpu-map → gpu-convert → kernel-outline
#
# 依赖: /home/lwy/download/llvm-project/install/bin/mlir-opt
set -euo pipefail

MLIR_OPT="/home/lwy/download/llvm-project/install/bin/mlir-opt"
INPUT="${1:?Usage: $0 <input.mlir> [output.mlir]}"
OUTPUT="${2:-/dev/stdout}"

echo "[lower_to_gpu] input: $INPUT" >&2
echo "[lower_to_gpu] output: $OUTPUT" >&2

"$MLIR_OPT" \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  "$INPUT" \
  -o "$OUTPUT" 2>&1

echo "[lower_to_gpu] done" >&2