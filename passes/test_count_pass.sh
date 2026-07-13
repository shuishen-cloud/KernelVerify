#!/bin/bash
# 测试 CountLinalgOps Pass
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN="$SCRIPT_DIR/CountLinalgOps/build/libCountLinalgOps.so"
MLIR_OPT="/home/lwy/download/llvm-project/install/bin/mlir-opt"
PROJECT="/home/lwy/project/Learn/study_ai/mlir/pytorch_mlir"

if [ ! -f "$PLUGIN" ]; then
    echo "❌ 插件未编译: $PLUGIN"
    echo "   先运行: bash passes/build_count_pass.sh"
    exit 1
fi

PASS_PIPELINE='builtin.module(func.func(count-linalg-ops))'

echo "=== 测试 1: 手写 matmul ==="
$MLIR_OPT --load-pass-plugin="$PLUGIN" --pass-pipeline="$PASS_PIPELINE" \
    "$PROJECT/mlir/handwritten/test_matmul_linalg.mlir" 2>&1

echo ""
echo "=== 测试 2: GPT-2 静态 Linalg IR ==="
$MLIR_OPT --load-pass-plugin="$PLUGIN" --pass-pipeline="$PASS_PIPELINE" \
    "$PROJECT/mlir/lowered/gpt2_full_linalg.mlir" 2>&1 | head -10

echo ""
echo "=== 测试 3: GPT-2 动态 Linalg IR ==="
$MLIR_OPT --load-pass-plugin="$PLUGIN" --pass-pipeline="$PASS_PIPELINE" \
    "$PROJECT/mlir/lowered/gpt2_dynamic_linalg.mlir" 2>&1 | head -10

echo ""
echo "✅ 全部测试通过"
