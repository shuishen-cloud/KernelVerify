#!/bin/bash
# 编译 CountLinalgOps Pass 插件
set -euo pipefail

PASS_DIR="$(cd "$(dirname "$0")" && pwd)/CountLinalgOps"
BUILD_DIR="$PASS_DIR/build"
MLIR_INSTALL="/home/lwy/download/llvm-project/install"

echo "=== 编译 CountLinalgOps Pass ==="
echo "Source: $PASS_DIR"
echo "Build:  $BUILD_DIR"
echo "MLIR:   $MLIR_INSTALL"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DCMAKE_PREFIX_PATH="$MLIR_INSTALL" \
  -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -3

make -j$(nproc) 2>&1

echo ""
echo "=== 编译完成 ==="
echo "插件: $BUILD_DIR/libCountLinalgOps.so"
ls -lh "$BUILD_DIR/libCountLinalgOps.so" 2>/dev/null || echo "⚠️ .so 未生成，检查编译错误"