#!/bin/bash
# 编译 SimNewBackend Pass 插件
set -euo pipefail

PASS_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PASS_DIR/build"
MLIR_INSTALL="/home/lwy/download/llvm-project/install"

echo "=== 编译 SimNewBackend Pass ==="
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
SO_FILE="$BUILD_DIR/libSimNewBackend.so"
if [ -f "$SO_FILE" ]; then
    ls -lh "$SO_FILE"
    echo ""
    echo "验证 Pass 注册:"
    torch-mlir-opt --load-pass-plugin="$SO_FILE" --help 2>&1 | grep -A1 "sim-"
else
    echo "⚠️  $SO_FILE 未生成，请检查编译错误"
    exit 1
fi
