#!/bin/bash
# verify_mlir.sh — W2: Verify Torch Dialect MLIR syntax and lowering pipeline
# Usage: ./tools/verify_mlir.sh <mlir_file>

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[OK]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }
stage() { echo -e "\n${BOLD}--- $* ---${NC}"; }

MLIR_FILE="${1:?Usage: $0 <mlir_file>}"
BASENAME=$(basename "$MLIR_FILE" .mlir)
OUT_DIR="mlir/lowered"

mkdir -p "$OUT_DIR"

# Stage 1: Syntax verification (torch-mlir-opt has torch dialect)
stage "Stage 1: Torch Dialect syntax verification"
if torch-mlir-opt --verify-diagnostics "$MLIR_FILE" > /dev/null 2>&1; then
    info "Torch Dialect syntax OK: $MLIR_FILE"
else
    fail "Torch Dialect verification failed: $MLIR_FILE"
fi

# Stage 2: Lowering to Linalg
stage "Stage 2: Lowering Torch Dialect → Linalg"
LOWERED="$OUT_DIR/${BASENAME}_lowered.mlir"
torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline \
    "$MLIR_FILE" > "$LOWERED" 2>&1
info "Lowered to Linalg: $LOWERED"
echo "  $(wc -l < "$LOWERED") lines"

# Stage 3: Linalg IR verification
stage "Stage 3: Linalg IR verification"
if mlir-opt --verify-diagnostics "$LOWERED" > /dev/null 2>&1; then
    info "Linalg IR verification passed"
else
    fail "Linalg IR verification failed"
fi

# Stage 4: Apply optimizations
stage "Stage 4: Optimize (fuse-elementwise + canonicalize + cse)"
OPTIMIZED="$OUT_DIR/${BASENAME}_optimized.mlir"
mlir-opt --linalg-fuse-elementwise-ops --canonicalize --cse \
    "$LOWERED" > "$OPTIMIZED" 2>&1
info "Optimized IR: $OPTIMIZED"
echo "  $(wc -l < "$OPTIMIZED") lines"

# Summary
stage "Summary"
BEFORE_OPS=$(grep -c "linalg\." "$LOWERED" || true)
AFTER_OPS=$(grep -c "linalg\." "$OPTIMIZED" || true)
echo "  linalg ops before: $BEFORE_OPS"
echo "  linalg ops after:  $AFTER_OPS"

if [ "$AFTER_OPS" -le "$BEFORE_OPS" ]; then
    info "Optimization reduced or maintained linalg op count"
else
    info "Note: op count increased (may be due to canonicalization decomposition)"
fi

echo ""
info "All stages passed for $MLIR_FILE"
