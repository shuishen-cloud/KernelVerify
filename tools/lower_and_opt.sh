#!/bin/bash
# lower_and_opt.sh — W4: Torch Dialect → Linalg lowering + configurable optimization pipeline
#
# Usage:
#   ./tools/lower_and_opt.sh <mlir_file>                    # default strategy
#   ./tools/lower_and_opt.sh <mlir_file> --preset=light      # light optimization
#   ./tools/lower_and_opt.sh <mlir_file> --preset=aggressive # aggressive optimization
#   ./tools/lower_and_opt.sh mlir/exported/ --batch          # batch all .mlir in dir

set -euo pipefail

# ── Configuration ──────────────────────────────────────────
OUT_DIR="mlir/lowered"
PRESET="default"

# ── Optimization presets ───────────────────────────────────
# Each preset is a space-separated list of mlir-opt passes
declare -A PASSES
PASSES[light]="--canonicalize --cse"
PASSES[default]="--linalg-fuse-elementwise-ops --canonicalize --cse"
PASSES[aggressive]="--linalg-fuse-elementwise-ops --linalg-generalize-named-ops --canonicalize --cse --eliminate-empty-tensors"

LOWERING_PIPELINE="--torch-backend-to-linalg-on-tensors-backend-pipeline"

# ── Helpers ────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${RED}[ERR]${NC} $*"; }
stage() { echo -e "\n${BOLD}${CYAN}── $* ──${NC}"; }

count_linalg_ops() { grep -c 'linalg\.' "$1" 2>/dev/null || echo 0; }
count_scf_ops()    { grep -c 'scf\.'    "$1" 2>/dev/null || echo 0; }
count_arith_ops()  { grep -c 'arith\.'  "$1" 2>/dev/null || echo 0; }

# ── Parse args ─────────────────────────────────────────────
BATCH_MODE=false
TARGET=""

for arg in "$@"; do
    case $arg in
        --preset=*) PRESET="${arg#*=}" ;;
        --batch)   BATCH_MODE=true ;;
        --out-dir=*) OUT_DIR="${arg#*=}" ;;
        *)         TARGET="$arg" ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Usage: $0 <mlir_file|directory> [--preset=light|default|aggressive] [--batch]"
    exit 1
fi

PASS_LIST="${PASSES[$PRESET]:-${PASSES[default]}}"
mkdir -p "$OUT_DIR"

# ── Process single file ────────────────────────────────────
process_file() {
    local input="$1"
    local basename
    basename=$(basename "$input" .mlir)
    local lowered="$OUT_DIR/${basename}_lowered.mlir"
    local optimized="$OUT_DIR/${basename}_opt_${PRESET}.mlir"

    # Stage 1: Lower Torch → Linalg
    torch-mlir-opt "$LOWERING_PIPELINE" "$input" > "$lowered" 2>&1
    if [ $? -ne 0 ]; then
        warn "Lowering failed: $input"
        return 1
    fi

    local before_ops
    before_ops=$(count_linalg_ops "$lowered")

    # Stage 2: Optimize
    # shellcheck disable=SC2086
    mlir-opt $PASS_LIST "$lowered" > "$optimized" 2>&1
    if [ $? -ne 0 ]; then
        warn "Optimization failed: $input"
        return 1
    fi

    local after_ops
    after_ops=$(count_linalg_ops "$optimized")
    local reduced=$((before_ops - after_ops))
    local pct=0
    if [ "$before_ops" -gt 0 ]; then
        pct=$((100 * reduced / before_ops))
    fi

    printf "  %-25s  preset=%-10s  linalg: %2d → %2d  (-%d, %d%%)\n" \
        "$basename" "$PRESET" "$before_ops" "$after_ops" "$reduced" "$pct"

    # Write stats file
    local stats="$OUT_DIR/${basename}_opt_${PRESET}.stats"
    cat > "$stats" <<EOF
file:       $input
preset:     $PRESET
passes:     $PASS_LIST
lowered:    $lowered
optimized:  $optimized
linalg_ops_before: $before_ops
linalg_ops_after:  $after_ops
scf_ops_before:    $(count_scf_ops "$lowered")
scf_ops_after:     $(count_scf_ops "$optimized")
arith_ops_before:  $(count_arith_ops "$lowered")
arith_ops_after:   $(count_arith_ops "$optimized")
EOF
}

# ── Main ───────────────────────────────────────────────────
stage "Torch Dialect → Linalg Lowering + Optimization"
echo "  Preset: $PRESET"
echo "  Passes: $PASS_LIST"
echo "  Output: $OUT_DIR/"
echo ""

if $BATCH_MODE && [ -d "$TARGET" ]; then
    shopt -s nullglob
    files=("$TARGET"/*.mlir)
    shopt -u nullglob
    if [ ${#files[@]} -eq 0 ]; then
        warn "No .mlir files found in $TARGET"
        exit 1
    fi
    echo "  Batch processing ${#files[@]} files..."
    for f in "${files[@]}"; do
        process_file "$f"
    done
else
    process_file "$TARGET"
fi

echo ""
info "Done. Output files in $OUT_DIR/"
ls -1 "$OUT_DIR"/*_opt_"$PRESET".mlir 2>/dev/null || true
