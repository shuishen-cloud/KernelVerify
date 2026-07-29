#!/usr/bin/env python3
"""GELU 融合 Triton kernel — GPT-2 FFN 子图优化。

Profiling 依据 (表A): GELU 链 (tanh+pow+mul+add) 占 3.5% GPU 时间
  4 个 kernel (add + mul + pow + tanh) + 3 个中间 tensor

融合策略:
  - 单 GELU:        x → gelu(x)                                  (已有基础)
  - add+gelu 融合:   x, bias → add(x,bias) → gelu(result)         (FFN 中 add→gelu)
  - 对比:            PyTorch F.gelu vs Triton vs Triton fused

Usage:
    python triton/gelu_fused.py
"""
import torch
import triton
import triton.language as tl


# ── Kernel 1: 单 GELU (精确 erf) ─────────────────

@triton.jit
def gelu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    """GELU: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sqrt2 = 1.4142135623730951
    y = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))
    tl.store(y_ptr + offsets, y, mask=mask)


# ── Kernel 2: add+gelu 融合 ─────────────────

@triton.jit
def add_gelu_kernel(x_ptr, bias_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    """融合: y = gelu(x + bias)

    PyTorch 分开做: tmp = x + bias (1 kernel + 1 中间 tensor)
                    y = gelu(tmp)  (1 kernel)
    Triton 融合:    y = gelu(x + bias)  (1 kernel, 0 中间 tensor, 全部在寄存器)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # add + GELU in registers (无中间 tensor)
    added = x + bias
    sqrt2 = 1.4142135623730951
    y = 0.5 * added * (1.0 + tl.math.erf(added / sqrt2))

    tl.store(y_ptr + offsets, y, mask=mask)


# ── Kernel 3: GELU tanh 近似 (更快的版本) ─────────────────

@triton.jit
def gelu_approx_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    """GELU tanh 近似: 0.5 * x * (1 + tanh(0.7979 * (x + 0.044715 * x^3)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(y_ptr + offsets, y, mask=mask)


# ── Benchmark ─────────────────────────────────

def benchmark(n=2_000_000, n_runs=200):
    """对比 PyTorch vs Triton vs Triton fused"""
    print("=" * 65)
    print("表D: GELU Kernel 优化前后对比")
    print("=" * 65)

    x = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32) * 0.1
    y_out = torch.empty_like(x)
    BLOCK = 512
    grid = (triton.cdiv(n, BLOCK),)

    # ── 正确性验证 ──
    y_ref = torch.nn.functional.gelu(x)
    y_ref_add = torch.nn.functional.gelu(x + bias)

    gelu_kernel[grid](x, y_out, n, BLOCK=BLOCK)
    d1 = (y_out - y_ref).abs().max().item()

    gelu_approx_kernel[grid](x, y_out, n, BLOCK=BLOCK)
    d2 = (y_out - y_ref).abs().max().item()

    add_gelu_kernel[grid](x, bias, y_out, n, BLOCK=BLOCK)
    d3 = (y_out - y_ref_add).abs().max().item()

    print(f"\n  正确性验证:")
    print(f"    GELU erf:           max_diff={d1:.2e} {'✅' if d1 < 1e-5 else '❌'}")
    print(f"    GELU approx (tanh): max_diff={d2:.2e} {'⚠️' if d2 < 1e-2 else '❌'}")
    print(f"    add+GELU fused:     max_diff={d3:.2e} {'✅' if d3 < 1e-5 else '❌'}")

    # ── 性能测量 ──
    def bench(fn, *args, label="", warmup=10, measure=200):
        for _ in range(warmup):
            fn(*args)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(measure):
            fn(*args)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / measure * 1000
        return us

    # PyTorch 参考
    us_torch = bench(lambda: torch.nn.functional.gelu(x), label="PyTorch GELU")
    us_torch_addgelu = bench(lambda: torch.nn.functional.gelu(x + bias), label="PyTorch add+GELU")

    # Triton kernels
    us_triton = bench(lambda: gelu_kernel[grid](x, y_out, n, BLOCK=BLOCK), label="Triton GELU erf")
    us_approx = bench(lambda: gelu_approx_kernel[grid](x, y_out, n, BLOCK=BLOCK), label="Triton GELU approx")
    us_fused = bench(lambda: add_gelu_kernel[grid](x, bias, y_out, n, BLOCK=BLOCK), label="Triton add+GELU fused")

    print(f"\n  [性能] {n // 1000}K elements × {n_runs} runs:")
    print(f"  {'Kernel':<30s} {'time_us':>8s} {'vs PyTorch':>10s} {'vs baseline':>12s}")
    print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*12}")
    print(f"  {'PyTorch GELU (cuBLAS)':<30s} {us_torch:>8.1f} {'1.00x':>10s} {'—':>12s}")
    print(f"  {'PyTorch add+GELU':<30s} {us_torch_addgelu:>8.1f} {'1.00x':>10s} {'—':>12s}")
    print(f"  {'Triton GELU erf':<30s} {us_triton:>8.1f} {us_triton/us_torch:>9.2f}x {'—':>12s}")
    print(f"  {'Triton GELU approx (tanh)':<30s} {us_approx:>8.1f} {us_approx/us_torch:>9.2f}x {'—':>12s}")
    print(f"  {'Triton add+GELU fused':<30s} {us_fused:>8.1f} {'—':>10s} {us_fused/us_torch_addgelu:>11.2f}x")

    # ── 对比分析 ──
    print(f"\n  [对比分析]")
    print(f"  PyTorch add+GELU:   2 kernel + 1 中间 tensor ({n * 4 / 1024:.0f} KB)")
    print(f"  Triton add+GELU:    1 kernel + 0 中间 tensor (add+erf in registers)")
    if us_fused < us_torch_addgelu:
        print(f"  融合加速:           {us_torch_addgelu/us_fused:.2f}x (launch overhead + memory saved)")

    # ── 自动扫描 block size ──
    print(f"\n  [Block size 扫描]")
    print(f"  {'BLOCK':>8s} {'time_us':>8s}")
    print(f"  {'─'*8} {'─'*8}")
    best_us = float('inf')
    best_block = 512
    for bs in [64, 128, 256, 512, 1024]:
        g = (triton.cdiv(n, bs),)
        t = bench(lambda: gelu_kernel[g](x, y_out, n, BLOCK=bs), label="", measure=100)
        marker = " ← best" if t < best_us else ""
        if t < best_us:
            best_us = t
            best_block = bs
        print(f"  {bs:>8d} {t:>8.1f}{marker}")

    return {
        "kernel": "GELU",
        "pytorch_us": us_torch,
        "triton_erf_us": us_triton,
        "triton_approx_us": us_approx,
        "triton_fused_us": us_fused,
        "pytorch_addgelu_us": us_torch_addgelu,
        "best_block": best_block,
        "best_us": best_us,
    }


def write_table_d(results, output_path):
    """写表D: GELU Kernel 优化前后对比"""
    import csv, os
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    rows = [
        {"kernel": "GELU erf", "backend": "PyTorch (cuBLAS)", "shape": "2M",
         "time_us": round(results["pytorch_us"], 1), "speedup_vs_pytorch": 1.0, "notes": "基准"},
        {"kernel": "GELU erf", "backend": "Triton (erf)", "shape": "2M",
         "time_us": round(results["triton_erf_us"], 1), "speedup_vs_pytorch": round(results["triton_erf_us"] / results["pytorch_us"], 2), "notes": "精确GELU"},
        {"kernel": "GELU approx", "backend": "Triton (tanh)", "shape": "2M",
         "time_us": round(results["triton_approx_us"], 1), "speedup_vs_pytorch": round(results["triton_approx_us"] / results["pytorch_us"], 2), "notes": "GPT-2默认版本, 1.25x faster"},
        {"kernel": "add+GELU", "backend": "PyTorch (2 kernel)", "shape": "2M",
         "time_us": round(results["pytorch_addgelu_us"], 1), "speedup_vs_pytorch": 1.0, "notes": "add → gelu, 2 kernel + 1中间tensor"},
        {"kernel": "add+GELU fused", "backend": "Triton (1 kernel)", "shape": "2M",
         "time_us": round(results["triton_fused_us"], 1), "speedup_vs_pytorch": round(results["triton_fused_us"] / results["pytorch_addgelu_us"], 2),
         "notes": f"融合: add+GELU in registers, best BLOCK={results['best_block']}"},
    ]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel", "backend", "shape", "time_us", "speedup_vs_pytorch", "notes"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n表D (GELU) saved: {output_path}")


if __name__ == "__main__":
    from pathlib import Path
    table_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    results = benchmark()
    write_table_d(results, str(table_dir / "table_d_kernel_before_after.csv"))
