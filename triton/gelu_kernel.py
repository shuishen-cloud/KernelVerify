#!/usr/bin/env python3
"""Triton GELU kernel — GPT-2 核心激活函数的 GPU 实现。

对比 MLIR 分解 (tanh/pow/mul 5 个 generic op) vs Triton 单 kernel。

Usage:
    python triton/gelu_kernel.py
"""
import torch
import triton
import triton.language as tl


# ── Triton 实现 (单 kernel，无临时 tensor) ─────────────────

@triton.jit
def gelu_triton(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """GELU: x * 0.5 * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)

    # 精确 GELU (用 erf)
    sqrt2 = 1.4142135623730951
    y = 0.5 * x * (1.0 + tl.math.erf(x / sqrt2))

    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def gelu_approx_triton(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """GELU tanh 近似: 0.5*x*(1+tanh(0.7979*(x+0.044715*x^3)))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)

    # MLIR 中分解出的 tanh 近似公式
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))

    tl.store(y_ptr + offsets, y, mask=mask)


def benchmark():
    print("=" * 60)
    print("Triton GELU Kernel Benchmark")
    print("=" * 60)

    n = 1024 * 1024  # 1M elements (~ 768 hidden dim 的 batch)
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y_triton = torch.empty_like(x)
    y_approx = torch.empty_like(x)
    n_runs = 500
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)

    # ── 正确性验证 ──
    # PyTorch 参考
    y_ref = torch.nn.functional.gelu(x)

    gelu_triton[grid](x, y_triton, n, BLOCK=BLOCK)
    diff = (y_triton - y_ref).abs().max().item()
    print(f"\n  精确 GELU (erf):    diff={diff:.2e} {'✅' if diff < 1e-5 else '❌'}")

    gelu_approx_triton[grid](x, y_approx, n, BLOCK=BLOCK)
    diff_approx = (y_approx - y_ref).abs().max().item()
    print(f"  近似 GELU (tanh):   diff={diff_approx:.2e} {'⚠️ (预期差异)' if diff_approx < 1e-2 else '❌'}")

    # ── 性能对比 ──
    def bench_triton(kernel, label):
        for _ in range(10):
            kernel[grid](x, y_triton, n, BLOCK=BLOCK)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_runs):
            kernel[grid](x, y_triton, n, BLOCK=BLOCK)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) / n_runs * 1000
        print(f"  {label:<25} {us:>8.1f} μs")
        return us

    def bench_torch():
        for _ in range(10):
            _ = torch.nn.functional.gelu(x)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_runs):
            _ = torch.nn.functional.gelu(x)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n_runs * 1000

    print(f"\n  [性能] {n_runs} runs, {n} elements:")
    us_triton = bench_triton(gelu_triton, "GELU erf (Triton)")
    us_approx = bench_triton(gelu_approx_triton, "GELU tanh approx (Triton)")
    us_torch = bench_torch()
    print(f"  {'GELU PyTorch':<25} {us_torch:>8.1f} μs")

    # ── MLIR 对比 (回顾) ──
    print(f"\n  [MLIR 层面对比]")
    print(f"  MLIR Linalg 分解:       5 个 linalg.generic (sub→mul→pow→tanh→mul)")
    print(f"  MLIR 优化后:              1 个 linalg.generic (fuse-elementwise-ops)")
    print(f"  Triton 单 kernel:        1 个 GPU kernel (无临时 tensor)")
    print(f"  MLIR GPU pipeline:       1 个 gpu.launch (但无 tiling)")
    print(f"  Triton vs PyTorch:       {us_triton/us_torch:.2f}x ({'=' if abs(us_triton-us_torch) < 10 else '>' if us_triton < us_torch else '<'})")


if __name__ == "__main__":
    benchmark()
