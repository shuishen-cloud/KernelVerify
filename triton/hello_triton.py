#!/usr/bin/env python3
"""Triton Hello World — 验证 4060 GPU 上 Triton kernel 可用。

Usage:
    conda activate novel_llm
    python triton/hello_triton.py
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """向量加法: out = x + y"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    print("=" * 50)
    print("Triton Hello World — RTX 4060")
    print("=" * 50)

    print(f"\n[env] PyTorch {torch.__version__}, Triton {triton.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    print(f"      GPU: {torch.cuda.get_device_name(0)}")

    n = 1024 * 1024  # 1M elements
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()

    expected = x + y
    diff = (out - expected).abs().max().item()
    print(f"\n[result] max diff: {diff:.2e} ({'✅ PASS' if diff < 1e-5 else '❌ FAIL'})")

    # 性能对比
    n_runs = 1000
    for _ in range(10):
        add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_runs):
        add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / n_runs

    start.record()
    for _ in range(n_runs):
        _ = x + y
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_runs

    print(f"\n[perf] {n_runs} runs avg:")
    print(f"       Triton:  {triton_ms*1000:.1f} μs")
    print(f"       PyTorch: {torch_ms*1000:.1f} μs")


if __name__ == "__main__":
    main()
