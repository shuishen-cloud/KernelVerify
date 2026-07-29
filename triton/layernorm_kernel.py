#!/usr/bin/env python3
"""LayerNorm Triton kernel — GPT-2 normalization 层优化。

Profiling 依据 (表A): LayerNorm 占 1.2% GPU 时间 (1250 次调用)
  PyTorch 分解为: mean → sub → pow → mean → add(eps) → rsqrt → mul → mul(w) → add(b)

融合策略:
  Triton 单 kernel: load row → warp reduce(mean, var) → normalize → scale → store

Usage:
    python triton/layernorm_kernel.py
"""
import torch
import triton
import triton.language as tl


# ── LayerNorm forward kernel ─────────────────

@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    """LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias

    x:   [M, N]    输入 (M rows, N features)
    w/b: [N]       权重/偏置
    y:   [M, N]    输出

    每行独立处理: 每个 program 处理一行
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # 加载整行
    x_ptrs = x_ptr + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # ── Welford 在线 mean/var (单 pass) ──
    # 支持 BLOCK_N > N 的情况 (padding)
    mean = tl.sum(x, axis=0) / N

    # var = mean((x - mean)^2)
    diff = (x - mean)
    var = tl.sum(diff * diff, axis=0) / N

    # ── Normalize ──
    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # ── Scale + bias ──
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    y = normalized * w + b

    tl.store(y_ptr + row * N + cols, y, mask=mask)


# ── LayerNorm + residual add kernel (GPT-2 常用 pattern) ──

@triton.jit
def layernorm_add_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, y_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    """LayerNorm + residual: y = layernorm(x + residual) * weight + bias

    融合了 3 个操作:
    - add(x, residual)  — 1 kernel, 1 中间 tensor (PyTorch)
    - layernorm(result)  — ~8 kernel (PyTorch)
    → Triton 1 kernel (load → add → reduce → norm → scale → store)
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load x and residual → add in registers
    x_ptrs = x_ptr + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    res_ptrs = residual_ptr + row * N + cols
    residual = tl.load(res_ptrs, mask=mask, other=0.0)

    added = x + residual  # 融合: 无中间 tensor!

    # Welford
    mean = tl.sum(added, axis=0) / N
    diff = added - mean
    var = tl.sum(diff * diff, axis=0) / N

    rstd = 1.0 / tl.sqrt(var + eps)
    normalized = diff * rstd

    # Scale + bias
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    y = normalized * w + b

    tl.store(y_ptr + row * N + cols, y, mask=mask)


# ── Benchmark ─────────────────────────────────

def benchmark():
    print("=" * 60)
    print("表D: LayerNorm Kernel 优化前后对比")
    print("=" * 60)

    # GPT-2 hidden_dim = 768
    for shape_desc, M_val, N_val in [
        ("GPT-2 single row (1×768)", 1, 768),
        ("GPT-2 full seq (128×768)", 128, 768),
        ("GPT-2 full batch (64×128×768)", 64 * 128, 768),
    ]:
        print(f"\n  ── {shape_desc} ──")

        x = torch.randn(M_val, N_val, device="cuda", dtype=torch.float32)
        residual = torch.randn(M_val, N_val, device="cuda", dtype=torch.float32)
        weight = torch.randn(N_val, device="cuda", dtype=torch.float32)
        bias = torch.randn(N_val, device="cuda", dtype=torch.float32)
        eps = 1e-5

        y_triton = torch.empty_like(x)
        y_fused = torch.empty_like(x)

        BLOCK_N = triton.next_power_of_2(N_val)
        grid = (M_val,)

        # ── 正确性验证 ──
        # PyTorch 参考
        y_ref = torch.nn.functional.layer_norm(x, (N_val,), weight=weight, bias=bias, eps=eps)
        y_ref_add = torch.nn.functional.layer_norm(x + residual, (N_val,), weight=weight, bias=bias, eps=eps)

        layernorm_kernel[grid](x, weight, bias, y_triton, M_val, N_val, eps, BLOCK_N=BLOCK_N)
        diff1 = (y_triton - y_ref).abs().max().item()

        layernorm_add_kernel[grid](x, residual, weight, bias, y_fused, M_val, N_val, eps, BLOCK_N=BLOCK_N)
        diff2 = (y_fused - y_ref_add).abs().max().item()

        print(f"    Layernorm:          max_diff={diff1:.2e} {'✅' if diff1 < 1e-3 else '❌'}")
        print(f"    Layernorm+add fused: max_diff={diff2:.2e} {'✅' if diff2 < 1e-3 else '❌'}")

        # ── 性能 ──
        def bench(fn, warmup=20, measure=200):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(measure):
                fn()
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) / measure * 1000  # us

        us_torch = bench(lambda: torch.nn.functional.layer_norm(x, (N_val,), weight=weight, bias=bias, eps=eps))
        us_triton = bench(lambda: layernorm_kernel[grid](x, weight, bias, y_triton, M_val, N_val, eps, BLOCK_N=BLOCK_N))
        us_torch_add = bench(lambda: torch.nn.functional.layer_norm(x + residual, (N_val,), weight=weight, bias=bias, eps=eps))
        us_triton_fused = bench(lambda: layernorm_add_kernel[grid](x, residual, weight, bias, y_fused, M_val, N_val, eps, BLOCK_N=BLOCK_N))

        print(f"    {'Kernel':<25s} {'us':>8s} {'vs PyTorch':>10s}")
        print(f"    {'PyTorch F.layer_norm':<25s} {us_torch:>8.1f} {'1.00x':>10s}")
        print(f"    {'Triton layernorm':<25s} {us_triton:>8.1f} {us_triton/us_torch:>9.2f}x")
        print(f"    {'PyTorch add+LN':<25s} {us_torch_add:>8.1f} {'1.00x':>10s}")
        print(f"    {'Triton add+LN fused':<25s} {us_triton_fused:>8.1f} {us_triton_fused/us_torch_add:>9.2f}x")

    # ── 对比分析 ──
    print(f"\n  [对比分析]")
    print(f"  PyTorch LayerNorm: mean→sub→pow→mean→add→rsqrt→mul→mul→add (9 kernel)")
    print(f"  Triton LayerNorm:  1 kernel (warp reduce + elementwise)")
    print(f"  PyTorch add+LN:    10 kernel (1 add + 9 LN) + 1 中间 tensor")
    print(f"  Triton add+LN:     1 kernel (load→add→reduce→norm→store，全部在寄存器/SMEM)")


if __name__ == "__main__":
    benchmark()
