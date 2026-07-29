#!/usr/bin/env python3
"""GPT-2 模块替换 — 用 Triton kernel 替换 GELU/LayerNorm，测端到端效果 (→ 表F).

Usage:
    python scripts/patch_gpt2_triton.py [--tiny] [--full]
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import GPT2Model, GPT2Config
import time
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ── Triton GELU (erf, 已在 triton/gelu_fused.py 验证) ──

@triton.jit
def _triton_gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """GELU tanh 近似 (GPT-2 默认)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _triton_add_gelu_kernel(x_ptr, bias_ptr, y_ptr, n, BLOCK: tl.constexpr):
    """add+GELU tanh 近似融合 (GPT-2 默认)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    b = tl.load(bias_ptr + offsets, mask=mask)
    added = x + b
    inner = 0.7978845608028654 * (added + 0.044715 * added * added * added)
    y = 0.5 * added * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(y_ptr + offsets, y, mask=mask)


class TritonGELU(nn.Module):
    """nn.Module wrapper: GELU via Triton"""
    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _triton_gelu_kernel[grid](x, y, n, BLOCK=BLOCK)
        return y


# ── nn.Module wrappers ──

@triton.jit
def _triton_layernorm_kernel(x_ptr, w_ptr, b_ptr, y_ptr, M, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    y = diff * rstd * tl.load(w_ptr + cols, mask=mask, other=0.0) + tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * N + cols, y, mask=mask)


class TritonLayerNorm(nn.Module):
    """nn.Module wrapper: LayerNorm via Triton (warp reduce)"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # x: [*, N]
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        BLOCK_N = triton.next_power_of_2(N)
        grid = (M,)
        _triton_layernorm_kernel[grid](x_2d, self.weight, self.bias, y, M, N, self.eps, BLOCK_N=BLOCK_N)
        return y.reshape(orig_shape)


# ── GPT-2 Surgery ──

def patch_gpt2(model):
    """替换 GPT-2 内部的 GELU 和 LayerNorm 为 Triton 版本"""
    import transformers.models.gpt2.modeling_gpt2 as gpt2_mod

    class GPT2MLPWithTriton(gpt2_mod.GPT2MLP):
        """GPT2MLP 但用 Triton GELU 代替默认 GELU"""
        def __init__(self, config, parent_mlp=None):
            # Copy state from parent
            super().__init__(config)
            if parent_mlp is not None:
                self.load_state_dict(parent_mlp.state_dict())
            self.act = TritonGELU()

    class GPT2BlockWithTriton(gpt2_mod.GPT2Block):
        """GPT2Block 但用 Triton LayerNorm 代替"""
        def __init__(self, config, parent_block=None, patch_ln=False, patch_gelu=False):
            super().__init__(config)
            if parent_block is not None:
                self.load_state_dict(parent_block.state_dict(), strict=False)
            if patch_ln:
                self.ln_1 = TritonLayerNorm(config.n_embd, config.layer_norm_epsilon)
                self.ln_2 = TritonLayerNorm(config.n_embd, config.layer_norm_epsilon)
            if patch_gelu:
                self.mlp.act = TritonGELU()

    patches = []
    for i, block in enumerate(model.h):
        block.mlp.act = TritonGELU()
        patches.append(f"h[{i}].mlp.act → TritonGELU")
    return model, patches


# ── Benchmark ──

def bench_model(model, input_ids, name, n_warmup=20, n_runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_runs):
            _ = model(input_ids)
        e.record()
        torch.cuda.synchronize()
        latency_ms = s.elapsed_time(e) / n_runs

        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()

    return {"name": name, "latency_ms": round(latency_ms, 2), "peak_memory_mb": round(peak_mem, 1)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    if not args.tiny and not args.full:
        args.tiny = True  # default to tiny (fast)

    print(f"[GPT-2 Patch] Device: {torch.cuda.get_device_name(0)}")

    if args.tiny:
        config = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64)
        model = GPT2Model(config).cuda()
        model_desc = "GPT2-tiny (2L/128h)"
    else:
        model = GPT2Model.from_pretrained("gpt2").cuda()
        model_desc = "GPT2 (12L/768h/124M)"

    input_ids = torch.randint(0, 50257, (1, args.seq_len)).cuda()
    print(f"[GPT-2 Patch] Model: {model_desc}, input: {input_ids.shape}")

    rows = []

    # 1. Baseline
    print(f"\n{'='*60}")
    print(f"表F: GPT-2 端到端 Triton 模块替换对比 — {model_desc}")
    print(f"{'='*60}")

    r = bench_model(model, input_ids, "GPT-2 baseline (PyTorch)", n_runs=args.runs)
    r["num_ops"] = "—"
    r["variant"] = "baseline"
    rows.append(r)
    print(f"  baseline: {r['latency_ms']:.2f}ms, peak {r['peak_memory_mb']:.0f}MB")

    # Validate reference output
    with torch.no_grad():
        ref_out = model(input_ids)[0]

    # 2. + Triton GELU
    model_gelu = GPT2Model(config).cuda() if args.tiny else GPT2Model.from_pretrained("gpt2").cuda()
    _, patches = patch_gpt2(model_gelu)
    print(f"  Patches: {patches[:2]}... (+{len(patches)-2} more)")

    r = bench_model(model_gelu, input_ids, "+ Triton GELU", n_runs=args.runs)
    r["num_ops"] = f"-{len(patches)}"
    r["variant"] = "gelu"
    rows.append(r)

    with torch.no_grad():
        gelu_out = model_gelu(input_ids)[0]
    acc_gelu = torch.allclose(gelu_out, ref_out, atol=1e-3)
    print(f"  +GELU:  {r['latency_ms']:.2f}ms, peak {r['peak_memory_mb']:.0f}MB, acc={acc_gelu}")

    # 3. Stats
    baseline_ms = rows[0]["latency_ms"]
    print(f"\n  {'Variant':<30s} {'ms':>8s} {'vs base':>8s} {'mem_mb':>8s} {'acc':>6s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    for r in rows:
        vs = f"{r['latency_ms']/baseline_ms:.2f}x" if baseline_ms > 0 else "N/A"
        acc_str = str(r.get("accuracy", "—"))
        print(f"  {r['name']:<30s} {r['latency_ms']:>8.2f} {vs:>8s} {r['peak_memory_mb']:>8.1f} {acc_str:>6s}")

    # Write CSV
    output_path = TABLE_DIR / "table_f_fx_fusion.csv"
    fieldnames = ["variant", "name", "latency_ms", "peak_memory_mb"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
