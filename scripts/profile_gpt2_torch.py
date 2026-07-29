#!/usr/bin/env python3
"""GPT-2 torch.profiler — 采集 op 级 CUDA 耗时和显存 (→ 表A).

Usage:
    python scripts/profile_gpt2_torch.py [--tiny] [--runs 100]
"""
import os
import sys
import csv
import argparse
import time
from pathlib import Path

import torch
import torch.profiler
from transformers import GPT2Model, GPT2Config

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def run_profile(model, input_ids, n_warmup=10, n_runs=100):
    """运行 torch.profiler 采集 CUDA kernel 级数据"""
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()

    # Profile
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for _ in range(n_runs):
                _ = model(input_ids)
        torch.cuda.synchronize()

    return prof


def parse_profile(prof, model_desc):
    """从 profiler 结果提取关键数据 → 表A"""
    key_avgs = prof.key_averages()

    # PyTorch 2.x 属性: self_device_time_total (μs), self_cpu_time_total (μs),
    #                     self_device_memory_usage (bytes), count
    total_device_time = sum(e.self_device_time_total for e in key_avgs)
    if total_device_time == 0:
        print("WARNING: 0 CUDA time recorded. GPU may not be enabled.")
        return []

    rows = []
    for e in key_avgs:
        dev_time_us = e.self_device_time_total
        if dev_time_us == 0:
            continue
        pct = (dev_time_us / total_device_time) * 100
        avg_us = dev_time_us / e.count if e.count > 0 else 0

        rows.append({
            "model": model_desc,
            "op_name": e.key,
            "cuda_time_us": round(dev_time_us, 1),
            "pct_total": round(pct, 1),
            "calls": e.count,
            "avg_us": round(avg_us, 1),
            "cpu_time_us": round(e.self_cpu_time_total, 1),
            "cuda_memory_b": round(e.self_device_memory_usage, 0),
            "cuda_memory_mb": round(e.self_device_memory_usage / 1024 / 1024, 2),
        })

    rows.sort(key=lambda r: r["cuda_time_us"], reverse=True)
    return rows


def write_table(rows, output_path):
    """写入 CSV"""
    fieldnames = [
        "model", "op_name", "cuda_time_us", "pct_total", "calls", "avg_us",
        "cpu_time_us", "cuda_memory_mb",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Also write a human-readable summary
    print(f"\n{'='*80}")
    print(f"表A: GPT-2 Op 耗时排名 (top 25 by GPU time)")
    print(f"{'='*80}")
    print(f"{'op_name':<55s} {'gpu_us':>9s} {'pct':>6s} {'calls':>6s} {'avg_us':>8s} {'mem_mb':>7s}")
    print("-" * 80)
    for r in rows[:25]:
        print(f"{r['op_name']:<55s} {r['cuda_time_us']:>9.1f} {r['pct_total']:>5.1f}% {r['calls']:>6d} {r['avg_us']:>8.1f} {r['cuda_memory_mb']:>7.2f}")
    print(f"\nSaved: {output_path} ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 torch.profiler → 表A")
    parser.add_argument("--tiny", action="store_true", help="Use tiny GPT-2 (2 layers)")
    parser.add_argument("--full", action="store_true", help="Use standard GPT-2 (12 layers, 124M)")
    parser.add_argument("--runs", type=int, default=100, help="Number of profiled iterations")
    parser.add_argument("--seq-len", type=int, default=128, help="Input sequence length")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not args.tiny and not args.full:
        args.full = True  # default to full GPT-2

    print(f"[torch.profiler] Device: {torch.cuda.get_device_name(0)}")
    print(f"[torch.profiler] CUDA available: {torch.cuda.is_available()}")

    # Build model
    if args.tiny:
        config = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64)
        model = GPT2Model(config).cuda()
        model_desc = "GPT2-tiny (2L/128h)"
    else:
        print("[torch.profiler] Loading gpt2 from HuggingFace ...")
        model = GPT2Model.from_pretrained("gpt2").cuda()
        model_desc = "GPT2 (12L/768h/124M)"

    input_ids = torch.randint(0, 50257, (1, args.seq_len)).cuda()
    print(f"[torch.profiler] Input shape: {input_ids.shape}, runs: {args.runs}")

    # Profile
    t0 = time.time()
    prof = run_profile(model, input_ids, n_runs=args.runs)
    elapsed = time.time() - t0
    print(f"[torch.profiler] Profiling done in {elapsed:.1f}s")

    # Parse and save
    rows = parse_profile(prof, model_desc)

    if not rows:
        print("ERROR: No profiling data collected")
        sys.exit(1)

    output_path = args.output or str(TABLE_DIR / "table_a_op_timing.csv")
    write_table(rows, output_path)

    # Also export Chrome trace for interactive visualization
    trace_path = str(TABLE_DIR.parent / "results" / "gpt2_torch_trace.json")
    os.makedirs(str(TABLE_DIR.parent / "results"), exist_ok=True)
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved: {trace_path}")
    print("  Open chrome://tracing in Chrome and load this file.")


if __name__ == "__main__":
    main()
