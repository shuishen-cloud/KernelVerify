#!/usr/bin/env python3
"""GPT-2 nsys profiling helper —— 运行 GPT-2 推理供 nsys 采集 GPU kernel trace.

Usage (配合 nsys):
    nsys profile --stats=true -o benchmarks/results/gpt2_nsys \
      python scripts/profile_gpt2_nsys.py [--tiny] [--runs 100]

然后解析 nsys 输出:
    python scripts/profile_gpt2_nsys.py --parse benchmarks/results/gpt2_nsys.sqlite
"""
import os
import sys
import csv
import argparse
import sqlite3
import time
from pathlib import Path

import torch
from transformers import GPT2Model, GPT2Config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_inference(args):
    """供 nsys 采集的推理循环"""
    print(f"[nsys] Device: {torch.cuda.get_device_name(0)}")

    if args.tiny:
        config = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64)
        model = GPT2Model(config).cuda()
        model_desc = "GPT2-tiny (2L/128h)"
    else:
        model = GPT2Model.from_pretrained("gpt2").cuda()
        model_desc = "GPT2 (12L/768h/124M)"

    print(f"[nsys] Model: {model_desc}")

    input_ids = torch.randint(0, 50257, (1, args.seq_len)).cuda()
    print(f"[nsys] Input: {input_ids.shape}, warmup={args.warmup}, runs={args.runs}")

    model.eval()
    with torch.no_grad():
        # Warmup
        for i in range(args.warmup):
            _ = model(input_ids)
            if i == 0:
                print(f"[nsys] First inference done (compilation trigged)")
        torch.cuda.synchronize()

        # Timed runs
        print(f"[nsys] Starting {args.runs} timed runs ...")
        t0 = time.time()
        for _ in range(args.runs):
            _ = model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        avg_ms = (elapsed / args.runs) * 1000
        print(f"[nsys] Done: {elapsed:.2f}s total, {avg_ms:.2f}ms avg per inference")


def parse_nsys_sqlite(db_path, output_path):
    """解析 nsys SQLite 输出 → 表B"""
    print(f"[nsys] Parsing {db_path} ...")

    if not os.path.exists(db_path):
        print(f"ERROR: {db_path} not found. Run nsys profile first:")
        print(f"  nsys profile --stats=true -o {RESULTS_DIR}/gpt2_nsys python scripts/profile_gpt2_nsys.py")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # nsys 表结构: CUPTI_ACTIVITY_KIND_KERNEL 表包含 GPU kernel 信息
    # 尝试查找包含 kernel 信息的表
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    print(f"[nsys] Found {len(tables)} tables")

    # 尝试多个可能的表名
    kernel_table = None
    for candidate in [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "StringIds", "TARGET_INFO_SESSION_START_TIME",
    ]:
        if candidate in tables:
            kernel_table = candidate
            # 检查是否有 kernel 表
            if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
                # 新版本 nsys 使用不同的表结构
                cursor.execute("SELECT name FROM sqlite_master WHERE name LIKE '%KERNEL%' OR name LIKE '%kernel%'")
                kt = cursor.fetchall()
                if kt:
                    kernel_table = kt[0][0]
                else:
                    print("[nsys] No kernel table found. Trying generic schema ...")
                    # 尝试打印所有表的前几行
                    for t in tables[:10]:
                        cursor.execute(f"SELECT * FROM \"{t}\" LIMIT 1")
                        cols = [d[0] for d in cursor.description]
                        print(f"  {t}: {cols}")
                    break
            break

    rows = []

    if kernel_table and "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        cursor.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        cols = [c[1] for c in cursor.fetchall()]
        print(f"[nsys] Kernel table columns: {cols}")

        # 构建查询: 提取 kernel name, duration, grid, block, registers
        select_cols = []
        for c in cols:
            if c == "shortName" or c == "demangledName":
                select_cols.append("shortName as kernel_name")
            elif c == "start":
                select_cols.append("start")
            elif c == "end":
                select_cols.append("end")
            elif c == "duration":
                select_cols.append("duration")
            elif "grid" in c.lower():
                select_cols.append(f"{c} as grid")
            elif "block" in c.lower():
                select_cols.append(f"{c} as block")
            elif "register" in c.lower():
                select_cols.append(f"{c} as registers")
            elif "shared" in c.lower():
                select_cols.append(f"{c} as shared_mem")

        if not select_cols:
            # Fallback: select all
            query = f"SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start LIMIT 100"
        else:
            query_cols = ", ".join(set(select_cols))
            query = f"SELECT {query_cols} FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"

        cursor.execute(query)
        col_names = [d[0] for d in cursor.description]
        for row in cursor.fetchall():
            r = dict(zip(col_names, row))
            duration_ns = r.get("duration", 0) or 0

            # 判断 bound type
            ai_ratio = 0
            bound_type = "UNKNOWN"

            rows.append({
                "kernel_name": str(r.get("kernel_name", "unknown")),
                "duration_us": round(duration_ns / 1000.0, 2),
                "grid": str(r.get("grid", "?")),
                "block": str(r.get("block", "?")),
                "registers": r.get("registers", "?"),
                "shared_mem_kb": r.get("shared_mem", "?"),
            })

    conn.close()

    if not rows:
        # Fallback: use nsys stats CSV export
        print("[nsys] SQLite parsing returned no rows.")
        print(f"[nsys] Try: nsys stats --report cuda_gpu_kern_sum --format csv -o {RESULTS_DIR}/gpt2_kernels.csv {db_path}")
        print("[nsys] Skipping table generation — create table manually from nsys stats CSV.")
        return

    # Sort by duration descending
    rows.sort(key=lambda r: r["duration_us"], reverse=True)

    # Write table B
    fieldnames = ["kernel_name", "duration_us", "grid", "block", "registers", "shared_mem_kb"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[nsys] Table B saved: {output_path} ({len(rows)} kernels)")

    # Summary
    print(f"\n{'='*80}")
    print(f"表B: GPT-2 GPU Kernel 特征 (top 20 by duration)")
    print(f"{'='*80}")
    print(f"{'kernel_name':<50s} {'dur_us':>8s} {'grid':>15s} {'block':>15s} {'regs':>5s}")
    print("-" * 80)
    for r in rows[:20]:
        print(f"{r['kernel_name']:<50s} {r['duration_us']:>8.1f} {r['grid']:>15s} {r['block']:>15s} {str(r['registers']):>5s}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 nsys profiling → 表B")
    parser.add_argument("--tiny", action="store_true", help="Use tiny GPT-2")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--parse", type=str, default=None, help="Parse nsys SQLite output to CSV")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.parse:
        output_path = args.output or str(TABLE_DIR / "table_b_kernel_breakdown.csv")
        parse_nsys_sqlite(args.parse, output_path)
    else:
        # Run inference for nsys to capture
        run_inference(args)


if __name__ == "__main__":
    main()
