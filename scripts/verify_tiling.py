#!/usr/bin/env python3
"""Tiling 对比度量框架（CPU，两个后端）。

用途
----
为 tiling 的结构改动建立**可重复的度量**：每改一次 tiling 策略（切哪些维、切多大），
都能立刻看到「数值是否还正确」+「CPU 上是否变快」。

为什么用 CPU 而不是 GPU
-----------------------
GPU 实测需要 NVPTX 后端，当前被 LLVM 编译 OOM 阻塞（见 work/task/分析-NVPTX_LLVM编译OOM.md）。
CPU 是**代理指标**，边界必须说清：

  能验证：数值正确性、tile 形状/tile size 对缓存复用的影响
  不能验证：shared memory promotion、occupancy、并行度 —— 这些是 GPU 独有机制

因此本框架回答「tiling 有没有让 IR 变好」，**不是**「GPU 上快多少」。结论不可外推。

两个后端
--------
* **runner**（默认，推荐）：`mlir-opt` 降到 LLVM dialect → `mlir-runner` 执行。
  用 `--convert-linalg-to-loops` 得到**朴素循环、无自动 tiling/vectorize**，
  这样 tiling 自身的效果才被隔离出来。也最贴近本项目 Phase 3「原始 MLIR 直落 GPU」的场景。
* **iree**（保留作参照）：IREE `llvm-cpu`。**经实测不适用于本用途** —— IREE 自身就是
  完整编译器（会做 tiling / vectorize / 并行化），喂它预 tiled 的 IR 等于打断它自己的能力，
  测到的是「被打断的代价」而非「tiling 的收益」。见 work/task/分析-Tiling概念与算子验证进度.md 第十一节。

已知限制
--------
* `mlir-cpu-runner` 在本机不存在（LLVM 只装了 mlir-opt），新版改名后的 **`mlir-runner`** 可用。
* IREE 的 flow pipeline 在 **`scf.forall`（张量上的 forall + tensor.parallel_insert_slice）上段错误**
  （崩在 mlir::Value::getParentBlock）。故 forall 形态只能用 runner 后端度量。
* IREE 必须传 `--iree-llvmcpu-target-cpu=host`，否则默认 generic CPU，性能数字没有意义。
* runner 后端的校验值是 A/B 全 1 时的 checksum（=`K × reps`）。它能抓出重复计数/漏算 tile，
  但抓不出置换类错误 —— 完整随机数值校验走 iree 后端（5 个变体已全部通过）。
* 本机 `conda activate novel_llm` 不生效，需用带 `iree.compiler` 的 Python（本机为
  PythonProjectStarter 的 venv，Python 3.13）。

用法
----
    python scripts/verify_tiling.py                            # 默认 runner 后端, 512^3
    python scripts/verify_tiling.py --size 1024
    python scripts/verify_tiling.py --backend iree --size 256
    python scripts/verify_tiling.py --variants base 32x32x16
"""
import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MLIR_OPT = "/home/lwy/download/llvm-project/install/bin/mlir-opt"
DEFAULT_MLIR_RUNNER = "/home/lwy/download/llvm-project/install/bin/mlir-runner"
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"

# 变体：(名字, tile_sizes 或 None)。tile_sizes 三位 = M, N, K；0 表示该维不切。
VARIANTS = [
    ("base", None),
    ("tile_32x32_K0", (32, 32, 0)),
    ("tile_32x32x16", (32, 32, 16)),
    ("tile_64x64x32", (64, 64, 32)),
    ("tile_16x16x8", (16, 16, 8)),
]

# ── payload 生成 ────────────────────────────────────────────
# iree 后端：单次 matmul，入参出参都是 tensor，便于与 numpy 全量比对
PAYLOAD_IREE = """func.func @main(%A: tensor<{M}x{K}xf32>, %B: tensor<{K}x{N}xf32>) -> tensor<{M}x{N}xf32> {{
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<{M}x{N}xf32>
  %zero = linalg.fill ins(%cst : f32) outs(%init : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%zero : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  return %0 : tensor<{M}x{N}xf32>
}}
"""

# runner 后端：main 内部循环 reps 次 matmul 并累加，返回 C[0,0] 作 checksum。
# 循环用来摊销进程启动/JIT 开销；累加是为了避免 DCE 掉前面的迭代。
PAYLOAD_RUNNER = """func.func @main() -> f32 {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %R = arith.constant {REPS} : index
  %one = arith.constant 1.0 : f32
  %zero = arith.constant 0.0 : f32
  %ea = tensor.empty() : tensor<{M}x{K}xf32>
  %A = linalg.fill ins(%one : f32) outs(%ea : tensor<{M}x{K}xf32>) -> tensor<{M}x{K}xf32>
  %eb = tensor.empty() : tensor<{K}x{N}xf32>
  %B = linalg.fill ins(%one : f32) outs(%eb : tensor<{K}x{N}xf32>) -> tensor<{K}x{N}xf32>
  %ec = tensor.empty() : tensor<{M}x{N}xf32>
  %init = linalg.fill ins(%zero : f32) outs(%ec : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %res = scf.for %i = %c0 to %R step %c1 iter_args(%acc = %init) -> (tensor<{M}x{N}xf32>) {{
    %ec2 = tensor.empty() : tensor<{M}x{N}xf32>
    %z = linalg.fill ins(%zero : f32) outs(%ec2 : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
    %m = linalg.matmul ins(%A, %B : tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>) outs(%z : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
    %s = linalg.add ins(%acc, %m : tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) outs(%acc : tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
    scf.yield %s : tensor<{M}x{N}xf32>
  }}
  %r = tensor.extract %res[%c0, %c0] : tensor<{M}x{N}xf32>
  return %r : f32
}}
"""

SCRIPT_TMPL = """module @tm attributes {{transform.target_tag="linalg", transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {{
    %f = transform.structured.match ops{{["func.func"]}} in %root : (!transform.op<"builtin.module">) -> !transform.op<"func.func">
    %m = transform.structured.match ops{{["linalg.matmul"]}} in %f : (!transform.op<"func.func">) -> !transform.op<"linalg.matmul">
{op}
    transform.yield
  }}
}}
"""

LOWER_PASSES = [
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    # tiling 会引入 tensor.extract_slice/insert_slice → bufferize 后是 memref.subview/copy，
    # 必须先展开 strided metadata 才能降到 LLVM
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-to-llvm",
    "--reconcile-unrealized-casts",
]

# tiled IR 里会残留 memref.copy → 需要 runner utils 提供 memrefCopy 符号
DEFAULT_RUNNER_LIBS = "/home/lwy/download/llvm-project/install/lib/libmlir_c_runner_utils.so"


def payload_iree(m, k, n):
    return PAYLOAD_IREE.format(M=m, K=k, N=n)


def payload_runner(m, k, n, reps):
    return PAYLOAD_RUNNER.format(M=m, K=k, N=n, REPS=reps)


def make_transform_script(tile_sizes):
    """生成 tile_using_for 的 transform 脚本。

    tile_using_for 每个"被切的维"返回一个 scf.for，所以结果类型个数必须
    与实际切出来的循环数一致（0 表示不切该维）。
    """
    n_loops = sum(1 for s in tile_sizes if s)
    results = ", ".join(['!transform.op<"linalg.matmul">'] + ['!transform.op<"scf.for">'] * n_loops)
    sizes = ", ".join(str(s) for s in tile_sizes)
    lhs = "%tiled" + "".join(f", %l{i}" for i in range(n_loops))
    op = (
        f"    {lhs} = transform.structured.tile_using_for %m tile_sizes [{sizes}]\n"
        f'      : (!transform.op<"linalg.matmul">) -> ({results})'
    )
    return SCRIPT_TMPL.format(op=op)


def strip_transform_module(text):
    """删掉内嵌的 `module @tm { ... }`（按括号配平）。"""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith("module @tm"):
            depth = 0
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
                if depth <= 0:
                    break
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + "\n"


def apply_transform(mlir_opt, payload, script):
    """把 transform 脚本与 payload 写进同一文件后运行解释器。

    mlir-opt 只接受 1 个位置参数，所以脚本必须以 `module @tm {...}` 的形式
    内嵌在同一份 IR 里（本地实测唯一跑通的形态）。
    """
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "combined.mlir"
        f.write_text(payload + "\n" + script)
        r = subprocess.run([mlir_opt, "--transform-interpreter", str(f)],
                           capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"transform 失败: {r.stderr.strip()[:300]}")
    return strip_transform_module(r.stdout)


def lower_to_llvm(mlir_opt, ir):
    r = subprocess.run([mlir_opt, *LOWER_PASSES], input=ir, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"lowering 失败: {r.stderr.strip()[:300]}")
    return r.stdout


# ── 后端 1: mlir-runner（朴素循环） ────────────────────────
def bench_runner(mlir_opt, mlir_runner, m, k, n, sizes, r1, r2, shared_libs=DEFAULT_RUNNER_LIBS):
    """两点法测每次 matmul 耗时：T(R2) - T(R1)，扣掉进程启动/JIT 开销。"""
    lowered = {}
    for reps in (r1, r2):
        ir = payload_runner(m, k, n, reps)
        if sizes is not None:
            ir = apply_transform(mlir_opt, ir, make_transform_script(sizes))
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "ll.mlir"
            f.write_text(lower_to_llvm(mlir_opt, ir))
            cmd = [mlir_runner, "-e", "main", "-entry-point-result=f32"]
            if shared_libs:
                cmd.append(f"--shared-libs={shared_libs}")
            t0 = time.perf_counter()
            r = subprocess.run(cmd + [str(f)], capture_output=True, text=True)
            wall_ms = (time.perf_counter() - t0) * 1000
        if r.returncode != 0:
            raise RuntimeError(f"mlir-runner 失败: {r.stderr.strip()[:200]}")
        got = float(r.stdout.strip().splitlines()[-1])
        expect = float(k * reps)
        if abs(got - expect) > abs(expect) * 1e-3:
            raise RuntimeError(f"checksum 不符: got={got} expect={expect}")
        lowered[reps] = wall_ms
    per_iter = (lowered[r2] - lowered[r1]) / (r2 - r1)
    return {"ok": True, "median_ms": per_iter, "min_ms": per_iter,
            "gflops": 2 * m * k * n / (per_iter / 1000) / 1e9,
            "raw": f"T({r1})={lowered[r1]:.1f}ms T({r2})={lowered[r2]:.1f}ms"}


# ── 后端 2: IREE（参照，已证不适用） ───────────────────────
def bench_iree(mlir_opt, ir, m, k, n, reps, warmup):
    import numpy as np
    import iree.compiler
    import iree.runtime

    t0 = time.perf_counter()
    compiled = iree.compiler.compile_str(
        ir, target_backends=["llvm-cpu"],
        input_type=iree.compiler.InputType.AUTO,
        extra_args=["--iree-llvmcpu-target-cpu=host"],
    )
    compile_ms = (time.perf_counter() - t0) * 1000
    cfg = iree.runtime.Config("local-task")
    vm = iree.runtime.load_vm_module(
        iree.runtime.VmModule.from_flatbuffer(cfg.vm_instance, compiled, warn_if_copy=False), cfg)

    rng = np.random.default_rng(0)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)
    ref = a @ b
    ok = bool(np.allclose(vm.main(a, b).to_host(), ref, rtol=1e-3, atol=1e-3))
    for _ in range(warmup):
        vm.main(a, b)
    s = []
    for _ in range(reps):
        t = time.perf_counter()
        vm.main(a, b)
        s.append((time.perf_counter() - t) * 1000)
    med = statistics.median(s)
    return {"ok": ok, "median_ms": med, "min_ms": min(s),
            "gflops": 2 * m * k * n / (med / 1000) / 1e9, "compile_ms": compile_ms}


def main():
    ap = argparse.ArgumentParser(description="Tiling 对比度量（CPU）")
    ap.add_argument("--backend", choices=("runner", "iree"), default="runner")
    ap.add_argument("--size", type=int, default=512, help="M=K=N（默认 512）")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--reps", type=int, default=10, help="iree: 计时重复次数")
    ap.add_argument("--warmup", type=int, default=3, help="iree: 预热次数")
    ap.add_argument("--r1", type=int, default=5, help="runner: 低点循环次数")
    ap.add_argument("--r2", type=int, default=25, help="runner: 高点循环次数")
    ap.add_argument("--mlir-opt", default=DEFAULT_MLIR_OPT)
    ap.add_argument("--mlir-runner", default=DEFAULT_MLIR_RUNNER)
    ap.add_argument("--variants", nargs="*", default=None,
                    help=f"只跑指定变体，可选: {[v[0] for v in VARIANTS]}")
    ap.add_argument("--tiles", nargs="*", default=None,
                    help='自定义 tile 规格（给出则替代内置变体，base 仍保留），'
                         '如 --tiles "16,16,8" "32,32,16"')
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    m = args.m or args.size
    k = args.k or args.size
    n = args.n or args.size
    csv_path = Path(args.csv or TABLE_DIR / f"table_h_tiling_{args.backend}.csv")

    if args.tiles:
        todo = [("base", None)] + [
            (f"tile_{'_'.join(t.split(','))}", tuple(int(x) for x in t.split(",")))
            for t in args.tiles
        ]
    else:
        todo = [v for v in VARIANTS if args.variants is None or v[0] in args.variants]
    if not todo:
        print(f"没有匹配的变体。可选: {[v[0] for v in VARIANTS]}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"Tiling 对比度量 — backend={args.backend}  |  matmul {m}x{k}x{n}")
    print("=" * 78)
    print("⚠️  CPU 代理指标：只能验证数值与缓存复用，不能验证 shared memory / occupancy")
    if args.backend == "runner":
        print(f"    朴素循环（--convert-linalg-to-loops），两点法 R={args.r1}/{args.r2}")
    else:
        print("    ⚠️  IREE 自身会 tiling/vectorize —— 测得的是「打断它」的代价，非 tiling 收益")
    print()

    rows = []
    for name, sizes in todo:
        try:
            if args.backend == "runner":
                r = bench_runner(args.mlir_opt, args.mlir_runner, m, k, n,
                                 sizes, args.r1, args.r2)
            else:
                ir = payload_iree(m, k, n)
                if sizes is not None:
                    ir = apply_transform(args.mlir_opt, ir, make_transform_script(sizes))
                r = bench_iree(args.mlir_opt, ir, m, k, n, args.reps, args.warmup)
        except Exception as e:
            print(f"  {name:<16} 失败: {str(e)[:110]}")
            rows.append({"variant": name, "tile_sizes": str(sizes), "status": "failed"})
            continue
        rows.append({
            "variant": name, "tile_sizes": str(sizes),
            "status": "ok" if r["ok"] else "MISMATCH",
            "median_ms": round(r["median_ms"], 4),
            "min_ms": round(r["min_ms"], 4),
            "gflops": round(r["gflops"], 2),
        })

    base = next((r for r in rows if r["variant"] == "base" and r.get("median_ms")), None)
    print(f"  {'variant':<16} {'tile(M,N,K)':<16} {'数值':<6} {'每次(ms)':>11} {'vs base':>9} {'GFLOPS':>8}")
    print("  " + "-" * 74)
    for r in rows:
        if "median_ms" not in r:
            print(f"  {r['variant']:<16} {r['tile_sizes']:<16} {r['status']}")
            continue
        ratio = f"{base['median_ms'] / r['median_ms']:.2f}x" if base else "-"
        print(f"  {r['variant']:<16} {r['tile_sizes']:<16} "
              f"{'✅' if r['status'] == 'ok' else '❌':<6} {r['median_ms']:>11.4f} {ratio:>9} {r['gflops']:>8.2f}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "tile_sizes", "status",
                                         "median_ms", "min_ms", "gflops"])
        w.writeheader()
        for r in rows:
            w.writerow({k2: r.get(k2, "") for k2 in w.fieldnames})
    print(f"\n已写入: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
