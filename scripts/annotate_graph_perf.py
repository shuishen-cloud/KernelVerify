#!/usr/bin/env python3
"""将运行时性能信息附加到计算图节点上。

结合三层数据：
  1. 计算图结构  — torch.export → FX Graph（节点/边/consumer）
  2. 运行时性能  — torch.profiler（每个 aten op 的 GPU 耗时/调用次数）
  3. 名称映射    — "aten.addmm.default" (FX) ↔ "aten::addmm" (profiler)

产出：
  - benchmarks/tables/table_annotated_graph.csv  — 每个图节点 + 性能列
  - 控制台热力表 — 按耗时排序的节点视图
  - (可选) 带性能标注的 DOT/SVG 可视化

Usage:
    python scripts/annotate_graph_perf.py --block 0          # 单层
    python scripts/annotate_graph_perf.py --full             # 完整 GPT-2
    python scripts/annotate_graph_perf.py --viz              # 生成 SVG
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.profiler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"
GRAPH_DIR = PROJECT_ROOT / "mlir" / "graphs"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ── 名称映射 ──────────────────────────────────────────────

def fx_to_prof_name(target: str) -> str:
    """FX 图 target → torch.profiler key。

    "aten.addmm.default"  → "aten::addmm"
    "aten.mul.Tensor"     → "aten::mul"
    "aten.softmax.int"    → "aten::softmax"
    "aten.pow.Tensor_Scalar" → "aten::pow"
    """
    s = str(target)
    # 处理 FxGraphDrawer 的 "torch.ops.aten.addmm.default" 前缀
    if s.startswith("torch.ops.aten."):
        s = s[len("torch.ops."):]
    if not s.startswith("aten."):
        return s
    # "aten.addmm.default" → "aten::addmm.default"
    s = s.replace("aten.", "aten::", 1)
    # 去掉 overload 后缀 → "aten::addmm"
    base = s.split("::", 1)
    if len(base) == 2:
        op_and_overload = base[1]
        # "addmm.default" → "addmm"
        if "." in op_and_overload:
            op_and_overload = op_and_overload.rsplit(".", 1)[0]
        s = f"aten::{op_and_overload}"
    return s


# torch.profiler 中的 op 名 vs FX 图的 op 名别名映射
PROF_ALIASES = {
    "aten::matmul": ["aten::bmm", "aten::mm", "aten::matmul"],
    "aten::layer_norm": ["aten::native_layer_norm", "aten::layer_norm"],
    "aten::softmax": ["aten::_softmax", "aten::softmax"],
    "aten::add": ["aten::add", "aten::add_"],
    "aten::mul": ["aten::mul", "aten::mul_"],
}


def resolve_prof_key(fx_op_name: str, prof_perf: dict) -> str:
    """将 FX op 名解析到 profiler 中实际存在的 key（含别名匹配）。"""
    if fx_op_name in prof_perf:
        return fx_op_name
    for alias in PROF_ALIASES.get(fx_op_name, []):
        if alias in prof_perf:
            return alias
    return None


def classify_op(target: str) -> str:
    """归类算子类型（复用 analyze_dataflow 逻辑）"""
    t = str(target).lower()
    if any(k in t for k in ["addmm", "mm", "bmm", "matmul", "scaled_dot_product"]):
        return "MATMUL"
    if any(k in t for k in ["mul", "add", "div", "sub", "pow", "tanh", "gelu",
                             "erf", "where", "clamp", "relu", "sigmoid"]):
        return "ELEMENTWISE"
    if any(k in t for k in ["softmax", "sum", "mean", "layer_norm", "norm", "reduce"]):
        return "REDUCTION"
    if any(k in t for k in ["view", "reshape", "permute", "transpose", "unsqueeze",
                             "squeeze", "split", "expand", "repeat", "slice", "contiguous"]):
        return "VIEW"
    if any(k in t for k in ["copy", "clone", "to"]):
        return "MEMORY"
    return "OTHER"


# ── 性能采集 ──────────────────────────────────────────────

def collect_prof(model, inputs, n_warmup=10, n_runs=30):
    """用 torch.profiler 采集每个 aten op 的性能 → {op_name: {avg_us, total_us, calls, memory_mb}}"""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(n_runs):
                model(*inputs)
        torch.cuda.synchronize()

    # 提取 aten op 级别的 device 时间 + 显存占用
    perf = {}
    for e in prof.key_averages():
        key = e.key
        # 只取 aten op（CPU 侧记录的高层 op 名），跳过 CUDA kernel 名
        if not key.startswith("aten::"):
            continue
        dev_time_us = e.self_device_time_total  # μs
        if dev_time_us <= 0 and e.self_device_memory_usage <= 0:
            continue
        perf[key] = {
            "total_us": dev_time_us,
            "avg_us": dev_time_us / e.count if e.count > 0 else 0,
            "calls": e.count,
            "memory_mb": e.self_device_memory_usage / 1024 / 1024,  # bytes → MB
        }
    return perf


# ── 图提取 + 附加性能 ─────────────────────────────────────

def extract_and_annotate(model, inputs, prof_perf, model_desc):
    """提取 FX 图，遍历节点，附加性能数据"""
    prog = torch.export.export(model, inputs)
    graph = prog.graph_module.graph

    nodes = []
    total_profiled_us = sum(v["total_us"] for v in prof_perf.values())

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = str(node.target)
        prof_name = fx_to_prof_name(target)

        # shape
        meta = node.meta.get("val", None)
        if meta is not None and hasattr(meta, "shape"):
            shape = str(list(meta.shape))
        else:
            shape = "?"

        # consumers
        consumers = [u for u in node.users]

        # 性能数据（名称映射 + 别名解析）
        resolved_key = resolve_prof_key(prof_name, prof_perf)
        if resolved_key is not None:
            p = prof_perf[resolved_key]
            avg_us = round(p["avg_us"], 1)
            total_us = round(p["total_us"], 1)
            calls = p["calls"]
            memory_mb = round(p.get("memory_mb", 0), 2)
            pct = round(p["total_us"] / total_profiled_us * 100, 2) if total_profiled_us else 0
            matched = True
        else:
            avg_us = total_us = 0
            calls = 0
            memory_mb = 0
            pct = 0
            matched = False

        nodes.append({
            "model": model_desc,
            "op_name": prof_name,
            "op_type": classify_op(target),
            "shape": shape,
            "n_consumers": len(consumers),
            "avg_us": avg_us,
            "total_us": total_us,
            "calls": calls,
            "memory_mb": memory_mb,
            "pct_total": pct,
            "matched": matched,
        })

    return nodes, prog.graph_module


# ── 输出 ─────────────────────────────────────────────────

def write_csv(nodes, output_path):
    fieldnames = ["model", "op_name", "op_type", "shape", "n_consumers",
                  "avg_us", "total_us", "calls", "memory_mb", "pct_total", "matched"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(nodes)
    print(f"性能标注 CSV saved: {output_path} ({len(nodes)} 节点)")


def print_heatmap(nodes, top_n=25):
    """控制台热力表：按 op 类型聚合去重，避免同名 op 重复累计"""
    # 按 op_name 聚合（同名 op 的总耗时只计一次）
    agg = {}
    for n in nodes:
        if not n["matched"]:
            continue
        key = n["op_name"]
        if key not in agg:
            agg[key] = {
                "op_name": key,
                "op_type": n["op_type"],
                "shapes": set(),
                "avg_us": n["avg_us"],
                "total_us": n["total_us"],
                "calls": n["calls"],
                "memory_mb": n["memory_mb"],
                "pct_total": n["pct_total"],
                "n_nodes": 0,
            }
        agg[key]["shapes"].add(n["shape"])
        agg[key]["n_nodes"] += 1

    matched = sorted(agg.values(), key=lambda a: -a["total_us"])

    print(f"\n{'='*90}")
    print(f"性能标注计算图 — op 类型聚合 (按 GPU 总耗时排序)")
    print(f"{'='*90}")
    print(f"{'op':<22s} {'type':<12s} {'total_us':>9s} {'calls':>6s} {'mem_mb':>8s} {'pct':>6s} {'节点':>4s}")
    print("-" * 90)
    for a in matched[:top_n]:
        print(f"{a['op_name']:<22s} {a['op_type']:<12s} {a['total_us']:>9.1f} "
              f"{a['calls']:>6d} {a['memory_mb']:>8.2f} {a['pct_total']:>5.2f}% {a['n_nodes']:>4d}")

    # 未匹配的节点
    unmatched = [n for n in nodes if not n["matched"]]
    if unmatched:
        print(f"\n  未匹配到性能数据的节点 ({len(unmatched)} 个):")
        types = {}
        for n in unmatched:
            types[n["op_name"]] = types.get(n["op_name"], 0) + 1
        for op, cnt in sorted(types.items(), key=lambda x: -x[1])[:15]:
            print(f"    {op:<30s} × {cnt}")


def write_viz(graph_module, nodes, output_path, color_by="time"):
    """生成带性能标注的 SVG。

    color_by: "time" → 颜色按耗时(红=慢), "memory" → 颜色按显存(红=占用大)
    """
    try:
        import pydot
        from torch.fx.passes.graph_drawer import FxGraphDrawer
    except ImportError:
        print("跳过可视化: pydot 未安装")
        return

    # 建立 op_name → (avg_us, memory_mb) 映射（用 op 类型级别）
    perf_map = {}
    mem_map = {}
    for n in nodes:
        if n["matched"]:
            perf_map[n["op_name"]] = n["avg_us"]
            mem_map[n["op_name"]] = n["memory_mb"]

    drawer = FxGraphDrawer(graph_module, "GPT-2 性能标注计算图")
    dot = drawer.get_dot_graph()

    # 选择上色维度
    if color_by == "memory":
        values = mem_map
        max_v = max(mem_map.values(), default=0.01) or 0.01
    else:
        values = perf_map
        max_v = max(perf_map.values(), default=0.01) or 0.01

    for dot_node in dot.get_nodes():
        label = dot_node.get_label()
        # 从 label 提取 target（格式: "...|target=aten.addmm.default\\n|..."）
        target = None
        for part in label.split("|"):
            if part.startswith("target="):
                target = part.split("=", 1)[1].replace("\\n", "").strip()
                break

        if target is None:
            continue

        # 规范化 target → op_name，匹配性能
        prof_name = fx_to_prof_name(target) if target.startswith(("aten.", "torch.ops.aten.")) else None
        if prof_name is None:
            continue

        v = values.get(prof_name, 0)
        if v <= 0:
            continue

        ratio = v / max_v
        red = int(255 * ratio)
        blue = int(255 * (1 - ratio))
        fillcolor = f"#{red:02x}60{blue:02x}"  # 60 = alpha (半透明)
        dot_node.set("fillcolor", fillcolor)
        dot_node.set("style", "filled")
        # 在 label 末尾追加耗时 + 显存
        avg_us = perf_map.get(prof_name, 0)
        mem = mem_map.get(prof_name, 0)
        new_label = label.rstrip("}") + f"|⏱ {avg_us:.1f}μs, 💾 {mem:.2f}MB\\n}}"
        dot_node.set("label", new_label)

    try:
        svg = dot.create_svg()
        with open(output_path, "wb") as f:
            f.write(svg)
        print(f"可视化 SVG saved: {output_path} ({len(svg)} bytes)")
    except Exception as e:
        print(f"可视化失败: {e}")


# ── 模型构建 ──────────────────────────────────────────────

def build_model(args):
    from transformers import GPT2Config, GPT2Model
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    if args.full:
        model = GPT2Model.from_pretrained("gpt2").cuda()
        inputs = (torch.randint(0, 50257, (1, args.seq_len)).cuda(),)
        desc = "GPT2 (12L/768h/124M)"
    else:
        config = GPT2Config(n_layer=1, n_head=4, n_embd=args.hidden, n_positions=64)
        block = GPT2Block(config, layer_idx=args.block).cuda()

        class BlockWrapper(nn.Module):
            def __init__(self, blk):
                super().__init__()
                self.blk = blk
            def forward(self, h, mask):
                return self.blk(h, attention_mask=mask)

        model = BlockWrapper(block)
        inputs = (torch.randn(1, args.seq_len, args.hidden).cuda(),
                  torch.ones(1, 1, args.seq_len, args.seq_len).cuda())
        desc = f"GPT2Block-{args.block} ({args.hidden}h)"
    return model, inputs, desc


def main():
    parser = argparse.ArgumentParser(description="将运行时性能附加到计算图")
    parser.add_argument("--block", type=int, default=0, help="分析第 N 层 (0-based)")
    parser.add_argument("--full", action="store_true", help="完整 GPT-2")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--viz", action="store_true", help="生成 SVG 可视化")
    parser.add_argument("--color-by", choices=["time", "memory"], default="memory",
                        help="SVG 上色维度: time=按耗时, memory=按显存 (默认 memory)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"[annotate] Device: {torch.cuda.get_device_name(0)}")

    model, inputs, desc = build_model(args)

    # 1. 采集性能
    print(f"[annotate] Profiling {desc} ({args.runs} runs) ...")
    perf = collect_prof(model, inputs, n_runs=args.runs)
    print(f"[annotate] 采集到 {len(perf)} 个 aten op 的性能数据")

    # 2. 提取图 + 附加性能
    print(f"[annotate] Extracting FX graph ...")
    nodes, graph_module = extract_and_annotate(model, inputs, perf, desc)

    # 3. 输出
    output_path = args.output or str(TABLE_DIR / "table_annotated_graph.csv")
    write_csv(nodes, output_path)
    print_heatmap(nodes)

    if args.viz:
        # 按选定维度上色
        suffix = "mem" if args.color_by == "memory" else "time"
        viz_path = str(GRAPH_DIR / f"gpt2_annotated_graph_{suffix}.svg")
        write_viz(graph_module, nodes, viz_path, color_by=args.color_by)

    # 4. 统计
    matched = sum(1 for n in nodes if n["matched"])
    print(f"\n[annotate] 图节点 {len(nodes)} 个，匹配到性能 {matched} 个 ({matched/len(nodes)*100:.0f}%)")


if __name__ == "__main__":
    main()
