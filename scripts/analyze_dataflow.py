#!/usr/bin/env python3
"""算子数据流与计算图分析工具。

分析 PyTorch 模型计算图中的 producer→consumer 关系，
识别可融合的算子链、多 consumer 冲突、跨库边界。

Usage:
    # 分析 GPT-2 单层
    python scripts/analyze_dataflow.py --model gpt2 --block 0

    # 分析自定义模型
    python scripts/analyze_dataflow.py --export my_model.py:MyModel

输出 (stdout + benchmarks/tables/table_c_subgraph_ops.csv):
    1. 子图拓扑: 每个 op 的 shape + consumer 列表
    2. 融合候选: 单 consumer 的 elementwise 链
    3. 冲突点: 多 consumer、跨库调用的 op
"""
import sys
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = PROJECT_ROOT / "benchmarks" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OpInfo:
    """单个算子信息"""
    name: str           # aten::addmm.default
    op_type: str        # MATMUL / ELEMENTWISE / REDUCTION / VIEW / OTHER
    shape: str          # "[1, 8, 768]"
    consumers: list = field(default_factory=list)  # [(consumer_name, shape)]
    producer: str = ""


def classify_op(target: str) -> str:
    """归类算子类型"""
    t = target.lower()
    if any(k in t for k in ["addmm", "mm", "bmm", "scaled_dot_product", "matmul"]):
        return "MATMUL"
    if any(k in t for k in ["mul", "add.tensor", "div", "sub", "pow", "tanh", "gelu",
                             "erf", "where", "clamp", "relu", "sigmoid"]):
        return "ELEMENTWISE"
    if any(k in t for k in ["softmax", "sum", "mean", "layer_norm", "norm", "reduce",
                             "max.dim", "min.dim", "argmax"]):
        return "REDUCTION"
    if any(k in t for k in ["view", "reshape", "permute", "transpose", "unsqueeze",
                             "squeeze", "split", "expand", "repeat"]):
        return "VIEW"
    if any(k in t for k in ["copy", "clone", "contiguous", "to"]):
        return "MEMORY"
    return "OTHER"


def extract_dataflow(exported_program, model_name="model"):
    """从 torch.export 的 ExportedProgram 提取完整数据流。

    遍历 graph.nodes，对每个 call_function 节点：
    - 提取 target (算子名)
    - 提取 shape (从 node.meta['val'])
    - 提取 consumers (node.users)
    - 分类算子类型
    """
    graph = exported_program.graph_module.graph
    ops = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue

        target = str(node.target)
        op_type = classify_op(target)

        # Shape from node metadata
        meta = node.meta.get("val", None)
        if meta is not None:
            if hasattr(meta, "shape"):
                shape = str(list(meta.shape))
            elif isinstance(meta, (list, tuple)) and len(meta) > 0:
                shape = str(list(meta[0].shape)) if hasattr(meta[0], "shape") else "?"
            else:
                shape = "?"
        else:
            shape = "?"

        # Consumers
        consumers = []
        for user in node.users:
            user_meta = user.meta.get("val", None)
            if user_meta is not None and hasattr(user_meta, "shape"):
                user_shape = str(list(user_meta.shape))
            else:
                user_shape = "?"
            consumers.append({
                "name": str(user.target)[:80],
                "shape": user_shape,
                "op_type": classify_op(str(user.target)),
            })

        ops.append(OpInfo(
            name=target,
            op_type=op_type,
            shape=shape,
            consumers=consumers,
        ))

    return ops


def find_fusion_chains(ops: list[OpInfo]) -> list[dict]:
    """识别可融合的算子链。

    规则：
    - 连续的 ELEMENTWISE op 链（无 MATMUL/REDUCTION 打断）
    - 每个中间 op 只有 1 个 consumer
    - 不跨 VIEW 边界（view 通常不可与 elementwise 融合）
    """
    chains = []
    current_chain = []
    chain_start_shape = ""

    for op in ops:
        n_consumers = len(op.consumers)
        is_single_consumer = n_consumers == 1
        is_elementwise = op.op_type == "ELEMENTWISE"
        next_is_elementwise = (
            n_consumers == 1
            and op.consumers[0]["op_type"] == "ELEMENTWISE"
        )

        if is_elementwise and next_is_elementwise and is_single_consumer:
            if not current_chain:
                chain_start_shape = op.shape
            current_chain.append(op.name.split("::")[-1][:40])
        else:
            if len(current_chain) >= 2:  # 至少 2 个 op 的链才值得融合
                # 检查：最后一个 consumer 是否也是 elementwise
                last = current_chain[-1] if current_chain else ""
                chains.append({
                    "length": len(current_chain),
                    "start_shape": chain_start_shape,
                    "chain": " → ".join(current_chain),
                    "savings": f"{-len(current_chain) - 1} kernel launch, {-len(current_chain)} 中间tensor",
                })
            current_chain = []

    # Don't forget the last chain
    if len(current_chain) >= 2:
        chains.append({
            "length": len(current_chain),
            "start_shape": chain_start_shape,
            "chain": " → ".join(current_chain),
            "savings": f"{-len(current_chain) - 1} kernel launch, {-len(current_chain)} 中间tensor",
        })

    return chains


def find_conflicts(ops: list[OpInfo]) -> list[dict]:
    """识别融合冲突点：多 consumer 或跨库边界"""
    conflicts = []
    for op in ops:
        n = len(op.consumers)
        if n > 1:
            # 多 consumer — 需要检查 consumer 是否都是 elementwise
            consumer_types = [c["op_type"] for c in op.consumers]
            conflicts.append({
                "op": op.name.split("::")[-1][:50],
                "shape": op.shape,
                "issue": f"多consumer ({n}个): {', '.join(consumer_types)}",
                "fusible": "❌" if len(set(consumer_types)) > 1 or "MATMUL" in consumer_types else "⚠️ 需检查",
            })
        if op.op_type == "VIEW" and len(op.consumers) > 0:
            if any(c["op_type"] == "MATMUL" for c in op.consumers):
                conflicts.append({
                    "op": op.name.split("::")[-1][:50],
                    "shape": op.shape,
                    "issue": "VIEW→MATMUL 跨库边界",
                    "fusible": "❌",
                })
    return conflicts


def print_analysis(ops, chains, conflicts):
    """Human-readable analysis output"""
    # Op type summary
    from collections import Counter
    type_counts = Counter(op.op_type for op in ops)
    print(f"\n{'='*60}")
    print(f"算子分类统计 (共 {len(ops)} 个算子)")
    print(f"{'='*60}")
    for t in ["MATMUL", "ELEMENTWISE", "REDUCTION", "VIEW", "MEMORY", "OTHER"]:
        if type_counts.get(t, 0) > 0:
            print(f"  {t:<20s} × {type_counts[t]}")

    # Top ops by type
    print(f"\n{'='*60}")
    print(f"MATMUL 算子明细")
    print(f"{'='*60}")
    print(f"  {'op':<50s} {'shape':>15s} {'consumers':>10s}")
    for op in ops:
        if op.op_type == "MATMUL":
            consumer_short = ", ".join(c["op_type"][:8] for c in op.consumers[:2])
            print(f"  {op.name.split('::')[-1][:48]:<50s} {op.shape:>15s} {consumer_short:>10s}")

    # Producer→Consumer chains (key ops only)
    print(f"\n{'='*60}")
    print(f"关键数据流 (producer → consumer)")
    print(f"{'='*60}")
    for op in ops:
        if op.op_type in ("MATMUL", "REDUCTION") or len(op.consumers) > 1:
            n_cons = len(op.consumers)
            flag = " ⚠️" if n_cons > 1 else ""
            print(f"  [{op.op_type}] {op.name.split('::')[-1][:45]} {op.shape}")
            for c in op.consumers[:3]:
                print(f"    → [{c['op_type']}] {c['name'][:40]} {c['shape']}")
            if n_cons > 3:
                print(f"    ... +{n_cons - 3} more")

    # Fusion chains
    print(f"\n{'='*60}")
    print(f"可融合的 ELEMENTWISE 链 (≥2 ops, 单consumer)")
    print(f"{'='*60}")
    if chains:
        for i, ch in enumerate(chains):
            print(f"  [{i+1}] {ch['length']} ops, start={ch['start_shape']}")
            print(f"      {ch['chain'][:100]}")
            print(f"      融合收益: {ch['savings']}")
    else:
        print("  (未发现可融合的 elementwise 链)")

    # Conflicts
    print(f"\n{'='*60}")
    print(f"融合冲突点")
    print(f"{'='*60}")
    if conflicts:
        for c in conflicts:
            print(f"  {c['op']:<40s} {c['shape']:>12s}  {c['issue']:<40s} {c['fusible']}")
    else:
        print("  (未发现融合冲突)")


def main():
    parser = argparse.ArgumentParser(description="算子数据流与计算图分析")
    parser.add_argument("--model", choices=["gpt2", "gpt2-tiny"], default="gpt2-tiny",
                        help="分析哪个模型")
    parser.add_argument("--block", type=int, default=0, help="只分析第 N 层 (0-based)")
    parser.add_argument("--seq-len", type=int, default=8, help="序列长度")
    parser.add_argument("--hidden", type=int, default=128, help="hidden dim (tiny=128, full=768)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"[analyze_dataflow] model={args.model}, block={args.block}")

    # Build a single GPT-2 block for analysis
    from transformers import GPT2Config
    from transformers.models.gpt2.modeling_gpt2 import GPT2Block

    n_embd = args.hidden
    config = GPT2Config(n_layer=args.block + 1, n_head=4, n_embd=n_embd,
                        n_positions=64, n_ctx=64)
    full_model = torch.nn.Module()  # placeholder type
    block = GPT2Block(config, layer_idx=args.block)

    # Trace the block
    hidden = torch.randn(1, args.seq_len, n_embd)
    attn_mask = torch.ones(1, 1, args.seq_len, args.seq_len)

    class BlockWrapper(nn.Module):
        def __init__(self, blk):
            super().__init__()
            self.blk = blk
        def forward(self, h, mask):
            return self.blk(h, attention_mask=mask)

    wrapped = BlockWrapper(block)
    prog = torch.export.export(wrapped, (hidden, attn_mask))
    ops = extract_dataflow(prog, f"GPT2Block-{args.block}")

    chains = find_fusion_chains(ops)
    conflicts = find_conflicts(ops)

    print_analysis(ops, chains, conflicts)

    # Write table C
    output = args.output or str(TABLE_DIR / "table_c_subgraph_ops.csv")
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["op_name", "op_type", "shape", "n_consumers",
                                           "consumers", "fusion_notes"])
        w.writeheader()
        for op in ops:
            consumer_str = "; ".join(
                f"{c['name'][:40]} [{c['op_type']}]" for c in op.consumers[:3]
            )
            n = len(op.consumers)
            notes = ""
            if n > 1:
                notes = f"⚠️ multi-consumer ({n})"
            elif op.op_type == "ELEMENTWISE" and n == 1 and op.consumers[0]["op_type"] == "ELEMENTWISE":
                notes = "✅ chain candidate"
            elif op.op_type == "VIEW":
                notes = "reshape boundary"
            w.writerow({
                "op_name": op.name,
                "op_type": op.op_type,
                "shape": op.shape,
                "n_consumers": n,
                "consumers": consumer_str,
                "fusion_notes": notes,
            })

    print(f"\n表C (新版) saved: {output}")

    # Fusion chain summary
    if chains:
        chain_path = str(TABLE_DIR / "table_c_fusion_chains.csv")
        with open(chain_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["length", "start_shape", "chain", "savings"])
            w.writeheader()
            w.writerows(chains)
        print(f"融合链 saved: {chain_path}")


if __name__ == "__main__":
    main()
