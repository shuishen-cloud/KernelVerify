#!/usr/bin/env python3
"""演示：在 PyTorch FX 图层面操作计算图。

计算图的本质: 每个 node 是一个算子，每条 edge 是数据流向。
操作图 = 增删改 node/edge。

你所处的状态:
  ① FX Graph 层 (Python) — 本脚本演示
     torch.export → ExportedProgram.graph → 增删改 node

  ② MLIR 层 (C++ Pass) — 你 W10 的 CountLinalgOps
     在这一层用 C++ 操作 MLIR op

  ③ IREE 编译流程 (Python)
     iree.compiler.compile_str → 自定义 pipeline

用法:
    conda activate novel_llm
    python scripts/demo_graph_manipulation.py
"""
import torch
import torch.fx as fx


def label(n):
    """简短描述一个 node 的计算."""
    t = n.target
    return t.__name__ if hasattr(t, '__name__') else str(t)


# ============================================================
def demo_1_print_graph():
    """演示 1: 计算图长什么样？

    MyModel 的 FX 图:
      x → add(1.0) → mul(2.0) → relu → add(x+1.0) → output
    """
    print("=" * 60)
    print("演示 1: 打印计算图结构")

    class MyModel(torch.nn.Module):
        def forward(self, x):
            a = x + 1.0        # call_function add
            b = a * 2.0        # call_function mul
            c = b.relu()       # call_method   relu
            d = c + a          # call_function add (residual)
            return d

    gm = fx.symbolic_trace(MyModel())
    gm.graph.print_tabular()
    print(f"\n共 {len(list(gm.graph.nodes))} 个节点 (含 placeholder 和 output)")


# ============================================================
def demo_2_dce():
    """演示 2: 死代码消除 (Dead Code Elimination).

    手动构建图，插入一个不被输出引用的分支，然后消除它。
    这和 MLIR 的 -canonicalize 做的是同一件事。
    """
    print("\n" + "=" * 60)
    print("演示 2: 死代码消除 (DCE)")

    graph = fx.Graph()
    x = graph.placeholder("x")

    # 活代码: x + 1 → * 2 → output
    add_n = graph.call_function(torch.add, args=(x, torch.tensor(1.0)))
    mul_n = graph.call_function(torch.mul, args=(add_n, torch.tensor(2.0)))

    # 死代码: 算了一个 sin，但没人用它
    dead = graph.call_function(torch.sin, args=(mul_n,))

    # 另一个死代码: 两个常量相加但结果也没人用
    c1 = graph.call_function(torch.tensor, args=(1.0,))
    c2 = graph.call_function(torch.tensor, args=(2.0,))
    dead2 = graph.call_function(torch.add, args=(c1, c2))

    graph.output(mul_n)

    print("优化前:")
    for n in graph.nodes:
        if n.op != "output":
            print(f"  {n.op:15s} {n.name:12s} = {label(n)}")

    before = len(list(graph.nodes))
    graph.eliminate_dead_code()
    graph.lint()
    after = len(list(graph.nodes))

    print(f"\n优化后 (DCE, 消除 {before - after} 个死节点):")
    for n in graph.nodes:
        if n.op != "output":
            print(f"  {n.op:15s} {n.name:12s} = {label(n)}")
    print("  sin + 常量 add 被删除 → 只有被 output 引用的节点保留")


# ============================================================
def demo_3_fusion():
    """演示 3: 算子融合 (Operator Fusion).

    找到 add → relu 的 pattern，合并为一个节点。
    MLIR 的 -linalg-fuse-elementwise-ops 在 IR 层做同样的事。
    """
    print("\n" + "=" * 60)
    print("演示 3: 算子融合 — add+relu 合并")

    graph = fx.Graph()
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    add_n = graph.call_function(torch.add, args=(x, y))
    relu_n = graph.call_function(torch.relu, args=(add_n,))
    graph.output(relu_n)

    print("融合前 (2 个计算节点):")
    for n in graph.nodes:
        if n.op not in ("placeholder", "output"):
            print(f"  {n.name:8s} = {label(n)}")

    # 融合变换: 找 add→relu 链，替换 consumer
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.add:
            continue
        for consumer in list(node.users):
            if consumer.op == "call_function" and consumer.target == torch.relu:
                print(f"\n  发现可融合: {node.name}(add) → {consumer.name}(relu)")
                with graph.inserting_before(consumer):
                    fused = graph.call_function(torch.relu, args=(node,))
                consumer.replace_all_uses_with(fused)

    graph.eliminate_dead_code()
    graph.lint()

    print("\n融合后:")
    for n in graph.nodes:
        if n.op not in ("placeholder", "output"):
            print(f"  {n.name:8s} = {label(n)}")
    print("  fused 节点在真实场景中 = 单个 CUDA kernel, 一次访存完成 add+relu")


# ============================================================
def demo_4_export():
    """演示 4: torch.export 产生的真实模型图 — 你项目用的方式。"""
    print("\n" + "=" * 60)
    print("演示 4: torch.export 产生的真实 FX 图")

    class LinearReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return torch.relu(self.linear(x))

    model = LinearReLU().eval()
    example = torch.randn(2, 10)

    prog = torch.export.export(model, (example,))
    gm = prog.graph_module
    print(f"图中 {len(list(gm.graph.nodes))} 个节点:")
    gm.graph.print_tabular()

    print("\n这就是 export_and_import() 内部产生的图结构。")
    print("在这里你可以: 新增节点 / 删除节点 / 替换节点 / 改变边")
    print("全部在 Python 层面完成，不需要写 C++。")


# ============================================================
def main():
    demo_1_print_graph()
    demo_2_dce()
    demo_3_fusion()
    demo_4_export()

    print("\n" + "=" * 60)
    print("总结: 你可以操作计算图的三个层面")
    print()
    print("  1  FX Graph 层  (Python)")
    print("     torch.export -> ExportedProgram.graph -> 增删改 node")
    print()
    print("  2  MLIR 层  (C++ Pass)")
    print("     你 W10 的 CountLinalgOps 就是 MLIR Pass")
    print()
    print("  3  IREE 编译流程  (Python)")
    print("     iree.compiler.compile_str -> 自定义 pipeline")


if __name__ == "__main__":
    main()
