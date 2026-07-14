#!/usr/bin/env python3
"""将计算图渲染为 SVG 图像。

用法:
    conda activate novel_llm
    python scripts/visualize_graph.py                # 默认生成 SVG
    python scripts/visualize_graph.py --format png   # 生成 PNG (需 graphviz)
"""
import argparse
import subprocess
from pathlib import Path

import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

OUTPUT_DIR = Path("mlir/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def render_svg(dot_source: str, output_path: str) -> bool:
    """用 graphviz dot 渲染 SVG/PNG."""
    fmt = Path(output_path).suffix.lstrip(".")
    try:
        result = subprocess.run(
            ["dot", f"-T{fmt}", "-o", output_path],
            input=dot_source, text=True, capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def visualize_model(name: str, model: torch.nn.Module,
                    example_inputs, fmt: str = "svg"):
    """对模型进行 symbolic_trace 并渲染图."""
    try:
        gm = fx.symbolic_trace(model)
    except Exception as e:
        print(f"  {name}: symbolic_trace 失败: {e}")
        return

    # FxGraphDrawer 生成 dot 源码
    drawer = FxGraphDrawer(gm, name)
    dot_source = drawer.get_dot_graph().to_string()

    output_path = OUTPUT_DIR / f"{name}.{fmt}"
    # 也保存一份 dot 源码，方便手动修改
    dot_path = OUTPUT_DIR / f"{name}.dot"
    dot_path.write_text(dot_source)

    if render_svg(dot_source, str(output_path)):
        size = Path(output_path).stat().st_size
        print(f"  {name}: -> {output_path} ({size:,} bytes)")
    else:
        print(f"  {name}: 渲染失败，dot 源文件在 {dot_path}")


# ============================================================
def demo1_simple_model(fmt: str):
    """可视化: add -> mul -> relu -> add (简单计算图)."""
    print(f"\n[1] 简单计算图: add -> mul -> relu -> add")

    class MyModel(torch.nn.Module):
        def forward(self, x):
            a = x + 1.0
            b = a * 2.0
            c = b.relu()
            return c + a

    visualize_model("simple_graph", MyModel(), (torch.randn(1),), fmt)


def demo2_linear_relu(fmt: str):
    """可视化: Linear -> ReLU (真实模型)."""
    print(f"\n[2] Linear -> ReLU")

    class LinearReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)

        def forward(self, x):
            return torch.relu(self.linear(x))

    visualize_model("linear_relu", LinearReLU(), (torch.randn(2, 10),), fmt)


def demo3_exported_graph(fmt: str):
    """可视化: torch.export 导出的真实 Aten 图."""
    print(f"\n[3] torch.export 产生的 Aten 图")

    class ConvBNReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(16)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)))

    model = ConvBNReLU().eval()
    example = torch.randn(1, 3, 32, 32)
    prog = torch.export.export(model, (example,))

    gm = prog.graph_module
    drawer = FxGraphDrawer(gm, "conv_bn_relu_exported")
    dot_source = drawer.get_dot_graph().to_string()

    output_path = OUTPUT_DIR / f"conv_bn_relu_exported.{fmt}"
    (OUTPUT_DIR / "conv_bn_relu_exported.dot").write_text(dot_source)

    if render_svg(dot_source, str(output_path)):
        size = Path(output_path).stat().st_size
        print(f"  -> {output_path} ({size:,} bytes)")
        print(f"  注意: 这是 torch.export 导出的图，所有参数都是 placeholder")
        print(f"  这正是你项目里 export_and_import 的输入")
    else:
        print(f"  渲染失败")


def demo4_gpt2_exported(fmt: str):
    """可视化: 极小 GPT-2 的 exported 图."""
    print(f"\n[4] 极小 GPT-2 的 exported 图")
    print(f"  (多层嵌套，图较大)")

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=50257,
        n_embd=64,
        n_layer=1,
        n_head=4,
        n_positions=64,
    )
    model = GPT2LMHeadModel(config).eval()
    example = torch.randint(0, 50257, (1, 4))
    prog = torch.export.export(model, (example,))

    gm = prog.graph_module
    drawer = FxGraphDrawer(gm, "gpt2_tiny_exported")
    dot_source = drawer.get_dot_graph().to_string()

    output_path = OUTPUT_DIR / f"gpt2_tiny_exported.{fmt}"
    (OUTPUT_DIR / "gpt2_tiny_exported.dot").write_text(dot_source)

    if render_svg(dot_source, str(output_path)):
        size = Path(output_path).stat().st_size
        print(f"  -> {output_path} ({size:,} bytes)")
    else:
        print(f"  渲染失败，dot 源文件可用")


def main():
    parser = argparse.ArgumentParser(description="可视化计算图")
    parser.add_argument("--format", default="svg", choices=["svg", "png", "pdf"])
    parser.add_argument("--all", action="store_true", default=True)
    args = parser.parse_args()

    print(f"输出格式: {args.format}")
    print(f"输出目录: {OUTPUT_DIR.resolve()}/")

    demo1_simple_model(args.format)
    demo2_linear_relu(args.format)
    demo3_exported_graph(args.format)
    demo4_gpt2_exported(args.format)

    print(f"\n{'='*50}")
    print(f"所有图片已保存到 {OUTPUT_DIR.resolve()}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.suffix[1:] in ("svg", "png", "pdf"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
