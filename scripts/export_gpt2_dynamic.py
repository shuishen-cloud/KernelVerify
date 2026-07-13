"""W9: 动态 shape 导出 GPT-2 → Torch Dialect + Linalg.

动态 shape 让序列长度和 batch size 保持为符号变量，
编译器不会展开循环，保留 scf.for 结构。

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
    python scripts/export_gpt2_dynamic.py
"""
import torch, time
from pathlib import Path
from transformers import GPT2Model
from torch.export import Dim
from torch_mlir import fx

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # 1. 加载模型
    print("[1/4] 加载 GPT-2 small (12层/124M)...", flush=True)
    t0 = time.time()
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}, 耗时: {time.time()-t0:.1f}s")

    # 示例输入 + 动态维度
    example_input = torch.randint(0, 50257, (2, 16))
    batch = Dim("batch", min=1, max=32)
    seq = Dim("seq", min=1, max=512)
    dynamic_shapes = {"input_ids": {0: batch, 1: seq}}
    print(f"  示例输入: {tuple(example_input.shape)}, 动态轴: batch(1-32), seq(1-512)")

    # 2. 导出 Torch Dialect
    print("[2/4] 导出 Torch Dialect (dynamic)...", flush=True)
    t0 = time.time()
    result = fx.export_and_import(
        model, example_input, output_type="torch",
        dynamic_shapes=dynamic_shapes,
    )
    elapsed = time.time() - t0
    mlir = result.operation.get_asm()
    print(f"  ✅ {mlir.count(chr(10))+1} 行, {mlir.count('torch.operator')} fallback, {elapsed:.1f}s")
    out = PROJECT_ROOT / "mlir/exported/gpt2_dynamic_torch.mlir"
    out.write_text(mlir)
    print(f"  → {out}")

    # 3. Lowering 到 Linalg
    print("[3/4] Lowering 到 Linalg (dynamic)...", flush=True)
    t0 = time.time()
    result = fx.export_and_import(
        model, example_input, output_type="linalg-on-tensors",
        dynamic_shapes=dynamic_shapes,
    )
    elapsed = time.time() - t0
    mlir = result.operation.get_asm()
    n_scf = mlir.count("scf.")
    print(f"  ✅ {mlir.count(chr(10))+1} 行, {mlir.count('torch.operator')} fallback, {elapsed:.1f}s")
    print(f"  scf ops: {n_scf} (动态 shape 关键指标)")
    out = PROJECT_ROOT / "mlir/lowered/gpt2_dynamic_linalg.mlir"
    out.write_text(mlir)
    print(f"  → {out}")

    # 4. 检查
    print("[4/4] 结构检查...", flush=True)
    lines = mlir.split("\n")
    print(f"  符号维度 '?': {sum(1 for l in lines if '?' in l)} 处")
    for op in ["scf.for", "scf.parallel", "scf.while"]:
        c = mlir.count(op)
        if c: print(f"  {op}: {c}")

    func_line = next((l for l in lines if "func.func @" in l), "")
    if "?" in func_line:
        print("  ✅ 函数签名含 '?' — 动态导出成功")
    else:
        print("  ⚠️ 函数签名无 '?' — 可能未被标记为动态")


if __name__ == "__main__":
    main()
