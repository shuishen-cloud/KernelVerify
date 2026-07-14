#!/usr/bin/env python3
"""用 IREE 编译并执行 GPT-2 模型

用法:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm

    # 运行 tiny GPT-2 (1层, 最快)
    python scripts/test_iree_gpt2.py --tiny

    # 运行完整 GPT-2 (12层, 124M参数, CPU可能几分钟)
    python scripts/test_iree_gpt2.py --full

    # 指定输入长度
    python scripts/test_iree_gpt2.py --tiny --seq-len 8
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent


def load_model_and_ref():
    """加载 tiny GPT-2 PyTorch 模型作为参考."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=50257,
        n_embd=64,       # 极小: 64 hidden
        n_layer=1,        # 1 层 transformer
        n_head=4,         # 4 个 attention head
        n_positions=64,   # 最大 64 token
    )
    model = GPT2LMHeadModel(config).eval()
    return model


def export_mlir(model: GPT2LMHeadModel, seq_len: int, output_path: str):
    """导出模型到 Torch Dialect MLIR."""
    import torch_mlir.fx as fx

    input_ids = torch.randint(0, 50257, (1, seq_len))
    prog = torch.export.export(model, (input_ids,))
    prog = prog.run_decompositions()

    result = fx.export_and_import(
        prog,
        output_type="torch",
    )
    Path(output_path).write_text(str(result))
    print(f"  MLIR exported -> {output_path} ({len(str(result))} chars)")
    return input_ids, model(input_ids).logits.detach()


def test_iree_compile_run(mlir_path: str, input_data: dict, backend: str = "llvm-cpu"):
    """用 IREE 编译 MLIR 并执行推理.

    Args:
        mlir_path: Torch Dialect MLIR 文件路径
        input_data: dict of {arg_name: np.ndarray}
        backend: "llvm-cpu" (CPU) 或 "cuda"

    Returns:
        (success, output_array, compile_ms, exec_ms)
    """
    import iree.compiler, iree.runtime

    mlir_text = Path(mlir_path).read_text()

    # === 编译 ===
    target = {"llvm-cpu": "local-task", "cuda": "cuda"}[backend]
    print(f"  编译 ({backend}, {len(mlir_text)} chars)...", flush=True)
    t0 = time.time()
    try:
        compiled = iree.compiler.compile_str(
            mlir_text,
            target_backends=[backend],
            input_type=iree.compiler.InputType.TORCH,
        )
        compile_ms = (time.time() - t0) * 1000
        print(f"    .vmfb: {len(compiled):,} bytes, {compile_ms:.1f}ms")
    except Exception as e:
        print(f"    编译失败: {e}")
        return False, None, 0, 0

    # === 执行 ===
    try:
        config = iree.runtime.Config(target)
        vm_module = iree.runtime.load_vm_module(
            iree.runtime.VmModule.from_flatbuffer(config.vm_instance, compiled),
            config,
        )

        t0 = time.time()
        result = vm_module.main(*input_data.values())
        exec_ms = (time.time() - t0) * 1000

        output = result.to_host()
        print(f"    执行: {exec_ms:.1f}ms, output shape={output.shape}")
        return True, output, compile_ms, exec_ms
    except Exception as e:
        print(f"    运行失败: {e}")
        return False, None, compile_ms, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", action="store_true", help="运行 tiny GPT-2 (1层)")
    parser.add_argument("--full", action="store_true", help="运行完整 GPT-2 (已有 IR)")
    parser.add_argument("--seq-len", type=int, default=4, help="输入序列长度")
    parser.add_argument("--backend", default="llvm-cpu", choices=["llvm-cpu", "cuda"])
    args = parser.parse_args()

    if not args.tiny and not args.full:
        args.tiny = True  # 默认 tiny

    print("=" * 60)
    print(f"IREE GPT-2 推理测试 ({'tiny' if args.tiny else 'full'}, {args.backend})")
    print("=" * 60)

    # === 获取 PyTorch 参考输出 ===
    print("\n[1] PyTorch 参考推理...")
    model = load_model_and_ref()
    input_ids = torch.randint(0, 50257, (1, args.seq_len))
    with torch.no_grad():
        ref_output = model(input_ids).logits
    print(f"  输入: {input_ids.shape}, 输出: {ref_output.shape}")

    # === 导出或使用已有 MLIR ===
    if args.tiny:
        mlir_path = PROJECT_ROOT / "mlir/exported/gpt2_tiny_torch.mlir"
        if not mlir_path.exists():
            print("\n[2] 导出 tiny GPT-2 Torch MLIR...")
            export_mlir(model, args.seq_len, str(mlir_path))
        else:
            print(f"\n[2] 使用已有 MLIR: {mlir_path}")
    else:
        mlir_path = PROJECT_ROOT / "mlir/exported/gpt2_full_torch.mlir"
        print(f"\n[2] 使用已有 MLIR: {mlir_path}")
        if not mlir_path.exists():
            print("  ❌ 找不到 gpt2_full_torch.mlir，请先导出")
            return 1

    # === IREE 编译 + 执行 ===
    print(f"\n[3] IREE 编译 + 执行 ({args.backend})...")
    input_data = {"arg0": input_ids.numpy().astype(np.int64)}
    ok, output, compile_ms, exec_ms = test_iree_compile_run(
        str(mlir_path), input_data, args.backend
    )

    if not ok:
        return 1

    # === 验证结果 ===
    print(f"\n[4] 结果验证...")
    iree_out = torch.from_numpy(output)
    diff = (ref_output - iree_out).abs().max().item()
    allclose = torch.allclose(ref_output, iree_out, atol=1e-3)

    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  IREE output shape:      {iree_out.shape}")
    print(f"  Max difference:         {diff:.6f}")
    print(f"  Allclose (atol=1e-3):   {'✅ PASS' if allclose else '❌ FAIL'}")

    # === 汇总 ===
    print(f"\n{'='*60}")
    print(f"  编译: {compile_ms:.0f}ms | 执行: {exec_ms:.1f}ms")
    print(f"  .vmfb: {Path(mlir_path).with_suffix('.vmfb').exists()} ({len(compiled) if ok else 0:,} bytes)")
    print(f"  精度: {'✅ 一致' if allclose else f'❌ 偏差={diff:.4f}'}")
    return 0 if allclose else 1


if __name__ == "__main__":
    sys.exit(main())
