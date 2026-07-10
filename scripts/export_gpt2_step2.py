"""W6: Export GPT-2 model to Torch Dialect MLIR.

Step 2 - First bare export attempt with minimal config.

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
    python scripts/export_gpt2_step2.py
"""
import torch
from transformers import GPT2Model, GPT2Config
from torch_mlir import fx


def main() -> None:
    # 极小配置，快速迭代
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64)
    print(f"[config] n_layer={config.n_layer}, n_head={config.n_head}, "
          f"n_embd={config.n_embd}, n_positions={config.n_positions}")

    model = GPT2Model(config)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] 参数量: {n_params:,}")

    # 小序列长度
    example_input = torch.randint(0, config.vocab_size, (1, 8))
    print(f"[input] shape={example_input.shape}")

    print("[export] 开始导出 (output_type='torch')...", flush=True)
    try:
        result = fx.export_and_import(
            model, example_input,
            output_type="torch",
            verbose=True,
        )
        print("\n✅ 导出成功!")
        mlir_text = result.operation.get_asm()
        print(f"[结果] {mlir_text.count(chr(10)) + 1} lines")
        # 只打印前 100 行
        lines = mlir_text.split("\n")
        for i, line in enumerate(lines[:100]):
            print(f"  {line}")
        if len(lines) > 100:
            print(f"  ... ({len(lines) - 100} more lines)")
    except Exception as e:
        print(f"\n❌ 导出失败: {e}")


if __name__ == "__main__":
    main()
