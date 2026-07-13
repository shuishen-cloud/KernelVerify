"""下载 Qwen 2.5 1.5B 到项目目录。

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
    python scripts/download_qwen.py

下载位置: models/.cache/
预计大小: ~3.1GB
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = str(PROJECT_ROOT / "models" / ".cache")


def main():
    print("=" * 50)
    print("下载 Qwen 2.5 1.5B")
    print("=" * 50)
    print(f"目标目录: {CACHE_DIR}")
    print(f"预计大小: ~3.1GB")
    print()

    # 1. 加载配置 (秒级)
    print("[1/3] 下载 config.json...", flush=True)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen2.5-1.5B", cache_dir=CACHE_DIR
        )
        print(f"  n_layer={config.num_hidden_layers}, "
              f"n_embd={config.hidden_size}, "
              f"n_head={config.num_attention_heads}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return 1

    # 2. 下载模型权重 (~3.1GB)
    print("[2/3] 下载模型权重 (safetensors, ~3.1GB)...", flush=True)
    print("  可能持续数分钟，请耐心等待", flush=True)
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "Qwen/Qwen2.5-1.5B",
            torch_dtype=torch.float32,
            cache_dir=CACHE_DIR,
        )
        model.eval()
        elapsed = time.time() - t0
        speed_mbps = 3100 / elapsed if elapsed > 0 else 0
        print(f"  耗时: {elapsed:.0f}s ({elapsed/60:.1f}min) "
              f"≈ {speed_mbps:.1f} MB/s")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {n_params:,}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return 1

    # 3. 下载 tokenizer
    print("[3/3] 下载 tokenizer...", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B", cache_dir=CACHE_DIR
        )
        print(f"  vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        print(f"  ⚠️ tokenizer 下载失败: {e}")

    # 汇总
    cache_path = Path(CACHE_DIR)
    total_size = sum(
        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
    )
    print(f"\n缓存目录: {cache_path}")
    print(f"总大小: {total_size / 1e9:.2f} GB")
    print("文件列表:")
    for f in sorted(cache_path.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(cache_path)} ({f.stat().st_size/1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
