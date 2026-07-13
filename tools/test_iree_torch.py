#!/usr/bin/env python3
"""测试 IREE 对 Torch Dialect IR 的编译和执行。

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
    python tools/test_iree_torch.py [--cpu | --cuda | --all]
"""
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def test_add_relu(backend: str = "llvm-cpu"):
    """用 add_relu 测试 IREE 编译+执行，返回 (成功, 耗时ms)."""
    ir_path = PROJECT_ROOT / "mlir/handwritten/add_relu.mlir"
    mlir_text = ir_path.read_text()

    import iree.compiler, iree.runtime

    target = {"llvm-cpu": "local-task", "cuda": "cuda"}[backend]

    try:
        print(f"  编译 (target={backend})...", flush=True)
        t0 = time.time()
        compiled = iree.compiler.compile_str(
            mlir_text,
            target_backends=[backend],
            input_type=iree.compiler.InputType.TORCH,
        )
        compile_ms = (time.time() - t0) * 1000
        print(f"    编译完成: {len(compiled)} bytes, {compile_ms:.1f}ms")
    except Exception as e:
        print(f"    编译失败: {e}")
        return False, 0

    try:
        config = iree.runtime.Config(target)
        vm_module = iree.runtime.load_vm_module(
            iree.runtime.VmModule.from_flatbuffer(config.vm_instance, compiled),
            config,
        )

        a = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
        b = np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32)

        t0 = time.time()
        result = vm_module.main(a, b)
        exec_ms = (time.time() - t0) * 1000

        expected = np.maximum(a + b, 0.0)
        ok = np.allclose(result.to_host(), expected, atol=1e-5)
        if ok:
            print(f"    ✅ 结果正确, 执行: {exec_ms:.3f}ms")
        else:
            print(f"    ❌ 结果不匹配: got={result.to_host()}, expected={expected}")
        return ok, exec_ms
    except Exception as e:
        print(f"    运行失败: {e}")
        return False, 0


def check_cuda_available() -> bool:
    """检查 IREE CUDA 后端是否可用."""
    import iree.compiler
    try:
        targets = iree.compiler.query_available_targets()
        print(f"IREE 可用 targets: {targets}")
        return "cuda" in str(targets).lower() or "vulkan" in str(targets).lower()
    except Exception as e:
        print(f"查询 targets 失败: {e}")
        return False


def main():
    print("=" * 50)
    print("IREE Torch IR 编译执行测试")
    print("=" * 50)
    print()

    # 检查可用后端
    check_cuda_available()
    print()

    # 测试 CPU
    print("[Test 1] CPU 后端 (add_relu)")
    cpu_ok, cpu_ms = test_add_relu("llvm-cpu")
    print()

    # 尝试 CUDA
    print("[Test 2] CUDA 后端 (add_relu)")
    cuda_ok, cuda_ms = test_add_relu("cuda")
    print()

    # 汇总
    print("=" * 50)
    print(f"CPU:  {'✅' if cpu_ok else '❌'} ({cpu_ms:.1f}ms)")
    print(f"CUDA: {'✅' if cuda_ok else '❌'} ({cuda_ms:.1f}ms)")
    return 0 if (cpu_ok or cuda_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
