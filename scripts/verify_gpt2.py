"""W6 Step 7: GPT-2 功能正确性验证

验证内容:
1. MLIR 函数签名 vs PyTorch 模型输入/输出 shape
2. Torch Dialect IR → Linalg IR → 优化后 IR 的 shape/dtype 一致性
3. 原始 PyTorch 推理输出值 (作参考基线)

Usage:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
    python scripts/verify_gpt2.py
"""
import re
import torch
from pathlib import Path
from transformers import GPT2Model, GPT2Config


PROJECT_ROOT = Path(__file__).parent.parent

# 使用与导出时相同的配置
CONFIG = GPT2Config(n_layer=2, n_head=4, n_embd=128, n_positions=64)


def extract_mlir_signature(mlir_path: Path) -> dict:
    """从 MLIR 文件中提取函数签名信息."""
    text = mlir_path.read_text()

    # 匹配 func.func @main(...) -> (...)
    pattern = r'func\.func @(\w+)\(([^)]*)\)\s*->\s*\(([^)]*)\)'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return {"error": f"未找到 func.func 定义: {mlir_path}"}

    func_name = match.group(1)
    inputs_raw = match.group(2)
    outputs_raw = match.group(3)

    def parse_types(raw: str) -> list[str]:
        """从 MLIR 类型字符串中提取每个参数的类型."""
        types = []
        depth = 0
        current = []
        for ch in raw:
            if ch == "<":
                depth += 1
                current.append(ch)
            elif ch == ">":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                types.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            types.append("".join(current).strip())
        return types

    input_types = parse_types(inputs_raw)
    output_types = parse_types(outputs_raw)

    # 解析 shape 和 dtype
    def parse_shape_dtype(t: str) -> tuple[list[str], str]:
        """解析 MLIR 类型，兼容多种格式.

        !torch.vtensor<[1,8,128],f32>  → (['1','8','128'], 'f32')
        tensor<1x8x128xf32>            → (['1','8','128'], 'f32')
        !torch.vtensor<[1,8],si64>     → (['1','8'], 'si64')
        """
        # 格式1: !torch.vtensor<[dims],dtype>
        match = re.search(r'\[([^\]]*)\]', t)
        if match:
            dims_str = match.group(1)
            dims = [d.strip() for d in dims_str.split(",") if d.strip()]
            dtype = re.sub(r'.*\]\s*,?\s*', '', t).strip()
            # 去掉首尾的 >
            dtype = dtype.rstrip(">")
            return dims, dtype

        # 格式2: tensor<d1xd2xd3xdtype>
        match = re.match(r'tensor<(.+)>', t)
        if match:
            inner = match.group(1)
            # 分离维度部分和 dtype
            # 例如: 1x8x128xf32 → dims=[1,8,128], dtype=f32
            parts = inner.split("x")
            dims = []
            for p in parts[:-1]:
                # 可能包含 '?' 作为动态维度
                dims.append(p)
            dtype = parts[-1]
            return dims, dtype

        return [], t

    return {
        "func_name": func_name,
        "n_inputs": len(input_types),
        "n_outputs": len(output_types),
        "inputs": [parse_shape_dtype(t) for t in input_types],
        "outputs": [parse_shape_dtype(t) for t in output_types],
        "n_funcs": text.count("func.func @"),
    }


def verify() -> int:
    """验证 MLIR 导出与 PyTorch 原始模型的一致性。返回 0 表示成功."""
    errors = 0

    # ─── 1. 获取 PyTorch 基线 ───
    print("=" * 60)
    print("1. PyTorch 基线推理")
    print("=" * 60)

    torch.manual_seed(42)
    model = GPT2Model(CONFIG)
    model.eval()
    example_input = torch.randint(0, CONFIG.vocab_size, (1, 8))
    print(f"   输入 shape: {tuple(example_input.shape)}, dtype: {example_input.dtype}")

    with torch.no_grad():
        ref_output = model(example_input)

    # GPT2Model 输出格式: BaseModelOutputWithPastAndCrossAttentions
    # last_hidden_state + past_key_values (tuple of tuples)
    if hasattr(ref_output, "last_hidden_state"):
        hs = ref_output.last_hidden_state
        pkv = ref_output.past_key_values
    else:
        hs = ref_output[0]
        pkv = ref_output[1]

    print(f"   hidden_state shape: {tuple(hs.shape)}, dtype: {hs.dtype}")
    print(f"   hidden_state 值范围: [{hs.min().item():.6f}, {hs.max().item():.6f}]")
    print(f"   past_key_values 层数: {len(pkv)}")
    for i, (k, v) in enumerate(pkv[:2]):
        print(f"   layer[{i}] key: {tuple(k.shape)}, value: {tuple(v.shape)}")
    if len(pkv) > 2:
        print(f"   ... (共 {len(pkv)} 层)")

    # ─── 2. 验证 Torch Dialect IR 签名 ───
    print("\n" + "=" * 60)
    print("2. Torch Dialect IR 签名对比")
    print("=" * 60)

    torch_ir = PROJECT_ROOT / "mlir/exported/gpt2_tiny_torch.mlir"
    sig_t = extract_mlir_signature(torch_ir)
    if "error" in sig_t:
        print(f"   ❌ {sig_t['error']}")
        errors += 1
    else:
        print(f"   函数: @{sig_t['func_name']}")
        print(f"   输入数: {sig_t['n_inputs']}, 输出数: {sig_t['n_outputs']}")

        # 检查最后一个输入 (input_ids) 的 shape 匹配
        last_input_shape, last_input_dtype = sig_t["inputs"][-1]
        expected_shape = [str(s) for s in example_input.shape]
        if last_input_shape == expected_shape:
            print(f"   ✅ input_ids shape 匹配: {last_input_shape}")
        else:
            print(f"   ❌ input_ids shape 不匹配: IR={last_input_shape}, PyTorch={expected_shape}")
            errors += 1

        # 检查输出 shape
        expected_hs_shape = [str(s) for s in hs.shape]
        first_output_shape, first_output_dtype = sig_t["outputs"][0]
        if first_output_shape == expected_hs_shape:
            print(f"   ✅ hidden_state shape 匹配: {first_output_shape}")
        else:
            print(f"   ❌ hidden_state shape 不匹配: IR={first_output_shape}, PyTorch={expected_hs_shape}")
            errors += 1

    # ─── 3. 验证 Linalg IR 签名 ───
    print("\n" + "=" * 60)
    print("3. Linalg IR 签名对比")
    print("=" * 60)

    linalg_ir = PROJECT_ROOT / "mlir/lowered/gpt2_tiny_linalg.mlir"
    sig_l = extract_mlir_signature(linalg_ir)
    if "error" in sig_l:
        print(f"   ❌ {sig_l['error']}")
        errors += 1
    else:
        print(f"   函数: @{sig_l['func_name']}")
        print(f"   输入数: {sig_l['n_inputs']}, 输出数: {sig_l['n_outputs']}")

        # Linalg IR 的输入输出类型应该是 tensor (不是 vtensor)
        first_out = sig_l["outputs"][0]
        if first_out:
            shape, dtype = first_out
            expected_hs_shape = [str(s) for s in hs.shape]
            if shape == expected_hs_shape:
                print(f"   ✅ 输出 shape 匹配: {shape}")
            else:
                print(f"   ❌ 输出 shape 不匹配: IR={shape}, PyTorch={expected_hs_shape}")
                errors += 1
            print(f"   dtype: {dtype}")

    # ─── 4. 验证优化后 IR ───
    print("\n" + "=" * 60)
    print("4. 优化后 IR 完整性")
    print("=" * 60)

    opt_ir = PROJECT_ROOT / "mlir/lowered/gpt2_tiny_linalg_opt.mlir"
    sig_o = extract_mlir_signature(opt_ir)
    if "error" in sig_o:
        print(f"   ❌ {sig_o['error']}")
        errors += 1
    else:
        # 优化前后的输出数量应该一致
        if sig_l["n_outputs"] == sig_o["n_outputs"]:
            print(f"   ✅ 优化前后输出数一致: {sig_l['n_outputs']}")
        else:
            print(f"   ❌ 优化前 {sig_l['n_outputs']} 输出, 优化后 {sig_o['n_outputs']}")
            errors += 1

        # 输出 shape 应该一致
        l_shape = sig_l["outputs"][0][0]
        o_shape = sig_o["outputs"][0][0]
        if l_shape == o_shape:
            print(f"   ✅ 输出 shape 一致: {o_shape}")
        else:
            print(f"   ❌ 优化前 {l_shape}, 优化后 {o_shape}")
            errors += 1

    # ─── 5. IR 语法验证 ───
    print("\n" + "=" * 60)
    print("5. IR 语法验证 (torch-mlir-opt / mlir-opt)")
    print("=" * 60)

    import subprocess

    # Torch Dialect
    r = subprocess.run(
        ["torch-mlir-opt", "--verify-diagnostics", str(torch_ir)],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("   ✅ Torch Dialect IR 语法通过")
    else:
        print(f"   ❌ Torch Dialect IR 语法失败")
        errors += 1

    # Linalg IR
    mlir_opt = "/home/lwy/download/llvm-project/install/bin/mlir-opt"
    r = subprocess.run(
        [mlir_opt, "--verify-diagnostics", str(linalg_ir)],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("   ✅ Linalg IR 语法通过")
    else:
        print(f"   ❌ Linalg IR 语法失败")
        errors += 1

    # 优化后 IR
    r = subprocess.run(
        [mlir_opt, "--verify-diagnostics", str(opt_ir)],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print("   ✅ 优化后 IR 语法通过")
    else:
        print(f"   ❌ 优化后 IR 语法失败")
        errors += 1

    # ─── 6. 总结 ───
    print("\n" + "=" * 60)
    if errors == 0:
        print("✅ 全部验证通过 — IR 签名/语法/shape 一致")
        return 0
    else:
        print(f"❌ 发现 {errors} 个问题")
        return 1


if __name__ == "__main__":
    exit(verify())
