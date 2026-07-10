# 任务&讨论：GPT-2 模型导出到 Torch MLIR

> W6 详细实施方案 | 基于 2026-07-10 分析

---

## 一、背景回顾

W0~W5 已成功完成简单模型（linear_relu / conv_bn_relu / two_layer_mlp）的全流程：
PyTorch → torch_mlir → Torch Dialect → Linalg → 优化。

现在进入阶段二：将 GPT-2 small（~124M 参数）导入同一流程，验证 MLIR 优化对真实 NLP 模型的效果。

## 二、GPT-2 架构与算子清单

GPT-2 small 配置：
- `n_layer=12`, `n_head=12`, `n_embd=768`, `vocab_size=50257`
- 总参数量约 124M

### 2.1 核心子模块与涉及算子

```
GPT2Model
├── wte (nn.Embedding)          → aten.embedding
├── wpe (nn.Embedding)          → aten.embedding
├── drop (nn.Dropout)           → aten.native_dropout (eval 模式下为 pass-through)
├── h (×12, GPT2Block)
│   ├── ln_1 (nn.LayerNorm)     → aten.native_layer_norm
│   ├── attn (GPT2Attention)
│   │   ├── c_attn (nn.Linear)  → aten.linear / aten.addmm
│   │   ├── c_proj (nn.Linear)  → aten.linear / aten.addmm
│   │   ├── split heads         → aten.split / aten.reshape / aten.permute
│   │   └── SDPA / manual:
│   │       ├── Q @ K^T        → aten.matmul / aten.bmm
│   │       ├── scale           → aten.mul / aten.div
│   │       ├── mask            → aten.masked_fill
│   │       └── softmax         → aten.softmax
│   │       └── attn @ V        → aten.matmul / aten.bmm
│   ├── ln_2 (nn.LayerNorm)     → aten.native_layer_norm
│   └── mlp (GPT2MLP)
│       ├── c_fc (nn.Linear)    → aten.linear / aten.addmm
│       ├── act (nn.GELU)       → aten.gelu
│       ├── c_proj (nn.Linear)  → aten.linear / aten.addmm
│       └── drop (nn.Dropout)   → aten.native_dropout
└── ln_f (nn.LayerNorm)         → aten.native_layer_norm
└── lm_head (nn.Linear)         → aten.linear (ties with wte weight)
```

### 2.2 算子分类

| 分类 | 算子 | torch-mlir 支持 |
|------|------|:---:|
| **A 类：默认支持** | `aten.linear` / `aten.addmm` / `aten.matmul` / `aten.bmm` / `aten.softmax` / `aten.embedding` / `aten.permute` / `aten.reshape` / `aten.view` / `aten.split` / `aten.transpose` / `aten.mul` / `aten.div` / `aten.add` / `aten.masked_fill` / `aten.expand` | ✅ |
| **B 类：默认分解表中已有** | `aten.native_layer_norm` / `aten._native_batch_norm_legit_*` / `aten.native_dropout` / `aten.scaled_dot_product_attention` / `aten._log_softmax` | ✅ |
| **C 类：需手动加入分解表** | `aten.gelu` / `aten.gelu_backward`（若存在） | ⚠️ |
| **D 类：自动兜底为 torch.operator** | 任何未注册 op → 自动转为 `torch.operator` 节点（不崩溃但无法 lowering） | ⚡ 兜底 |
| **E 类：动态控制流** | 自回归循环（token-by-token generation）、KV Cache 更新 | 🔴 难点 |

### 2.3 关键发现（源码分析确认）

1. **不会崩溃**：任何未注册的 op 都会被 `_emit_operation()` 自动转为 `torch.operator` 通用节点（携带原始 op 名作为 attribute），导入不会失败
2. **无法 lowering 的 op 会被保留**：`torch.operator` 节点在 RAW/TORCH 输出中可见，但无法被 linalg/tosa/stablehlo pipeline 处理
3. **`torchdynamo-export-to-torch-backend-pipeline`** 是核心 pipeline，负责分解、shape/dtype 推导等
4. **默认分解表不含 `gelu`**：需手动加入，但 PyTorch 标准库有 `gelu` 的精确分解（基于 `erf`）
5. **GPT-2 的 LayerNorm 和 Dropout** 已在默认分解表中，SDPA 也已有分解支持

## 三、五个层次的手动干预机制

通过分析当前安装的 torch-mlir 20260531.828 源码，确认了五个递进的干预点：

### 理解导出流程（关键！）

```
PyTorch Model
    │
    ▼ torch.export.export()
ExportedProgram (FX Graph)
    │
    ▼ prog.run_decompositions(table)    ← 干预点 ① + ②
ExportedProgram (已分解)
    │
    ▼ FxImporter.import_frozen_program()  ← 干预点 ③
Torch Dialect MLIR (Raw)
    │
    ▼ torchdynamo-export-to-torch-backend-pipeline  ← 干预点 ④
Torch Backend IR
    │
    ▼ torch-backend-to-linalg-on-tensors-backend-pipeline
Linalg IR
```

**核心发现：`torch.operator` 自动兜底机制** — 即使某个 op 在 MLIR 中没有注册的 lowering，`_emit_operation()` 也会自动将其生成为通用 `torch.operator` 节点，携带 `name` 属性标记原始 op 名。这意味着**任何 op 都不会导致导入崩溃**，但未注册的 op 无法被 subsequent lowering pass 处理。

### 3.1 Decomposition Table（最推荐）

在 `run_decompositions()` 阶段，把复杂 op 分解为基础 op。

```python
from torch_mlir import fx
from torch._decomp import get_decompositions

gpt2_extra_decomps = [
    torch.ops.aten.gelu,        # GELU 激活函数
    torch.ops.aten.gelu_backward,
]
my_table = fx.get_decomposition_table()  # 默认 50 个分解
my_table.update(get_decompositions(gpt2_extra_decomps))

fx.export_and_import(
    model, example_input,
    output_type="torch",
    decomposition_table=my_table,
)
```

**适用场景**：op 有现成的 PyTorch 官方分解。查阅 `torch/_decomp/decompositions.py` 确认是否有可用分解。

### 3.2 `register_canonicalize` 节点重写（FX 图层面）

在 FX 图被导入 MLIR 之前，直接将节点目标改写为另一个已支持的 op。

```python
from torch_mlir.extras.fx_importer import register_canonicalize

# 将 unsupported op 重写为 equivalent supported op
@register_canonicalize(torch.ops.aten.my_unsupported_op.default)
def rewrite_my_op(node):
    # 修改 node.target 和 node.args 来实现等价替换
    node.target = torch.ops.aten.my_supported_op.default
```

已有示例（源码中）：
- `lift_fresh_copy.default` → `clone.default`
- `_local_scalar_dense.default` → `Float.Tensor` 或 `Int.Tensor`
- `empty.memory_format` → `zeros.default`

**适用场景**：两个 op 语义等价，只需简单的 target 替换，不需要分解为多个 op。

### 3.3 FxImporterHooks（导入时拦截）

在 FX 图导入 MLIR 的过程中拦截特定操作。

```python
from torch_mlir.extras.fx_importer import FxImporterHooks

class GPT2Hooks(FxImporterHooks):
    def resolve_literal(self, gni, literal, info):
        """拦截常量/literal 的导入，手动创建 IR Value"""
        return None  # None = 交给默认处理

    def resolve_input(self, gni, value, info):
        """拦截 buffer/parameter 输入的导入"""
        return None

    def store_produced_value(self, gni, py_value, produced_ir_value, info):
        """处理 buffer mutation 的存储"""
        pass

fx.export_and_import(model, example_input, hooks=GPT2Hooks())
```

**适用场景**：需要在 IR 层面精确控制特定值的导入方式（如 weight tying、KV cache 初始化）。

### 3.4 `backend_legal_ops` + 自定义 C++ Lowering（Pipeline 层面）

告诉 lowering pipeline 保留某些 op（不分解），配合自定义 C++ pass 提供手工 lowering：

```python
fx.export_and_import(
    model, example_input,
    output_type="torch",
    backend_legal_ops=["torch.aten.gelu", "torch.aten.scaled_dot_product_attention"],
)
# 同时可通过 FxImportOptions.extra_library_file_name 加载外部 .so 提供 lowering
```

**适用场景**：需要在 MLIR C++ 层面提供自定义 lowering pattern（重量级，本阶段不推荐）。

### 3.5 FX Graph 预处理（兜底方案）

直接在 `torch.export` 产出的 FX 图上手术。

```python
prog = torch.export.export(model, args)

for node in prog.graph.nodes:
    if node.op == "call_function" and node.target in UNSUPPORTED_OPS:
        with prog.graph.inserting_before(node):
            new_node = prog.graph.call_function(
                torch.ops.aten.matmul, args=(...)
            )
        node.replace_all_uses_with(new_node)

prog.graph.eliminate_dead_code()
```

**适用场景**：以上四种方式都无法处理的边缘情况。

## 四、分步实施计划

### Step 1：环境与基线确认（0.5h）

```bash
# 确认 transformers 可用
python -c "from transformers import GPT2Model, GPT2LMHeadModel; print('OK')"
# 确认 GPT-2 small 模型大小
python -c "
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
"
```

**产出**：`models/gpt2_info.txt` — 模型参数量、层数、隐藏维度等基本信息。

### Step 2：首次裸导出尝试（1h）

不添加任何自定义分解，直接用默认配置导出，收集失败信息。

```python
# scripts/export_gpt2.py - 第一版
import torch
from transformers import GPT2Model, GPT2Config
from torch_mlir import fx

def main():
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128)  # 从极小配置开始
    model = GPT2Model(config)
    model.eval()
    example_input = torch.randint(0, 50257, (1, 8))  # seq_len=8

    try:
        result = fx.export_and_import(
            model, example_input,
            output_type="torch",        # 先只到 Torch Dialect
            verbose=True,
        )
        print("✅ 导出成功")
        print(result.operation.get_asm())
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        # 记录错误信息用于后续分析

if __name__ == "__main__":
    main()
```

**关键参数说明**：
- `output_type="torch"` — 先不下沉到 linalg，避免双重失败
- `n_layer=2, n_embd=128` — 极小配置，快速迭代
- 失败信息会通过 `TorchMlirCompilerError` 自动输出 repro 命令

**产出**：
- `scripts/export_gpt2.py`（第一版）
- 错误日志 → `work/task/gpt2_export_errors.md`

### Step 3：逐个补充分解（1.5h）

根据 Step 2 的错误日志，按算子分类处理：

```
遇到 aten.gelu → 加入 decomposition_table
遇到 aten.native_dropout → 在 eval 模式下替换为恒等映射
遇到 未知 op → 查 torch._decomp 是否有现成分解
               → 有：加入 extra_decompositions
               → 无：用 torch.library 注册自定义分解
               → 无法分解：用 FX Graph 预处理重写子图
```

**补充分解代码**：

```python
# 在 export_gpt2.py 中
from torch._decomp import get_decompositions

GPT2_EXTRA_DECOMPOSITIONS = [
    # GELU 激活 — GPT-2 核心算子
    torch.ops.aten.gelu,
    torch.ops.aten.gelu_backward,
    # Scaled Dot Product Attention — 若 PyTorch 版本支持
    # (某些版本提供高效的融合 attention 实现)
    torch.ops.aten.scaled_dot_product_attention,
    # 其他可能不支持的 op
    torch.ops.aten._log_softmax,
    torch.ops.aten._softmax,
]

# 注意: 只添加 torch-mlir 默认表没有的、且当前 PyTorch 版本存在的 op
def build_decomposition_table():
    base_table = fx.get_decomposition_table()
    for op in GPT2_EXTRA_DECOMPOSITIONS:
        if hasattr(torch.ops.aten, op.name().split('.')[-1]):
            try:
                extra = get_decompositions([op])
                base_table.update(extra)
            except Exception:
                pass  # op 存在但无分解，跳过
    return base_table
```

**产出**：完善后的 `scripts/export_gpt2.py`

### Step 4：逐子模块验证（1h）

将 GPT-2 拆分为独立子模块逐一导出，降低调试复杂度：

```python
# scripts/export_gpt2_modules.py

SUB_MODULES = {
    "embedding":    lambda cfg: GPT2Model(cfg).wte,        # token embedding
    "layernorm":    lambda cfg: nn.LayerNorm(cfg.n_embd),  # LayerNorm standalone
    "attention":    lambda cfg: GPT2Attention(cfg),         # 单头注意力块
    "mlp":          lambda cfg: GPT2MLP(cfg),              # FFN 块
    "single_block": lambda cfg: GPT2Block(cfg),            # 完整 Transformer Block
}

# 对每个子模块独立导出，记录成功/失败
for name, builder in SUB_MODULES.items():
    try:
        m = builder(config)
        m.eval()
        mlir = fx.export_and_import(m, dummy_input(name), output_type="torch")
        print(f"[{name}] ✅")
    except Exception as e:
        print(f"[{name}] ❌ {e}")
```

**产出**：子模块导出结果矩阵 `mlir/exported/gpt2_{module_name}.mlir`

### Step 5：完整模型导出 + Lowering（1h）

成功导出到 Torch Dialect 后，执行完整的 lowering + 优化 pipeline：

```bash
# 1. 导出完整 GPT-2 (Torch Dialect)
python scripts/export_gpt2.py --output-type torch

# 2. Lowering 到 Linalg (按 function 拆分处理大 IR)
python scripts/export_gpt2.py --output-type linalg-on-tensors

# 3. 优化
bash tools/lower_and_opt.sh mlir/exported/gpt2_full_linalg.mlir
```

**关键策略**：
- 如果完整模型 IR 过大（预计数千行），按 function 拆分 lowering
- 使用 `torch-mlir-opt` 的 `--split-input-file` 选项
- 只聚焦 attention 和 FFN 子图进行详细分析（参考项目设计 §五-困难4）

**产出**：
- `mlir/exported/gpt2_full_torch.mlir`
- `mlir/lowered/gpt2_full_linalg.mlir`
- `mlir/lowered/gpt2_full_linalg_opt.mlir`

### Step 6：优化效果量化（0.5h）

对 GPT-2 的 lowering 结果执行与 W5 相同的量化分析：

```bash
# 统计关键 op 数量
python tools/count_ops.py mlir/lowered/gpt2_full_linalg.mlir       # 优化前
python tools/count_ops.py mlir/lowered/gpt2_full_linalg_opt.mlir    # 优化后
```

**产出**：`work/优化效果对比_gpt2.md`

### Step 7：功能正确性验证（0.5h）

验证 lowering 前后计算结果一致：

```python
# scripts/verify_gpt2.py
import torch
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2")
model.eval()
x = torch.randint(0, 50257, (1, 16))

# 1. PyTorch 原始推理
with torch.no_grad():
    ref_output = model(x)

# 2. 通过 torch-mlir + 执行引擎 得到的输出（如果有 IREE）
# 或者对比 intermediate IR 的 shape/dtype 信息

# 3. 使用 torch.allclose 或分析输出差异
```

**产出**：`scripts/verify_gpt2.py` + 验证日志

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|:---:|:---:|----------|
| `aten.gelu` 无现成分解 | 中 | 中 | 用 `tanh` 近似公式手动分解：`x * 0.5 * (1 + tanh(0.7979 * x + 0.044715 * x^3))` |
| `aten.scaled_dot_product_attention` 分解失败 | 中 | 高 | 手动展开为 matmul + scale + softmax + matmul 序列 |
| KV Cache 涉及动态形状导致 export 失败 | 高 | 中 | 阶段二先只导出 single forward pass（prefill），不处理自回归循环 |
| IR 过大导致 torch-mlir-opt OOM | 中 | 高 | 按 function 拆分处理；使用 `n_layer=2` 的极小配置 |
| weight tying（wte/lm_head 共享）导致 IR 错误 | 低 | 中 | 检查 `get_attr` 节点是否重复引用同一参数 |
| 导出 `ExportedProgram` 时的动态控制流 | 高 | 中 | 使用 `strict=False` 放宽约束 |

### 5.1 GELU 手动分解（备用方案）

如果 `aten.gelu` 分解失败，提供精确/近似两种方案：

```python
# 精确版 (PyTorch 内置分解)
# torch._decomp.decompositions.gelu 已提供分解，
# 只需确保其在 decomposition_table 中

# 近似版 (tanh 近似，用于快速验证)
def gelu_approx(x):
    return 0.5 * x * (1.0 + torch.tanh(
        0.7978845608028654 * (x + 0.044715 * x**3)
    ))

# 注册为分解
@torch.library.impl("aten::gelu", "CompositeImplicitAutograd")
def gelu_decomp(x, approximate="none"):
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))
```

### 5.2 控制流策略

GPT-2 的自回归生成涉及 `while` 循环（逐个 token 生成），当前 torch-mlir 的 FX importer 已支持 `torch._higher_order_ops.while_loop`（见 `fx_importer.py:_import_hop_while_loop`），但建议：

- **W6 只做单次 forward**（prefill 阶段，无需循环）
- **W7 再尝试**带 while_loop 的完整自回归导出

## 六、产出一览

| 产出物 | 路径 | 说明 |
|--------|------|------|
| GPT-2 导出脚本 | `scripts/export_gpt2.py` | 含自定义分解表 |
| 子模块导出脚本 | `scripts/export_gpt2_modules.py` | 逐模块调试 |
| 验证脚本 | `scripts/verify_gpt2.py` | 功能正确性 |
| GPT-2 Torch Dialect IR | `mlir/exported/gpt2_*.mlir` | 导出产物 |
| GPT-2 Linalg IR | `mlir/lowered/gpt2_*.mlir` | Lowering 产物 |
| 错误/问题日志 | `work/task/gpt2_export_errors.md` | 过程记录 |
| 优化效果对比 | `work/优化效果对比_gpt2.md` | 对标 W5 |

## 七、讨论点

1. **先从极小 GPT-2（2层/128维）开始还是直接用标准 gpt2（12层/768维）？**
   - 建议：从极小开始调试，通过后再升级到标准配置

2. **是否需要支持 weight tying？**
   - GPT-2 的 `wte` 和 `lm_head` 共享权重，torch-mlir 应能通过参数绑定处理

3. **是否需要导出 GPT2LMHeadModel 还是先只做 GPT2Model？**
   - 建议：先只做 GPT2Model（纯 encoder/decoder），成功后再加 lm_head

4. **阶段二的最终目标是什么？**
   - A) 完整 lowering 到 Linalg（验证可行性）
   - B) 量化优化效果（对标 W5）
   - C) 端到端正确性验证（包含 runtime 执行）
