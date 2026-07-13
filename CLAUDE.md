# PyTorch → MLIR 优化加速验证

> 验证 MLIR 编译器优化对 PyTorch 模型推理加速效果的研究项目。

## 项目目标

探索 MLIR 作为编译器基础设施对 PyTorch 模型的优化潜力，分三阶段：
1. **阶段一**（W0~W5 ✅）：简单模型 → Torch MLIR → Linalg 优化
2. **阶段二**（W6~W8 ✅）：GPT-2 → Torch MLIR → Linalg 优化 → 总结报告
3. **阶段三**（W9~）：聚焦 Linalg 层 — GPU Lowering / 自定义 Pass / kernel 性能

最终目的是学习 MLIR 编译器优化原理，为大模型推理加速积累技术储备。

## 技术链路

```
PyTorch 模型
  → torch.export.export()           # 导出为 ExportedProgram (FX Graph)
  → prog.run_decompositions()       # ATen op 分解
  → FxImporter                     # FX → Torch Dialect MLIR
  → torch-mlir-opt                 # Torch Dialect → Linalg/SCF/Arith
  → mlir-opt                       # Linalg 层优化 (fuse/canonicalize/cse)
  → 优化后的 MLIR
```

## 项目结构

```
pytorch_mlir/
├── work/                          # 项目文档（权威来源）
│   ├── 项目设计.md                  # 架构设计、验证方法、环境依赖
│   ├── 工作划分.md                  # 任务分解与风险评估
│   ├── 进度管理.md                  # 进度追踪与异常记录
│   ├── archive/                    # 已完成阶段的产出归档
│   │   ├── 优化效果对比_简单模型.md   # W5: 阶段一优化效果报告
│   │   ├── 优化效果对比_GPT2.md      # W6-W7: 阶段二优化效果报告
│   │   └── 总结报告_阶段一二.md       # W8: 阶段一二汇总
│   └── task/                       # 专项任务讨论
│       └── 任务&讨论-GPT2导出.md     # W6: GPT-2 导出实施方案
├── scripts/                       # Python 导出/验证脚本
│   ├── export_simple_model.py     # W3: 简单模型 → Torch MLIR
│   ├── export_gpt2_step2.py       # W6: GPT-2 → Torch MLIR
│   └── verify_gpt2.py             # W6: GPT-2 功能验证
├── models/                        # PyTorch 模型定义
│   └── simple_models.py           # W3: LinearReLU, ConvBNReLU, TwoLayerMLP
├── mlir/
│   ├── handwritten/               # W2: 手写 Torch Dialect MLIR
│   ├── exported/                  # W3/W6: 导出的 Torch Dialect IR
│   ├── lowered/                   # W4-W7: Lowering + 优化后的 IR
│   └── models/                    # 🆕 按模型组织的新 IR
│       ├── qwen/
│       ├── llama/
│       └── bert/
├── passes/                        # 🆕 自定义 MLIR Pass (Phase 3)
├── triton/                        # 🆕 Triton kernel 实验 (Phase 3)
├── benchmarks/                    # 🆕 性能基准测试数据
├── tools/                         # Shell 工具脚本
│   ├── verify_mlir.sh             # W2: MLIR 语法验证
│   └── lower_and_opt.sh           # W4: Lowering + 优化 pipeline
└── .gitignore
```

## 环境

- **Python**: Conda env `novel_llm`, Python 3.10.20
- **PyTorch**: 2.12.0, Transformers 4.33.0
- **torch-mlir**: nightly wheel 20260531.828 (`pip install --pre torch-mlir`)
- **mlir-opt**: `/home/lwy/download/llvm-project/install/bin/mlir-opt`
- **torch-mlir-opt**: conda env `novel_llm` bin 目录下

所有 Python 脚本需在 `novel_llm` 环境下运行：
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate novel_llm
```

## 当前状态

| 阶段 | 状态 | 关键产出 |
|------|:---:|------|
| W0~W2 环境与验证 | ✅ | torch-mlir 安装、手写 MLIR 验证通过 |
| W3 简单模型导出 | ✅ | 3 个模型 (linear_relu/conv_bn_relu/two_layer_mlp) 全部导出 |
| W4 Lowering Pipeline | ✅ | 3 策略对比，default (fuse+cse) 最优 |
| W5 优化效果对比 | ✅ | 平均 25.8% linalg op 降低，最高 50% (add_relu) |
| W6 GPT-2 导出 | ✅ | 极小 GPT-2 全链路通过，linalg ops 降低 52.3% |
| W7 GPT-2 优化 | ✅ | 标准 GPT-2 (12层/124M) 全链路通过，linalg ops 降低 58.5% |
| W8 总结报告 | ✅ | 阶段一二汇总，见 `work/archive/总结报告_阶段一二.md` |
| W9 Linalg→GPU | ⏳ | 学习 Linalg → GPU/LLVM Lowering，在 4060 上编译执行 |
| W10 自定义 Pass | ⏳ | Linalg tiling/fusion/vectorization Pass 开发 |
| W11 性能基准 | ⏳ | IREE + Triton 对比，kernel 级性能测量 |
| W12 LeetGPU 集成 | ⏳ | GPU 加速基础设施，为后续工作提供算力 |

## 关键发现

1. **torch-mlir 安装**：nightly wheel 预编译可用，1 分钟安装，无需源码编译
2. **导出 API**：正确入口是 `torch_mlir.fx.export_and_import()`，不是 `torch_mlir.compile()`
3. **最优 Pass 组合**：`--linalg-fuse-elementwise-ops --canonicalize --cse`，平均降低 25.8% linalg ops
4. **反效果 Pass**：`--linalg-generalize-named-ops` 将命名 op 展开为 generic，op 计数反而增加
5. **工具分工**：系统 `mlir-opt` 不含 Torch 方言，必须用 `torch-mlir-opt` 验证 Torch Dialect
6. **自动兜底**：未注册的 op 会被自动转为 `torch.operator` 节点，导入不会崩溃，但无法 lowering

## 导出 API 使用

```python
from torch_mlir import fx

# 基本用法
result = fx.export_and_import(model, example_input, output_type="torch")

# 带自定义分解表
decomp_table = fx.get_decomposition_table()
decomp_table.update(get_decompositions([torch.ops.aten.gelu]))
result = fx.export_and_import(
    model, example_input,
    output_type="linalg-on-tensors",
    decomposition_table=decomp_table,
    verbose=True,
)
```

`output_type` 选项：`raw` → `torch` → `linalg-on-tensors` → `tosa` → `stablehlo`

## 提交规范

- 每完成一个 W 任务立即 `git commit`
- 格式：`feat(W{n}): {中文描述}`
- 示例：`feat(W3): 批量导出 3 个简单 PyTorch 模型到 Torch Dialect MLIR`
