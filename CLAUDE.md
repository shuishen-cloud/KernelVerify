# PyTorch → MLIR 优化加速验证

> 验证 MLIR 编译器优化对 PyTorch 模型推理加速效果的研究项目。

## 项目目标

探索 MLIR 作为编译器基础设施对 PyTorch 模型的优化潜力，分三阶段：
1. **阶段一**（W0~W5 ✅）：简单模型 → Torch MLIR → Linalg 优化
2. **阶段二**（W6~W8 ✅）：GPT-2 → Torch MLIR → Linalg 优化 → 总结报告
3. **阶段三**（W9~）：多模型验证 + 自定义 Pass + 实际性能测量

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
| W3 简单模型导出 | ✅ | 3 个模型全部导出 |
| W4 Lowering Pipeline | ✅ | 3 策略对比，default (fuse+cse) 最优 |
| W5 优化效果对比 | ✅ | 平均 25.8% linalg op 降低 |
| W6 GPT-2 导出 | ✅ | 极小 GPT-2 全链路通过，ops 降低 52.3% |
| W7 GPT-2 优化 | ✅ | 标准 GPT-2 全链路通过，ops 降低 58.5% |
| W8 总结报告 | ✅ | 阶段一二汇总 |
| W9 Linalg→GPU | ✅ | GPU pipeline 打通；832 kernel，27032 行 |
| W10 自定义 Pass | ✅ | CountLinalgOps + transform tiling 探明 |
| W11 性能基准 | ✅ | IREE CPU / Triton GELU / NVPTX 受阻 |
| W12 模拟新后端 | ✅ | SimNewBackend 融合+展开全链路 |
| W13 Profiling 基线 | ✅ | torch.profiler + nsys + torch.fx → 表A/B/C |
| W14 Triton 手动优化 | ⏳ | Attention/GELU/LayerNorm Triton kernel |
| W15 验证与报告 | ⏳ | 数值验证 + 最终报告 |

当前分支：`phase4-kernel-optimization`

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

## 工作流：文档驱动开发

> **核心原则：先写文档，再开发，再跟踪。所有开发任务必须先在 work/ 三份文档中定义，不允许跳过文档直接写代码。**

### 三份核心文档

| 文档 | 职责 | 更新时机 |
|------|------|------|
| `work/工作划分.md` | 任务定义：每个 W 任务的描述、输入、产出、难点、优先级 | 开始新任务前 |
| `work/进度管理.md` | 进度追踪：状态表（⏳🚧✅❌）、异常记录、执行顺序 | 任务状态变更时 |
| `work/项目设计.md` | 方法论：架构设计、关键发现、技术决策、验证方法 | 方法论变更时 |
| `work/项目设计.md` | **设计讨论记录**：文档更新后、编码前，用户提出的问题与讨论 | 每次讨论发生时（文档更新后、编码前） |

### 执行流程

```
1. 写文档
   ├── 工作划分.md: 定义新 W 任务（描述/产出/优先级）
   ├── 进度管理.md: 添加进度表行（状态=⏳ 或 🚧）
   └── 项目设计.md: 如有新方法论/架构变更，更新对应章节

2. 讨论
   └── 在开始编码前，用户提出的任何问题（如"怎么做？""为什么？""可行吗？"）
       这些问题与讨论 → 作为讨论记录写入 项目设计.md 对应章节
       包括：问题背景、讨论结论、技术决策、方案选型理由

3. 开发
   └── 严格按照工作划分.md 中的"产出"清单编码
       每个产出对应一个具体文件路径

3. 跟踪
   ├── 任务完成 → 进度管理.md: 状态改为 ✅，填写完成时间
   ├── 遇到异常 → 进度管理.md: 添加异常记录
   └── 有发现 → 项目设计.md: 添加关键发现

4. 提交
   └── git commit -m "feat(W{n}): {中文描述}"
       每个 W 任务至少一次 commit
```

### 示例：开始 W13 的完整流程

```
Step 1: 编辑 工作划分.md → 添加 W13 任务定义（描述/产出/优先级）
Step 2: 编辑 进度管理.md → 进度表添加 W13 行（状态=🚧）+ 执行顺序更新
Step 3: 编辑 项目设计.md → 如有新方法论（如 Phase 4 表格驱动），添加章节
Step 4: git commit -m "docs: Phase 4 W13 任务定义"
Step 5: 按工作划分.md 的产出清单编写代码
Step 6: 运行验证 → 数据写入 benchmarks/ 对应目录
Step 7: 编辑 进度管理.md → W13 状态改为 ✅ + 完成时间
Step 8: git commit -m "feat(W13): {任务描述}"
```

## 提交规范

- **每次提交前必须获得用户确认**：展示拟提交的 commit message，等待用户批准后再执行 `git commit`
- 每完成一个 W 任务立即 `git commit`
- 文档更新和代码开发分开提交（先 docs: 后 feat:）
- 格式：`feat(W{n}): {中文描述}` 或 `docs: {中文描述}`
- 示例：`feat(W3): 批量导出 3 个简单 PyTorch 模型到 Torch Dialect MLIR`
- 所有 commit 末尾加 `Co-Authored-By: Claude <noreply@anthropic.com>`
- 执行流程：`git add` → 展示 commit message → 等用户确认 → `git commit`

## Phase 4 工具链（2026-07-29+）

Phase 4 使用纯 PyTorch 生态工具链（MLIR 不参与 profiling/优化循环）：

| 工具 | 用途 | 层面 |
|------|------|------|
| `torch.profiler` | op 级耗时、显存 | 计算图 |
| `nsys` / `ncu` | GPU kernel 级 timeline、occupancy | GPU kernel |
| `torch.fx` | 计算图捕获、可视化、手动改图 | 计算图 |
| **Triton 3.7.0** | 手写 GPU kernel、benchmark | GPU kernel |

Phase 4 的 Python 脚本同样在 `novel_llm` 环境下运行。
