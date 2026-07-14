# 讨论：推理引擎选型 — IREE vs vLLM vs SGLang

> 2026-07-14 | 当前阶段：Phase 3 Linalg-GPU

---

## 一、两种推理引擎的分野

vLLM **是**推理引擎，但"推理引擎"这个词被两种截然不同的架构共用了。区别在于**优化的方向不同**：

```
        编译器路线                              服务层路线
   (IREE / TensorRT)                       (vLLM / SGLang)

  模型 IR ──→ 优化 ──→ 生成 kernel         模型权重 ──→ 加载到显存
              │                                       │
              ├─ 算子融合                             ├─ 调度请求
              ├─ tiling                               ├─ 管理 KV Cache
              ├─ codegen                              ├─ Continuous Batching
              └─ 生成 .so/.ptx                        └─ 调用 PyTorch kernel
                    │                                       │
                    ▼                                       ▼
              自定义 kernel                          PyTorch 自带的 kernel
            (编译器生成的)                        (cuBLAS/cuDNN/手写)
```

### 各自优化的核心问题

```
IREE/TensorRT 优化:   "一个 matmul 怎么算最快？"       → 编译器层
vLLM/SGLang 优化:     "1000 个请求的 matmul 怎么排队？" → 服务层
```

### vLLM 做了什么 vs 没做什么

| vLLM 做了 (调度层) | vLLM 没做 (编译器层) |
|---|---|
| PagedAttention — KV Cache 虚拟内存管理 | 算子融合 — 交给 PyTorch 的 cuBLAS |
| Continuous Batching — 请求动态拼批 | codegen — 不需要，直接用 cuBLAS |
| Prefix Caching — 相同前缀复用 KV | IR lowering — 没有多层 IR |
| Tensor Parallelism — 多卡切分 | tiling — 不自己生成 kernel |

vLLM 的核心假设：**单次矩阵乘法 kernel 已经足够好**（cuBLAS 优化了几十年），真正浪费性能的地方在"请求排队等显存"、"KV Cache 预分配浪费"这类系统层问题。

### 三个引擎在栈上的位置

```
                    IREE                  vLLM / SGLang
                  ┌───────┐              ┌────────────┐
    用户接口      │ Python │              │  Python API │
                  ├───────┤              ├────────────┤
    调度/服务     │   VM   │  ← 有       │ Scheduler  │  ← 核心价值
                  ├───────┤              ├────────────┤
    Kernel 执行   │  HAL   │  ← 有       │ PyTorch     │  ← 不自己生成
                  ├───────┤              │ (cuBLAS等)  │
    编译器        │  MLIR  │  ← 积累     │    ❌        │  ← 没有
                  └───────┘              └────────────┘

    看到：        全链路，每一层          调度层 + 服务层
    看不到：      没有 LLM 服务优化       没有编译器/IR/kernel 生成
```

---

## 二、三维对比

### 2.1 功能维度

| | IREE | vLLM | SGLang |
|---|---|---|---|
| **开发者** | Google | UC Berkeley | 社区 (源自 Stanford) |
| **语言** | C++ / Python | Python + C++/CUDA | Python |
| **模型→可执行代码** | AOT 编译 IR → .vmfb | 运行时加载 HF 模型 | 同 vLLM |
| **kernel 来源** | MLIR 自动生成 | PyTorch 自带 (cuBLAS/cuDNN) | 同 vLLM |
| **编译器知识** | ✅ 核心 | ❌ 几乎没有 | ❌ 几乎没有 |
| **LLM 服务化** | ❌ 不支持 | ✅ 最成熟 | ✅ 更新更激进 |
| **多后端** | CUDA/Vulkan/Metal/CPU | 仅 NVIDIA GPU | 仅 NVIDIA GPU |

### 2.2 入门难度

| 引擎 | 难度 | 原因 |
|---|---|---|
| **IREE** | ⭐⭐⭐ | MLIR 知识已掌握，只差 VM→HAL 一层窗户纸 |
| **vLLM** | ⭐⭐ | Python 为主，3 行代码跑起来 |
| **SGLang** | ⭐⭐ | 同 vLLM，但生态更小 |
| **ncnn** | ⭐⭐⭐⭐ | C++ 手写 kernel，与 MLIR 知识脱节 |

### 2.3 结论

- **IREE**：已入门，继续深挖收益最大。学习目标是看懂 VM→HAL→Driver 运行时全貌。
- **vLLM**：作为对比参考，理解"编译器路线 vs 手写 kernel 路线"的差异。核心学 PagedAttention / Continuous Batching。
- **SGLang**：暂不需要。先理解 vLLM 的 PagedAttention，再看 RadixAttention 才有意义。

---

## 三、vLLM 快速上手记录

### 3.1 环境

| 项目 | 值 |
|---|---|
| GPU | RTX 4060 Laptop, 8GB 显存 |
| CUDA | 13.0 (随 PyTorch 2.12.0) |
| Python | 3.10.20 |

### 3.2 安装踩坑

1. **pip 26.1.2 与 Python 3.10 不兼容** → 降级 pip < 25 解决
2. **CUDA 13.0 兼容性未知** → vLLM 预编译 wheel 可能不匹配，需关注运行时是否报 CUDA 库错误

### 3.3 8GB 能跑什么

| 级别 | 模型 | 显存需求 |
|------|------|---------|
| ✅ 轻松 | GPT-2 (124M), Llama-3.2-1B, Qwen2-0.5B | <2GB |
| ✅ 没问题 | Qwen2-1.5B, Gemma-2B | ~4GB |
| ⚠️ 刚好 | Qwen2-7B (INT4), Llama-3-8B (INT4) | ~6-7GB |
| ❌ 跑不了 | Llama-3-8B (FP16), Qwen2-72B | >16GB |

---

## 四、GPT-2 对话踩坑

### 问题

GPT-2 (124M) 是**文本补全模型**，不是聊天模型。输入 hello → 输出随机新闻续写。

### 原因

GPT-2 未经指令微调 (Instruction Tuning) 和 RLHF，不理解"问答"行为模式。它只会续写你给它的文字。

### 解决

换 Instruct 模型（如 `Qwen/Qwen2-1.5B-Instruct`），使用 `AutoModelForCausalLM` + `chat_template`。

### 相关脚本

- `scripts/chat.py` — 通用 LLM 对话脚本，支持 GPT-2/Qwen/Llama/Phi

---

## 五、与项目当前阶段的关联

```
Phase 3 重点: Linalg 层 → GPU kernel 生成 → 性能调优

IREE 角色:  编译器 + 运行时，验证自动生成 kernel 的正确性和性能
vLLM 角色:  对比基线 — 手写 kernel + PyTorch 调度的性能天花板
Triton 角色: kernel 层手写实验，验证"编译器生成 vs 手写"的差距
```

### 待探索

- [ ] IREE 深入：跟一遍 GPT-2 的 VM → HAL → Driver 执行链路
- [ ] IREE vs vLLM 性能对比：同一 GPT-2 模型，两种路径的延迟/吞吐量
- [ ] ncnn：移动端部署场景暂不需要，先跳过

---

## 六、背景知识：AI 编译器

### 6.1 AI 编译器是什么

连接**深度学习框架**（PyTorch、TensorFlow）和**硬件后端**（CPU、GPU、TPU）的中间层。核心任务是将高层神经网络代码自动转换为可在目标硬件上高效执行的底层代码。

```
高层框架 (PyTorch/TF/JAX)
        │
        ▼
   AI 编译器
        │
        ▼
   硬件后端 (CUDA/ROCm/Metal/Vulkan/…)
```

和传统编译器（gcc/LLVM）的区别：传统编译器处理标量/循环代码，AI 编译器处理**张量计算图**——大量矩阵乘法、卷积、逐元素操作。

### 6.2 为什么需要 AI 编译器

| 问题 | AI 编译器的解法 |
|------|----------------|
| 算子太多（PyTorch ~2000+ ATen op） | 将高层 op **分解**为少量基础 op（decomposition） |
| 硬件太多 | **多层 IR** 实现"一次编写，多处编译" |
| 融合靠人 | **自动算子融合**（element-wise fusion、tiling） |
| 写 kernel 难 | 自动生成底层代码（codegen） |

### 6.3 多层 IR 渐进式 Lowering（以 MLIR 为例）

```
Layer 1: 框架层 IR       Torch Dialect / TF Dialect / ONNX
                         靠近用户代码，语义丰富

Layer 2: 高层计算 IR     Linalg / TOSA / StableHLO
                         与框架无关的线性代数表示

Layer 3: 中层循环 IR     SCF (Structured Control Flow)
                         显式循环、条件分支

Layer 4: 低层并行 IR     GPU / Affine
                         硬件相关的并行抽象

Layer 5: 代码生成        LLVM IR / NVPTX / SPIR-V
                         最终可执行代码
```

### 6.4 核心技术概念速查

**Decomposition**
```
aten.gelu(x) → x * 0.5 * (1 + erf(x / sqrt(2)))  → 5 个基础 op
```

**Linalg（线性代数 IR）**
- Named op: `linalg.matmul` — 有明确语义
- Generic op: `linalg.generic` + `affine_map` — 描述任意计算，循环隐式

**关键优化 Pass**

| Pass | 作用 |
|------|------|
| `-canonicalize` | 死代码消除、常量折叠、代数化简 |
| `-cse` | 公共子表达式消除 |
| `-linalg-fuse-elementwise-ops` | 融合逐元素操作 |
| `-linalg-tile` | 循环分块，利用缓存层次 |

**Bufferization**：tensor SSA（值语义）→ memref（内存语义），GPU lowering 的前置步骤。

**GPU Lowering 链路**
```
Linalg on tensors → bufferize → parallel-loops → gpu-map → gpu-outline
```

### 6.5 主流项目

| 项目 | 定位 | 核心技术 |
|------|------|---------|
| MLIR (Google/LLVM) | 编译器基础设施 | 多层 IR、Dialect、Pass |
| torch-mlir (LLVM) | PyTorch → MLIR | Torch Dialect、FxImporter |
| IREE (Google) | MLIR 运行时 | HAL 抽象层、多后端 |
| XLA (Google) | TF/JAX 编译器 | HLO IR、融合、codegen |
| TVM (Apache) | 端到端编译 | Halide-like schedule、AutoTVM |
| Triton (OpenAI) | GPU kernel 语言 | Pythonic DSL、block-level |

---

## 七、背景知识：推理引擎

### 7.1 推理引擎 vs AI 编译器

| 维度 | AI 编译器 | 推理引擎 |
|------|----------|---------|
| **产物** | 生成代码（.so, .ptx, .mlir） | 执行推理（输出 tensor） |
| **时机** | 编译时（ahead-of-time） | 运行时（online） |
| **核心问题** | "如何把 matmul 变成高效指令" | "如何把 1000 个 op 高效编排执行" |
| **类比** | gcc/LLVM | 操作系统 + 标准库 |

### 7.2 推理引擎核心模块

```
模型加载器 → 图优化器 → 内存规划器 → 算子调度器 → Kernel 执行器
```

**图优化器** — 推理引擎最有价值的部分之一：

```
Conv → BatchNorm → ReLU  →  ConvBN → ReLU  →  ConvBNReLU
                            (逐层融合)
```

常见图优化：算子融合、常量折叠、死分支消除、布局优化(NHWC↔NCHW)、量化、子图替换。

**内存规划器** — 核心竞争力，决定能跑多大模型：

```
Tensor A (op 1-3)  ████████░░░░░░░░
Tensor B (op 4-6)  ░░░░████████░░░░
Tensor C (op 7-9)  ░░░░░░░░████████

内存复用: 峰值 = max(size(A), size(B), size(C))
而非 size(A) + size(B) + size(C)
```

**算子调度器** — 决定执行顺序和并行策略（多 stream、CUDA Graph）。

### 7.3 LLM 推理的特殊挑战

**KV Cache — 最大的内存瓶颈**

自回归生成中，每个 step 的 Key/Value 可复用：
- 7B 模型, 2048 context, FP16 → ~0.5GB KV Cache
- 7B 模型, 128K context → ~32GB KV Cache

**PagedAttention (vLLM 核心创新)** — 借鉴 OS 页表机制，KV Cache 分 block 非连续存储，显存利用率从 20-40% → ~96%。

**Continuous Batching** — 请求级别的动态拼批，某请求结束立即补位，不等整个 batch。

**Speculative Decoding** — 小模型猜 N 个 token → 大模型一次性验证，2-3x 加速。

### 7.4 主流推理引擎

| 引擎 | 定位 | 优势 | 劣势 |
|------|------|------|------|
| ONNX Runtime | 通用跨框架 | 生态最大，多后端 | 极限性能不如专用 |
| TensorRT | NVIDIA GPU 专用 | 极致性能 | 只支持 NVIDIA |
| IREE | MLIR-native | 架构先进，与本项目直接相关 | 生态年轻 |
| vLLM | LLM 专用 | PagedAttention, 吞吐极高 | 只做 LLM |
| ncnn | 移动端推理 | 极轻量，ARM 优化好 | 手写 kernel，无 IR |
| llama.cpp | CPU LLM | CPU 也能跑大模型 | 量化精度有限 |

---

## 八、常量折叠 / 死代码消除 — 发生在哪个阶段？

### 答案：两个阶段都可以做

```
AI 编译器阶段 (编译时)          推理引擎阶段 (运行时)
     │                              │
  IR 级别                      计算图级别
  MLIR Dialect 上的 op         ONNX Graph / TensorRT Network
     │                              │
  -canonicalize pass            Graph Optimizer
  -sccp pass                    ConstantFolding pass
```

关键是 **任何有图/IR 的地方都可以做**，没有硬性边界：

| | AI 编译器 | 推理引擎图优化器 |
|---|---|---|
| 常量折叠 | ✅ IR 级，提前算好 | ✅ 图级，同样提前算好 |
| 死代码消除 | ✅ 消除未使用的 op | ✅ 消除不可达分支 |
| 区别 | 离 target 更近，可用更低层信息 | 离用户更近，能看到高层语义 |

TensorRT 是典型例子——既是编译器又是推理引擎，常量折叠和 DCE 做了不止一轮。

---

## 九、如何操作计算图 — 项目中的三个切入点

### 9.1 你的技术链路中的图操作点

```
PyTorch 模型
    │
    ▼  ← ① FX Graph 层   — Python，最易上手
torch.export.export()  →  ExportedProgram (图结构)
    │
    ▼  ← ② MLIR 层       — C++ Pass，W10 已入门
Torch Dialect MLIR  →  custom Pass → Linalg → ...
    │
    ▼  ← ③ IREE 编译流程  — Python pipeline
iree.compiler.compile_str()  →  .vmfb
```

### 9.2 三层详解

| 层 | 语言 | 操作对象 | 项目中的入口 |
|---|---|---|---|
| ① FX Graph | Python | `torch.fx.Graph` (node/edge 增删改) | `prog.graph_module.graph` |
| ② MLIR Pass | C++ | MLIR Operation/Block/Region | `passes/CountLinalgOps/` |
| ③ IREE Pipeline | Python | Pass 编排顺序 | `iree.compiler.compile_str()` |

### 9.3 FX Graph 层的核心操作

FX Graph 是有向无环图，每个 node 有 op (placeholder/call_function/call_method/get_attr/output) 和 target (具体的函数/方法)：

```python
# 遍历节点
for node in graph.nodes:
    print(node.op, node.name, node.target, node.args, node.users)

# 变换操作
node.replace_all_uses_with(other)     # 替换消费者
graph.eliminate_dead_code()           # 死代码消除
graph.call_function(fn, args=(...))   # 新增节点
```

### 9.4 演示脚本

- `scripts/demo_graph_manipulation.py` — 四个演示：
  1. 打印计算图结构
  2. 死代码消除 (DCE) — 5 节点 → 3 节点
  3. 算子融合 — add+relu 合并为单个 fused 节点
  4. torch.export 产生的真实 Aten 图

### 9.5 何为 FX 图

FX 是 PyTorch 的图捕获和变换系统。普通 PyTorch 是 eager mode（执行即计算），FX 把它变成"先记录图，再执行"。

```
Eager mode:  x + 1.0  →  立刻算出结果
FX:          x + 1.0  →  记录 "add" 节点，不计算
```

**捕获原理**：`symbolic_trace` 给模型喂 Proxy 对象（假 tensor），利用 Python 运算符重载记录每一步操作：

```python
proxy_x = Proxy("x")         # 假输入
proxy_a = proxy_x + 1.0      # 触发 __add__ → 记录 "add" 节点
proxy_b = proxy_a * 2.0      # 触发 __mul__ → 记录 "mul" 节点
# 全程零计算，只建图
```

**FX 图即 DAG**（有向无环图），每个节点有固定 5 种 opcode：

| opcode | 含义 | 例子 |
|--------|------|------|
| `placeholder` | 输入节点 | 模型参数、输入 tensor |
| `call_function` | 函数调用 | `torch.add(x, y)` |
| `call_method` | 方法调用 | `x.relu()` |
| `get_attr` | 属性读取 | `self.weight` |
| `output` | 输出节点 | `return z` |

**图变换操作**：

```python
node.replace_all_uses_with(other)   # 替换所有消费者
graph.eliminate_dead_code()         # 死代码消除
graph.call_function(fn, args=(...)) # 插入新节点
graph.erase_node(node)              # 删除节点
```

### 9.6 FX 与 MLIR 的关系

```
FX Graph (Python 层)              MLIR (编译器层)
┌──────────────────┐         ┌──────────────────────┐
│ x = placeholder   │         │ func.func @main(%x)  │
│ add = add(x, 1.0) │  ───→   │   %0 = aten.add(%x) │
│ relu = relu(add)  │  翻译   │   %1 = aten.relu(%0)│
│ output(relu)      │         │   return %1          │
└──────────────────┘         └──────────────────────┘
  操作简单 (Python)              操作强大 (C++ Pass)
  适合高层变换                   适合低层优化
```

### 9.7 计算图可视化

图可以渲染为 SVG/PNG 图像，直观看到节点和边的拓扑结构。

**渲染工具链**：
```
symbolic_trace(model) → FxGraphDrawer → DOT 源码 → graphviz dot → SVG/PNG
```

**相关脚本**：
- `scripts/demo_graph_manipulation.py` — 图操作演示（4 个例子）
- `scripts/visualize_graph.py` — 图可视化导出

**已生成的可视化**（`mlir/graphs/`）：

| 文件 | 内容 | 大小 |
|------|------|------|
| `simple_graph.svg` | add→mul→relu→add 简单图 | 10KB |
| `linear_relu.svg` | Linear→ReLU (symbolic_trace) | 9KB |
| `conv_bn_relu_exported.svg` | Conv→BN→ReLU (Aten 图) | 31KB |
| `gpt2_tiny_exported.svg` | 极小 GPT-2 (1层64hidden) | 249KB |

图中每个节点标注了 name、op_code、target、args、num_users，和 MLIR Pass 中的 op 结构是同一个概念。

### 9.8 关键认知

> 常量折叠和 DCE 在 AI 编译器和推理引擎两处都会做。你在 FX 图（Python）和 MLIR Pass（C++）两个层面都能操作计算图。W10 的 CountLinalgOps 已经在做 MLIR 层的事了，FX 层更简单——不用编译 C++，Python 直接改。

---

## 十、讨论记录

| 时间 | 观点 | 来源 |
|------|------|------|
| 2026-07-14 | ncnn 是手写 kernel 库 + 轻量调度器，没有 IR lowering，与当前 MLIR 学习路径不匹配 | Claude |
| 2026-07-14 | IREE 深入优先级最高：MLIR 积累可复用，只差 VM→HAL 一层 | Claude |
| 2026-07-14 | vLLM 作为推理服务层参考，学调度优化而非编译器 | Claude |
| 2026-07-14 | SGLang 等理解 vLLM PagedAttention 后再看 RadixAttention | Claude |
| 2026-07-14 | vLLM 安装受阻：pip 26.x + Python 3.10 不兼容，需降级 pip 后重试 | 用户 |
| 2026-07-14 | 澄清：vLLM 是推理引擎但走服务层路线，优化调度而非 kernel；IREE 走编译器路线，优化 kernel 生成。两者都是推理引擎，优化方向不同 | 用户+Claude |
