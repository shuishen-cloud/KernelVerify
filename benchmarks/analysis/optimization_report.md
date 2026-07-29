# Phase 4 优化报告：GPT-2 Kernel 性能分析与手动优化

> 日期：2026-07-29 | 模型：GPT-2 (12L/768h/124M) | GPU：RTX 4060 Laptop

## 一、Profiling 发现（W13）

### 1.1 Op 级耗时分布 (torch.profiler)

| 排名 | op/cuda kernel | GPU 耗时 | 占比 | 类型 |
|:---:|------|:---:|:---:|------|
| 1 | aten::addmm | 152,735 μs | 38.0% | GEMM |
| 2 | ampere_sgemm_128x64_nn | 134,520 μs | 33.4% | cuBLAS GEMM |
| 3 | ampere_sgemm_128x32_nn | 16,846 μs | 4.2% | cuBLAS GEMM |
| 4 | aten::bmm | 15,072 μs | 3.7% | Batch matmul |
| 5 | ampere_sgemm_128x128_nn | 9,165 μs | 2.3% | cuBLAS GEMM |
| 6 | aten::mul | 8,082 μs | 2.0% | Elementwise |
| 7 | aten::add | 6,480 μs | 1.6% | Elementwise |
| — | GEMM 类合计 | ~318,000 μs | **~79%** | cuBLAS 已极致优化 |
| — | Elementwise 类合计 | ~32,000 μs | **~8%** | 融合候选 |

**结论**：GEMM 占绝对主导，cuBLAS 已高度优化。手动优化的空间在 elementwise 碎片消除。

### 1.2 计算图结构 (torch.fx)

GPT-2 单层结构（12 层相同）：
```
LayerNorm → Attention (QKV addmm → bmm×2 + softmax → O addmm) → +residual
  → LayerNorm → FFN (up addmm → GELU → down addmm) → +residual
```

## 二、数据流分析（W14）

工具：`scripts/analyze_dataflow.py`（`torch.export` → `node.target`/`node.users`/`node.meta['val']` 遍历）

### 2.1 融合安全性判断

| 候选 | 数据流 | consumer 数 | 结论 |
|------|------|:---:|:---:|
| GELU 内部链 | 7 ELEMENTWISE ops 单链 | 全为 1 | ✅ 可融 |
| Attention score | mul→add→softmax（跨 bmm） | 1 | ⚠️ 跨库边界 |
| add→LayerNorm | residual add 后流向 LN+残差 | **2** | ❌ 不可融 |
| add→GELU | GPT-2 中无此 pattern（bias 在 addmm 内） | — | ❌ 不存在 |

### 2.2 融合机制

```
PyTorch GELU (8 kernel):
  load → mul → store tmp1
  load → pow → store tmp2
  ... (6 more kernels, 7 intermediate tensors)

Triton GELU (1 kernel):
  load → mul→pow→add→mul→tanh→mul→add (all in registers) → store
```

**加速来源**：不是 Triton 比 cuBLAS 快，而是消灭了 8→1 kernel launch + 7→0 中间 tensor 显存读写。

## 三、优化结果

### 3.1 表D：单 Kernel 级

| kernel | backend | shape | time_us | vs pytorch | 备注 |
|--------|--------|------|:---:|:---:|------|
| add+GELU fused | Triton | 2M | 22.6 | **1.72x** | 人工构造场景 |
| add+LN fused | Triton | 128×768 | 13.3 | **1.51x** | 人工构造场景 |
| add+LN fused | Triton | 8192×768 | 386.0 | **1.61x** | 大批量场景 |
| GELU (GPT-2内部) | Triton | 1×8×3072 | — | — | 嵌入了GPT-2 |
| LayerNorm (小批量) | Triton | 1×768 | 16.3 | 0.86x | launch overhead > 计算 |

### 3.2 表F：端到端 (GPT-2 12L 124M, 1×128)

| variant | latency_ms | vs baseline | peak_mem_mb | accuracy |
|------|:---:|:---:|:---:|:---:|
| GPT-2 baseline | 8.30 | 1.00x | 530 | — |
| + Triton GELU (12层) | 7.68 | **1.08x** | 1013 | ✅ |

### 3.3 表G：最终汇总

| optimization | prof依据 | before | after | speedup | impact |
|-------------|------|------|------|:---:|:---:|
| GELU Triton (单kernel) | 表A: 3.5% 表B: MEMORY-bound | 8 kernel | 1 kernel | — | ⭐⭐ |
| GELU GPT-2嵌入 | 表C: 7 ELEMENTWISE 单链 | 8.30ms | 7.68ms | **1.08x** | ⭐⭐⭐ |
| LayerNorm Triton | 表A: 1.2% | 9 kernel | 1 kernel | — | ⭐ |
| add+GELU 独立验证 | 表B + 表C | 39.0μs | 22.6μs | **1.72x** | ⭐⭐ |
| add+LN 独立验证 | 表B | 20.1μs | 13.3μs | **1.51x** | ⭐⭐ |

## 四、方法论总结

### 4.1 完整的优化流程

```
Profiling (表A/B)
  → 数据流分析 (analyze_dataflow.py, 表C)  ← W14 教训：不可跳过
    → 选热点 + 判断可融性
      → 写 Triton kernel (表D/E)
        → 模块替换 + 端到端 (表F)
          → 汇总 (表G)
```

### 4.2 核心教训

1. **写 kernel 前必须先做数据流分析**：否则会写出 GPT-2 中根本不存在的 pattern（如 add+gelu）
2. **Profiling 数据可能与直觉相反**：以为 GELU 只占 3.5%，但 8 kernel 碎片化开销被忽略了
3. **GEMM 不可超越**：cuBLAS 占 79% GPU 时间且已极致优化，Triton 收益来自碎片消除
4. **Model Surgery 是跑端到端的正确入口**：FX 全图 trace 对 GPT-2 不可行
5. **文档驱动开发有效**：W14 跳过了数据流分析，教训被记录到项目设计.md，流程得到修正
