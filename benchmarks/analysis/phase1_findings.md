# W13 Profiling 分析结果

> 日期：2026-07-29 | 状态：✅ 表A/B/C 生成完成

## 表A：Op 耗时排名（torch.profiler, Full GPT-2 12L/768h/124M, 1×128 tokens）

| 排名 | op | GPU 耗时 (μs) | 占比 | 调用次数 | 类型 |
|:---:|------|:---:|:---:|:---:|------|
| 1 | **aten::addmm** | 152,735 | **38.0%** | 2,400 | GEMM (QKV+FFN) |
| 2 | **ampere_sgemm_128x64_nn** | 134,520 | **33.4%** | 1,800 | cuBLAS GEMM |
| 3 | ampere_sgemm_128x32_nn | 16,846 | 4.2% | 600 | cuBLAS GEMM |
| 4 | aten::bmm | 15,072 | 3.7% | 1,200 | Batch matmul |
| 5 | ampere_sgemm_128x128_nn | 9,165 | 2.3% | 600 | cuBLAS GEMM |
| 6 | aten::mul | 8,082 | 2.0% | 2,400 | Elementwise |
| 7 | aten::add | 6,480 | 1.6% | 2,450 | Elementwise |
| 8 | ampere_sgemm_128x128_tn | 5,907 | 1.5% | 600 | cuBLAS GEMM |
| 9 | aten::native_layer_norm | 4,662 | 1.2% | 1,250 | LayerNorm |
| 10 | aten::where | 2,811 | 0.7% | 600 | Elementwise |
| 11 | aten::_softmax | 2,644 | 0.7% | 600 | Reduction |
| 12 | aten::div | 2,056 | 0.5% | 600 | Elementwise |
| 13 | aten::tanh | 2,033 | 0.5% | 600 | Elementwise |
| 14 | aten::pow | 1,905 | 0.5% | 600 | Elementwise |
| 15 | aten::copy_ | 1,702 | 0.4% | 1,200 | Memcpy |
| 16 | aten::gather | 336 | 0.1% | 100 | Indexing |
| 17 | Memcpy HtoD | 199 | 0.0% | 600 | Host→Device |
| — | **其他** | ~25,000 | ~6% | — | — |
| | | | | | |
| | **总计** | ~402,000 | 100% | | |

### 核心发现

```
GEMM 类 (addmm + sgemm):    ~75% GPU 时间  ← cuBLAS 已经高度优化，Triton 难以超越
Elementwise (mul+add+...):   ~8%  GPU 时间  ← 融合收益：减少 launch overhead + 中间 tensor
Attention 链 (bmm+softmax):  ~6%  GPU 时间  ← bmm 是瓶颈(3.7%)，周围 op 可融合
LayerNorm:                   ~1.2% GPU 时间 ← 简单 kernel，适合做 Triton 入门练习
GELU 链 (tanh+pow+mul+add):  ~3.5% GPU 时间 ← 4 个 kernel → 1 个 Triton，消除中间 tensor
```

### 优化优先级

| 优先级 | 目标 | 依据 | 预期单 kernel 加速 | 端到端收益 |
|:---:|---|---|:---:|:---:|
| P0 | Attention Score 融合 | bmm(3.7%) + scale+mul+mask+softmax(~2%) → 1 kernel | 2-3x | ~3% |
| P1 | GELU 激活融合 | tanh+pow+mul+add(3.5%) → 1 kernel | 2x | ~1.7% |
| P2 | LayerNorm Triton | 简单 reduction kernel，1.2% | 1.2x | ~0.2% |
| P3 | addmm → Triton matmul | 38%，但 cuBLAS 已优化到极致 | ~1x | 0% |

**结论**：GPT-2 中 cuBLAS GEMM 已经是 dominant factor (75%)，Triton 手写优化的端到端加速空间约 5-10%。主要收益来自减少 kernel launch overhead 和消除中间 tensor 显存流量。

## 表C：计算图结构（torch.fx）

GPT-2 (tiny, 2 层) FX 图：154 节点，其中 120 个 call_function，33 个 placeholder（模型参数+输入）

每层结构（12 层相同）：
```
Layer {n}:
  └─ LayerNorm (ln_1) → Attention
  │    ├─ QKV: 3× addmm (weight[768,2304])
  │    ├─ Score: bmm → mul(scale) → add(mask) → softmax → bmm
  │    └─ Output: addmm (weight[768,768])
  └─ LayerNorm (ln_2) → FFN
       ├─ addmm (weight[768,3072]) → GELU (tanh+pow+mul+add)
       └─ addmm (weight[3072,768])
```

融合机会：
- Attention Score：5 个 op → 1 个 Triton kernel（bmm 本身占 3.7%，融合周围 op）
- GELU：4 个 op → 1 个 Triton kernel（tanh+pow+mul+add）
- FFN + GELU：addmm → GELU → addmm 可整体融合

## 下一步

进入 W14：基于以上 profiling 数据，按优先级实现 Triton kernel 手动优化。
