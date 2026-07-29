# W14 数据流与依赖分析

> 分析 GPT-2 计算图中的 producer-consumer 关系，判断哪些算子链可以安全融合。

## 一、GPT2MLP 源码结构

```python
# transformers GPT2MLP.forward()
hidden_states = self.c_fc(hidden_states)     # Linear(768→3072): addmm + bias
hidden_states = self.act(hidden_states)       # GELU (tanh近似)
hidden_states = self.c_proj(hidden_states)    # Linear(3072→768): addmm + bias
hidden_states = self.dropout(hidden_states)
```

关键事实：`c_fc` 和 `c_proj` 中的 bias add 是 `F.linear()` 内部的 `addmm`，**不是**独立的 `aten::add`。所以 FFN 中没有独立的 add→gelu 链——GELU 的输入直接来自 addmm。

## 二、从 torch.export 提取的实际数据流

### 2.1 FFN 子图（每层 1 次，共 12 层）

```
c_fc: addmm  [8, 512]
  → view    [1, 8, 512]       ← reshape for elementwise
    ├→ mul   [1, 8, 512]      ← GELU: x * 0.5 * (1+tanh(...))
    ├→ pow   [1, 8, 512]      ← GELU: x^3
    └→ add   [1, 8, 512]      ← GELU: x + 0.044715*x^3
         → mul [1, 8, 512]    ← GELU: 0.7979 * (...)
           → tanh              ← GELU: tanh(...)
             → mul             ← GELU: 0.5 * x * (1+tanh(...))
    → view    [8, 512]         ← reshape for addmm
      → addmm [8, 128]         ← c_proj
```

**GELU 分解链**：view 的输出有 3 个 consumer（mul, pow, add），随后逐级单 consumer 链至最终 output。

**融合判断**：GELU 整体的输入→输出是单 producer 单 consumer（view 之后 → view 之前），可以安全融合为单 kernel。

**我们做的**：`block.mlp.act = TritonGELU()` 替换的是整个 `self.act()` 调用。Triton kernel 在寄存器中完成全部 GELU 计算，无中间 tensor。

**Profiling 依据（表A）**：GELU 链（tanh+pow+mul+add 等）总耗时 ~3.5% GPU 时间，6+ 个 kernel 调用，4+ 个中间 tensor。

### 2.2 Attention 子图（每层 1 次，共 12 层）

```
QKV: addmm [8, 128] → view → split → permute × 3  (Q, K, V heads)
Score:
  bmm(Q,K^T) [4, 8, 8]
    → mul(scale)     [4, 8, 8]
      → add(mask)    [4, 8, 8]      ← 1 consumer ✅
        → softmax   [4, 8, 8]       ← 1 consumer ✅
          → bmm(×V) [4, 8, 32]
Output: view → addmm [8, 128] → view → dropout → add(residual)
```

**融合判断**：mul→add→softmax 链是单 consumer 链（每个 op 的输出只有一个 consumer），可以融合。但 **bmm 是 cuBLAS 调用，跨库融合不可行**。softmax 包含 reduction（max, sum, exp），需要特殊处理数值稳定性。

**我们没有做**：Attention 融合需要打破 bmm↔elementwise↔softmax 的边界，且 softmax 的 warp-level reduction 实现需要较高的 Triton 熟练度。

### 2.3 Residual + LayerNorm（每层 2 次，共 25 次）

```
add(residual, hidden) [1, 8, 128]    ← 2 consumers!
  ├→ layer_norm  [1, 8, 128]        ← LN 消费者
  └→ add(residual_for_next)          ← 残差连接消费者 ⚠️
```

**融合判断**：add 的输出有 **2 个 consumer**——这是残差连接的本质。不能简单地将 add+LN 融合为单个 kernel，因为 add 的结果还被下游残差路径使用。融合后残差路径会丢失输入。

**我们没有在模型替换中做 add+LN 融合**——只做了纯 LayerNorm 替换。`triton/layernorm_kernel.py` 中有 add+LN kernel，但它是独立的 benchmark，**未嵌入 GPT-2**。

## 三、融合安全性判断规则

| 条件 | 检查 |
|------|------|
| 单 consumer | `len(list(op.users)) == 1` — 输出只被一个 op 消费 |
| 同类型 | 被融合的 op 都是 elementwise / reduction（无跨库调用如 cuBLAS） |
| 形状一致 | producer 和 consumer 的 tensor shape 兼容 |
| 无副作用 | 融合后的 kernel 不改变其他 consumer 的行为 |

## 四、W14 实际完成的优化

| 优化 | 融合对象 | 理由 | 表A 依据 | 实际效果 (表D/F) |
|------|------|------|:---:|:---:|
| GELU Triton | GELU 分解链 (6+ elementwise ops) | 单输入单输出，无多 consumer | 3.5% GPU 时间 | **端到端 1.08x** (full GPT-2) |
| LayerNorm Triton | LayerNorm 分解链 (~9 ops) | 单输入单输出 | 1.2% GPU 时间 | 端到端几乎无变化 |
| **add+GELU fused** (benchmark only) | add + GELU | 人工构造的 add→gelu 链，单 consumer | — | **1.72x** (单 kernel) |
| **add+LN fused** (benchmark only) | add + LN | 人工构造的 add→LN 链，单 consumer | — | **1.51x** (128×768) |

> add+gelu 和 add+LN 融合只在独立 benchmark 中验证了性能，**未嵌入 GPT-2**，因为 GPT-2 的实际数据流中 add 的 bias 在 addmm 内部，而 residual add 有 2 个 consumer。

## 五、结论

1. **GPT-2 的算子融合空间有限**：GEMM 占 75%（cuBLAS 不可融合），残差连接天然有多 consumer（不可融合），真正能安全融合的只有 GELU 和 LayerNorm 内部的 elementwise 链。
2. **Model Surgery 是正确的策略**：在 FX 图层面能做的融合，通过模块替换一样能做到，而且避免了 trace 失败的问题。
3. **add+gelu 融合在独立 benchmark 中有 1.72x 加速**——这意味着如果模型中有独立的 add→gelu pattern（如不带 bias 的 Linear + GELU），融合收益会更大。
