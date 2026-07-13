# 优化效果对比报告 — GPT-2

> 阶段二输出，对比极小 GPT-2 vs 标准 GPT-2 的 lowering + 优化效果

---

## 一、模型信息

| 属性 | 值 |
|------|-----|
| 模型 | GPT-2 (GPT2Model) |
| 配置 | n_layer=2, n_head=4, n_embd=128, n_positions=64 |
| 参数量 | 6,837,888 |
| 输入 | (1, 8) int64 token ids |
| 输出 | hidden_state (1,8,128) + 4 个 attention KV tensors (1,4,8,32) |

## 二、优化效果总览

| IR 阶段 | linalg.generic | matmul | batch_matmul | fill | transpose | index | **总计** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 优化前 (Linalg) | 113 | 8 | 4 | 9 | 10 | 5 | **149** |
| 优化后 (fuse+cse+canon) | 47 | 0 | 0 | 9 | 10 | 5 | **71** |
| **变化** | ↓66 (-58%) | ↓8 (-100%) | ↓4 (-100%) | =0 | =0 | =0 | **↓78 (52.3%)** |

**总 linalg ops 降低 52.3%**，显著优于阶段一简单模型的平均 25.8%。

## 三、优化机制分析

### 3.1 matmul/batch_matmul → generic（-100%）

`--canonicalize` 将 12 个命名 matmul op 全部转换为 `linalg.generic`（带 reduction 迭代器），效果等同于 `--linalg-generalize-named-ops`。

在阶段一简单模型中，此转换被视为反效果（op 计数反而增加），但在 GPT-2 中：
- matmul op 的下游有大量 elementwise op（add/mul/div/tanh/softmax）
- 转换为 generic 后，`--linalg-fuse-elementwise-ops` 可以将它们与相邻 elementwise op 融合
- 结果：**12 个 matmul 被融合进 47 个 generic 宏内核中**，净效果为正

### 3.2 linalg.generic 融合（-58%）

113 个 generic → 47 个 generic，减少了 66 个独立 kernel。

典型融合链：
```
优化前 (5 个 kernel):
  %a = linalg.matmul %q, %k
  %b = linalg.generic { mulf }          # scale
  %c = linalg.generic { addf }          # mask
  %d = linalg.generic { exp; divf }     # softmax
  %e = linalg.matmul %d, %v

优化后 (1 个融合 kernel):
  %result = linalg.generic {
    mulf, addf, expf, divf, mulf, addf   # 全部融合
  } ins(...) outs(...)
```

### 3.3 fill/transpose/index 不变

这些是元数据/reshape 操作，不参与 elementwise 融合，保持不变。

## 四、标准 GPT-2 优化效果 (12层/768维/124M)

| IR 阶段 | linalg.generic | matmul | batch_matmul | fill | transpose | index | **总计** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 优化前 (Linalg) | 583 | 48 | 24 | 9 | 60 | 15 | **739** |
| 优化后 (fuse+cse+canon) | 237 | 0 | 0 | 9 | 60 | 5 | **311** |
| **变化** | ↓346 (-59.3%) | ↓48 (-100%) | ↓24 (-100%) | =0 | =0 | ↓10 (-66.7%) | **↓428 (57.9%)** |

**总 linalg ops 降低 57.9%**（包括 yield 则为 58.5%，1322→548），再次超过极小模型的 52.3%。

## 五、两种规模对比

| 指标 | 极小 GPT-2 | 标准 GPT-2 | 趋势 |
|------|:---:|:---:|:---:|
| 配置 | 2层/128维 | 12层/768维 | 6× |
| 参数量 | 6.8M | 124M | 18× |
| IR 行数 | 785 | 3,794 | 4.8× |
| 优化前 linalg ops | 149 | 739 | 5× |
| 优化后 linalg ops | 71 | 311 | 4.4× |
| **降低比例** | **52.3%** | **57.9%** | ↑ |
| generic 融合 | 113→47 (-58%) | 583→237 (-59.3%) | ↑ |
| matmul 融合 | 12→0 | 72→0 | 全融合 |

**趋势：模型越大，融合机会越多，优化比例越高。**

## 六、与阶段一对比

| 指标 | 阶段一 (6 简单模型) | 极小 GPT-2 | 标准 GPT-2 |
|------|:---:|:---:|:---:|
| linalg op 降低 | 25.8% (平均) | 52.3% | **57.9%** |
| 最高单模型降低 | 50% (add_relu) | - | - |
| 最优策略 | fuse+cse+canon | fuse+cse+canon | fuse+cse+canon |
| 关键差异 | matmul 少，融合机会少 | matmul→generic→融合 | 更大规模=更多融合 |

**结论**：MLIR Linalg 优化对真实 NLP 模型的效果远超简单模型，且呈**规模递增**趋势。

## 七、局限

1. **固定序列长度** (4/8)：循环被展开，无 `scf.for` 循环融合测试
2. **优化后 matmul=0 的隐忧**：命名 matmul → generic 后失去 BLAS 库加速机会，实际执行性能需 runtime 验证
3. **这是 IR 级别的静态分析**：op 数量减少 ≠ 实际 wall-clock 加速，需 IREE/MLIR 执行引擎测量
4. **IR 文件过大**：标准 GPT-2 的 IR 达 950MB（权重内联），不适合文本对比分析
