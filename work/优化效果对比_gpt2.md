# 优化效果对比报告 — GPT-2

> 阶段二输出，极小 GPT-2 (2层/128维/6.8M 参数) 的 lowering + 优化效果分析

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

## 四、与阶段一对比

| 指标 | 阶段一 (6 简单模型) | 阶段二 (GPT-2 tiny) |
|------|:---:|:---:|
| 平均 linalg op 降低 | 25.8% | **52.3%** |
| 最高单模型降低 | 50% (add_relu) | 52.3% |
| 最优策略 | fuse+cse+canon | fuse+cse+canon |
| 关键差异 | matmul 少，融合机会少 | **matmul→generic→融合** 链路额外收益 |

**结论**：GPT-2 的优化效果 (52.3%) 远超简单模型 (25.8%)，原因是 GPT-2 有更密集的 matmul+elementwise 组合，融合机会更多。

## 五、优化前关键 op 明细

| Op | 数量 | 来源 |
|----|:---:|------|
| linalg.generic | 113 | BN/LN 统计量、GELU 分解、softmax、mask、add 等 |
| linalg.matmul | 8 | QKV 投影 (c_attn)、输出投影 (c_proj)、FFN (c_fc/c_proj) |
| linalg.batch_matmul | 4 | attention score 计算 (Q@K^T, attn@V) |
| linalg.fill | 9 | 零初始化临时 tensor |
| linalg.transpose | 10 | head 拆分/合并、permute |
| linalg.index | 5 | embedding 查表 |

## 六、局限

1. **极小配置**：2 层 128 维远小于标准 GPT-2 (12 层 768 维)，优化比例可能随规模变化
2. **固定序列长度** (8)：循环被展开，无 `scf.for` 循环融合测试
3. **优化后 matmul=0 的隐忧**：命名 matmul → generic 后失去 BLAS 库加速机会，实际执行性能需 runtime 验证
4. **这是 IR 级别的静态分析**：op 数量减少 ≠ 实际 wall-clock 加速，需 IREE/MLIR 执行引擎测量

## 七、下一步

- [ ] 在标准 GPT-2 (12层/768维) 上重复此实验
- [ ] 使用 IREE 或其他 runtime 测量实际推理时间
- [ ] 对比 `generalize-named-ops` 启用/禁用的 GELU-level 性能差异
