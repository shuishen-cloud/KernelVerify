# 动态 Shape 导出 GPT-2 记录

> W9 | 2026-07-13

## 一、目的

静态 shape 导出导致编译器将循环全部展开（序列长度固定），无法产生 `scf.for`。动态 shape 导出保留符号维度，为 Linalg tiling → GPU kernel 提供条件。

## 二、关键命令

```python
from torch.export import Dim
from torch_mlir import fx

batch = Dim("batch", min=1, max=32)
seq = Dim("seq", min=1, max=512)

result = fx.export_and_import(
    model, example_input,
    output_type="linalg-on-tensors",
    dynamic_shapes=({0: batch, 1: seq},),  # tuple 匹配位置参数
)
```

## 三、结果

| 指标 | 静态 (seq=4) | 动态 (seq=?) |
|------|:---:|:---:|
| Torch IR 行数 | 1,729 | 1,831 |
| Linalg IR 行数 | 3,794 | 5,386 |
| scf.for | 0 | 0 |
| 符号维度 | 0 | 2,032 |
| 输出 shape | `1x4x768` | `?x?x768` |

## 四、为什么动态后仍无 scf.for？

Linalg 用单个 op 表达整个矩阵运算，不需要显式循环：

```mlir
// 静态: 固定 shape — 无循环
%0 = linalg.matmul ins(%a, %b : tensor<1x4x768xf32>, ...)

// 动态: 符号 shape — 仍无循环！
%0 = linalg.matmul ins(%a, %b : tensor<?x?x768xf32>, ...)
```

`linalg.matmul` 本身是隐式循环（affine_map 描述迭代空间）。`scf.for` 只在下层 lower（如 tiling）后才出现。

## 五、遇到的问题

| 问题 | 原因 | 解决 |
|------|------|------|
| transformers 5.13 导出失败 | `DynamicCache` + `aten.diff` 不兼容 | 降级到 4.33.0 |
| `dynamic_shapes` key 错误 | dict key 必须匹配参数名 | 改用 tuple: `({0: batch, 1: seq},)` |
