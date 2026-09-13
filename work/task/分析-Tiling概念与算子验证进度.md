# 分析：Tiling 概念澄清与项目 Tiling 验证进度

> 日期：2026-09-13（含多轮验证更新） | 状态：✅ L1/L2/L3 验证完成；K 维 tiling 配方已验证；CPU 度量框架就绪（IREE 经实测**不适合作度量后端**，改用 mlir-cpu-runner）

## 一、背景

讨论中提出两个问题：

1. 本项目（W4~W9）的算子验证**有没有到 tiling 阶段**？
2. **Tiling 是不是"排布"的意思**？

本文给出概念澄清与项目进度核查结论。

---

## 二、概念澄清：Tiling / Layout / Mapping 是三件事

先纠一个用词：是 **Tiling**（单 L），不是 `Tilling`（耕作）。检索文档时用 `tiling`。

**Tiling 不是"排布"。** 三个概念常被混用，但各自管不同的事：

| 概念 | 中文 | 管什么 | 本项目中的体现 |
|------|------|--------|----------------|
| **Tiling** | 分块 / 瓦片化 | 迭代空间怎么切、数据怎么复用 | `tile_sizes`、嵌套 `scf.for` |
| **Layout** | 排布 / 布局 | 数据在内存里怎么放 | `memref<...>` 的 layout map、strided layout |
| **Mapping** | 映射 / 调度 | 哪一块交给哪个 block/thread | `gpu.block_id` / `gpu.thread_id` |

Tiling 的本质是一个**循环变换**：

```
for i in 0..127        →    for io in 0..3 {        # 外层：给并行
                               for ii in 0..31       # 内层：给复用 / 向量化
```

其目的不是"排整齐"，而是**把一块数据搬进快速存储（寄存器 / shared memory / cache）后重复使用**，从而减少访存。

三者容易混，是因为**必须协同**：tile 的形状要和内存布局匹配才能合并访存，tile 的层级要和 block/thread 层级对齐。但概念上是三件事。

---

## 三、项目 Tiling 验证进度核查

**结论：算子链路（W4~W9）全程没有 tiling；tiling 只在 W10 用单个手写 matmul 验证过用法。**

### 3.1 W4~W7 算子链路：停在 tensor 级，无 tiling

`tools/lower_and_opt.sh` 只做 lowering + `fuse / canonicalize / cse`，全部作用于 linalg-on-tensors，未进入循环层。

### 3.2 W9 GPU pipeline：只有 Mapping，没有 Tiling

`tools/lower_to_gpu.sh:20-27` 的全部 pass：

```
--one-shot-bufferize
--convert-linalg-to-parallel-loops
--gpu-map-parallel-loops          ← 这是 Mapping，不是 Tiling
--convert-parallel-loops-to-gpu
--gpu-kernel-outlining
```

**其中没有任何 tiling pass。** 后果见 `work/文档-产出-工作以及实现和验证方式.md:201`：

> 遗留问题：无 tiling → 832 kernel 全是 1 block × 1 thread，性能低。

即：每个 parallel loop 直接映射成一个 block、一个线程，`block=(1,1,1)`。

### 3.3 W10：tiling 仅做了孤立验证

产物只有一个文件：`mlir/handwritten/tile_matmul_transform.mlir`。

- 形状：`matmul 128×256×64`
- `transform.structured.tile_using_for %matmul tile_sizes [32, 32, 16]`
- 结果：3 层嵌套 `scf.for`（M:4 / N:2 / K:16 个 tile）

**该文件未接入任何真实模型。** W10 的产出重心是 `passes/CountLinalgOps/`，tiling 属于"探明用法"级别的验证。

### 3.4 有一半是"做不了"，不是"没做"

`work/进度管理.md:69` 记录的阻塞：

> `scf-parallel-loop-tiling` 在 GPT-2 上破坏 GPU 映射结构（`parallel` → `cf.br` 不可 GPU map）→ 无法为 GPT-2 生成 tiled GPU kernel。

即 GPT-2 的 tiling 路径当时被 pass 行为**卡住**，而非单纯未开展。

---

## 四、与 832 kernel 性能问题的关系

本项目的 GPU pipeline 是 **有 Mapping、没有 Tiling**：循环直接映射到 gpu block/thread，但未先分块 → `block=(1,1,1)`，一个线程干完整个 128×256。

因此 `work/开发经验.md:131` 的表述需要精确化：

> 不是"没做 tiling 所以慢"，而是"**没做 tiling，测出来的数根本没有分析价值（因果倒置）**"。

这是相位顺序问题：`scf-parallel-loop-tiling`（或 transform dialect tiling）必须在 GPU 映射**之前**完成，否则映射出来的结构无法再分块。

---

## 五、文档不一致记录（待修）

关于 W10 tiling 有两条互相矛盾的记录：

| 出处 | 表述 |
|------|------|
| `work/工作划分.md:113` | 测试 transform dialect tiling（**该版本 mlir-opt 缺 `linalg-tile` op**） |
| `work/进度管理.md:70` | LinalgTransformOps **已存在**，但 transform 脚本 op 名写错 + 参数不匹配；修正为 `tile_using_for` 返回 3 个 `scf.for`，根参数改为 `builtin.module` |

以产物（`tile_matmul_transform.mlir`）和 `work/archive/项目进度总结_2026-07-13.md:49`（"Transform dialect tiling pipeline 已打通，matmul 32×32×16"）为准——**`tile_using_for` 是打通了的**。`工作划分.md:113` 的记录应属过时，需修正。

---

## 六、结论

1. **概念**：Tiling = 分块（迭代空间切分），**不是排布**；排布是 Layout；谁算哪块是 Mapping。
2. **进度**：算子验证**未到 tiling**。W4~W7 停在 tensor 级 + elementwise 融合；W9 的 GPU pipeline 只做 Mapping 未做 Tiling；tiling 仅在 W10 用单个手写 matmul 验证了 transform dialect 的用法。
3. **归因**：832 kernel `block=(1,1,1)` 的直接原因是"Mapping 先于 Tiling"，而非 tiling 本身缺失这一孤立事实。

---

## 七、后续可选动作

- [x] 重跑 `mlir-opt --transform-interpreter` 执行 `tile_matmul_transform.mlir` —— **链路可用**，见第八节 L1。
- [x] 评估在 GPU 映射**之前**插入 tiling —— 用 `tile_using_forall` 可映射，但**加 K 循环后会退回 (1,1,1)**，见第八节 L3 及其更正。
- [x] L2 语义验证（tiled 数值 == 未 tiled）—— **已完成**，`scripts/verify_tiling.py` 的 5 个变体数值全部一致（见第十一节）。
- [x] 建立 CPU 度量框架 —— `scripts/verify_tiling.py` 已就绪；实测发现 IREE 不适合作度量后端，改用 mlir-cpu-runner。
- [x] 用 mlir-cpu-runner（朴素循环，无自动 tiling）重做度量 —— **已完成**，见 11.6：K 切分带来 1.25~1.46x 收益，只切并行维反而略慢。
- [ ] 将 K 维 tiling 脚本接入 GPU pipeline，配合 `transform.gpu.map_nested_forall_to_threads` 修线程映射。
- [ ] 加 `transform.structured.promote` 做 shared memory promotion（K tiling 收益的真正来源）。
- [ ] 重新导出 GPT-2 Linalg IR（被 `.gitignore` 忽略）后在真实模型上跑 L3。
- [ ] 修正 `work/工作划分.md:113` 的过时记录。

---

## 八、验证结果更新（L1 / L3 已通过）

### 8.1 L1 结构性：tiling 链路可用 ✅

```bash
mlir-opt --transform-interpreter mlir/handwritten/tile_matmul_transform.mlir
```

产物：3 层嵌套 `scf.for`（M/32、N/32、K/16）+ 内层 `linalg.matmul 32×16×32` + `tensor.extract_slice`/`insert_slice`。

**结论**：`transform.structured.tile_using_for` 在本版本（LLVM 23.0.0git）可用。这**修正**了第五节中 `工作划分.md:113` 的说法——本版本确实不存在 `--linalg-tile` 独立 pass，但 tiling 可通过 **transform dialect** 实现（`tile_using_for` / `tile_using_forall`）。

### 8.2 L3 映射结构：tiling 改变 GPU 线程维度 ✅

| 路径 | GPU 映射结果 |
|------|-------------|
| 基线（无 tiling） | `threads in (%c1, %c1, %c1)` → **(1,1,1)**，一个线程干完 ❌ |
| `tile_using_for [32,32,16]`（W10 文件） | 产**串行** `scf.for` → 进不了 GPU 映射链 ❌ |
| `tile_using_forall [32,32]` + `--scf-forall-to-parallel` | `threads in (%2, %3, %c1)` → **不再恒为 1** ✅ |

```bash
# 基线
mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
         --convert-linalg-to-parallel-loops --gpu-map-parallel-loops \
         --convert-parallel-loops-to-gpu --gpu-kernel-outlining in.mlir
# 加 tiling
mlir-opt --transform-interpreter tile_forall.mlir \
  | mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
             --scf-forall-to-parallel \
             --convert-linalg-to-parallel-loops --gpu-map-parallel-loops \
             --convert-parallel-loops-to-gpu --gpu-kernel-outlining
```

**机理**：`tile_using_for` 产**串行** `scf.for`，GPU 映射链要的是可并行结构——这正是 GPT-2 上"`parallel → cf.br` 不可 GPU map"的成因。`tile_using_forall` 产 `scf.forall`（可并行），且本版本存在 `--scf-forall-to-parallel`，故可复用现有 GPU 映射管线。

> **⚠️ 后续更正（同日）**：上表的 `forall` 结论**有前提**——它只在**不插串行 K 循环**时成立。
> 把 `forall[32,32]` 与 `for[0,0,16]`（K 维）组合后，GPU kernel 的
> `known_block_size` **退回 `(1,1,1)`**：因为带 `iter_args` 累加器的串行 K 循环插在
> parallel 嵌套中间，legacy `--gpu-map-parallel-loops` 只映射**顶层** parallel，嵌套被打断后
> 它整体放弃，内层 32×32 留在未映射的 `scf.parallel`。
> 修法是改用 `transform.gpu.map_nested_forall_to_threads` / `map_forall_to_blocks`
> （能处理嵌套），且映射层级必须先于 K 分阶段确定。

---

## 九、K 维 tiling（"补 K 维"）

### 9.1 是什么

矩阵乘 `C[M,N] += A[M,K] × B[K,N]`，三维分两类：

| 维度 | 类型 | 含义 |
|------|------|------|
| M, N | 并行 | 输出空间坐标，各输出元素互不依赖 |
| **K** | **归约** | 每个输出元素都要累加完整条 K |

`tile_using_forall [32, 32]` 只切 M/N（forall 只能装并行维），内层仍是 **`32×256×32`——K 全宽**。补 K 维即把 K 也切块：

```
for k in 0..256 step 16:
    C_tile += A[32, k:k+16] × B[k:k+16, 32]
```

### 9.2 为什么（K tiling 为何是复用关键）

1. **让 tile 装进快速存储**：不切 K 时一个 32×32 输出 tile 要读 A 面板 `32×256×4B = 32KB` + B 面板 `256×32×4B = 32KB` + 累加器 4KB ≈ **68KB**，超过 shared memory 常见 48KB 上限；切 K=16 后 ≈ **8KB**，可放进 shared memory / 寄存器。
2. **才能做寄存器分块**：经典 GEMM 层次 = 寄存器累加小块 + shared memory 暂存 K 面板；无 K 循环则无法表达"加载面板→累加→换下一面板"。
3. **才能软件流水 / 双缓冲**：K 分多段才能预取下一段同时算当前段。
4. **才能 split-K 并行**（M·N 太小填不满 GPU 时）。

**一句话：M/N tiling 决定"能不能并行"，K tiling 决定"数据能不能复用"。**

### 9.3 配方（已验证）

**坑**：`tile_reduction_using_forall` **不能用于 `linalg.matmul`**：

```
error: 'linalg.matmul' op tiling parallel dimensions is not supported
       with partial reduction tiling strategies
```

它只支持**纯归约 op**（softmax 的 sum/max、LayerNorm 的 mean）。

**可行写法**：`forall` 切并行 + `for` 切 K，`0` 表示该维不切：

```mlir
// M,N → 并行
%t, %fa = transform.structured.tile_using_forall %m tile_sizes [32, 32]
  : (!transform.op<"linalg.matmul">) -> (!transform.op<"linalg.matmul">, !transform.op<"scf.forall">)
// K → 串行归约（0,0 表示 M,N 不再切）
%t2, %k = transform.structured.tile_using_for %m2 tile_sizes [0, 0, 16]
  : (!transform.op<"linalg.matmul">) -> (!transform.op<"linalg.matmul">, !transform.op<"scf.for">)
```

实测产物即**标准 GEMM 结构**：

```mlir
scf.forall (%i, %j) in (4, 2) {              // M/N 并行（可 GPU 映射）
   A 面板 32×256 / B 面板 256×32 / C 32×32
  scf.for %k = 0 to 256 step 16 {            // K 串行归约（局部性）
    A_k 32×16 / B_k 16×32
    linalg.matmul 32×16×32                   // 真正的 tile
    累加进累加器
  }
}
```

**语法注意**：这两个 transform op 的 `assemblyFormat` 用 **`by`** 且参数带 `=`，如
`... %m by num_threads = [4, 2] tile_sizes = [16]`；写成 `tile_sizes [16]` 会被解析器拒绝（`expected 'by'`）。

### 9.4 三种组合对比

| 变换 | 切了哪些维 | 并行性（threads ≠ 1） | 复用性（tile 装得下） |
|------|-----------|:--:|:--:|
| 无 tiling（现状） | — | ❌ | ❌ |
| `tile_using_forall [32,32]` | M, N | ✅ | ❌ |
| `tile_using_for [32,32,16]`（W10） | M, N, K | ❌（全串行） | ✅ |
| **`forall [32,32]` + `for [0,0,16]`** | M/N 并行 + K 串行 | ✅ | ✅ |

---

## 十、概念澄清：Tiling ≠ "切 K 维"

**Tiling 是对迭代空间做分块**（产生循环 + 切片进出）这一**变换本身**；切哪些维、切多大是**策略选择**，不是"切某一维"。

- `tile_using_forall [32, 32]` **已经是 tiling**（切了 M/N），不是"还没 tiling"。
- "补 K 维"是**在已有 tiling 上再加一维**，不是 tiling 的前提条件。
- **tiling 不保证性能**：还需 tile 能真正放进快速存储（需进一步的 shared memory promotion / vectorize）、tile size 合适、以及算子本身是 memory-bound 还是 compute-bound。

按算子类型看"该切哪些维"：

| 算子类型 | 归约维 | tiling 策略 |
|----------|--------|-------------|
| matmul（contraction） | K | M/N 并行 + K 归约（本节的组合写法） |
| elementwise（add/relu/GELU） | 无 | 只切并行维 |
| softmax / LayerNorm（纯归约） | sum/max/mean 维 | 可用 `tile_reduction_using_forall` |

---

## 十一、CPU 度量框架与首版数据（2026-09-13）

### 11.1 框架

`scripts/verify_tiling.py` —— 变体化对比：每个 tiling 策略一个变体，自动完成
「生成 payload → 跑 transform → 剥掉 transform module → 编译 → 运行 → 数值校验 → 计时」，
输出对比表 + `benchmarks/tables/table_h_tiling_cpu.csv`。

```bash
python scripts/verify_tiling.py --size 512 --reps 20
python scripts/verify_tiling.py --size 256 --variants base 32x32x16
```

**L2 语义验证随之完成**：5 个变体数值与 numpy `A@B` **全部一致** —— tiling 保语义确认。

### 11.2 首版数据（256³，reps=5）：预 tiling 一律大幅变慢

| 变体 | 中位(ms) | vs base | 数值 |
|------|---------:|--------:|:----:|
| base（不切） | 3.72 | 1.00x | ✅ |
| tile_32x32_K0 | 33.56 | 0.11x | ✅ |
| tile_64x64x32 | 73.18 | 0.05x | ✅ |
| tile_32x32x16 | 589.02 | 0.01x | ✅ |
| tile_16x16x8 | 3938.14 | 0.00x | ✅ |

### 11.3 诊断：同一 flatbuffer 换运行时配置可分离原因

| | local-task | local-sync | 差异 |
|---|---:|---:|---:|
| base | 0.49 ms | 0.32 ms | 1.5x |
| tile_32x32_K0 | 31.85 ms | 1.22 ms | **26x** |

**(a)** `tile_using_for` 产**串行** `scf.for`，抹掉了 IREE 本可提取的并行性；
**(b)** tiled IR 在 task 运行时下产生**病态的大量细粒度 dispatch**。

### 11.4 结论：IREE 不适合作 tiling 的度量后端

IREE 自身就是完整编译器（会做 tiling / vectorize / 并行化）。喂它预 tiled 的 IR，
等于**把它自己的能力打断**，测到的是「被打断的代价」，不是「tiling 的收益」。

要隔离 tiling 自身的缓存收益，需要**不做 tiling 的后端**：

```
mlir-opt --one-shot-bufferize --convert-linalg-to-loops ...  →  mlir-cpu-runner
```

这条路径也与本项目 Phase 3「原始 MLIR 直落 GPU」更可比；IREE 那条实际对应的是
torch.compile 一类的生产栈。

### 11.5 边界与遗留

- 只验证了「数值 + 缓存复用」这条轴；**shared memory promotion / occupancy 仍未验证**（GPU 独有，等 NVPTX）。
- base 绝对值在不同次运行间波动较大（0.49 ~ 3.72 ms），但 tiling 的惩罚远超噪声量级。
- **环境**：`conda activate novel_llm` 实际未生效，脚本跑在
  `/home/lwy/project/PythonProjectStarter/.../ProjectStarterVenv/bin/python`（Python 3.13）——
  该 venv 才有 `iree.compiler`。文档中「脚本都在 novel_llm 下跑」至少对 IREE 这条不成立。
- **IREE 限制**：`scf.forall`（张量上的 forall + `tensor.parallel_insert_slice`）会让
  IREE 段错误（崩在 `mlir::Value::getParentBlock()`），故 forall 形态无法用 IREE 度量。

### 11.6 runner 后端数据：K 切分才是收益来源

实现：`mlir-opt` 降到 LLVM dialect → `mlir-runner` 执行。`--convert-linalg-to-loops` 得**朴素循环
（无自动 tiling / vectorize）**，两点法（R=3/13）扣掉进程启动与 JIT 开销。校验值 = `K × reps`，全部通过。

| 变体 | 256³ vs base | 512³ vs base | 512³ GFLOPS |
|------|:---:|:---:|---:|
| base | 1.00x | 1.00x | 1.32 |
| tile_32x32_**K0**（只切并行） | 0.95x | **0.92x** | 1.22 |
| tile_32x32x16 | 1.31x | **1.41x** | 1.86 |
| tile_64x64x32 | 1.18x | 1.25x | 1.65 |
| tile_16x16x8 | 1.45x | **1.46x** | 1.93 |

**结论**：

1. **只切 M/N、不切 K → 反而变慢**（0.92~0.95x）：tile 仍要流式扫完整条 K，工作集溢出缓存，
   只付了切片/拷贝开销却没拿到复用收益。
2. **切了 K 才有收益**（1.25~1.46x）—— 与第九节「M/N 决定并行、K 决定复用」的推理一致，现由数据证实。
3. **收益随规模增长**（32x32x16：1.31→1.41x），符合 cache blocking 的预期。
4. 本组最佳为**小 tile**（16x16x8，1.46x）。

**注意事项**：朴素标量循环基线仅 ~1.3 GFLOPS，绝对值不真实，**只有比值有意义**；
且无 vectorize / promotion，GPU 上的图景不同（GPU 独有轴，仍待 NVPTX）。

**一个副产品发现**：`tile_using_for` 在张量层引入 `tensor.insert_slice`，bufferize 后变成
`memref.copy`（需要 runner utils 提供 `memrefCopy` 符号）—— 也就是说**朴素路径下 tiling 会引入真实的数据搬运**。
这正解释了为什么两种后端里"只切并行"都变慢，也说明 ③ 的 promotion（把面板提升到 shared memory /
寄存器、消除这次拷贝）才是 K tiling 收益的真正落点。

### 11.7 tile-size 扫描（512³）：存在明确的内部最优

`--tiles` 让框架可做扫描；512³、runner 后端：

| tile (M,N,K) | vs base | GFLOPS |
|--------------|:---:|---:|
| base | 1.00x | 1.32 |
| 8, 8, 4 | 1.13x | 1.49 |
| **16, 16, 8** | **1.46x** | 1.92 |
| 32, 32, 16 | 1.43x | 1.88 |
| 64, 64, 32 | 1.26x | 1.66 |
| 128, 128, 64 | 1.13x | 1.50 |
| 32, 32, 0（不切 K） | **0.96x** | 1.27 |

**典型 cache-blocking 曲线**：tile 太小被循环开销吃掉（8,8,4 → 1.13x），太大放不进缓存
（128,128,64 → 1.13x），**最优落在 16~32**；K 不切的变体是唯一倒退项，与 11.6 结论一致。

### 11.8 ② 映射修复的 IR 级探索（未完成，但约束已探明）

改用现代 transform dialect 路径（`scf.forall` → `gpu.launch` → threads），已查清以下**硬约束**：

| 约束 | 出处 |
|------|------|
| `map_nested_forall_to_threads` 必须作用在 **`gpu.launch`** 上 | `GPUTransformOps.td:149` |
| 它只支持**已 bufferize 的** `scf.forall`；遇到 **tensor forall 必然失败** | `GPUTransformOps.td:173, 188` |
| `map_forall_to_blocks` 的 `grid_dims` 必须是**空或 3 维**（实测报错） | 实测 |
| 它要求 `scf.forall` **自带 `mapping` 属性**（实测 `error: scf.forall op requires a mapping attribute`） | 实测 |

最后一条是关键卡点：本版本的 `transform.structured.tile_using_forall` **不产出 `mapping` 属性**
（其 arguments 只有 target / dynamic_sizes / static_sizes / interchange / scalable_sizes），
而 `map_forall_to_blocks` 又要求它已存在。

**因此 ② 还需要打通**：`tile_using_forall` → 给 forall 附加 `#gpu.block<...>` 映射 → bufferize
→ `map_forall_to_blocks` → `map_nested_forall_to_threads` → `gpu-kernel-outlining`。
这是一个独立的小工作包，**且只能做 IR 结构验证**（无法执行，NVPTX 未通）。
