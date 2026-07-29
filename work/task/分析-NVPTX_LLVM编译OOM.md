# 分析：NVPTX LLVM 重新编译 OOM

> 日期：2026-07-29 | 状态：🔴 阻塞（分析已完成，待重建）

## 一、背景

当前 LLVM 安装在 `/home/lwy/download/llvm-project/install/`（版本 23.0.0git），只编译了 X86 target。Phase 3 需要将 Linalg GPU IR 继续 Lowering 到 PTX/可执行代码，这依赖 LLVM 的 NVPTX 后端。

重编译 LLVM 加入 NVPTX 时，遭遇了 **make 卡死 + 链接器 OOM + lld 不兼容** 的连锁问题。

## 二、当前系统环境

| 项目 | 数值 |
|------|------|
| 物理内存 | 15GiB (约 16GB) |
| Swap | 4.0GiB |
| 总可用 | ~20GB |
| CPU | 32 核 |
| LLVM 源码 | `/home/lwy/download/llvm-project/` |
| 现有构建目录 | `build/`（6.5GB，1669 个 .o，Debug） |
| 现有安装目录 | `install/`（Release，仅 X86，Shared Libs） |

## 三、失败构建的 CMake 配置

从 `/home/lwy/download/llvm-project/build/CMakeCache.txt` 中提取的关键参数：

| 参数 | 失败构建的值 | 后果 |
|------|:---:|------|
| `CMAKE_BUILD_TYPE` | **Debug** | `-g` 全量调试符号 + 无优化 → `.o` 文件体积暴增 (10~50×) |
| `BUILD_SHARED_LIBS` | **OFF** | 30+ 个 `.a` 静态库，链接时全量加载到内存 |
| `LLVM_BUILD_LLVM_DYLIB` | **OFF** | 无单一共享库，每个工具独立链接所有静态库 |
| `LLVM_ENABLE_ASSERTIONS` | **ON** | 大量断言代码，进一步膨胀二进制体积 |
| `LLVM_PARALLEL_LINK_JOBS` | **空（无限制）** | 多个链接进程可同时运行 |
| `LLVM_TARGETS_TO_BUILD` | `X86;NVPTX` | 需编译两个后端（比仅 X86 更多代码） |
| 生成器 | **make（Unix Makefiles）** | 不强制线程限制，job 调度粗糙 |
| `LLVM_ENABLE_LLD` | **OFF** | 使用 GNU ld（比 lld 更耗内存） |

构建目录现状：**6.5GB**，**1669 个 .o 文件**（这是 Debug 模式下仅 X86+NVPTX 的编译产物，还没包含 tools）。

## 四、OOM 的三层根因

### 第 1 层：Debug 静态库链接 = 内存黑洞

在 Debug + 静态库模式下，最终链接 `mlir-opt` 需要将 30+ 个 `.a` 文件全部解析并链接成一个可执行文件。每个 `.a` 在 Debug 模式下的估算体积：

```
libMLIRLinalgDialect.a     ~800MB  (含 -g 符号)
libMLIRAffineDialect.a     ~500MB
libMLIRIR.a                ~400MB
libMLIRTransformDialect.a  ~300MB
libLLVMX86CodeGen.a        ~600MB
libLLVMNVPTXCodeGen.a      ~300MB
libLLVMSelectionDAG.a      ~400MB
... 其余 20+ 个静态库 ...
──────────────────────────────────
单次链接 ≈ 需要 8-12GB RSS
```

GNU ld 链接时会把每个 `.a` 的符号表全部加载进内存，Debug 符号（DWARF）让这个负担放大了 10~50 倍。

### 第 2 层：make 无并发上限 → 多链接内存叠加

`LLVM_PARALLEL_LINK_JOBS` 为空时，make 对链接任务并行度没有限制。LLVM 构建会同时链接多个目标：

```
同时运行: mlir-opt 链接 (10GB) + mlir-runner 链接 (8GB) + llvm-config 链接 (3GB)
        + llvm-tblgen 链接 (5GB) + mlir-tblgen 链接 (6GB) + ...
```

在 32 核机器上 `make` 可能同时启动 10+ 个链接进程：

```
最坏情况估算: 10 个链接器 × 10GB = 100GB 虚拟内存
实际物理内存: 16GB + 4GB swap = 20GB
→ 系统疯狂 swap → 界面完全卡死 ("make 卡死")
→ OOM Killer 最终杀掉链接进程 ("链接 OOM")
```

这就是为什么看起来 make"卡住"了 — 其实没有死，只是在等 swap 来回换页，但一页一页地换到天荒地老。

### 第 3 层：GNU ld 比 lld 更吃内存

`LLVM_ENABLE_LLD=OFF` 使用的是系统 GNU ld。GNU ld 的静态库链接比 lld 多消耗 30-50% 内存。`install-debug.sh` 脚本也提到过 "lld 不兼容"，这意味着即使想切 lld 也有障碍。

## 五、为什么 install.sh 能成功（对比分析）

同一个源码树，`install.sh` 一次就过，`build/` 的 Debug 配置却 OOM，差异如下：

| 参数 | install.sh（✅ 成功） | build/ 当前配置（❌ OOM） |
|------|------|------|
| `CMAKE_BUILD_TYPE` | **Release** | **Debug** |
| `LLVM_BUILD_SHARED_LIBS` | **ON** | **OFF** |
| `LLVM_BUILD_LLVM_DYLIB` | **ON** | **OFF** |
| `LLVM_ENABLE_ASSERTIONS` | **OFF** | **ON** |
| `LLVM_TARGETS_TO_BUILD` | `X86` | `X86;NVPTX` |
| `LLVM_PARALLEL_LINK_JOBS` | `2` | 空（无限制） |
| `LLVM_PARALLEL_COMPILE_JOBS` | `6` | 空（无限制，32 核） |
| 生成器 | make | make |
| 文件总数 | ~300 .o（Release 优化后） | 1669 .o（Debug） |
| 单链接器 RSS | ~200MB（1 个 libLLVM.so） | ~8-12GB（30+ 个 .a） |

**结论**：install.sh 的成功秘诀是 `Release + Shared Libs`，这组配置把链接内存需求从 **10GB 级直接压到 200MB 级**。不需要更强大的硬件，只需正确的 CMake 参数。

## 六、验证方法

在重建前，可用以下命令验证根因（在构建目录下操作）：

### 6.1 检查静态库膨胀程度

```bash
# 看单个 .a 多大 — Debug 下往往 500MB+
ls -lh /home/lwy/download/llvm-project/build/lib/libMLIR*.a 2>/dev/null | sort -k5 -hr | head -10
```

**预期**：单个文件 300-800MB — 确认 Debug 膨胀是真实的。

### 6.2 测量单次链接的内存

```bash
# 在 build 目录中只链接 mlir-opt，用 1 个 job（避免并发干扰）
cd /home/lwy/download/llvm-project/build
/usr/bin/time -v cmake --build . --target mlir-opt -j1 2>&1 | grep "Maximum resident"
```

**预期**：RSS 超过 8000000 (KB) = 8GB — 确认单个链接就接近物理内存极限。

### 6.3 监控系统 OOM 事件

```bash
# 重建过程中在另一终端运行，观察是否出现 OOM kill
watch -n 1 'free -h; echo ---; dmesg | grep -i "oom\|out of memory\|killed" | tail -5'
```

### 6.4 验证修复后的内存占用

用 Release + Shared Libs 参数重建后，重复 6.2 的测量：

```bash
cd /home/lwy/download/llvm-project/build-nvptx
/usr/bin/time -v cmake --build . --target mlir-opt -j1 2>&1 | grep "Maximum resident"
```

**预期**：RSS 在 200-500MB 范围，相比 8GB+ 降低 95%+。

## 七、解决方案

### 完整的 Ninja + Release 构建脚本

```bash
#!/bin/bash
set -e

THREADS=6                               # 保守线程数（避免编译阶段也 OOM）
BUILD_DIR="./build-nvptx"
INSTALL_DIR="./install-nvptx"

# CMake 配置（在 llvm-project 根目录执行）
cmake -G Ninja -S llvm -B ${BUILD_DIR} \
  -DCMAKE_BUILD_TYPE=Release \          # 🔑 Release 代替 Debug（去掉 -g，开启 -O3）
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DLLVM_BUILD_SHARED_LIBS=ON \         # 🔑 共享库代替 30+ 静态库
  -DLLVM_BUILD_LLVM_DYLIB=ON \          # 🔑 单一 libLLVM.so，链接只需 ~200MB
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_PLUGINS=ON \
  -DLLVM_ENABLE_PIC=ON \
  -DLLVM_ENABLE_ASSERTIONS=OFF \        # 关闭断言（Release 下不应开启）
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \ # 目标：加入 NVPTX
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_INCLUDE_UTILS=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_USE_SPLIT_DWARF=OFF \
  -DLLVM_PARALLEL_COMPILE_JOBS=${THREADS} \  # 编译阶段最多 6 线程
  -DLLVM_PARALLEL_LINK_JOBS=1 \          # 🔑 串行链接，杜绝多链接叠加
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DMLIR_BUILD_EXAMPLES=OFF \
  -DMLIR_BUILD_TESTS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF

# 编译
ninja -C ${BUILD_DIR} -j${THREADS}

# 安装
ninja -C ${BUILD_DIR} install

echo "Done: ${INSTALL_DIR}/bin/mlir-opt"
${INSTALL_DIR}/bin/mlir-opt --version
```

### 改动总结

| 改动 | 效果 | 预计内存节省 |
|------|------|:---:|
| `Debug → Release` | 去掉 `-g` 调试符号，开启 `-O3 -DNDEBUG` | ~60% |
| `STATIC → SHARED` | 1 个 `libLLVM.so` 替代 30+ 个 `.a` | **~95%** |
| `LINK_JOBS=1` | 串行链接，杜绝多进程内存叠加 | 防止叠加爆 |
| `Assertions=OFF` | 减少二进制中的断言代码 | ~10% |
| `make → Ninja` | 更智能的 job 调度，更快的增量编译 | 稳定可控 |

### 时间估算

- CMake 配置：~2 分钟
- 编译：~30-60 分钟（6 线程，取决于 NVPTX 代码量）
- 安装：~1 分钟

## 八、风险与对策

| 风险 | 概率 | 对策 |
|------|:---:|------|
| NVPTX 在 Release 下的代码生成有 bug | 低 | Release 优化级别足够成熟；如有问题回退到 `RelWithDebInfo` |
| lld 仍然不兼容（install-debug.sh 的备注） | 中 | 不强制 lld，使用 GNU ld；Release + Shared 模式下 GNU ld 也够用 |
| 编译阶段也 OOM（32 核 × Debug 编译线程） | 低 | `LLVM_PARALLEL_COMPILE_JOBS=6` 严格限制 |
| Ninja 未安装 | 低 | `sudo apt install ninja-build` |
| 磁盘空间不足 | 低 | Release 构建目录约 2-3GB，检查 `df -h` |

## 九、成功后验证

```bash
# 1. 确认 NVPTX 后端已编译进去
/home/lwy/download/llvm-project/install-nvptx/bin/mlir-opt --version

# 2. 生成 PTX 代码
/home/lwy/download/llvm-project/install-nvptx/bin/mlir-opt \
  mlir/lowered/gpt2_dynamic_gpu.mlir \
  --convert-gpu-to-nvvm \
  --gpu-to-cubin \
  -o /tmp/test.ptx

# 3. 确认 PTX 语法有效
ptxas --version
ptxas /tmp/test.ptx
```

## 十、教训

1. **永远不要在 Debug 模式下做全量 LLVM 静态链接** — 即使有 64GB 内存也会吃力。LLVM 的 Debug 构建只适合开发单个 pass 时用。
2. **`LLVM_PARALLEL_LINK_JOBS` 必须显式设置** — 空值意味着 "无限"，不意味着 "1"。
3. **Release + Shared Libs 是分发 LLVM 工具链的正确姿势** — install.sh 已经验证了这个公式，只需要加上 NVPTX target。
4. **make 的 "卡死" 通常是 swap 颠簸而非真死锁** — 可以用 `dmesg` 和 `free -h` 区分。
