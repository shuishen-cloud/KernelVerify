// ═══════════════════════════════════════════════════════════════════
// 测试输入: add → mul 的 pattern，用于验证 SimNewBackend Pass
//
// 用法:
//   # 1. 查看原始 IR
//   torch-mlir-opt test_fused_add_mul.mlir
//
//   # 2. 运行融合 Pass
//   torch-mlir-opt --load-pass-plugin=build/libSimNewBackend.so \
//     --pass-pipeline="builtin.module(sim-fuse-add-mul)" \
//     test_fused_add_mul.mlir
//
//   # 3. 融合 + 展开 (等价变换，验证语义保持)
//   torch-mlir-opt --load-pass-plugin=build/libSimNewBackend.so \
//     --pass-pipeline="builtin.module(sim-fuse-add-mul,sim-expand-fused)"
// ═══════════════════════════════════════════════════════════════════

// ── 测试 1: 基本 add → mul pattern ──
func.func @test_add_mul(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>, %c: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %init1 = tensor.empty() : tensor<4x8xf32>
  %add = linalg.add ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
                    outs(%init1 : tensor<4x8xf32>) -> tensor<4x8xf32>

  %init2 = tensor.empty() : tensor<4x8xf32>
  %mul = linalg.mul ins(%add, %c : tensor<4x8xf32>, tensor<4x8xf32>)
                    outs(%init2 : tensor<4x8xf32>) -> tensor<4x8xf32>

  return %mul : tensor<4x8xf32>
}

// ── 测试 2: 多个 add→mul (函数内有两组) ──
func.func @test_double(%a: tensor<10xf32>, %b: tensor<10xf32>,
                        %c: tensor<10xf32>, %d: tensor<10xf32>)
    -> (tensor<10xf32>, tensor<10xf32>) {
  // 第一组: a + b → mul c
  %init1 = tensor.empty() : tensor<10xf32>
  %add1 = linalg.add ins(%a, %b : tensor<10xf32>, tensor<10xf32>)
                     outs(%init1 : tensor<10xf32>) -> tensor<10xf32>
  %init2 = tensor.empty() : tensor<10xf32>
  %mul1 = linalg.mul ins(%add1, %c : tensor<10xf32>, tensor<10xf32>)
                     outs(%init2 : tensor<10xf32>) -> tensor<10xf32>

  // 第二组: a + d → mul c  (add 的输出被多处使用 -> 不会被融合)
  %init3 = tensor.empty() : tensor<10xf32>
  %add2 = linalg.add ins(%a, %d : tensor<10xf32>, tensor<10xf32>)
                     outs(%init3 : tensor<10xf32>) -> tensor<10xf32>
  %init4 = tensor.empty() : tensor<10xf32>
  %mul2 = linalg.mul ins(%add2, %c : tensor<10xf32>, tensor<10xf32>)
                     outs(%init4 : tensor<10xf32>) -> tensor<10xf32>
  // add2 还用在一个独立的地方
  %init5 = tensor.empty() : tensor<10xf32>
  %unrelated = linalg.add ins(%add2, %c : tensor<10xf32>, tensor<10xf32>)
                          outs(%init5 : tensor<10xf32>) -> tensor<10xf32>

  return %mul1, %mul2 : tensor<10xf32>, tensor<10xf32>
}

// ── 测试 3: 无匹配 (只有 add，没有 mul) ──
func.func @test_no_match(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>)
    -> tensor<4x8xf32> {
  %init = tensor.empty() : tensor<4x8xf32>
  %add = linalg.add ins(%a, %b : tensor<4x8xf32>, tensor<4x8xf32>)
                    outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
  return %add : tensor<4x8xf32>
}
