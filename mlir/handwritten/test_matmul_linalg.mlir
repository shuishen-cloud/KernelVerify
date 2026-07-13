// W9: 手写 Linalg matmul — 测试 Linalg → GPU lowering pipeline
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %C: tensor<128x64xf32>)
    -> tensor<128x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                     outs(%C : tensor<128x64xf32>)
     -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}
