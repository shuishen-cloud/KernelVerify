// W10: Transform dialect Linalg tiling — 修正版
// tile_using_for 自动产生 scf.for + tiled linalg op
// 不需要额外的 lower_using_for_to_scf_for

func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %C: tensor<128x64xf32>)
    -> tensor<128x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                     outs(%C : tensor<128x64xf32>)
     -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}

module @transform_module attributes {transform.target_tag="linalg", transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.op<"builtin.module">) {
    // 1. 从根模块匹配 func.func
    %func = transform.structured.match ops{["func.func"]} in %root
      : (!transform.op<"builtin.module">) -> !transform.op<"func.func">

    // 2. 在 func 中匹配 linalg.matmul
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
      : (!transform.op<"func.func">) -> !transform.op<"linalg.matmul">

    // 3. Tiling: M=32, N=32, K=16 → 自动产生 scf.for
    %tiled, %lm, %ln, %lk = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 16]
      : (!transform.op<"linalg.matmul">) -> (!transform.op<"linalg.matmul">, !transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)

    transform.yield
  }
}
