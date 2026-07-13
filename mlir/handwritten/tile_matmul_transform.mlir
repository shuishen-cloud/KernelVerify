// W10: Transform dialect 脚本 — Linalg tiling
// 在 linalg.matmul 上做 32x32 tiling，然后 lower 到 GPU

func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x64xf32>, %C: tensor<128x64xf32>)
    -> tensor<128x64xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<128x256xf32>, tensor<256x64xf32>)
                     outs(%C : tensor<128x64xf32>)
     -> tensor<128x64xf32>
  return %0 : tensor<128x64xf32>
}

module @transform_module attributes {transform.target_tag="linalg"} {
  transform.named_sequence @__transform_main(%arg0: !transform.op<"func.func">) {
    // 获取 matmul 并应用 tiling
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.op<"func.func">) -> !transform.op<"linalg.matmul">
    %tiled_linalg, %loops = transform.structured.tile_using_for %matmul tile_sizes [32, 32, 16]
      : (!transform.op<"linalg.matmul">) -> (!transform.op<"linalg.matmul">, !transform.any_op)

    // Lower tiled linalg to loops
    transform.structured.lower_using_for_to_scf_for %loops
      : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
