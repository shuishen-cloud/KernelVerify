#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_3_2_torch.float32> : tensor<3x2xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %0 = tensor.empty() : tensor<2x3xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<3x2xf32>) outs(%0 : tensor<2x3xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x3xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x2xf32>, tensor<2x3xf32>) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<1x3xf32>, tensor<3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<1x3xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x3xf32>) outs(%1 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %6 = arith.cmpf ugt, %in, %cst : f32
      %7 = arith.select %6, %in, %cst : f32
      linalg.yield %7 : f32
    } -> tensor<1x3xf32>
    return %5 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_2_torch.float32: "0x040000002AEB7BBEF78EAD3EDFCB333F96F206BD4C8D1A3F6BECCE3E",
      torch_tensor_3_torch.float32: "0x0400000079AA31BFA797913E2AB4AD3E"
    }
  }
#-}

