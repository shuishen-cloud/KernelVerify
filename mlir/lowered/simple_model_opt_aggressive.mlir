#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_3_2_torch.float32> : tensor<3x2xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>
    %0 = tensor.empty() : tensor<2x3xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<3x2xf32>) outs(%0 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x3xf32>
    %2 = tensor.empty() : tensor<1x3xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x3xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %1 : tensor<1x2xf32>, tensor<2x3xf32>) outs(%3 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<1x3xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_1 : tensor<1x3xf32>, tensor<3xf32>) outs(%2 : tensor<1x3xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      %7 = arith.cmpf ugt, %6, %cst : f32
      %8 = arith.select %7, %6, %cst : f32
      linalg.yield %8 : f32
    } -> tensor<1x3xf32>
    return %5 : tensor<1x3xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_3_2_torch.float32: "0x04000000A5C6433E21F7B8BE0EF5AD3DB81416BF053A153F27FBDB3E",
      torch_tensor_3_torch.float32: "0x0400000042B979BE97EFFABC2327D4BE"
    }
  }
#-}

