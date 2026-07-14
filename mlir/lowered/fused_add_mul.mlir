#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @test_add_mul(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = tensor.empty() : tensor<4x8xf32>
    %2 = tensor.empty() : tensor<4x8xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) outs(%2 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      %5 = arith.mulf %4, %in_1 : f32
      linalg.yield %5 : f32
    } -> tensor<4x8xf32>
    return %3 : tensor<4x8xf32>
  }
  func.func @test_double(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = tensor.empty() : tensor<10xf32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : tensor<10xf32>, tensor<10xf32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %9 = arith.addf %in, %in_0 : f32
      %10 = arith.mulf %9, %in_1 : f32
      linalg.yield %10 : f32
    } -> tensor<10xf32>
    %4 = tensor.empty() : tensor<10xf32>
    %5 = linalg.add ins(%arg0, %arg3 : tensor<10xf32>, tensor<10xf32>) outs(%4 : tensor<10xf32>) -> tensor<10xf32>
    %6 = tensor.empty() : tensor<10xf32>
    %7 = linalg.mul ins(%5, %arg2 : tensor<10xf32>, tensor<10xf32>) outs(%6 : tensor<10xf32>) -> tensor<10xf32>
    %8 = tensor.empty() : tensor<10xf32>
    return %3, %7 : tensor<10xf32>, tensor<10xf32>
  }
  func.func @test_no_match(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}

