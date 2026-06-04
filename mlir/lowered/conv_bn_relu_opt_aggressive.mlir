#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>
module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<i64>, %arg3: tensor<1x1x16x16xf32>) -> tensor<1x8x16x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_8_torch.float32_2> : tensor<8xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_8_torch.float32_1> : tensor<8xf32>
    %cst_3 = arith.constant 1.000000e-05 : f64
    %cst_4 = arith.constant dense_resource<torch_tensor_8_1_3_3_torch.float32> : tensor<8x1x3x3xf32>
    %cst_5 = arith.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %padded = tensor.pad %arg3 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg4: index, %arg5: index, %arg6: index, %arg7: index):
      tensor.yield %cst : f32
    } : tensor<1x1x16x16xf32> to tensor<1x1x18x18xf32>
    %0 = tensor.empty() : tensor<1x8x16x16xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<8xf32>) outs(%0 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8x16x16xf32>
    %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %cst_4 : tensor<1x1x18x18xf32>, tensor<8x1x3x3xf32>) outs(%1 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %4 = arith.mulf %in, %in_9 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1x8x16x16xf32>
    %expanded = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [8, 1, 1] : tensor<8xf32> into tensor<8x1x1xf32>
    linalg.generic {indexing_maps = [#map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<8x1x1xf32>) {
    ^bb0(%in: f32):
      %4 = arith.truncf %cst_3 : f64 to f32
      %5 = arith.addf %in, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf one, %6, %cst : f32
      cf.assert %7, "unimplemented: tensor with zero element"
      linalg.yield
    }
    %expanded_6 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [8, 1, 1] : tensor<8xf32> into tensor<8x1x1xf32>
    linalg.generic {indexing_maps = [#map1, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded : tensor<1x8x16x16xf32>, tensor<8x1x1xf32>) {
    ^bb0(%in: f32, %in_9: f32):
      %4 = arith.truncf %cst_3 : f64 to f32
      %5 = arith.addf %in_9, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf one, %6, %cst : f32
      cf.assert %7, "unimplemented: tensor with zero element"
      linalg.yield
    }
    %expanded_7 = tensor.expand_shape %cst_2 [[0, 1, 2]] output_shape [8, 1, 1] : tensor<8xf32> into tensor<8x1x1xf32>
    linalg.generic {indexing_maps = [#map1, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded : tensor<1x8x16x16xf32>, tensor<8x1x1xf32>) {
    ^bb0(%in: f32, %in_9: f32):
      %4 = arith.truncf %cst_3 : f64 to f32
      %5 = arith.addf %in_9, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf one, %6, %cst : f32
      cf.assert %7, "unimplemented: tensor with zero element"
      linalg.yield
    }
    %expanded_8 = tensor.expand_shape %cst_1 [[0, 1, 2]] output_shape [8, 1, 1] : tensor<8xf32> into tensor<8x1x1xf32>
    linalg.generic {indexing_maps = [#map1, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded : tensor<1x8x16x16xf32>, tensor<8x1x1xf32>) {
    ^bb0(%in: f32, %in_9: f32):
      %4 = arith.truncf %cst_3 : f64 to f32
      %5 = arith.addf %in_9, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf one, %6, %cst : f32
      cf.assert %7, "unimplemented: tensor with zero element"
      linalg.yield
    }
    %3 = linalg.generic {indexing_maps = [#map1, #map6, #map6, #map6, #map6, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded_6, %expanded, %expanded_7, %expanded_8 : tensor<1x8x16x16xf32>, tensor<8x1x1xf32>, tensor<8x1x1xf32>, tensor<8x1x1xf32>, tensor<8x1x1xf32>) outs(%0 : tensor<1x8x16x16xf32>) {
    ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %in_12: f32, %out: f32):
      %4 = arith.truncf %cst_3 : f64 to f32
      %5 = arith.addf %in_10, %4 : f32
      %6 = math.sqrt %5 : f32
      %7 = arith.cmpf one, %6, %cst : f32
      cf.assert %7, "unimplemented: tensor with zero element"
      %8 = arith.divf %cst_0, %6 : f32
      %9 = arith.subf %in, %in_9 : f32
      %10 = arith.mulf %9, %8 : f32
      %11 = arith.mulf %10, %in_11 : f32
      %12 = arith.addf %11, %in_12 : f32
      %13 = arith.cmpf ugt, %12, %cst : f32
      %14 = arith.select %13, %12, %cst : f32
      linalg.yield %14 : f32
    } -> tensor<1x8x16x16xf32>
    return %3 : tensor<1x8x16x16xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_torch.float32_2: "0x040000000000000000000000000000000000000000000000000000000000000000000000",
      torch_tensor_8_torch.float32_1: "0x040000000000803F0000803F0000803F0000803F0000803F0000803F0000803F0000803F",
      torch_tensor_8_1_3_3_torch.float32: "0x04000000F6CB0CBD668A743EB6ED93BDC6128F3DC69D6FBED7F29B3EA47D8DBE76A5CEBDF66CB23D5662923B98901FBEAB09B6BBBE3D09BE1B95D4BD4051A13CDBE6483EDF0599BE805499BCCB22493D005AF4BB1E9357BEE0E3A2BE2855213E00C0623E0E8280BE3BE865BEC084323E205A10BD5BE40CBEFBACE1BDBED3A53E8B5E293DB67B9FBE96F84EBD1070593E688711BE94EB93BEB8214F3EEE26683ECA3B9F3E6A6A91BE3E6C44BE5060A83E0010E93B5C2E813ECC8880BE104C903E304DD03D58C986BE7BD5973E0080EF3CF20997BE235A533E90E2B3BD1EB245BEDE596EBEBE3F4A3E002E84BE4B6671BE63FB69BE562BC9BDD625E93C4AD5A13E66B39C3E4B101D3D002628BC7746A2BEE6A45C3EF09219BE36041F3EFBD7A53EEECF55BE",
      torch_tensor_8_torch.float32: "0x04000000383389BE806B3CBC7F358D3E0606B2BDF36466BE002710BDD68A54BD4B63953E"
    }
  }
#-}

