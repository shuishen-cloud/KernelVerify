#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> ()>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_2_8_torch.float32> : tensor<2x8xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_8_4_torch.float32> : tensor<8x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<8x4xf32>) outs(%0 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x8xf32>
    %2 = tensor.empty() : tensor<1x8xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%2 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x8xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %1 : tensor<1x4xf32>, tensor<4x8xf32>) outs(%3 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.mulf %in, %in_4 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<1x8xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%4, %cst_3 : tensor<1x8xf32>, tensor<8xf32>) outs(%2 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.addf %in, %in_4 : f32
      %13 = arith.cmpf ugt, %12, %cst : f32
      %14 = arith.select %13, %12, %cst : f32
      linalg.yield %14 : f32
    } -> tensor<1x8xf32>
    %6 = tensor.empty() : tensor<8x2xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_1 : tensor<2x8xf32>) outs(%6 : tensor<8x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<8x2xf32>
    %8 = tensor.empty() : tensor<1x2xf32>
    %9 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%8 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x2xf32>
    %10 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %7 : tensor<1x8xf32>, tensor<8x2xf32>) outs(%9 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.mulf %in, %in_4 : f32
      %13 = arith.addf %out, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<1x2xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map6, #map1], iterator_types = ["parallel", "parallel"]} ins(%10, %cst_0 : tensor<1x2xf32>, tensor<2xf32>) outs(%8 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %12 = arith.addf %in, %in_4 : f32
      linalg.yield %12 : f32
    } -> tensor<1x2xf32>
    return %11 : tensor<1x2xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.float32: "0x04000000A5587ABE36B7B0BE",
      torch_tensor_2_8_torch.float32: "0x04000000C46C563E3A61B13D2793773E36324DBEB68C423C24F5F1BB534009BE126AB4BE9D6B7C3DD13990BEFD57D3BDE573D3BD3EF398BC14FAB33EB0A23E3E3FF4273D",
      torch_tensor_8_4_torch.float32: "0x0400000008C4ED3D407D7CBC9C4230BEF0C6553ECA0B93BE78E87DBEC6A6F23E7275BCBE9AF79E3EF898F03E94C2EDBEE0EDCD3DCC7DE7BEE059113EF0FB16BDA0F35F3EEAB4D2BE408F6D3C38A98FBE2C4D01BE3018853E5E6B8BBE28853A3E74EC7FBE503317BDD8AD923E7C0ECA3E4435BE3EFCB06DBE34BB14BE8E49C23EA09A80BE",
      torch_tensor_8_torch.float32: "0x0400000006B38EBE7C534C3E00E5A8BBB8EFD63D4CD04CBED8DAE33DB00AE9BE48104F3E"
    }
  }
#-}

