#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_2_8_torch.float32> : tensor<2x8xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_8_4_torch.float32> : tensor<8x4xf32>
    %cst_3 = arith.constant dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>
    %0 = tensor.empty() : tensor<4x8xf32>
    %transposed = linalg.transpose ins(%cst_2 : tensor<8x4xf32>) outs(%0 : tensor<4x8xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<1x8xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<1x4xf32>, tensor<4x8xf32>) outs(%2 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_3 : tensor<1x8xf32>, tensor<8xf32>) outs(%1 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %11 = arith.addf %in, %in_5 : f32
      linalg.yield %11 : f32
    } -> tensor<1x8xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<1x8xf32>) outs(%1 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.cmpf ugt, %in, %cst : f32
      %12 = arith.select %11, %in, %cst : f32
      linalg.yield %12 : f32
    } -> tensor<1x8xf32>
    %6 = tensor.empty() : tensor<8x2xf32>
    %transposed_4 = linalg.transpose ins(%cst_1 : tensor<2x8xf32>) outs(%6 : tensor<8x2xf32>) permutation = [1, 0] 
    %7 = tensor.empty() : tensor<1x2xf32>
    %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %9 = linalg.matmul ins(%5, %transposed_4 : tensor<1x8xf32>, tensor<8x2xf32>) outs(%8 : tensor<1x2xf32>) -> tensor<1x2xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %cst_0 : tensor<1x2xf32>, tensor<2xf32>) outs(%7 : tensor<1x2xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %11 = arith.addf %in, %in_5 : f32
      linalg.yield %11 : f32
    } -> tensor<1x2xf32>
    return %10 : tensor<1x2xf32>
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

