module {
  func.func @main(%arg0: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,3],f32> {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_3_2_torch.float32> : tensor<3x2xf32>) : !torch.vtensor<[3,2],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_3_torch.float32> : tensor<3xf32>) : !torch.vtensor<[3],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.aten.transpose.int %0, %int0, %int1 : !torch.vtensor<[3,2],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,3],f32>
    %3 = torch.aten.mm %arg0, %2 : !torch.vtensor<[1,2],f32>, !torch.vtensor<[2,3],f32> -> !torch.vtensor<[1,3],f32>
    %4 = torch.aten.add.Tensor %3, %1, %int1 : !torch.vtensor<[1,3],f32>, !torch.vtensor<[3],f32>, !torch.int -> !torch.vtensor<[1,3],f32>
    %5 = torch.aten.relu %4 : !torch.vtensor<[1,3],f32> -> !torch.vtensor<[1,3],f32>
    return %5 : !torch.vtensor<[1,3],f32>
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
