module {
  func.func @main(%arg0: !torch.vtensor<[1,4],f32>) -> !torch.vtensor<[1,2],f32> {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>) : !torch.vtensor<[2],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_2_8_torch.float32> : tensor<2x8xf32>) : !torch.vtensor<[2,8],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_8_4_torch.float32> : tensor<8x4xf32>) : !torch.vtensor<[8,4],f32>
    %3 = torch.vtensor.literal(dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %4 = torch.aten.transpose.int %2, %int0, %int1 : !torch.vtensor<[8,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,8],f32>
    %5 = torch.aten.mm %arg0, %4 : !torch.vtensor<[1,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[1,8],f32>
    %6 = torch.aten.add.Tensor %5, %3, %int1 : !torch.vtensor<[1,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[1,8],f32>
    %7 = torch.aten.relu %6 : !torch.vtensor<[1,8],f32> -> !torch.vtensor<[1,8],f32>
    %8 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[8,2],f32>
    %9 = torch.aten.mm %7, %8 : !torch.vtensor<[1,8],f32>, !torch.vtensor<[8,2],f32> -> !torch.vtensor<[1,2],f32>
    %10 = torch.aten.add.Tensor %9, %0, %int1 : !torch.vtensor<[1,2],f32>, !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
    return %10 : !torch.vtensor<[1,2],f32>
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
