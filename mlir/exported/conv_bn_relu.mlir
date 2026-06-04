module {
  func.func @main(%arg0: !torch.vtensor<[8],f32>, %arg1: !torch.vtensor<[8],f32>, %arg2: !torch.vtensor<[],si64>, %arg3: !torch.vtensor<[1,1,16,16],f32>) -> !torch.vtensor<[1,8,16,16],f32> {
    %false = torch.constant.bool false
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_8_torch.float32_2> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_8_torch.float32_1> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %int-1 = torch.constant.int -1
    %float1.000000e-05 = torch.constant.float 1.000000e-05
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_8_1_3_3_torch.float32> : tensor<8x1x3x3xf32>) : !torch.vtensor<[8,1,3,3],f32>
    %3 = torch.vtensor.literal(dense_resource<torch_tensor_8_torch.float32> : tensor<8xf32>) : !torch.vtensor<[8],f32>
    %int1 = torch.constant.int 1
    %4 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %6 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %7 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %8 = torch.aten.convolution %arg3, %2, %3, %4, %5, %6, %false, %7, %int1 : !torch.vtensor<[1,1,16,16],f32>, !torch.vtensor<[8,1,3,3],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,8,16,16],f32>
    %9 = torch.aten.add.Scalar %arg1, %float1.000000e-05, %int1 : !torch.vtensor<[8],f32>, !torch.float, !torch.int -> !torch.vtensor<[8],f32>
    %10 = torch.aten.sqrt %9 : !torch.vtensor<[8],f32> -> !torch.vtensor<[8],f32>
    %11 = torch.aten.reciprocal %10 : !torch.vtensor<[8],f32> -> !torch.vtensor<[8],f32>
    %12 = torch.aten.mul.Scalar %11, %int1 : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8],f32>
    %13 = torch.aten.unsqueeze %arg0, %int-1 : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8,1],f32>
    %14 = torch.aten.unsqueeze %13, %int-1 : !torch.vtensor<[8,1],f32>, !torch.int -> !torch.vtensor<[8,1,1],f32>
    %15 = torch.aten.unsqueeze %12, %int-1 : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8,1],f32>
    %16 = torch.aten.unsqueeze %15, %int-1 : !torch.vtensor<[8,1],f32>, !torch.int -> !torch.vtensor<[8,1,1],f32>
    %17 = torch.aten.sub.Tensor %8, %14, %int1 : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[8,1,1],f32>, !torch.int -> !torch.vtensor<[1,8,16,16],f32>
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[8,1,1],f32> -> !torch.vtensor<[1,8,16,16],f32>
    %19 = torch.aten.unsqueeze %1, %int-1 : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8,1],f32>
    %20 = torch.aten.unsqueeze %19, %int-1 : !torch.vtensor<[8,1],f32>, !torch.int -> !torch.vtensor<[8,1,1],f32>
    %21 = torch.aten.mul.Tensor %18, %20 : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[8,1,1],f32> -> !torch.vtensor<[1,8,16,16],f32>
    %22 = torch.aten.unsqueeze %0, %int-1 : !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[8,1],f32>
    %23 = torch.aten.unsqueeze %22, %int-1 : !torch.vtensor<[8,1],f32>, !torch.int -> !torch.vtensor<[8,1,1],f32>
    %24 = torch.aten.add.Tensor %21, %23, %int1 : !torch.vtensor<[1,8,16,16],f32>, !torch.vtensor<[8,1,1],f32>, !torch.int -> !torch.vtensor<[1,8,16,16],f32>
    %25 = torch.aten.relu %24 : !torch.vtensor<[1,8,16,16],f32> -> !torch.vtensor<[1,8,16,16],f32>
    return %25 : !torch.vtensor<[1,8,16,16],f32>
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
