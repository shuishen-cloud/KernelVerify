// W2: Hand-written Torch Dialect MLIR — elementwise Add + ReLU
// Tests: torch.aten.add.Tensor + torch.aten.relu lowering and fusion

module {
  func.func @main(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
    %int1 = torch.constant.int 1
    %sum = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>, !torch.int -> !torch.vtensor<[2,3],f32>
    %result = torch.aten.relu %sum : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
    return %result : !torch.vtensor<[2,3],f32>
  }
}
