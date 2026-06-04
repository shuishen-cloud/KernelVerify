// W2: Hand-written Torch Dialect MLIR — Matrix Multiply (2×3) × (3×2) = (2×2)
// Tests: torch.aten.mm lowering to linalg.matmul

module {
  func.func @main(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[3,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %result = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[3,2],f32> -> !torch.vtensor<[2,2],f32>
    return %result : !torch.vtensor<[2,2],f32>
  }
}
