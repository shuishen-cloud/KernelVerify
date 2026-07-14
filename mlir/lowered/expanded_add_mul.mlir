module {
  func.func @test_add_mul(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = tensor.empty() : tensor<4x8xf32>
    %2 = tensor.empty() : tensor<4x8xf32>
    %3 = tensor.empty() : tensor<4x8xf32>
    %4 = linalg.add ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%3 : tensor<4x8xf32>) -> tensor<4x8xf32>
    %5 = tensor.empty() : tensor<4x8xf32>
    %6 = linalg.mul ins(%4, %arg2 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%5 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %6 : tensor<4x8xf32>
  }
  func.func @test_double(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
    %0 = tensor.empty() : tensor<10xf32>
    %1 = tensor.empty() : tensor<10xf32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = tensor.empty() : tensor<10xf32>
    %4 = linalg.add ins(%arg0, %arg1 : tensor<10xf32>, tensor<10xf32>) outs(%3 : tensor<10xf32>) -> tensor<10xf32>
    %5 = tensor.empty() : tensor<10xf32>
    %6 = linalg.mul ins(%4, %arg2 : tensor<10xf32>, tensor<10xf32>) outs(%5 : tensor<10xf32>) -> tensor<10xf32>
    %7 = tensor.empty() : tensor<10xf32>
    %8 = linalg.add ins(%arg0, %arg3 : tensor<10xf32>, tensor<10xf32>) outs(%7 : tensor<10xf32>) -> tensor<10xf32>
    %9 = tensor.empty() : tensor<10xf32>
    %10 = linalg.mul ins(%8, %arg2 : tensor<10xf32>, tensor<10xf32>) outs(%9 : tensor<10xf32>) -> tensor<10xf32>
    %11 = tensor.empty() : tensor<10xf32>
    return %6, %10 : tensor<10xf32>, tensor<10xf32>
  }
  func.func @test_no_match(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = tensor.empty() : tensor<4x8xf32>
    %1 = linalg.add ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}

