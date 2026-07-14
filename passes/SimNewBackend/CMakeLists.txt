# SimNewBackend — 模拟新后端 Pass 插件
#
# 构建: bash passes/SimNewBackend/build.sh
# 使用:
#   # 融合
#   torch-mlir-opt --load-pass-plugin=./libSimNewBackend.so \
#     --pass-pipeline="builtin.module(sim-fuse-add-mul)" test.mlir
#   # 融合+展开 (验证语义保持)
#   torch-mlir-opt --load-pass-plugin=./libSimNewBackend.so \
#     --pass-pipeline="builtin.module(sim-fuse-add-mul,sim-expand-fused)"

cmake_minimum_required(VERSION 3.20)
project(SimNewBackend LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

add_library(SimNewBackend MODULE SimNewBackend.cpp)

target_include_directories(SimNewBackend PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(SimNewBackend PRIVATE
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRSupport
  MLIRPluginsLib
)
