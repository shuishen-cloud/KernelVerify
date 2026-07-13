# CountLinalgOps — 独立 MLIR Pass 插件
#
# 构建:
#   mkdir -p build && cd build
#   cmake .. -DCMAKE_PREFIX_PATH=/home/lwy/download/llvm-project/install
#   make
#
# 使用:
#   mlir-opt --load-pass-plugin=./libCountLinalgOps.so \
#            --pass-pipeline='builtin.module(func.func(count-linalg-ops))' input.mlir

cmake_minimum_required(VERSION 3.20)
project(CountLinalgOps LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# 构建共享库
add_library(CountLinalgOps MODULE CountLinalgOps.cpp)

# 设置 include 和链接
target_include_directories(CountLinalgOps PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(CountLinalgOps PRIVATE
  MLIRIR
  MLIRPass
  MLIRLinalgDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRPluginsLib
)
