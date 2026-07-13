//===-- CountLinalgOps.cpp - 统计 Linalg op 数量 ------------------ C++ --===//
//
// 第一个自定义 MLIR Pass — 插件版
// 构建: cmake + make
// 使用: mlir-opt --load-pass-plugin=./libCountLinalgOps.so \
//                 --pass-pipeline='builtin.module(func.func(count-linalg-ops))'
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace mlir;

namespace {

struct CountLinalgOps : public PassWrapper<CountLinalgOps, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountLinalgOps)

  StringRef getArgument() const override { return "count-linalg-ops"; }
  StringRef getDescription() const override { return "Count Linalg ops by type in each function"; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    std::map<StringRef, int> counts;

    func.walk([&](linalg::LinalgOp op) {
      counts[op->getName().getStringRef()]++;
    });

    llvm::errs() << "[count-linalg-ops] " << func.getSymName() << ":\n";
    int total = 0;
    for (auto &[name, count] : counts) {
      llvm::errs() << "  " << name << ": " << count << "\n";
      total += count;
    }
    llvm::errs() << "  TOTAL: " << total << "\n";
  }
};

} // namespace

// MLIR Pass Plugin API: 入口点
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION, "CountLinalgOps", "0.1",
    []() { PassRegistration<CountLinalgOps>(); }
  };
}
