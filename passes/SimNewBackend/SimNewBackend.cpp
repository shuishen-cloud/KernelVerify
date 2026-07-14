//===- SimNewBackend.cpp - 模拟新后端 Pass: fused_add_mul -----------------===//
//
// 模拟新后端指令 fused_add_mul = (a + b) * c，一次完成。
//
// 包含两个 Pass:
//   1. sim-fuse-add-mul  — 融合: linalg.add + linalg.mul → linalg.generic(add+mul)
//   2. sim-expand-fused  — 展开: fused generic → linalg.add + linalg.mul
//
// 构建: bash passes/SimNewBackend/build.sh
// 使用:
//   mlir-opt --load-pass-plugin=build/libSimNewBackend.so \
//     --pass-pipeline="builtin.module(func.func(sim-fuse-add-mul))" test.mlir
//   mlir-opt --load-pass-plugin=build/libSimNewBackend.so \
//     --pass-pipeline="builtin.module(func.func(sim-expand-fused))" fused.mlir
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

// helpers
static SmallVector<Value> vals(Value a, Value b) { return {a, b}; }
static SmallVector<Value> vals(Value a, Value b, Value c) { return {a, b, c}; }
static SmallVector<Value> vals(Value a) { return {a}; }

static bool isElementWise(linalg::GenericOp op) {
  for (auto it : op.getIteratorTypesArray())
    if (it != utils::IteratorType::parallel) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Pass 1: sim-fuse-add-mul
//===----------------------------------------------------------------------===//
struct FuseAddMulPass
    : public PassWrapper<FuseAddMulPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAddMulPass)

  StringRef getArgument() const override { return "sim-fuse-add-mul"; }
  StringRef getDescription() const override {
    return "Fuse linalg.add+linalg.mul into one linalg.generic";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    int fusedCount = 0;

    func.walk([&](linalg::MulOp mulOp) {
      Value mulInput = mulOp.getInputs()[0];
      auto *defOp = mulInput.getDefiningOp();
      if (!defOp) return;
      auto addOp = dyn_cast<linalg::AddOp>(defOp);
      if (!addOp) return;
      if (!addOp.getResult(0).hasOneUse()) return;

      llvm::errs() << "[fuse] " << addOp->getName().getStringRef()
                   << " -> " << mulOp->getName().getStringRef() << "\n";

      OpBuilder builder(mulOp);
      auto loc = mulOp.getLoc();
      Value a = addOp.getInputs()[0];
      Value b = addOp.getInputs()[1];
      Value c = (mulOp.getInputs()[0] == addOp.getResult(0))
                    ? mulOp.getInputs()[1] : mulOp.getInputs()[0];
      auto rtt = mlir::cast<RankedTensorType>(mulOp.getResult(0).getType());
      int rank = rtt.getRank();
      MLIRContext *ctx = builder.getContext();

      SmallVector<AffineMap> maps(4, AffineMap::getMultiDimIdentityMap(rank, ctx));
      SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

      Value init = builder.create<tensor::EmptyOp>(loc, rtt, ValueRange{});

      auto fusedOp = builder.create<linalg::GenericOp>(
          loc, rtt, vals(a, b, c), vals(init), maps, iterTypes,
          [&](OpBuilder &b, Location bl, ValueRange args) {
            Value av = b.create<arith::AddFOp>(bl, args[0], args[1]);
            Value mv = b.create<arith::MulFOp>(bl, av, args[2]);
            b.create<linalg::YieldOp>(bl, mv);
          });

      mulOp.getResult(0).replaceAllUsesWith(fusedOp.getResult(0));
      fusedCount++;
    });

    // 迭代 DCE (mul 被删后 add 才变 dead)
    if (fusedCount > 0) {
      int removed = 0;
      for (bool changed = true; changed; ) {
        changed = false;
        SmallVector<Operation *> dead;
        func.walk([&](Operation *op) {
          if (isa<linalg::AddOp, linalg::MulOp>(op) &&
              llvm::all_of(op->getResults(),
                           [](Value r) { return r.use_empty(); }))
            dead.push_back(op);
        });
        for (auto *op : dead) { op->erase(); removed++; changed = true; }
      }
      llvm::errs() << "[fuse] " << fusedCount << " pairs, "
                   << removed << " dead ops removed\n";
    } else {
      llvm::errs() << "[fuse] no add->mul pattern found\n";
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass 2: sim-expand-fused
//===----------------------------------------------------------------------===//
struct ExpandFusedPass
    : public PassWrapper<ExpandFusedPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExpandFusedPass)

  StringRef getArgument() const override { return "sim-expand-fused"; }
  StringRef getDescription() const override {
    return "Expand fused generic back to linalg.add+linalg.mul";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    int expandedCount = 0;

    func.walk([&](linalg::GenericOp genericOp) {
      if (!isElementWise(genericOp)) return;
      if (genericOp.getInputs().size() != 3) return;
      auto &region = genericOp.getRegion();
      if (!region.hasOneBlock()) return;

      bool hasAdd = false, hasMul = false;
      for (auto &op : region.front()) {
        if (isa<arith::AddFOp>(op)) hasAdd = true;
        if (isa<arith::MulFOp>(op)) hasMul = true;
      }
      if (!hasAdd || !hasMul) return;

      llvm::errs() << "[expand] fused generic -> add+mul\n";

      OpBuilder builder(genericOp);
      auto loc = genericOp.getLoc();
      Value a = genericOp.getInputs()[0];
      Value b = genericOp.getInputs()[1];
      Value c = genericOp.getInputs()[2];
      auto resultType = genericOp.getResult(0).getType();

      Value addInit = builder.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
      auto newAdd = builder.create<linalg::AddOp>(
          loc, TypeRange{resultType}, vals(a, b), vals(addInit));

      Value mulInit = builder.create<tensor::EmptyOp>(loc, resultType, ValueRange{});
      auto newMul = builder.create<linalg::MulOp>(
          loc, TypeRange{resultType}, vals(newAdd.getResult(0), c), vals(mulInit));

      genericOp.getResult(0).replaceAllUsesWith(newMul.getResult(0));
      genericOp.erase();
      expandedCount++;
    });

    llvm::errs() << "[expand] " << expandedCount << " ops expanded\n";
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "SimNewBackend", "0.1",
          []() {
            PassRegistration<FuseAddMulPass>();
            PassRegistration<ExpandFusedPass>();
          }};
}
