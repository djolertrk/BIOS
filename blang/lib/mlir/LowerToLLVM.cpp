//====- LowerToLLVM.cpp - Lowering from Blang+Affine+Std to LLVM ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Blang operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "blang/mlir/Dialect.h"
#include "blang/mlir/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BlangToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace {
/// Lowers `blang.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(blang::PrintOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto *llvmDialect =
        op->getContext()->getRegisteredDialect<LLVM::LLVMDialect>();
    assert(llvmDialect && "expected llvm dialect to be registered");

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule, llvmDialect);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule, llvmDialect);

    if (auto memRefType = (*op->operand_type_begin()).dyn_cast<MemRefType>()) {
      auto memRefShape = memRefType.getShape();
      auto elType = memRefType.getElementType();
      Value formatSpecifierCst;
      if (auto intType = elType.dyn_cast<IntegerType>()) {
        assert(intType.getWidth() == 8 && "must be i8");
        formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%c\0", 4), parentModule,
          llvmDialect);
      } else {
        // F64 type.
        formatSpecifierCst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule,
          llvmDialect);
      }

      // Create a loop for each of the dimensions within the shape.
      SmallVector<Value, 4> loopIvs;
      for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
        auto lowerBound = rewriter.create<ConstantIndexOp>(loc, 0);
        auto upperBound = rewriter.create<ConstantIndexOp>(loc, memRefShape[i]);
        auto step = rewriter.create<ConstantIndexOp>(loc, 1);
        auto loop =
            rewriter.create<loop::ForOp>(loc, lowerBound, upperBound, step);
        loop.getBody()->clear();
        loopIvs.push_back(loop.getInductionVar());

        // Terminate the loop body.
        rewriter.setInsertionPointToStart(loop.getBody());

        // Insert a newline after each of the inner dimensions of the shape.
        if (i != e - 1)
          rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                  newLineCst);
        rewriter.create<loop::TerminatorOp>(loc);
        rewriter.setInsertionPointToStart(loop.getBody());
      }

      // Generate a call to printf for the current element of the loop.
      auto printOp = cast<blang::PrintOp>(op);
      auto elementLoad = rewriter.create<LoadOp>(loc, printOp.input(), loopIvs);
      rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                              ArrayRef<Value>({formatSpecifierCst, elementLoad}));
    } else if (auto floatType =
                          (*op->operand_type_begin()).dyn_cast<FloatType>()) {
            // Handle print of constants.
      auto printOp = cast<blang::PrintOp>(op);

      if (floatType.getWidth() == 64) {
        Value formatSpecifierFloat64Cst = getOrCreateGlobalString(
          loc, rewriter, "frmt_spec", StringRef("%lf\0A\00", 4), parentModule,
          llvmDialect);
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
            ArrayRef<Value>({formatSpecifierFloat64Cst, printOp.input()}));
        // Put the endline.
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      } else {
        // 32 bits wide float number.
        Value formatSpecifierFloatCst = getOrCreateGlobalString(
          loc, rewriter, "float32_frmt_spec", StringRef("%f\0A\00"), parentModule,
          llvmDialect);

        // We use fpExtend to print the float value. The same way LLVM deals with
        // it.
        Value fpExtendFloatToDouble = rewriter.create<mlir::LLVM::FPExtOp>(
          loc, mlir::LLVM::LLVMType::getDoubleTy(llvmDialect),
          llvm::ArrayRef<mlir::Value>{printOp.input()});
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                ArrayRef<Value>({formatSpecifierFloatCst, fpExtendFloatToDouble}));
      }
    } else if (auto intType =
                          (*op->operand_type_begin()).dyn_cast<IntegerType>()) {
      // Handle print of constants.
      auto printOp = cast<blang::PrintOp>(op);
      if (intType.getWidth() == 32) {
        Value formatSpecifierIntCst = getOrCreateGlobalString(
          loc, rewriter, "int_frmt_spec", StringRef("%d\0A\00"), parentModule,
          llvmDialect);
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                    ArrayRef<Value>({formatSpecifierIntCst, printOp.input()}));
      } else if (intType.getWidth() == 64) {
        // 64 bits wide integer.
        Value formatSpecifierInt64Cst = getOrCreateGlobalString(
          loc, rewriter, "int64_frmt_spec", StringRef("%" PRId64 "\0A\00"),
          parentModule, llvmDialect);
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                    ArrayRef<Value>({formatSpecifierInt64Cst, printOp.input()}));
      } else {
        // 8 bits wide integer representing the char type.
        Value formatSpecifierInt8Cst = getOrCreateGlobalString(
          loc, rewriter, "int8_frmt_spec", StringRef("%c\0A\00"),
          parentModule, llvmDialect);
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                    ArrayRef<Value>({formatSpecifierInt8Cst, printOp.input()}));
      }
    } else {
      llvm_unreachable("Unhandled type within the printOp");
    }

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return matchSuccess();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module,
                                             LLVM::LLVMDialect *llvmDialect) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get("printf", context);

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(llvmDialect);
    auto llvmI8PtrTy = LLVM::LLVMType::getInt8PtrTy(llvmDialect);
    auto llvmFnType = LLVM::LLVMType::getFunctionTy(llvmI32Ty, llvmI8PtrTy,
                                                    /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get("printf", context);
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module,
                                       LLVM::LLVMDialect *llvmDialect) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt8Ty(llvmDialect), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value));
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, LLVM::LLVMType::getInt64Ty(llvmDialect),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMType::getInt8PtrTy(llvmDialect), globalPtr,
        ArrayRef<Value>({cst0, cst0}));
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// BlangToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct BlangToLLVMLoweringPass : public ModulePass<BlangToLLVMLoweringPass> {
  void runOnModule() final;
};
} // end anonymous namespace

void BlangToLLVMLoweringPass::runOnModule() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. Do perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `blang`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `blang` dialect, is the
  // PrintOp.
  patterns.insert<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getModule();
  if (failed(applyFullConversion(module, target, patterns, &typeConverter)))
    signalPassFailure();
}

/// Create a pass for lowering operations the remaining `Blang` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::blang::createLowerToLLVMPass() {
  return std::make_unique<BlangToLLVMLoweringPass>();
}
