//====- LowerToAffineLoops.cpp - Partial lowering from Blang to Affine+Std --===//
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

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as blang functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input a rewriter, an array of memRefOperands corresponding
/// to the operands of the input operation, and the set of loop induction
/// variables for the iteration. It returns a value to store at the current
/// index of the iteration.
using LoopIterationFn = function_ref<Value(PatternRewriter &rewriter,
                                           ArrayRef<Value> memRefOperands,
                                           ArrayRef<Value> loopIvs)>;

static void lowerOpToLoops(Operation *op, ArrayRef<Value> operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create an empty affine loop for each of the dimensions within the shape.
  SmallVector<Value, 4> loopIvs;
  for (auto dim : tensorType.getShape()) {
    auto loop = rewriter.create<AffineForOp>(loc, /*lb=*/0, dim, /*step=*/1);
    loop.getBody()->clear();
    loopIvs.push_back(loop.getInductionVar());

    // Terminate the loop body and update the rewriter insertion point to the
    // beginning of the loop.
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<AffineTerminatorOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }

  // Generate a call to the processing function with the rewriter, the memref
  // operands, and the loop induction variables. This function will return the
  // value to store at the current index.
  Value valueToStore = processIteration(rewriter, operands, loopIvs);
  rewriter.create<AffineStoreOp>(loc, valueToStore, alloc,
                                 llvm::makeArrayRef(loopIvs));

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the BinaryOp. This
          // allows for using the nice named accessors that are generated by the
          // ODS.
          typename BinaryOp::OperandAdaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the inner
          // loop.
          auto loadedLhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
          auto loadedRhs =
              rewriter.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

          // Create the binary operation performed on the loaded values.
          return rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });
    return matchSuccess();
  }
};
using AddOpLowering = BinaryOpLowering<blang::AddOp, AddFOp>;
using MulOpLowering = BinaryOpLowering<blang::MulOp, MulFOp>;

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<blang::ConstantOp> {
  using OpRewritePattern<blang::ConstantOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantOp op,
                                     PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.value();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (valueShape.size()) {
      for (auto i : llvm::seq<int64_t>(
              0, *std::max_element(valueShape.begin(), valueShape.end())))
       constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
    } else {
      // This is the case for the tensor scalars (tesnros of rank 0).
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.getValues<FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
            llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<blang::ReturnOp> {
  using OpRewritePattern<blang::ReturnOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ReturnOp op,
                                     PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return matchFailure();

    // We lower "blang.return" directly to "std.return".
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantFloat64 operations
//===----------------------------------------------------------------------===//

struct ConstantFloat32OpLowering
    : public OpRewritePattern<blang::ConstantFloat32Op> {
  using OpRewritePattern<blang::ConstantFloat32Op>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantFloat32Op op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    rewriter.replaceOpWithNewOp<ConstantOp>(op, FloatAttr::get(op.getType(),
                                            constantValue));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantFloat64 operations
//===----------------------------------------------------------------------===//

struct ConstantFloat64OpLowering
    : public OpRewritePattern<blang::ConstantFloat64Op> {
  using OpRewritePattern<blang::ConstantFloat64Op>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantFloat64Op op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    rewriter.replaceOpWithNewOp<ConstantOp>(op, FloatAttr::get(op.getType(),
                                            constantValue));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantInt operations
//===----------------------------------------------------------------------===//

struct ConstantIntOpLowering
    : public OpRewritePattern<blang::ConstantIntOp> {
  using OpRewritePattern<blang::ConstantIntOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantIntOp op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            IntegerAttr::get(op.getType(),
                                                             constantValue));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantInt64 operations
//===----------------------------------------------------------------------===//

struct ConstantInt64OpLowering
    : public OpRewritePattern<blang::ConstantInt64Op> {
  using OpRewritePattern<blang::ConstantInt64Op>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantInt64Op op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            IntegerAttr::get(op.getType(),
                                                             constantValue));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantChar operations
//===----------------------------------------------------------------------===//

struct ConstantCharOpLowering
    : public OpRewritePattern<blang::ConstantCharOp> {
  using OpRewritePattern<blang::ConstantCharOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantCharOp op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    rewriter.replaceOpWithNewOp<ConstantOp>(op,
                                            IntegerAttr::get(op.getType(),
                                                             constantValue));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: ConstantString operations
//===----------------------------------------------------------------------===//

struct ConstantStringOpLowering
    : public OpRewritePattern<blang::ConstantStringOp> {
  using OpRewritePattern<blang::ConstantStringOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(blang::ConstantStringOp op,
                                     PatternRewriter &rewriter) const final {
    auto constantValue = op.value();
    Location loc = op.getLoc();

    // Get the memref<Nxi8> type.
    auto memRefType = op.getResult().getType().cast<MemRefType>();
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    for (auto i : llvm::seq<int64_t>(
            0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.begin();
    auto dataType = rewriter.getIntegerType(8);
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<AffineStoreOp>(
            loc,
            rewriter.create<ConstantOp>(loc,
                                        IntegerAttr::get(dataType, *valueIt++)),
            alloc, llvm::makeArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlangToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(blang::TransposeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](PatternRewriter &rewriter, ArrayRef<Value> memRefOperands,
              ArrayRef<Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS.
          blang::TransposeOpOperandAdaptor transposeAdaptor(memRefOperands);
          Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<AffineLoadOp>(loc, input, reverseIvs);
        });
    return matchSuccess();
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// BlangToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the blang operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Blang dialect.
namespace {
struct BlangToAffineLoweringPass : public FunctionPass<BlangToAffineLoweringPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void BlangToAffineLoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "main")
    return;

  // Verify that the given main has no inputs and results.
  if (function.getNumArguments() || function.getType().getNumResults()) {
    function.emitError("expected 'main' to have 0 inputs and 0 results");
    return signalPassFailure();
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();

  // We also define the Blang dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Blang operations that don't want
  // to lower, `blang.print`, as `legal`.
  target.addIllegalDialect<blang::blangDialect>();
  target.addLegalOp<blang::PrintOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Blang operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, ConstantOpLowering, ConstantFloat64OpLowering,
                  ConstantFloat32OpLowering, ConstantIntOpLowering,
                  ConstantInt64OpLowering, ConstantCharOpLowering,
                  MulOpLowering, ConstantStringOpLowering,
                  ReturnOpLowering, TransposeOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Blang IR (e.g. matmul).
std::unique_ptr<Pass> mlir::blang::createLowerToAffinePass() {
  return std::make_unique<BlangToAffineLoweringPass>();
}
