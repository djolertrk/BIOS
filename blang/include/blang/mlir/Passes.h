//===- Passes.h - blang Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for blang.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_PASSES_H
#define MLIR_TUTORIAL_TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;

namespace blang {
std::unique_ptr<Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Blang IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

/// Create a pass for lowering operations the remaining `Blang` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // end namespace blang
} // end namespace mlir

#endif // MLIR_TUTORIAL_TOY_PASSES_H
