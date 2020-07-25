//===- MLIRGen.h - MLIR Generation from a blang AST -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the blang language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_blang_MLIRGEN_H_
#define MLIR_blang_MLIRGEN_H_

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace blang {
class ModuleAST;

/// Emit IR for the given blang moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace blang

#endif // MLIR_blang_MLIRGEN_H_
