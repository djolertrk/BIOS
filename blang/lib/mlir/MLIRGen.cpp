//===- MLIRGen.cpp - MLIR Generation from a blang AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the blang language.
//
//===----------------------------------------------------------------------===//

#include "blang/mlir/MLIRGen.h"
#include "blang/Frontend/AST.h"
#include "blang/mlir/Dialect.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::blang;
using namespace blang;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the blang AST.
///
/// This will emit operations that are specific to the blang language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context)
    : mlirContext(context), builder(&context) {}

  /// Public API: convert the AST for a blang module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &record : moduleAST) {
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        auto func = mlirGen(*funcAST);
        if (!func)
          return nullptr;

        theModule.push_back(func);
        functionMap.insert({func.getName(), func});
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str)))
          return nullptr;
      } else {
        llvm_unreachable("unknown record type");
      }
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the blang operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::MLIRContext &mlirContext;

  /// A "module" matches a blang source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, VarDeclExprAST *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, VarDeclExprAST *>>;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::FuncOp> functionMap;

  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;

  /// Helper conversion for a blang AST location to an MLIR location.
  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(VarDeclExprAST &var, mlir::Value value) {
    if (symbolTable.count(var.getName()))
      return mlir::failure();
    symbolTable.insert(var.getName(), {value, &var});
    return mlir::success();
  }

  /// Create an MLIR type for the given struct.
  mlir::LogicalResult mlirGen(StructAST &str) {
    if (structMap.count(str.getName()))
      return emitError(loc(str.loc())) << "error: struct type with name `"
                                       << str.getName() << "' already exists";

    auto variables = str.getVariables();
    std::vector<mlir::Type> elementTypes;
    elementTypes.reserve(variables.size());
    for (auto &variable : variables) {
      if (variable->getInitVal())
        return emitError(loc(variable->loc()))
               << "error: variables within a struct definition must not have "
                  "initializers";
      auto varType = variable->getType();
      if (auto tensorType = llvm::dyn_cast<TensorType>(varType)) {
        if (!tensorType->getShape().empty())
          return emitError(loc(variable->loc()))
                 << "error: variables within a struct definition must not have "
                    "initializers";
      }

      mlir::Type type = getType(varType, variable->loc());
      if (!type)
        return mlir::failure();
      elementTypes.push_back(type);
    }

    structMap.try_emplace(str.getName(), StructType::get(elementTypes), &str);
    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided blang AST prototype.
  mlir::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    argTypes.reserve(proto.getArgs().size());
    for (auto &arg : proto.getArgs()) {
      mlir::Type type = getType(arg->getType(), arg->loc());
      if (!type)
        return nullptr;
      argTypes.push_back(type);
    }
    auto func_type = builder.getFunctionType(argTypes, llvm::None);
    return mlir::FuncOp::create(location, proto.getName(), func_type);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    SymbolTableScopeT var_scope(symbolTable);

    // Create an MLIR function for the given prototype.
    mlir::FuncOp function(mlirGen(*funcAST.getProto()));
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto &entryBlock = *function.addEntryBlock();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto &name_value :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(*std::get<0>(name_value), std::get<1>(name_value))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(function.getType().getInputs(),
                                               *returnOp.operand_type_begin()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setVisibility(mlir::FuncOp::Visibility::Private);

    return function;
  }

  /// Return the struct type that is the result of the given expression, or null
  /// if it cannot be inferred.
  StructAST *getStructFor(ExprAST *expr) {
    llvm::StringRef structName;
    if (auto *decl = llvm::dyn_cast<VariableExprAST>(expr)) {
      auto varIt = symbolTable.lookup(decl->getName());
      if (!varIt.first)
        return nullptr;
      structName = varIt.second->getType()->getName();
    } else if (auto *access = llvm::dyn_cast<BinaryExprAST>(expr)) {
      if (access->getOp() != '.')
        return nullptr;
      // The name being accessed should be in the RHS.
      auto *name = llvm::dyn_cast<VariableExprAST>(access->getRHS());
      if (!name)
        return nullptr;
      StructAST *parentStruct = getStructFor(access->getLHS());
      if (!parentStruct)
        return nullptr;

      // Get the element within the struct corresponding to the name.
      VarDeclExprAST *decl = nullptr;
      for (auto &var : parentStruct->getVariables()) {
        if (var->getName() == name->getName()) {
          decl = var.get();
          break;
        }
      }
      if (!decl)
        return nullptr;
      structName = decl->getType()->getName();
    }
    if (structName.empty())
      return nullptr;

    // If the struct name was valid, check for an entry in the struct map.
    auto structIt = structMap.find(structName);
    if (structIt == structMap.end())
      return nullptr;
    return structIt->second.second;
  }

  /// Return the numeric member index of the given struct access expression.
  llvm::Optional<size_t> getMemberIndex(BinaryExprAST &accessOp) {
    assert(accessOp.getOp() == '.' && "expected access operation");

    // Lookup the struct node for the LHS.
    StructAST *structAST = getStructFor(accessOp.getLHS());
    if (!structAST)
      return llvm::None;

    // Get the name from the RHS.
    VariableExprAST *name = llvm::dyn_cast<VariableExprAST>(accessOp.getRHS());
    if (!name)
      return llvm::None;

    auto structVars = structAST->getVariables();
    auto it = llvm::find_if(structVars, [&](auto &var) {
      return var->getName() == name->getName();
    });
    if (it == structVars.end())
      return llvm::None;
    return it - structVars.begin();
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    auto location = loc(binop.loc());

    // If this is an access operation, handle it immediately.
    if (binop.getOp() == '.') {
      llvm::Optional<size_t> accessIndex = getMemberIndex(binop);
      if (!accessIndex) {
        emitError(location, "invalid access into struct expression");
        return nullptr;
      }
      return builder.create<StructAccessOp>(location, lhs, *accessIndex);
    }

    // Otherwise, this is a normal binary op.
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case '*':
      return builder.create<MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()).first)
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().hasValue()) {
      if (!(expr = mlirGen(*ret.getExpr().getValue())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location, expr ? makeArrayRef(expr)
                                            : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a constant for a literal/constant array. It will be emitted as a
  /// flattened array of data in an Attribute attached to a `blang.constant`
  /// operation. See documentation on [Attributes](LangRef.md#attributes) for
  /// more details. Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "blang.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::DenseElementsAttr getConstantAttr(LiteralExprAST &lit) {
    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    return mlir::DenseElementsAttr::get(dataType, llvm::makeArrayRef(data));
  }
  mlir::DenseElementsAttr getConstantAttr(NumberExprAST *lit) {
    auto tensorNum = llvm::dyn_cast<TensorNumberExprAST>(lit);

    // The type of this attribute is tensor of 64-bit floating-point with no
    // shape.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get({}, elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    return mlir::DenseElementsAttr::get(dataType,
                                        llvm::makeArrayRef(tensorNum->getValue()));
  }
  /// Emit a constant for a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `blang.struct_constant`
  /// operation. This function returns the generated constant, along with the
  /// corresponding struct type.
  std::pair<mlir::ArrayAttr, mlir::Type>
  getConstantAttr(StructLiteralExprAST &lit) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;

    for (auto &var : lit.getValues()) {
      if (auto *number = llvm::dyn_cast<NumberExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(number));
        typeElements.push_back(getType(llvm::None));
      } else if (auto *lit = llvm::dyn_cast<LiteralExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*lit));
        typeElements.push_back(getType(llvm::None));
      } else {
        auto *structLit = llvm::cast<StructLiteralExprAST>(var.get());
        auto attrTypePair = getConstantAttr(*structLit);
        attrElements.push_back(attrTypePair.first);
        typeElements.push_back(attrTypePair.second);
      }
    }
    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    mlir::Type dataType = StructType::get(typeElements);
    return std::make_pair(dataAttr, dataType);
  }

  /// Emit an array literal.
  mlir::Value mlirGen(LiteralExprAST &lit) {
    mlir::Type type = getType(lit.getDims());
    mlir::DenseElementsAttr dataAttribute = getConstantAttr(lit);

    // Build the MLIR op `blang.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  /// Emit a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `blang.struct_constant`
  /// operation.
  mlir::Value mlirGen(StructLiteralExprAST &lit) {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(lit);

    // Build the MLIR op `blang.struct_constant`. This invokes the
    // `StructConstantOp::build` method.
    return builder.create<StructConstantOp>(loc(lit.loc()), dataType, dataAttr);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    auto numExpr = llvm::dyn_cast<NumberExprAST>(&expr);
    auto tensorNum = llvm::dyn_cast<TensorNumberExprAST>(numExpr);
    assert((tensorNum != nullptr) && "must be a number of tensor type");
    data.push_back(tensorNum->getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builting calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: blang.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return builder.create<TransposeOp>(location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to ser-defined
    // functions are mapped to a custom call that takes the callee name as an
    // attribute.
    auto calledFuncIt = functionMap.find(callee);
    if (calledFuncIt == functionMap.end()) {
      emitError(location) << "no defined function found for '" << callee << "'";
      return nullptr;
    }
    mlir::FuncOp calledFunc = calledFuncIt->second;
    return builder.create<GenericCallOp>(
        location, calledFunc.getType().getResult(0),
        builder.getSymbolRefAttr(callee), operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    builder.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num) {
    if (auto floatNum = llvm::dyn_cast<FloatNumberExprAST>(&num)) {
      return builder.create<ConstantFloat32Op>(loc(num.loc()),
                                             floatNum->getValue());
    } else if (auto float64Num = llvm::dyn_cast<Float64NumberExprAST>(&num)) {
      return builder.create<ConstantFloat64Op>(loc(num.loc()),
                                               float64Num->getValue());
    } else if (auto intNum = llvm::dyn_cast<IntNumberExprAST>(&num)) {
      return builder.create<ConstantIntOp>(loc(num.loc()),
                                           intNum->getValue());
    } else if (auto int64Num = llvm::dyn_cast<Int64NumberExprAST>(&num)) {
      return builder.create<ConstantInt64Op>(loc(num.loc()),
                                           int64Num->getValue());
    } else if (auto charNum = llvm::dyn_cast<CharNumberExprAST>(&num)) {
      return builder.create<ConstantCharOp>(loc(num.loc()),
                                           charNum->getValue());
    } else if (auto stringVar = llvm::dyn_cast<StringExprAST>(&num)) {
      return builder.create<ConstantStringOp>(loc(num.loc()),
                                           stringVar->getValue());
    } else if (auto tensorNum = llvm::dyn_cast<TensorNumberExprAST>(&num)) {
      return builder.create<ConstantOp>(loc(num.loc()), tensorNum->getValue());
    } else if (auto stringVar = llvm::dyn_cast<StringExprAST>(&num)) {
      return builder.create<ConstantStringOp>(loc(num.loc()), stringVar->getValue());
    }

    llvm_unreachable("unknown type of number");
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case blang::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr));
    case blang::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case blang::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case blang::ExprAST::Expr_StructLiteral:
      return mlirGen(cast<StructLiteralExprAST>(expr));
    case blang::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case blang::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    // Handle the case where we are initializing a struct value.
    auto varType = vardecl.getType();
    if (auto tensorVarType = llvm::dyn_cast<TensorType>(varType)) {
      if (!tensorVarType->getShapeName().empty()) {
        // Check that the initializer type is the same as the variable
        // declaration.
        mlir::Type type = getType(varType, vardecl.loc());
        if (!type)
          return nullptr;
        if (type != value.getType()) {
          emitError(loc(vardecl.loc()))
              << "struct type of initializer is different than the variable "
                 "declaration. Got "
              << value.getType() << ", but expected " << type;
          return nullptr;
        }
  
        // Otherwise, we have the initializer value, but in case the variable was
        // declared with specific shape, we emit a "reshape" operation. It will
        // get optimized out later as needed.
      } else if (!tensorVarType->getShape().empty()) {
        value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                          getType(tensorVarType->getShape()), value);
      }
    } else if (auto floatVarType = llvm::dyn_cast<FloatType>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else if (auto doubleVarType = llvm::dyn_cast<Float64Type>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else if (auto intVarType = llvm::dyn_cast<IntType>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else if (auto int64VarType = llvm::dyn_cast<Int64Type>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else if (auto charVarType = llvm::dyn_cast<CharType>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else if (auto strVarType = llvm::dyn_cast<StringType>(varType)) {
      // TODO: Can we do something additional here????

      // Just continue with this function for now.
    } else {
      llvm_unreachable("Unknown mlir type");
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl, value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    SymbolTableScopeT var_scope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a blang AST variable type (forward to the generic
  /// getType above for non-struct types).
  mlir::Type getType(const VarType *type, const Location &location) {
    if (auto tensorVarType = llvm::dyn_cast<TensorType>(type)) {
      if (!tensorVarType->getShapeName().empty()) {
        auto it = structMap.find(tensorVarType->getShapeName());
        if (it == structMap.end()) {
          emitError(loc(location))
              << "error: unknown struct type '" << tensorVarType->getShapeName() << "'";
          return nullptr;
        }
        return it->second.first;
      }
      return getType(tensorVarType->getShape());
    } else if (auto floatVarType = llvm::dyn_cast<FloatType>(type)) {
      // Build a 'f32' type.
      return mlir::FloatType::get(mlir::StandardTypes::F32, &mlirContext);
    } else if (auto doubleVarType = llvm::dyn_cast<Float64Type>(type)) {
      // Build a 'f64' type.
      return mlir::FloatType::getF64(&mlirContext);
    } else if (auto intVarType = llvm::dyn_cast<IntType>(type)) {
      // Build an 'integer' type.
      return mlir::IntegerType::get(32, &mlirContext);
    } else if (auto int64VarType = llvm::dyn_cast<Int64Type>(type)) {
      // Build an 'integer64' type.
      return mlir::IntegerType::get(64, &mlirContext);
    } else if (auto charVarType = llvm::dyn_cast<CharType>(type)) {
      // Build an 'integer8' type.
      return mlir::IntegerType::get(8, &mlirContext);
    }

    llvm_unreachable("Unknown mlir type");
  }
};

} // namespace

namespace blang {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace blang
