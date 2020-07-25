//===- AST.h - Node definition for the blang AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the blang language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef BLANG_AST_H_
#define BLANG_AST_H_

#include "blang/Frontend/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <vector>

namespace blang {

/// A variable type.
class VarType {
public:
  enum VarTypeKind {
    Type_Unknown,
    Type_Tensor,
    Type_Float,
    Type_Float64,
    Type_Int,
    Type_Int64,
    Type_Char,
    Type_String,
  };

  virtual ~VarType() = default;
  VarType() = delete;

  VarType(VarTypeKind Kind, std::string name)
    : kind(Kind), name(std::move(name)) {}

  VarTypeKind getKind() const { return kind; }

  const std::string &getName() const { return name; }

private:
  VarTypeKind kind = Type_Unknown;
  std::string name;
};

class TensorType : public VarType {
public:
  TensorType()
    : VarType(Type_Tensor, "tensor") {}
  TensorType(std::string name)
    : VarType(Type_Tensor, "tensor"), tensorShape(name) {}
  TensorType(std::string tensorShape, std::vector<int64_t> shape)
    : VarType(Type_Tensor, "tensor"),
      tensorShape(std::move(tensorShape)), shape(std::move(shape)) {}

  const std::string &getShapeName() const {
    return tensorShape;
  }

  const std::vector<int64_t> &getShape() const {
    return shape;
  }

  std::vector<int64_t> &getShape() {
    return shape;
  }

  void setTypeName(std::string name) {
    tensorShape = std::move(name);
  }

  void setTypeShape(std::vector<int64_t> typeShape) {
    shape = std::move(typeShape);
  }

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Tensor;
  }

private:
  std::string tensorShape;
  std::vector<int64_t> shape;
};

class FloatType : public VarType {
public:
  FloatType()
    : VarType(Type_Float, "float") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Float;
  }
};

class Float64Type : public VarType {
public:
  Float64Type()
    : VarType(Type_Float64, "float64") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Float64;
  }
};

class IntType : public VarType {
public:
  IntType()
    : VarType(Type_Int, "int") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Int;
  }
};

class Int64Type : public VarType {
public:
  Int64Type()
    : VarType(Type_Int64, "int64") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Int64;
  }
};

class CharType : public VarType {
public:
  CharType()
    : VarType(Type_Char, "char") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_Char;
  }
};

class StringType : public VarType {
public:
  StringType()
    : VarType(Type_String, "string") {}

  /// LLVM style RTTI.
  static bool classof(const VarType *var) {
    return var->getKind() == Type_String;
  }
};


/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_StructLiteral,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
public:
  enum NumberASTKind {
    Num_Unknown,
    Num_Float,
    Num_Float64,
    Num_Int,
    Num_Int64,
    Num_Char,
    Num_String,
    Num_Tensor,
  };

  NumberExprAST(Location loc, NumberASTKind kind)
    : ExprAST(Expr_Num, loc), TypeKind(kind) {}

  NumberASTKind getTypeKind() const { return TypeKind; }
  NumberASTKind getTypeKind() { return TypeKind; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
private:
  NumberASTKind TypeKind = Num_Unknown;
};

/// Float (float) numeric literals.
class FloatNumberExprAST : public NumberExprAST {
  float Val;
public:
  FloatNumberExprAST(Location loc, float val)
    : NumberExprAST(loc, Num_Float), Val(val) {}

  float getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Float;
  }
};

/// Double (float64) numeric literals.
class Float64NumberExprAST : public NumberExprAST {
  double Val;
public:
  Float64NumberExprAST(Location loc, double val)
    : NumberExprAST(loc, Num_Float64), Val(val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Float64;
  }
};

/// Integer numeric literals.
class IntNumberExprAST : public NumberExprAST {
  int Val;
public:
  IntNumberExprAST(Location loc, int val)
    : NumberExprAST(loc, Num_Int), Val(val) {}

  int getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Int;
  }
};

/// Integer64 numeric literals.
class Int64NumberExprAST : public NumberExprAST {
  int64_t Val;
public:
  Int64NumberExprAST(Location loc, int64_t val)
    : NumberExprAST(loc, Num_Int64), Val(val) {}

  int64_t getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Int64;
  }
};

class CharNumberExprAST : public NumberExprAST {
  char Val;
public:
  CharNumberExprAST(Location loc, char val)
    : NumberExprAST(loc, Num_Char), Val(val) {}

  char getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Char;
  }
};

// Strings are arrays of i8, so it is a "number".
class StringExprAST : public NumberExprAST {
  std::string Val;
public:
  StringExprAST(Location loc, std::string val)
    : NumberExprAST(loc, Num_String), Val(val) {}

  std::string& getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_String;
  }
};

/// Tensor numeric literals.
class TensorNumberExprAST : public NumberExprAST {
  double Val;
public:
  TensorNumberExprAST(Location loc, double val)
    : NumberExprAST(loc, Num_Tensor), Val(val) {}

  double getValue() { return Val; }

  /// LLVM style RTTI
  static bool classof(const NumberExprAST *n) {
    return n->getTypeKind() == Num_Tensor;
  }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// Expression class for a literal struct value.
class StructLiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;

public:
  StructLiteralExprAST(Location loc,
                       std::vector<std::unique_ptr<ExprAST>> values)
      : ExprAST(Expr_StructLiteral, loc), values(std::move(values)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) {
    return c->getKind() == Expr_StructLiteral;
  }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  std::string name;
  std::unique_ptr<VarType> type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name,
                 std::unique_ptr<VarType> type,
                 std::unique_ptr<ExprAST> initVal = nullptr)
      : ExprAST(Expr_VarDecl, loc), name(name), type(std::move(type)),
        initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType* getType() { return type.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  llvm::Optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, llvm::Optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, loc), expr(std::move(expr)) {}

  llvm::Optional<ExprAST *> getExpr() {
    if (expr.hasValue())
      return expr->get();
    return llvm::None;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char Op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, loc), op(Op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  std::string callee;
  std::vector<std::unique_ptr<ExprAST>> args;

public:
  CallExprAST(Location loc, const std::string &callee,
              std::vector<std::unique_ptr<ExprAST>> args)
      : ExprAST(Expr_Call, loc), callee(callee), args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, loc), arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VarDeclExprAST>> args)
      : location(location), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getArgs() { return args; }
};

/// This class represents a top level record in a module.
class RecordAST {
public:
  enum RecordASTKind {
    Record_Function,
    Record_Struct,
  };

  RecordAST(RecordASTKind kind) : kind(kind) {}
  virtual ~RecordAST() = default;

  RecordASTKind getKind() const { return kind; }

private:
  const RecordASTKind kind;
};

/// This class represents a function definition itself.
class FunctionAST : public RecordAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : RecordAST(Record_Function), proto(std::move(proto)),
        body(std::move(body)) {}
  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }

  /// LLVM style RTTI
  static bool classof(const RecordAST *R) {
    return R->getKind() == Record_Function;
  }
};

/// This class represents a struct definition.
class StructAST : public RecordAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VarDeclExprAST>> variables;

public:
  StructAST(Location location, const std::string &name,
            std::vector<std::unique_ptr<VarDeclExprAST>> variables)
      : RecordAST(Record_Struct), location(location), name(name),
        variables(std::move(variables)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VarDeclExprAST>> getVariables() {
    return variables;
  }

  /// LLVM style RTTI
  static bool classof(const RecordAST *R) {
    return R->getKind() == Record_Struct;
  }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  std::vector<std::unique_ptr<RecordAST>> records;

public:
  ModuleAST(std::vector<std::unique_ptr<RecordAST>> records)
      : records(std::move(records)) {}

  auto begin() -> decltype(records.begin()) { return records.begin(); }
  auto end() -> decltype(records.end()) { return records.end(); }
};

void dump(ModuleAST &);

} // namespace blang

#endif // BLANG_AST_H_
