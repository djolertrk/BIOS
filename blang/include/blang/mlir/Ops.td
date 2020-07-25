//===- Ops.td - blang dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the blang dialect.
//
//===----------------------------------------------------------------------===//

#ifndef blang_OPS
#define blang_OPS

include "mlir/Analysis/CallInterfaces.td"

include "blang/mlir/ShapeInferenceInterface.td"

// Provide a definition of the 'blang' dialect in the ODS framework so that we
// can define our operations.
def blang_Dialect : Dialect {
  let name = "blang";
  let cppNamespace = "blang";
}

// Base class for blang dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class blang_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<blang_Dialect, mnemonic, traits>;

// Provide a definition for the blang StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Blang_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "blang struct type">;

// Provide a definition of the types that are used within the blang dialect.
def Blang_Type : AnyTypeOf<[F64Tensor, Blang_StructType]>;

//===----------------------------------------------------------------------===//
// blang Operations
//===----------------------------------------------------------------------===//

// We define a blang operation by inheriting from our base 'blang_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'NoSideEffect' as it is a pure operation
// and may be removed if dead.
def ConstantOp : blang_Op<"constant",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"Builder *builder, OperationState &state, "
              "DenseElementsAttr value", [{
      build(builder, state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, double value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
}

// Constant operation containg floats (32 bits wide).
def ConstantFloat32Op : blang_Op<"constant_float32",
    [NoSideEffect]> {
  let summary = "constant_float32";
  let description = [{
    Constant operation turns a float32 literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant_float32"()
         { value = 1.0 : f32 }
        : () -> f32
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F32Attr:$value);

  // The constant operation returns a single value of a float like type.
  let results = (outs FloatLike);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantFloat32Op>(...)`.
  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "FloatAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(value.getType());
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, float value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

// Constant operation containg floats.
def ConstantFloat64Op : blang_Op<"constant_float",
    [NoSideEffect]> {
  let summary = "constant_float";
  let description = [{
    Constant operation turns a float literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant"()
         { value = 1.0 : f64 }
        : () -> f64
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64Attr:$value);

  // The constant operation returns a single value of a float like type.
  let results = (outs FloatLike);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantFloat64Op>(...)`.
  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "FloatAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(value.getType());
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, double value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

// Constant operation containg floats.
def ConstantIntOp : blang_Op<"constant_int",
    [NoSideEffect]> {
  let summary = "constant_int";
  let description = [{
    Constant operation turns an int literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant"()
         { value = 1 : i32 }
        : () -> i32
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I32Attr:$value);

  // The constant operation returns a single value of a float like type.
  let results = (outs IntegerLike);

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantIntOp>(...)`.
  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "IntegerAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(value.getType());
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, int value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

// Constant operation containg floats.
def ConstantInt64Op : blang_Op<"constant_int64",
    [NoSideEffect]> {
  let summary = "constant_int64";
  let description = [{
    Constant operation turns an int literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant_int64"()
         { value = 1 : i64 }
        : () -> i64
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I64Attr:$value);

  // The constant operation returns a single value of a int like type.
  let results = (outs IntegerLike);

  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "IntegerAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(value.getType());
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, int64_t value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

// Constant operation containg floats.
def ConstantCharOp : blang_Op<"constant_char",
    [NoSideEffect]> {
  let summary = "constant_char";
  let description = [{
    Constant operation turns an char literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant_char"()
         { value = 1 : i8 }
        : () -> i8
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins I8Attr:$value);

  // The constant operation returns a single value of a int like type.
  let results = (outs IntegerLike);

  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "IntegerAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(value.getType());
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, char value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

// Constant operation containg floats.
def ConstantStringOp : blang_Op<"constant_string",
    [NoSideEffect]> {
  let summary = "constant_string";
  let description = [{
    Constant operation turns an char literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "blang.constant_string"()
         { value =  string }
        : () -> i8*
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins StrAttr:$value);

  // The constant operation returns a single value of a int like type.
  let results = (outs I8MemRef);

  let builders = [
    // HACK: Revisit this.
    OpBuilder<"Builder *builder, OperationState &state, "
              "StringAttr value", [{
        state.addAttribute("value", value);
        state.addTypes(MemRefType::get(value.getValue().size(),
                       builder->getIntegerType(8)));
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<"Builder *builder, OperationState &state, std::string value">
  ];

  // Invoke a static verify method to verify this constant operation.
  let verifier = [{ return ::verify(*this); }];
}

def AddOp : blang_Op<"add",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def CastOp : blang_Op<"cast",
    [DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, NoSideEffect,
     SameOperandsAndResultShape]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types
    must both be tensor types with the same element type. If both are ranked
    then the rank should be the same and static dimensions should match. The
    operation is invalid if converting to a mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  // Set the folder bit so that we can fold redundant cast operations.
  let hasFolder = 1;
}

def GenericCallOp : blang_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = "blang.generic_call"(%1, %3) {callee = @my_func}
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Blang_Type>:$inputs);

  // The generic call operation returns a single value of TensorType or
  // StructType.
  let results = (outs Blang_Type);

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, "
              "StringRef callee, ArrayRef<Value> arguments">
  ];
}

def MulOp : blang_Op<"mul",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
  ];
}

def PrintOp : blang_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef, F32, F64, I8MemRef,
                                  I8, I32, I64]>:$input);
}

def ReshapeOp : blang_Op<"reshape", [NoSideEffect]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = "blang.reshape"(%arg1) : (tensor<10xf64>) -> tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);
  let hasCanonicalizer = 1;

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);
}

def ReturnOp : blang_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      func @foo() -> tensor<2xf64> {
        ...
        blang.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<Blang_Type>:$input);

  // Allow building a ReturnOp with no return operand.
  let builders = [OpBuilder<
    "Builder *b, OperationState &state", [{ build(b, state, llvm::None); }]
  >];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let verifier = [{ return ::verify(*this); }];
}

def StructAccessOp : blang_Op<"struct_access", [NoSideEffect]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Blang_StructType:$input, I64Attr:$index);
  let results = (outs Blang_Type);

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value input, size_t index">
  ];

  let verifier = [{ return ::verify(*this); }];

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}

def StructConstantOp : blang_Op<"struct_constant", [NoSideEffect]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = "blang.struct_constant"() {
        value = [dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>]
      } : () -> !blang.struct<tensor<*xf64>>
    ```
  }];

  let hasFolder = 1;
  let arguments = (ins ArrayAttr:$value);
  let results = (outs Blang_StructType);
  let verifier = [{ return ::verify(*this); }];
}

def TransposeOp : blang_Op<"transpose",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
  let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value input">
  ];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::verify(*this); }];
}

#endif // blang_OPS
