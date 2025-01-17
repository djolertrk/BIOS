// RUN: %blang %s -emit=mlir -opt 2>&1 | FileCheck %s

// Check the result of inlining+shape inference on an input module.

func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64>
    attributes { sym_visibility = "private" } {
  %0 = "blang.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64>
  %1 = "blang.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64>
  %2 = "blang.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
  "blang.return"(%2) : (tensor<*xf64>) -> ()
}
func @main() {
  %0 = "blang.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %1 = "blang.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
  %2 = "blang.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64>
  %3 = "blang.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64>
  %4 = "blang.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  %5 = "blang.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
  "blang.print"(%5) : (tensor<*xf64>) -> ()
  "blang.return"() : () -> ()
}

// CHECK-NOT: func @multiply_transpose
// CHECK-NOT: tensor<*xf64>

// CHECK-LABEL: func @main()
// CHECK:         [[VAL_0:%.*]] = "blang.constant"() {value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
// CHECK:         [[VAL_1:%.*]] = "blang.transpose"([[VAL_0]]) : (tensor<2x3xf64>) -> tensor<3x2xf64>
// CHECK:         [[VAL_2:%.*]] = "blang.mul"([[VAL_1]], [[VAL_1]]) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
// CHECK:         "blang.print"([[VAL_2]]) : (tensor<3x2xf64>) -> ()
// CHECK:         "blang.return"() : () -> ()
