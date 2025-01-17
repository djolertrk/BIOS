// RUN: %blang %s -emit=llvm -opt

func @main() {
  %0 = "blang.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
  %2 = "blang.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
  %3 = "blang.mul"(%2, %2) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
  "blang.print"(%3) : (tensor<3x2xf64>) -> ()
  "blang.return"() : () -> ()
}

// CHECK-LABEL: define void @main()
// CHECK: @printf
// CHECK-SAME: 1.000000e+00
// CHECK: @printf
// CHECK-SAME: 1.600000e+01
// CHECK: @printf
// CHECK-SAME: 4.000000e+00
// CHECK: @printf
// CHECK-SAME: 2.500000e+01
// CHECK: @printf
// CHECK-SAME: 9.000000e+00
// CHECK: @printf
// CHECK-SAME: 3.000000e+01
