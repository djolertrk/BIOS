// RUN: not %blang %s -emit=mlir 2>&1

// The following IR is not "valid":
// - blang.print should not return a value.
// - blang.print should take an argument.
// - There should be a block terminator.
func @main() {
  %0 = "blang.print"()  : () -> tensor<2x3xf64>
}
