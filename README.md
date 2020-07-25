# BIOS

## Now

BIOS is a toy language based on LLVM [0] and MLIR [1], and currently it is just an extension of the MLIR Toy language tutorial. It has its own build system defined, as well as testing infrastructure.

[0] http://llvm.org/
[1] https://mlir.llvm.org/

## Will be

BIOS is a special-purpose programming language. The language is specially designed for bioinformatics purpose, aiming to speed up the programs from that domain. blang is a compiler front end for the BIOS programming language, based on LLVM back end.

*blang* is the name of the project, and the blang provides a language front-end and tooling infrastructure for BIOS languages.

## How to build

    $ mkdir build && cd build
    $ cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;llvm;blang;" -DLLVM_ENABLE_LIBCXX=ON -DLLVM_TARGETS_TO_BUILD="X86" $PATH_TO_PROJECT//bios/llvm
    $ ninja && ninja check-blang

## Hello world

    $ cat hello.bs
    fn main() {
      var s : string = "Hello world!";
      print (s);
    }

    $ blang hello.bs
    Hello world!


