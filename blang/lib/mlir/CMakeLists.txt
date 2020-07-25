set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

include_directories(include/
                    ${LLVM_MAIN_INCLUDE_DIR}/
                    # This should be MLIR_MAIN_INCLUDE_DIR.
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include
                    ${BLANG_INCLUDE_DIR}/
                    ${LLVM_MAIN_INCLUDE_DIR}/../../blang/include
                    ${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/
                    ${LLVM_BINARY_DIR}/tools/mlir/include/
                    )

#TODO Take mlir-tblgen path from cmake
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)
set(LLVM_TARGET_DEFINITIONS BlangCombine.td)
mlir_tablegen(BlangCombine.inc -gen-rewriters "-I${CMAKE_CURRENT_SOURCE_DIR}/include" 
  # FIXME: CMAKE WORKAROUND
  "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
add_public_tablegen_target(BlangCombineIncGen)

add_llvm_library(blangMLIR
  MLIRGen.cpp
  Dialect.cpp
  BlangCombine.cpp
  LowerToAffineLoops.cpp
  LowerToLLVM.cpp
  ShapeInferencePass.cpp
  )

add_dependencies(blangMLIR BlangOpsIncGen)
add_dependencies(blangMLIR BlangCombineIncGen)
add_dependencies(blangMLIR MLIRCallOpInterfacesIncGen)

target_link_libraries(blangMLIR
  PRIVATE
    MLIRAllDialects
    MLIRAffineToStandard
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRLoopToStandard
    MLIRParser
    MLIRPass
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    )

if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
  target_compile_options(blangAST PRIVATE
    -Wno-unused-private-field
    )
endif()
if(MSVC)
  # ignore "warning LNK4199: /DELAYLOAD:shell32.dll ignored; no imports found from shell32.dll"
  target_link_libraries(blangAST PRIVATE "-ignore:4199")
endif()
