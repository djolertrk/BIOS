set(LLVM_TARGET_DEFINITIONS Ops.td)

#TODO Take mlir-tblgen path from cmake
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

include(TableGen)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${LLVM_BINARY_DIR}/include" "-I${PROJECT_SOURCE_DIR}/include" "-I${MLIR_MAIN_SRC_DIR}/include" 
  "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include"
  "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/blang" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

mlir_tablegen(Ops.h.inc -gen-op-decls "-I${CMAKE_CURRENT_SOURCE_DIR}/.." "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include" "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/blang" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
mlir_tablegen(Ops.cpp.inc -gen-op-defs "-I${CMAKE_CURRENT_SOURCE_DIR}/.." "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include" "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/blang" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
add_public_tablegen_target(BlangOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls  "-I${CMAKE_CURRENT_SOURCE_DIR}/.." "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include" "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/blang" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs  "-I${CMAKE_CURRENT_SOURCE_DIR}/.." "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/" "-I${LLVM_MAIN_INCLUDE_DIR}/../../blang/include/blang" "-I${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include/")
add_public_tablegen_target(BlangShapeInferenceInterfaceIncGen)
