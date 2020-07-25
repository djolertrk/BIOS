# BLANG project.
set(BLANG_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include) # --src-root
set(BLANG_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir

set(BLANG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BLANG_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# TODO Take mlir-tblgen path from cmake.
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

include_directories(include/
                    ${LLVM_MAIN_INCLUDE_DIR}/
                    # This should be MLIR_MAIN_INCLUDE_DIR.
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include
                    # FIXME: CMAKE WORKAROUND
                    ${CMAKE_CURRENT_BINARY_DIR}/include/
                    ${MLIR_MAIN_INCLUDE_DIR}/
                    ${LLVM_MAIN_INCLUDE_DIR}/../../blang/include
                    ${LLVM_MAIN_INCLUDE_DIR}/../../mlir/include
                    ${LLVM_BINARY_DIR}/tools/mlir/include/
                    )

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(blang
  blang.cpp
  )

add_dependencies(blang BlangOpsIncGen)
add_dependencies(blang BlangShapeInferenceInterfaceIncGen)

target_link_libraries(blang
  PRIVATE
    blangAST
    blangMLIR
    )
