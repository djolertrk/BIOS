set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(blangAST
  AST.cpp
  )

include_directories(include/
                    ${LLVM_MAIN_INCLUDE_DIR}/
                    # This should be MLIR_MAIN_INCLUDE_DIR.
                    ${CMAKE_CURRENT_SOURCE_DIR}/../mlir/include
                    # FIXME: CMAKE WORKAROUND
                    ${CMAKE_CURRENT_BINARY_DIR}/include/
                    ${MLIR_MAIN_INCLUDE_DIR}/
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
