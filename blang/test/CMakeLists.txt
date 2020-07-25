configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BLANG_TEST_DEPENDS
  FileCheck
  blang
  )

# Show progress bar!
list(APPEND LLVM_LIT_ARGS
    "-s"
)

add_lit_testsuite(check-blang "Running the blang regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BLANG_TEST_DEPENDS}
  )
set_target_properties(check-blang PROPERTIES FOLDER "Tests")

add_lit_testsuites(BLANG ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${BLANG_TEST_DEPENDS}
)
