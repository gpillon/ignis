# The kernel leaf's own test executable (ADR 0010): the reference's op-test
# harness, run by CTest against a real GPU.
#
# P1-06 (GitHub #42) lands the harness and the vendored-substrate smoke test;
# each later op ticket adds its reference op test as another source here.
#
# ADR 0006 / docs/agents/testing.md: compute tests FAIL when the GPU is
# missing, busy, or a kernel errors — no SKIP_RETURN_CODE is set on them, so a
# skip can never be mistaken for green (the reference's own suite uses 77 for
# CPU-only machines; the ignis rule is stricter).

add_executable(ignis_kernel_op_tests
  test_vendor_substrate.cu
)

# The vendored op-test harness lives at `tests/ops/` in the reference and
# includes itself as "ops/op_tester.h", so the vendored tests root is an
# include root of its own. `core/...` and `ops/common/...` come from
# ignis_vendor's PUBLIC include directory.
target_include_directories(ignis_kernel_op_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_op_tests PRIVATE ignis_vendor)

add_test(NAME ignis_kernel_op_tests COMMAND ignis_kernel_op_tests)

# BF16 and W8G32 linear reference op tests (GitHub #46, P1-10). Each vendored
# test file owns a `main()`, so each is its own executable/CTest entry rather
# than a source appended to ignis_kernel_op_tests. No SKIP_RETURN_CODE is set:
# the reference's own build treats "no CUDA" (exit 77) as a skip, but ADR 0006
# makes a missing/busy GPU a CTest FAILURE here.
add_executable(ignis_kernel_op_test_bf16_linear
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_bf16_a16.cpp
)
target_include_directories(ignis_kernel_op_test_bf16_linear PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_op_test_bf16_linear PRIVATE ignis_kernel)
add_test(NAME ignis_kernel_op_test_bf16_linear COMMAND ignis_kernel_op_test_bf16_linear)

add_executable(ignis_kernel_op_test_w8_linear
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_w8_a16.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_kernel_op_test_w8_linear PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_op_test_w8_linear PRIVATE ignis_kernel)
add_test(NAME ignis_kernel_op_test_w8_linear COMMAND ignis_kernel_op_test_w8_linear)
