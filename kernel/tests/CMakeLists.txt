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

# P1-08 (GitHub #44): the reference's own op tests for embedding (W8G32 gather,
# vendored alongside dense/Q6/FP8) and argmax, vendored verbatim from
# tests/ops/. Each reference op test is its own executable (its own main()),
# matching the reference's one-test-per-executable convention. Unlike the
# reference, no SKIP_RETURN_CODE is set: ADR 0006 / docs/agents/testing.md
# make a missing GPU a failure here, never a skip.
add_executable(ignis_op_test_embedding ${IGNIS_VENDOR_DIR}/tests/ops/test_embedding.cpp)
target_include_directories(ignis_op_test_embedding PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_op_test_embedding PRIVATE ignis_vendor)
add_test(NAME ignis_op_test_embedding COMMAND ignis_op_test_embedding)

add_executable(ignis_op_test_argmax ${IGNIS_VENDOR_DIR}/tests/ops/test_argmax.cpp)
target_include_directories(ignis_op_test_argmax PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_op_test_argmax PRIVATE ignis_vendor)
add_test(NAME ignis_op_test_argmax COMMAND ignis_op_test_argmax)
