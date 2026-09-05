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

# P1-09 (GitHub #45): the reference's own NVFP4 a16/small-T linear test, run
# unmodified. This is a *second* executable, not another source in
# ignis_kernel_op_tests, because the reference's test file brings its own
# main() (op_tester.h's convention is one test = one process; the reference's
# test suite is a set of small binaries, not one big one). It links the
# leaf's `ops::linear` dispatcher (ignis_kernel, kernel/src/linear.cu) so the
# vendored test drives real NVFP4 GEMV/small-T kernels, exercising the
# blockscale-k16-m128x4-v1 scale decoder and the weight divisor exactly as the
# acceptance criteria require — this test file is never rewritten to do that
# itself.
add_executable(ignis_kernel_nvfp4_linear_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_nvfp4_a16.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_kernel_nvfp4_linear_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_nvfp4_linear_tests PRIVATE ignis_kernel ignis_vendor)

# ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE here either. The
# reference's own main() returns 77 when it finds no usable CUDA device (its
# convention for a CPU-only machine); left unconfigured, CTest treats any
# nonzero exit — 77 included — as FAILED, never as a skip, so a busy or
# missing GPU cannot read as green.
add_test(NAME ignis_kernel_nvfp4_linear_tests COMMAND ignis_kernel_nvfp4_linear_tests)
