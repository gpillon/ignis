# The kernel leaf's own test executables (ADR 0010): the reference's op-test
# harness, run by CTest against a real GPU.
#
# P1-06 (GitHub #42) lands the harness and the vendored-substrate smoke test;
# each later op ticket adds its reference op test(s) as more executables here
# (each reference test file brings its own `main()`, so it cannot be merged
# into a shared binary).
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

# --- vendored reference op tests (P1-07 GitHub #43: norms + glue; P1-08
# GitHub #44: embedding + argmax) ---------------------------------------
#
# Each entry is a reference `tests/ops/test_<op>.cpp`, vendored verbatim
# (kernel/vendor/manifest.json) and built here as its own CTest executable.
# `ninfer/ops/...` (the public headers) and `core/...` / `ops/common/...`
# come from ignis_vendor's PUBLIC include directories; `ops/op_tester.h` and
# `ops/norm_test_common.h` come from the vendored tests root, same as above.
set(ignis_vendored_op_tests
  rmsnorm
  gated_rmsnorm
  l2norm
  residual_add
  silu_mul
  sigmoid_mul
  embedding
  argmax
)
foreach(op IN LISTS ignis_vendored_op_tests)
  set(target ignis_${op}_test)
  add_executable(${target} ${IGNIS_VENDOR_DIR}/tests/ops/test_${op}.cpp)
  target_include_directories(${target} PRIVATE ${IGNIS_VENDOR_DIR}/tests)
  target_link_libraries(${target} PRIVATE ignis_vendor)
  add_test(NAME ${target} COMMAND ${target})
endforeach()
