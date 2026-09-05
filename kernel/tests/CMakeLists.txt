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

# P1-10 (GitHub #46): the reference's own BF16 and W8G32 a16 linear tests, run
# unmodified, same pattern as the NVFP4 block above — each links ignis_kernel
# so it drives the leaf's real ops::linear dispatch (kernel/src/linear.cu),
# and neither sets SKIP_RETURN_CODE (ADR 0006: a missing/busy GPU fails here).
add_executable(ignis_kernel_bf16_linear_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_bf16_a16.cpp
)
target_include_directories(ignis_kernel_bf16_linear_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_bf16_linear_tests PRIVATE ignis_kernel)
add_test(NAME ignis_kernel_bf16_linear_tests COMMAND ignis_kernel_bf16_linear_tests)

add_executable(ignis_kernel_w8_linear_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_w8_a16.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_kernel_w8_linear_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_w8_linear_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_w8_linear_tests COMMAND ignis_kernel_w8_linear_tests)

# P1-11 (GitHub #47): the reference's own attn_input_proj test, patched
# (kernel/vendor/patches/tests/ops/test_attn_input_proj.cpp.diff) to drop the
# Q4/Q5, FP8 and W8 arms this ticket does not vendor — same pattern as the
# linear tests above, driving the leaf's own `ops::attn_input_proj`
# (kernel/src/attn_input_proj.cu) so it exercises the real NVFP4/BF16 fused
# q/k/gate/v projection at the model's 24q/4kv x 256 split
# ([14336,5120]).
add_executable(ignis_kernel_attn_input_proj_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/test_attn_input_proj.cpp
)
target_include_directories(ignis_kernel_attn_input_proj_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_attn_input_proj_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_attn_input_proj_tests COMMAND ignis_kernel_attn_input_proj_tests)

# P1-11 (GitHub #47): the reference's own linear_add tests, run unmodified —
# the NVFP4 arm ([5120,6144] and [5120,17408]) and the BF16 arm ([5120,6144],
# the attention output projection's residual add) — driving the leaf's own
# `ops::linear_add` (kernel/src/linear_add.cu).
add_executable(ignis_kernel_nvfp4_linear_add_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear_add/test_nvfp4.cpp
)
target_include_directories(ignis_kernel_nvfp4_linear_add_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_nvfp4_linear_add_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_nvfp4_linear_add_tests COMMAND ignis_kernel_nvfp4_linear_add_tests)

add_executable(ignis_kernel_bf16_linear_add_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear_add/test_bf16_a16.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear_add/linear_add_test_common.cpp
)
target_include_directories(ignis_kernel_bf16_linear_add_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_bf16_linear_add_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_bf16_linear_add_tests COMMAND ignis_kernel_bf16_linear_add_tests)
