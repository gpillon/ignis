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

# --- P1-16 (GitHub #52): the sequence-state pools ---------------------------
#
# The reference's own test_kv_cache.cpp / test_state_store.cpp, vendored
# verbatim (ADR 0010) rather than reimplemented, each its own `main()` and so
# its own executable. Same no-SKIP_RETURN_CODE rule as above: a missing GPU
# fails these, it does not skip them.
add_executable(ignis_vendor_kv_cache_test ${IGNIS_VENDOR_DIR}/tests/test_kv_cache.cpp)
target_link_libraries(ignis_vendor_kv_cache_test PRIVATE ignis_vendor)
add_test(NAME ignis_vendor_kv_cache_test COMMAND ignis_vendor_kv_cache_test)

add_executable(ignis_vendor_state_store_test ${IGNIS_VENDOR_DIR}/tests/test_state_store.cpp)
target_link_libraries(ignis_vendor_state_store_test PRIVATE ignis_vendor)
add_test(NAME ignis_vendor_state_store_test COMMAND ignis_vendor_state_store_test)

# Ours, not vendored: the leaf-level page-budget query (consumed by P1-19,
# GitHub #55). Pure host arithmetic over the vendored planner — no GPU
# needed, so this one is a normal test, not part of the GPU-only rule above.
add_executable(ignis_kernel_paged_kv_budget_test test_paged_kv_page_budget.cpp)
target_link_libraries(ignis_kernel_paged_kv_budget_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_paged_kv_budget_test COMMAND ignis_kernel_paged_kv_budget_test)
