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
#
# P1-14 (GitHub #50) adds the GDN family: causal_conv1d_silu, gdn_gating and
# gated_delta_net (recurrent + chunked, exercised through the same reference
# test by tokens counts that straddle the chunk boundary) and its
# replay-record entry point, each still a single vendored test file with no
# qtype dispatch, so the same loop applies.
set(ignis_vendored_op_tests
  rmsnorm
  gated_rmsnorm
  l2norm
  residual_add
  silu_mul
  sigmoid_mul
  embedding
  argmax
  causal_conv1d_silu
  gdn_gating
  gated_delta_net
  gated_delta_net_replay_record
  position
  rope
  qk_norm_rope
  sampling
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

# P2-03 (GitHub #85): the reference's own NVFP4 W4A4 linear test, run
# unmodified -- the activation-quantized route (CallForm::Policy AllowA4) at
# the model's attention-input [14336,5120], GDN-input [16384,5120], gate_up
# [34816,5120] and residual [5120,6144]/[5120,17408] geometries, driving the
# leaf's own `ops::linear` dispatcher (kernel/src/linear.cu) exactly like the
# a16/w8 linear tests above. Same no-SKIP_RETURN_CODE rule (ADR 0006): a
# missing or busy GPU fails here, it does not skip.
add_executable(ignis_kernel_nvfp4_linear_a4_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_nvfp4_a4.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_kernel_nvfp4_linear_a4_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_nvfp4_linear_a4_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_nvfp4_linear_a4_tests COMMAND ignis_kernel_nvfp4_linear_a4_tests)

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

# P1-12 (GitHub #48): the reference's own gdn_input_proj test, patched
# (kernel/vendor/patches/tests/ops/test_gdn_input_proj.cpp.diff) to drop the
# Q4/Q5, W8 and FP8 arms this ticket does not vendor (every GDN layer's
# query_key_value_z parent is NVFP4 per qwen3.8-27b-artifact.md §14.1, so no
# BF16 arm is needed here) — driving the leaf's own `ops::gdn_input_proj`
# (kernel/src/gdn_input_proj.cu) at the model's fused q/k/value/z split
# ([16384,5120]).
add_executable(ignis_kernel_gdn_input_proj_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/test_gdn_input_proj.cpp
)
target_include_directories(ignis_kernel_gdn_input_proj_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_gdn_input_proj_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_gdn_input_proj_tests COMMAND ignis_kernel_gdn_input_proj_tests)

# P1-12 (GitHub #48): the reference's own gdn_gating_proj test, run unmodified
# (BF16-only op, no excluded op family to trim) — both the Qwen3.8-27B
# ([96,5120] ab_weight) and Qwen3.6-35B geometries. The wrapper itself is
# vendored (kernel/vendor/src/ops/wrapper/gdn_gating_proj.cpp, built into
# ignis_vendor), so this links only ignis_vendor, same as the norm/glue tests
# above.
add_executable(ignis_kernel_gdn_gating_proj_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/test_gdn_gating_proj.cpp
)
target_include_directories(ignis_kernel_gdn_gating_proj_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_gdn_gating_proj_tests PRIVATE ignis_vendor)
add_test(NAME ignis_kernel_gdn_gating_proj_tests COMMAND ignis_kernel_gdn_gating_proj_tests)

# P1-13 (GitHub #49), A4 case restored in P2-03 (GitHub #85): the
# reference's own linear_swiglu NVFP4 test, now run unmodified -- the A4
# (large-T W4A4) case that #49's patch trimmed is back, because #85 turns
# the activation-quantized route on (the leaf's `ops::linear_swiglu`
# dispatcher, kernel/src/linear_swiglu.cu, takes the AllowA4 policy for
# NVFP4 at 27B geometry) -- so it exercises the real NVFP4 fused gate_up +
# SiLU-mul epilogue at [34816,5120] for T=1, small T and the A4 cases
# {5,48,49,128,1024}.
add_executable(ignis_kernel_nvfp4_linear_swiglu_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/linear_swiglu/test_nvfp4.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear_swiglu/linear_swiglu_test_common.cpp
)
target_include_directories(ignis_kernel_nvfp4_linear_swiglu_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_nvfp4_linear_swiglu_tests PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_nvfp4_linear_swiglu_tests COMMAND ignis_kernel_nvfp4_linear_swiglu_tests)

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

# P1-15 (GitHub #51): the reference's own gqa_attention test, patched
# (kernel/vendor/patches/tests/ops/test_gqa_attention.cpp.diff) to drop the I8
# dtype arms and their key-split/range cases — this ticket vendors the i8/hq
# decode/prefill launchers so the family compiles, but only the bf16 routes
# are gated here. Exercises both registered head geometries (24q/4kv and
# 16q/2kv, both x256) including multi-page paged-KV mappings (offset,
# fragmented). Links only ignis_vendor: gqa_attention's public API is fully
# vendored (ops/wrapper/gqa_attention.cpp), unlike attn_input_proj/linear_add
# above, which needed the leaf's own qtype dispatcher.
add_executable(ignis_kernel_gqa_attention_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/test_gqa_attention.cpp
)
target_include_directories(ignis_kernel_gqa_attention_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_gqa_attention_tests PRIVATE ignis_vendor)
add_test(NAME ignis_kernel_gqa_attention_tests COMMAND ignis_kernel_gqa_attention_tests)

# P1-15 (GitHub #51): the reference's own kv_cache_append_prefix test, run
# unmodified (BF16 only in the reference itself, no patch needed). Exercises
# both the paged and the DFlash-lane cyclic overload; the cyclic KV cache type
# is vendored (core/cyclic_kv_cache.{h,cpp}) only so the shared header
# compiles, not because ReplaySSM/DFlash2 (G5) is in scope here.
add_executable(ignis_kernel_kv_cache_append_prefix_tests
  ${IGNIS_VENDOR_DIR}/tests/ops/test_kv_cache_append_prefix.cpp
)
target_include_directories(ignis_kernel_kv_cache_append_prefix_tests PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_kv_cache_append_prefix_tests PRIVATE ignis_vendor)
add_test(NAME ignis_kernel_kv_cache_append_prefix_tests COMMAND ignis_kernel_kv_cache_append_prefix_tests)

# Ours, not vendored: the leaf-level page-budget query (consumed by P1-19,
# GitHub #55). Pure host arithmetic over the vendored planner — no GPU
# needed, so this one is a normal test, not part of the GPU-only rule above.
add_executable(ignis_kernel_paged_kv_budget_test test_paged_kv_page_budget.cpp)
target_link_libraries(ignis_kernel_paged_kv_budget_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_paged_kv_budget_test COMMAND ignis_kernel_paged_kv_budget_test)

# --- P1-19 (GitHub #55): the sequence handle ABI ----------------------------
#
# Ours, not vendored: exercises ignis_seq_alloc/release through the public
# ABI plus a direct device-memory zero-state check via
# kernel/include/ignis_seq_internal.h. Real device work (cudaMalloc,
# cudaMemset), so it follows the same no-SKIP_RETURN_CODE rule as the
# P1-16/#52 pool tests above.
add_executable(ignis_kernel_seq_alloc_test test_seq_alloc.cpp)
target_link_libraries(ignis_kernel_seq_alloc_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_alloc_test COMMAND ignis_kernel_seq_alloc_test)

# P4-03 (GitHub #119): the hq-e8-2b row codec's own op-level oracle. It was
# FOUND in the reference at `tools/test_kv/`, not `tests/ops/` — that is why
# no hq arm existed in the vendored `test_gqa_attention.cpp` (P1-15, GitHub
# #51): the reference never put codec-level coverage under `tests/ops/` in
# the first place, it lives here as its own tool with its own `main()`.
# Vendored verbatim (`tools/test_kv/test_hq_codec.cu`, `hq_dither_host.h`):
# exhaustive FP64 2*E8 nearest-point check, device encode/decode round-trip
# vs an FP64 oracle, bit-budget invariant, determinism, 8-lane group-decode
# equivalence, escalation + terminal-fallback census on a heavy-tailed
# corpus, Rice-k invariant, and a synthetic k>0 general-fallback arm.
# `hq_codec.cuh` is a header-only device header over the vendored substrate
# (already vendored by P1-15), so this links only ignis_vendor, not
# ignis_kernel. `tools/test_kv` is a private include directory so the oracle
# can include its sibling `hq_dither_host.h` unqualified, same pattern as
# the other vendored test roots above. Same no-SKIP_RETURN_CODE rule as
# every other GPU test here (ADR 0006).
add_executable(ignis_hq_codec_test ${IGNIS_VENDOR_DIR}/tools/test_kv/test_hq_codec.cu)
target_include_directories(ignis_hq_codec_test PRIVATE ${IGNIS_VENDOR_DIR}/tools/test_kv)
target_link_libraries(ignis_hq_codec_test PRIVATE ignis_vendor)
add_test(NAME ignis_hq_codec_test COMMAND ignis_hq_codec_test)

# P4-03 (GitHub #119): OURS, not vendored -- the measured-tolerance test at
# this model's real 27B geometry. The vendored ignis_hq_codec_test above
# already proves the codec's internal correctness against an FP64 brute
# force on a synthetic corpus with random signs and a fixed seed; this test
# covers only what that one cannot: round-trip error on real captured K/V
# rows (kernel/tests/fixtures/hq_kv_rows_27b.bin, captured by
# crates/core/tests/hq_kv_fixture_capture_gpu.rs), through the engine's real
# rotation and real per-row dither seeding, broken down per GQA layer
# ordinal and per role, with its enforced tolerance derived from that
# measurement (see the test file's own "derived tolerance" comment).
# IGNIS_HQ_KV_FIXTURE_PATH is an absolute path baked in at configure time so
# the test does not depend on CTest's working directory. Same
# no-SKIP_RETURN_CODE rule as every other GPU test here (ADR 0006): a
# missing or corrupt fixture is this test's own hard failure, not a skip.
add_executable(ignis_hq_codec_kv_rows_test test_hq_codec_kv_rows.cu)
target_compile_definitions(ignis_hq_codec_kv_rows_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_link_libraries(ignis_hq_codec_kv_rows_test PRIVATE ignis_vendor)
add_test(NAME ignis_hq_codec_kv_rows_test COMMAND ignis_hq_codec_kv_rows_test)

# P4-04 (GitHub #122): OURS, not vendored -- the KV append path under both
# storage formats. It builds two sequence pools differing only in
# `ignis_seq_pool_spec::kv_format`, appends the same real fixture rows to a
# sequence from each through the same vendored A2 (`gqa_kv_append`), and
# reads both back through one block-table-driven row-address helper, so "the
# hq append writes fixed-budget rows through the same page addressing BF16
# uses" is a check rather than a claim. It links ignis_kernel (for
# ignis_seq_pool_create and ignis_seq_internal.h) as well as ignis_vendor,
# unlike the codec tests above, and reuses the same committed fixture. Same
# no-SKIP_RETURN_CODE rule as every other GPU test here (ADR 0006).
add_executable(ignis_kernel_kv_append_format_test test_kv_append_format.cu)
target_compile_definitions(ignis_kernel_kv_append_format_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_link_libraries(ignis_kernel_kv_append_format_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_kv_append_format_test COMMAND ignis_kernel_kv_append_format_test)

# P4-05 (GitHub #123): OURS, not vendored -- the hq attention routes against
# ignis's own BF16 route on identical keys and values (ADR 0022's in-house
# acceptance for a lossy format). The same q/k/v/gate/positions go through A1
# twice against two pools differing only in `kv_format`, on both routes the
# GQA layer dispatches: the Prompt route at W=200 B=1, and the SmallT route at
# W=1 for every exact batch width 1..8. The tolerance is derived from the
# codec error ignis_hq_codec_kv_rows_test measured on the same fixture, which
# is why both read that one file. Links ignis_kernel for the pool ABI and the
# production view builders, and keeps the no-SKIP_RETURN_CODE rule (ADR 0006).
# P4-05 (GitHub #123): OURS, not vendored -- `ignis_model_load` refuses a KV
# format it does not recognize, and lets a recognized one through to the
# binding. Reachable only from a caller writing the flat C ABI directly (the
# Rust binding's `KvFormat` is a two-variant enum), and host-only: the check
# sits ahead of every device call the load makes, which the control arm pins.
add_executable(ignis_kernel_model_load_kv_format_test test_model_load_kv_format.cpp)
target_link_libraries(ignis_kernel_model_load_kv_format_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_kv_format_test
         COMMAND ignis_kernel_model_load_kv_format_test)

add_executable(ignis_kernel_hq_route_agreement_test test_hq_route_agreement.cu)
target_compile_definitions(ignis_kernel_hq_route_agreement_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_link_libraries(ignis_kernel_hq_route_agreement_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_hq_route_agreement_test COMMAND ignis_kernel_hq_route_agreement_test)
