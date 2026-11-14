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
  speculative_round
  dflash2_dynamic_conv
  dflash2_selector_predecessors
  dflash2_selector_scores
  dflash2_topk_walk
  swa
  prepare_masked_block
  cast
  # GitHub #177: the vision encoder's ops and the placeholder scatter.
  layer_norm
  gelu
  add_bias
  vision_pos_embed
  vision_attention
  scatter
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

# GitHub #177: the Q4/Q5/Q6 row-split linear families the vision tower is
# stored in, through the leaf's own ops::linear dispatch like the w8 test
# above (same no-SKIP_RETURN_CODE rule, ADR 0006).
foreach(q IN ITEMS q4 q5 q6)
  set(target ignis_kernel_${q}_linear_tests)
  add_executable(${target}
    ${IGNIS_VENDOR_DIR}/tests/ops/linear/test_${q}_a16.cpp
    ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
  )
  target_include_directories(${target} PRIVATE ${IGNIS_VENDOR_DIR}/tests)
  target_link_libraries(${target} PRIVATE ignis_kernel ignis_vendor)
  add_test(NAME ${target} COMMAND ${target})
endforeach()

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

# P1-12 (GitHub #48): the reference's own gdn_gating_proj test (BF16-only op,
# no excluded op family to trim) — both the Qwen3.8-27B ([96,5120] ab_weight)
# and Qwen3.6-35B geometries. Since GitHub #478 and #602 it carries a
# portability patch (ADR 0077, ADR 0091): it runs its case list at the card's
# own SM count, then under test-only caps of 84 and 36 on the SM count the
# planner reads, each below the card's own; where a cooperative schedule is not
# resident (fewer SMs, or an sm_86 build's fewer CTAs per SM) it names each
# case the planner runs on a fallback in a FALLBACK line, with its pass's SM
# count or cap, and holds it to the criterion of the route it runs;
# it prints each cooperative kernel's CTAs per SM and, on compute capability
# 12.0, fails if one is not the value the planner was tuned on, or on the 5090
# with no cap if any case falls back. The wrapper itself is
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

# GitHub #329: OURS -- the driver's libraries reached at run time, never
# linked: the forwarded driver API (kernel/src/cuda_driver_shim.cu), NVML
# opened by device.cu (kernel/src/shared_library.cu) and the driver probe. On
# this host it needs the driver; `--expect-no-driver` turns it into the check
# for a machine without one, which is how the Linux no-GPU proof runs it. It
# launches and allocates nothing, so it runs while another process holds the
# card.
add_executable(ignis_kernel_driver_libraries_test test_driver_libraries.cu)
target_include_directories(ignis_kernel_driver_libraries_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_driver_libraries_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_driver_libraries_test COMMAND ignis_kernel_driver_libraries_test)

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

# GitHub #257 (spec runtime/06): OURS -- the hq recent ring's validity-word
# masks (host arithmetic, the reference's rule) -- and the vendored kernel
# that applies them. Its last block runs on the device, so the
# no-SKIP_RETURN_CODE rule above applies.
add_executable(ignis_kernel_hq_ring_test test_hq_ring.cpp)
target_link_libraries(ignis_kernel_hq_ring_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_hq_ring_test COMMAND ignis_kernel_hq_ring_test)

# GitHub #315 (ADR 0048): OURS -- the reasoning redirect's cut, the one host
# function both leaves call wherever they assign a successor. Header-only and
# linked against nothing, so it cannot create a CUDA context: it runs while
# another process holds the card.
add_executable(ignis_kernel_reasoning_redirect_test test_reasoning_redirect.cpp)
target_include_directories(ignis_kernel_reasoning_redirect_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
add_test(NAME ignis_kernel_reasoning_redirect_test COMMAND ignis_kernel_reasoning_redirect_test)

# GitHub #267 (test-only): OURS -- a teacher-forced verify round's verdict,
# the host function the 27B verify round calls per forced lane. Header-only
# and linked against nothing, like the reasoning redirect's test above, so it
# too runs while another process holds the card.
add_executable(ignis_kernel_teacher_forcing_test test_teacher_forcing.cpp)
target_include_directories(ignis_kernel_teacher_forcing_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
                                   ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_test(NAME ignis_kernel_teacher_forcing_test COMMAND ignis_kernel_teacher_forcing_test)

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
# GitHub #571: it also holds the lean group decoder of the warp-specialized
# verify route (kernel/src/hq_decode_lean.cuh, a leaf-internal header, hence
# the include of kernel/src) to the vendored decode ways on every corpus, in
# both registrations; that decoder builds for every architecture.
add_executable(ignis_hq_codec_kv_rows_test test_hq_codec_kv_rows.cu)
target_compile_definitions(ignis_hq_codec_kv_rows_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_include_directories(ignis_hq_codec_kv_rows_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_hq_codec_kv_rows_test PRIVATE ignis_vendor)
add_test(NAME ignis_hq_codec_kv_rows_test COMMAND ignis_hq_codec_kv_rows_test)

# Spec flash-next/04 acceptance 7 (GitHub #302): the same measurement on real
# Flash-Next KV rows (kernel/tests/fixtures/hq_kv_rows_flash_next.bin, captured
# by crates/core/tests/flash_next_kv_fixture_capture_gpu.rs: 2 KV heads, QSA
# layers 3/19/35/47), held to the tolerances derived from that fixture
# (IGNIS_HQ_KV_FLASH_NEXT selects them in the test file).
add_executable(ignis_hq_codec_kv_rows_flash_next_test test_hq_codec_kv_rows.cu)
target_compile_definitions(ignis_hq_codec_kv_rows_flash_next_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_flash_next.bin"
  IGNIS_HQ_KV_FLASH_NEXT=1)
target_include_directories(ignis_hq_codec_kv_rows_flash_next_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_hq_codec_kv_rows_flash_next_test PRIVATE ignis_vendor)
add_test(NAME ignis_hq_codec_kv_rows_flash_next_test COMMAND ignis_hq_codec_kv_rows_flash_next_test)

# GitHub #335 (ADR 0061): OURS -- one prefill chunk's hq-e8-2b attention at a long history, A1's
# Prompt route against ignis's prompt attention without prompt planes, on one cache. A timing is a
# finding, so this is a tool and NOT registered with CTest.
add_executable(ignis_hq_prompt_bench bench_hq_prompt_attention.cu)
target_link_libraries(ignis_hq_prompt_bench PRIVATE ignis_kernel ignis_vendor)

# GitHub #268 (spec runtime/09): OURS -- the hq-e8-2b verify attention kernel
# timed alone at the engine's real launch, over code planes encoded from the
# same committed fixture rows, plus a hash of its partials so two builds can
# be compared bit for bit. A timing is a finding, not a pass/fail, so this is
# a tool and deliberately NOT registered with CTest. GitHub #571: the same
# binary times the reference route (--reference), today's throughput route
# and ignis's warp-specialized route (--ws12, kernel/src/hq_verify_ws.cuh).
add_executable(ignis_hq_verify_bench bench_hq_verify_attention.cu)
target_compile_definitions(ignis_hq_verify_bench PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_include_directories(ignis_hq_verify_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_hq_verify_bench PRIVATE ignis_vendor)

# GitHub #268: OURS -- ADR 0031's oracle for the patched hq verify kernel. The
# throughput route (staged code rows, the throughput group decoder, split
# K/V phases) against the reference route of the same kernel
# (IgnisThroughput = false), on caches encoded from the committed fixture:
# every partial byte and every cache byte the fused append writes must match.
# GitHub #571 (spec #570): ignis's warp-specialized route
# (kernel/src/hq_verify_ws.cuh) is held to the same reference on every shape,
# at compute capability 12.0 and up. On an older card, from a build whose list
# covers it (IGNIS_CUDA_ARCHITECTURES, #456), that route's kernel is a trap
# and the test runs the throughput route alone, so it takes no sm120a-only
# label. Same no-SKIP_RETURN_CODE rule as every other GPU test here (ADR 0006).
add_executable(ignis_kernel_hq_verify_exact_test test_hq_verify_exact.cu)
target_compile_definitions(ignis_kernel_hq_verify_exact_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_include_directories(ignis_kernel_hq_verify_exact_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_hq_verify_exact_test PRIVATE ignis_vendor)
add_test(NAME ignis_kernel_hq_verify_exact_test COMMAND ignis_kernel_hq_verify_exact_test)

# GitHub #572 (spec #570): OURS -- how a 27B load picks the attention route of
# its decode and verify rounds over hq-e8-2b (kernel/src/hq_decode_attention.h):
# the rule with the card's generation injected (Blackwell takes ignis's route
# unless IGNIS_HQ_DECODE_ATTENTION=vendored, every other card keeps today's
# whatever the switch says, a bad value is refused by name), then the image
# probe on the card at hand, which finds the route exactly when the build names
# the card's arch-specific target. Not sm120a-only: an older card's answer is
# "no image". No SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_hq_decode_attention_route_test test_hq_decode_attention_route.cu)
target_include_directories(ignis_kernel_hq_decode_attention_route_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_hq_decode_attention_route_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_hq_decode_attention_route_test COMMAND ignis_kernel_hq_decode_attention_route_test)

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

# P4-06 (GitHub #124): OURS, not vendored -- sequence state transfer. It
# checks the leaf-internal state-section table against the blob a snapshot
# actually writes, round-trips a dirtied sequence through release and
# re-allocation, and drives every refusal (stale version, foreign buffer,
# disagreeing size, edited layout, foreign geometry, too-small target,
# mid-chunk source) past the guarantee that a refused restore leaves the
# target untouched -- which needs ignis_seq_internal.h to set a sequence's
# frontier and read its device state, so it cannot live in Rust.
#
# It also PRINTS the measured snapshot/restore cost at the real 27B
# geometry for a short and a full-context sequence (the ticket's "measured
# rather than assumed"); the numbers are recorded in
# docs/findings/2026-09-12-sequence-snapshot-transfer-cost.md. That arm
# allocates a full-context KV pool plus a 144 MiB GDN slot and ~532 MB of
# pinned host memory, so it wants the card to itself -- which ADR 0006
# already requires of every test here. Same no-SKIP_RETURN_CODE rule.
add_executable(ignis_kernel_seq_snapshot_test test_seq_snapshot.cpp)
target_link_libraries(ignis_kernel_seq_snapshot_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_snapshot_test COMMAND ignis_kernel_seq_snapshot_test)

# P4-05 (GitHub #123): OURS, not vendored -- `ignis_model_load` refuses a KV
# format it does not recognize, and lets a recognized one through to the
# binding. Reachable only from a caller writing the flat C ABI directly (the
# Rust binding's `KvFormat` is a two-variant enum), and host-only: the check
# sits ahead of every device call the load makes, which the control arm pins.
add_executable(ignis_kernel_model_load_kv_format_test test_model_load_kv_format.cpp)
target_link_libraries(ignis_kernel_model_load_kv_format_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_kv_format_test
         COMMAND ignis_kernel_model_load_kv_format_test)

# P5-02 (GitHub #150): OURS, not vendored -- `ignis_model_load`'s speculation
# option: the options struct is refused by name before binding, and under
# DFLASH2 each of the 66 `dflash2/*` tensors is required, named when missing or
# mis-shaped, and an extra without the option. Host-only: every arm fails at
# binding, ahead of the load's device calls.
add_executable(ignis_kernel_model_load_speculative_options_test
               test_model_load_speculative_options.cpp)
target_link_libraries(ignis_kernel_model_load_speculative_options_test
                      PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_speculative_options_test
         COMMAND ignis_kernel_model_load_speculative_options_test)

# GitHub #227: OURS -- the text RoPE frequency table and its load option.
# Host-only twice over: the YaRN builder is pure math over the checkpoint's
# rope constants, and the load arms all fail at (or ahead of) binding, so
# this test needs no GPU at all, like the page-budget test above. It reaches
# into `kernel/src` for `rope_scaling.h`, which is a leaf-internal header,
# not part of the public include directory.
add_executable(ignis_kernel_rope_scaling_test test_rope_scaling.cpp)
target_include_directories(ignis_kernel_rope_scaling_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_rope_scaling_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_rope_scaling_test COMMAND ignis_kernel_rope_scaling_test)

# GitHub #177: OURS -- `ignis_model_load`'s vision envelope. The envelope is
# range-checked before binding, and with it each of the 333 `vision/*` tensors
# is required, named when missing or mis-shaped, and an extra without it.
# Host-only, the same way as the speculative options test above.
add_executable(ignis_kernel_model_load_vision_options_test
               test_model_load_vision_options.cpp)
target_link_libraries(ignis_kernel_model_load_vision_options_test
                      PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_vision_options_test
         COMMAND ignis_kernel_model_load_vision_options_test)

# GitHub #336: OURS -- the 27B's decode lanes as a load option. Host-only for
# the plan (0 is 8 byte for byte, 9 refused, every round line the formula at
# every lane count); then a no-layer BF16 load on the device captures its
# graphs at widths 1..lanes and refuses a round one wider.
add_executable(ignis_kernel_model_load_decode_lanes_test test_model_load_decode_lanes.cpp)
target_link_libraries(ignis_kernel_model_load_decode_lanes_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_decode_lanes_test
         COMMAND ignis_kernel_model_load_decode_lanes_test)

# OURS -- the vision encoder's workspace size, pinned against the serving
# load's own plan figure. Host-only layout arithmetic; it reaches into
# `kernel/src` for `model_internal.h`, like the RoPE scaling test.
add_executable(ignis_kernel_vision_workspace_test test_vision_workspace.cpp)
target_include_directories(ignis_kernel_vision_workspace_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_vision_workspace_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_vision_workspace_test COMMAND ignis_kernel_vision_workspace_test)

# GitHub #557 (ADR 0087): OURS -- the media encode at both text models' merger widths, 2,560
# (Flash-Next) and 5,120 (the 27B): every linear of an encode routed at every width it runs, for
# every item up to 32,768 merged tokens (host-only, `--cases routes`), and ignis_media_encode end to
# end through a no-layer load of each width, the companion's tower in its stored formats under each
# vision residency, against fp64 and host against device bit for bit. GPU (ADR 0006: no
# SKIP_RETURN_CODE); not sm120a-only, the tower's and the merger's routes build for every target.
add_executable(ignis_kernel_media_encode_test test_media_encode.cu)
target_include_directories(ignis_kernel_media_encode_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_media_encode_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_media_encode_test COMMAND ignis_kernel_media_encode_test)

# GitHub #296 / #335: OURS -- the GQA attention's arena: ignis's prompt
# attention's, flat in the context past one band and, at 524,288 keys, too
# small to hold one of the two prompt planes A1's holds (#603), and the vendored
# A1 capacity query against a layout-builder replay of what the hq Prompt
# route bumps, at every serving prompt width, in one scratch band and past it.
# Host-only layout arithmetic (the SM-count query creates no context); it pins
# the recorded correctness patch on kernel/vendor/src/ops/wrapper/
# gqa_attention.cpp (ADR 0037).
add_executable(ignis_kernel_gqa_workspace_capacity_test test_gqa_workspace_capacity.cpp)
target_link_libraries(ignis_kernel_gqa_workspace_capacity_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_gqa_workspace_capacity_test COMMAND ignis_kernel_gqa_workspace_capacity_test)

# P4-05 (GitHub #123): OURS, not vendored -- the hq attention routes against
# ignis's own BF16 route on identical keys and values (ADR 0022's in-house
# acceptance for a lossy format). The same q/k/v/gate/positions go through A1
# twice against two pools differing only in `kv_format`, on both routes the
# GQA layer dispatches: the Prompt route at W=200 and every narrow width
# 9..16 at B=1, and the SmallT route at W=1 for every exact batch width 1..8.
# The tolerance is derived from the codec error ignis_hq_codec_kv_rows_test
# measured on the same fixture, which is why both read that one file. Links
# ignis_kernel for the pool ABI, the production view builders and the
# engine's own attention-workspace sizing, and keeps the no-SKIP_RETURN_CODE
# rule (ADR 0006).
# OURS, not vendored: the DFlash2 drafter's top-k (kernel/src/dflash2_topk.h)
# against the vendored op it replaces. The vendored op specifies an exact
# deterministic selection, so the bound is bit equality and there is nothing
# to calibrate. Links ignis_kernel for our implementation and ignis_vendor for
# the oracle; no SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_dflash2_topk_test test_dflash2_topk.cu)
target_link_libraries(ignis_kernel_dflash2_topk_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_dflash2_topk_test COMMAND ignis_kernel_dflash2_topk_test)

# GitHub #586 (spec #287): OURS -- the permitted set's second form, the mask
# over the whole vocabulary a repair restricts a position to
# (kernel/src/permitted_tokens.h, leaf-internal). Bit-exact against the host
# walking the same bits. Not sm120a-only; no SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_permit_mask_bits_test test_permit_mask_bits.cu)
target_include_directories(ignis_kernel_permit_mask_bits_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_permit_mask_bits_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_permit_mask_bits_test COMMAND ignis_kernel_permit_mask_bits_test)

# GitHub #576 (spec #574, ADR 0090): OURS -- the sampled draft's kernels
# (kernel/src/sampled_draft.h, leaf-internal): the drafter's distribution, the
# draft's draw, and the accept, whose target distribution is the vendored
# sampler builder's bit for bit and whose greedy lanes and extent-0 lanes write
# the vendored accept's outputs. Links ignis_kernel for ours and ignis_vendor
# for the oracles. Not sm120a-only: the rule runs on every generation the 27B
# does. No SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_sampled_draft_test test_sampled_draft.cu)
target_include_directories(ignis_kernel_sampled_draft_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_sampled_draft_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_sampled_draft_test COMMAND ignis_kernel_sampled_draft_test)

# GitHub #507: OURS, an adapted op -- the GDN prefill's split conv (kernel/src/gdn_conv_split.h,
# leaf-internal) against what the layer body ran before it, the vendored causal conv and the three
# strided copies that split its plane, at every form the vendored wrapper switches between and the
# 27B's prefill chunks. The conv is exact, so the bound is bit equality. Links ignis_kernel for
# ours and ignis_vendor for the oracle; no SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_gdn_conv_split_test test_gdn_conv_split.cu)
target_include_directories(ignis_kernel_gdn_conv_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_gdn_conv_split_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_gdn_conv_split_test COMMAND ignis_kernel_gdn_conv_split_test)
# GitHub #507: OURS -- the GDN prefill copies no layout: the layer body at the prefill chunks and
# the chunked program prefill on a synthetic trellis body of the 27B's widths, against the body and
# loop as they ran before (the leaf's test-only switch, kernel/src/layer_internal.h), bit for bit,
# and the copy counter at zero. GPU; no SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_gdn_prefill_layout_test test_gdn_prefill_layout.cu)
target_include_directories(ignis_kernel_gdn_prefill_layout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_gdn_prefill_layout_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_gdn_prefill_layout_test COMMAND ignis_kernel_gdn_prefill_layout_test)
# GitHub #514 (spec #503): OURS -- the GDN prefill's chunked delta rule on our own full-rate kernels
# (kernel/src/gdn_prefill.h, leaf-internal) against the vendored chunked path (the oracle, ADR 0031)
# and both against an fp64 recurrence: outputs and carried state at the 27B's prefill chunks 1024,
# 2048 and 4096, three calls carrying the state in place, and ragged remainders, within the bound
# the test file argues. Not sm120a-only: it runs on 8.6. GPU; no SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_gdn_prefill_test test_gdn_prefill.cu)
target_include_directories(ignis_kernel_gdn_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_gdn_prefill_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_gdn_prefill_test COMMAND ignis_kernel_gdn_prefill_test)
# Its bench: one layer's GDN recurrence per chunk width (--tokens) and a prompt's 48 layers
# (--prompts), the vendored chunked kernels against ours. A tool and NOT a CTest; timing wants the
# card to itself.
add_executable(ignis_kernel_gdn_prefill_bench bench_gdn_prefill.cu)
target_include_directories(ignis_kernel_gdn_prefill_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_gdn_prefill_bench PRIVATE ignis_kernel ignis_vendor)

# ADR 0062: the token embedding in mapped pinned host memory (OURS: the
# device surface's allocation), gathered by the vendored W8 op bit for bit as
# from device memory, eager and graph-replayed. GPU; no SKIP_RETURN_CODE
# (ADR 0006).
add_executable(ignis_kernel_host_mapped_embedding_test test_host_mapped_embedding.cu)
target_link_libraries(ignis_kernel_host_mapped_embedding_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_host_mapped_embedding_test COMMAND ignis_kernel_host_mapped_embedding_test)
# The same gather timed from the mapped table against device memory: a tool,
# NOT a CTest (the 2026-10-21 host-embedding finding).
add_executable(ignis_kernel_host_mapped_embedding_bench bench_host_mapped_embedding.cu)
target_link_libraries(ignis_kernel_host_mapped_embedding_bench PRIVATE ignis_kernel ignis_vendor)

# OURS, not vendored: the A16 NVFP4 linear's tensor-core route
# (kernel/include/ignis_nvfp4_a16_mma.h), which takes every A16 call of 64
# columns or more (128 below 4,096 output rows), and on 8.x a prefill's
# DFlash2 feature taps from a width of their own (#521). Driven through the
# leaf's ops::linear with the vendored linear harness's FP64 oracle and its A16
# criterion, at prompt widths the vendored test never reaches, plus an arm
# that pins which kernel ops::linear ran, bit for bit, the widths per GPU
# generation, and the route at a reuse tail's 1-63 columns against the GEMVs.
# No SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_nvfp4_a16_mma_test
  test_nvfp4_a16_mma.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_kernel_nvfp4_a16_mma_test PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_kernel_nvfp4_a16_mma_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_nvfp4_a16_mma_test COMMAND ignis_kernel_nvfp4_a16_mma_test)

# OURS: the same route timed against the vendored small-T GEMVs on one weight.
# A timing is a finding, not a pass/fail, so a tool and NOT registered with
# CTest.
add_executable(ignis_nvfp4_a16_mma_bench
  bench_nvfp4_a16_mma.cpp
  ${IGNIS_VENDOR_DIR}/tests/ops/linear/linear_test_common.cpp
)
target_include_directories(ignis_nvfp4_a16_mma_bench PRIVATE ${IGNIS_VENDOR_DIR}/tests)
target_link_libraries(ignis_nvfp4_a16_mma_bench PRIVATE ignis_kernel ignis_vendor)

# GitHub #260 (ADR 0038): OURS, not vendored -- the attention readout
# (kernel/src/attention_readout.h, a leaf-internal header, hence the include
# of kernel/src). One query head's scores over a span of keys, from BF16 pages
# through a block table and from the hq prompt route's rotated-frame plane,
# against a double-precision host restatement of `q . k / 16`, plus every
# guard that leaves a readout unread. No SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_attention_readout_test test_attention_readout.cu)
target_include_directories(ignis_kernel_attention_readout_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_attention_readout_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_attention_readout_test COMMAND ignis_kernel_attention_readout_test)

add_executable(ignis_kernel_hq_route_agreement_test test_hq_route_agreement.cu)
target_compile_definitions(ignis_kernel_hq_route_agreement_test PRIVATE
  IGNIS_HQ_KV_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_27b.bin")
target_link_libraries(ignis_kernel_hq_route_agreement_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_hq_route_agreement_test COMMAND ignis_kernel_hq_route_agreement_test)

# GitHub #510 (spec #503): OURS -- the integer band (IgnisHqPromptPrecision::kInt8) against an fp64
# attention over the rows it reads, within a bound built from each of its roundings, its output
# and its carried state, at the 8K and 64K prefixes of a prompt prefilled at chunk 4096, on one
# split, the band's split rule and a narrow band. It runs on compute capability 8.6 as on 12.0:
# no sm120a-only label. No SKIP_RETURN_CODE (ADR 0006).
add_executable(ignis_kernel_hq_prompt_int8_test test_hq_prompt_int8.cu)
target_link_libraries(ignis_kernel_hq_prompt_int8_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_hq_prompt_int8_test COMMAND ignis_kernel_hq_prompt_int8_test)

# P4-10 (GitHub #126): OURS, not vendored -- device prefix reuse. It checks
# the two halves ADR 0024 describes against the device itself: that a
# claimant's block-table row addresses the *publisher's own physical pages*
# for the head (and that the pool is charged for them once, not once per
# claimant), and that its GDN slot, conv taps and penalty-count row come out
# byte-identical to the publisher's state at the prefix's end. Both need
# ignis_seq_internal.h / ignis_seq_prefix_internal.h to read a row and a slot
# back, so neither can live in Rust.
#
# It also PRINTS the measured device-to-device clone cost at the real 27B
# geometry under hq-e8-2b (the ticket's "measured rather than assumed"); the
# numbers are recorded in
# docs/findings/2026-09-12-device-prefix-clone-cost.md. That arm builds a
# full-context pool plus two 144 MiB GDN slots, so it wants the card to
# itself -- which ADR 0006 already requires of every test here. Same
# no-SKIP_RETURN_CODE rule.
add_executable(ignis_kernel_seq_prefix_test test_seq_prefix.cpp)
target_link_libraries(ignis_kernel_seq_prefix_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_prefix_test COMMAND ignis_kernel_seq_prefix_test)

# GitHub #186 (ADR 0029): OURS, not vendored -- prompt checkpoints. The
# shared-prefix test above proves what a prefix does; this proves the three
# things a checkpoint adds, and none of them is visible through the flat ABI:
# that the state is captured at the *generation opener* rather than at the
# page boundary below it, that the partial page the opener ends inside is
# copied while every whole page below stays shared, and that a capture reads
# the live sequence without perturbing a byte of it. It also asserts the
# acceptance criterion that only the ABI can check -- the penalty-count row a
# claimant receives is all zeros, because nothing has been sampled at the
# opener. All three need ignis_seq_internal.h / ignis_seq_checkpoint_internal.h
# to set a frontier and read a slot back, so none can live in Rust. Same
# no-SKIP_RETURN_CODE rule as every other GPU test here (ADR 0006).
add_executable(ignis_kernel_seq_checkpoint_test test_seq_checkpoint.cpp)
target_link_libraries(ignis_kernel_seq_checkpoint_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_checkpoint_test COMMAND ignis_kernel_seq_checkpoint_test)

# --- Flash-Next MoE ops (spec flash-next/02, GitHub #300): OURS -------------------------------
#
# Every op in kernel/include/ignis_moe.h is our own implementation (ADR 0043). The fixtures
# under fixtures/flash_next/ were recorded once from exllamav3's `reconstruct` and the
# checkpoint's transformers router by fixtures/flash_next/record.py; the tests read them through
# an absolute path baked in at configure time. Same no-SKIP_RETURN_CODE rule as every other GPU
# test here (ADR 0006).
set(IGNIS_FLASH_NEXT_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flash_next")
foreach(name IN ITEMS trellis_decode moe_router moe_experts fp8_linear moe_shared_combine moe_block
                     moe_prepare moe_decode_routes)
  add_executable(ignis_kernel_${name}_test test_${name}.cu)
  target_compile_definitions(ignis_kernel_${name}_test PRIVATE
    IGNIS_FLASH_NEXT_FIXTURE_DIR="${IGNIS_FLASH_NEXT_FIXTURE_DIR}")
  target_link_libraries(ignis_kernel_${name}_test PRIVATE ignis_kernel ignis_vendor)
  add_test(NAME ignis_kernel_${name}_test COMMAND ignis_kernel_${name}_test)
endforeach()
# What the expert kernels trap on rather than read as weights or store as a finite wrong answer:
# a selected slot with no record, an expert id outside [0, 512), a non-finite contribution. A
# trap kills the context, so each condition runs in its own process.
add_executable(ignis_kernel_moe_slot_trap_test test_moe_slot_trap.cu)
target_link_libraries(ignis_kernel_moe_slot_trap_test PRIVATE ignis_kernel ignis_vendor)
foreach(mode IN ITEMS decode prefill decode-id prefill-id decode-nan decode-clusters decode-id-clusters
                     decode-nan-clusters decode-registers decode-id-registers decode-nan-registers)
  add_test(NAME ignis_kernel_moe_slot_trap_${mode}_test COMMAND ignis_kernel_moe_slot_trap_test ${mode})
endforeach()
# The decode route's other implementations (IGNIS_MOE_DECODE_CLUSTERS, one thread-block cluster
# per expert; IGNIS_MOE_DECODE_REGISTERS, the ticket kernel before #306) held to the same tests as
# the default.
foreach(name IN ITEMS moe_experts moe_block)
  add_test(NAME ignis_kernel_${name}_clusters_test COMMAND ignis_kernel_${name}_test clusters)
  add_test(NAME ignis_kernel_${name}_registers_test COMMAND ignis_kernel_${name}_test registers)
endforeach()
# --- the 27B's dense trellis linear (spec 27b-16gb/02, GitHub #337): OURS ---------------------
#
# kernel/include/ignis_trellis_linear.h. The fixtures under fixtures/trellis_dense/ were recorded
# once from exllamav3's quantizer and `reconstruct` by fixtures/trellis_dense/record.py; the
# real-shape arms regenerate their counter-hash words on the device. Same no-SKIP_RETURN_CODE
# rule as every other GPU test here (ADR 0006).
set(IGNIS_TRELLIS_DENSE_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/trellis_dense")
foreach(name IN ITEMS trellis_dense_decode trellis_linear)
  add_executable(ignis_kernel_${name}_test test_${name}.cu)
  target_compile_definitions(ignis_kernel_${name}_test PRIVATE
    IGNIS_TRELLIS_DENSE_FIXTURE_DIR="${IGNIS_TRELLIS_DENSE_FIXTURE_DIR}")
  target_link_libraries(ignis_kernel_${name}_test PRIVATE ignis_kernel ignis_vendor)
  add_test(NAME ignis_kernel_${name}_test COMMAND ignis_kernel_${name}_test)
endforeach()
# GitHub #509 (spec #503): OURS -- the dense prefill (kernel/include/ignis_dense_prefill.h): its
# W8 codes against a host decode, its A8 against a host reference bit for bit, its int32
# accumulators against an exact integer reference on the CUDA cores at every tile plan, and the
# composed call's outputs within BF16 rounding of the fp64 epilogue, the tile plan's picks on the
# host and every tile plan's BF16 bytes against the op's own (GitHub #530); on synthetic matrices of
# every K class, and on real matrices of the base 16 GB artifact ($IGNIS_DENSE_PREFILL_ARTIFACT,
# SKIP outside the GPU profile when absent, a failure under it). Not sm120a-only: the kernels fit
# sm_86. The leaf-internal tile plan (dense_prefill_internal.cuh) is why it sees kernel/src.
add_executable(ignis_kernel_dense_prefill_test test_dense_prefill.cu)
target_include_directories(ignis_kernel_dense_prefill_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_dense_prefill_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_dense_prefill_test COMMAND ignis_kernel_dense_prefill_test)
# Its bench, against the fused prefill per body shape and per chunk at the quant's K mix (--tokens,
# --sweep for every tile plan). A tool and NOT a CTest; timing wants the card to itself.
add_executable(ignis_kernel_dense_prefill_bench bench_dense_prefill.cu)
target_include_directories(ignis_kernel_dense_prefill_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_dense_prefill_bench PRIVATE ignis_kernel ignis_vendor)
# The dense trellis microbenchmark: spec 02's report (DRAM bandwidth at 1..64 tokens, prefill
# TFLOP/s, under the debug SM caps). A tool and NOT a CTest; timing wants the card to itself.
add_executable(ignis_kernel_trellis_linear_bench bench_trellis_linear.cu)
target_link_libraries(ignis_kernel_trellis_linear_bench PRIVATE ignis_kernel ignis_vendor)
# The MoE microbenchmark: a report of every op's achieved bandwidth or throughput. The spec's
# decode floor (one-token routed decode >= 50% of the DRAM roofline) is its one pass/fail,
# opt-in with --check-floor; it is a tool and NOT a CTest while that floor is open, so the stage
# stays green for everyone else. Timing wants the card to itself.
add_executable(ignis_kernel_moe_bench bench_moe.cu)
target_link_libraries(ignis_kernel_moe_bench PRIVATE ignis_kernel ignis_vendor)

# GitHub #302 (spec flash-next/04): OURS -- the GDN layer count and value-head count as two
# topology numbers at the binder. Host-only (every arm fails at binding), like the other
# model-load option tests above.
add_executable(ignis_kernel_model_load_gdn_geometry_test test_model_load_gdn_geometry.cpp)
target_link_libraries(ignis_kernel_model_load_gdn_geometry_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_gdn_geometry_test COMMAND ignis_kernel_model_load_gdn_geometry_test)

# GitHub #302 (spec flash-next/04): OURS -- the Flash-Next program's internal
# contract (kernel/src/flash_next/flash_next_internal.h), its geometry read off
# the topology. Host-only; reaches into kernel/src for the leaf-internal header.
add_executable(ignis_kernel_flash_next_contract_test test_flash_next_contract.cpp)
target_include_directories(ignis_kernel_flash_next_contract_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_contract_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_contract_test COMMAND ignis_kernel_flash_next_contract_test)
# ADR 0059 (spec rope-scaling/02): OURS -- the rotary table a Flash-Next load rotates at, YaRN
# against HF's formula in fp64 and the linear one against the table before YaRN. Host-only.
add_executable(ignis_kernel_flash_next_rope_table_test test_flash_next_rope_table.cpp)
target_include_directories(ignis_kernel_flash_next_rope_table_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_rope_table_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_rope_table_test COMMAND ignis_kernel_flash_next_rope_table_test)
# A sequence pool of the topology's attention layers (kv_num_layers), planned
# without building it. Host-only.
add_executable(ignis_kernel_seq_pool_layers_test test_seq_pool_layers.cpp)
target_link_libraries(ignis_kernel_seq_pool_layers_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_pool_layers_test COMMAND ignis_kernel_seq_pool_layers_test)
# GitHub #302: a pool with Flash-Next's indexer and n-gram sections holds them,
# zeroes a slot's for its next sequence, and refuses every snapshot and clone
# by name. GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_seq_flash_next_sections_test test_seq_flash_next_sections.cpp)
target_link_libraries(ignis_kernel_seq_flash_next_sections_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_seq_flash_next_sections_test COMMAND ignis_kernel_seq_flash_next_sections_test)
# The Flash-Next binder (kernel/src/flash_next/bind.cu) at the real 48-layer
# schema: binds, places, refuses by name. Host-only.
add_executable(ignis_kernel_flash_next_bind_test test_flash_next_bind.cpp)
target_include_directories(ignis_kernel_flash_next_bind_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_bind_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_bind_test COMMAND ignis_kernel_flash_next_bind_test)
# fn_linear (kernel/src/flash_next/linear.cu): the BF16 route against fp64 at the
# program's shapes, both routes, and the FP8 routing. GPU.
add_executable(ignis_kernel_flash_next_linear_test test_flash_next_linear.cu)
target_include_directories(ignis_kernel_flash_next_linear_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_linear_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_linear_test COMMAND ignis_kernel_flash_next_linear_test)
# The hyper-connection mix and inject (kernel/src/flash_next/hc.cu) against a
# BF16-rounding fp64 reference of the checkpoint's module. GPU.
add_executable(ignis_kernel_flash_next_hc_test test_flash_next_hc.cu)
target_include_directories(ignis_kernel_flash_next_hc_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_hc_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_hc_test COMMAND ignis_kernel_flash_next_hc_test)
# Its microbenchmark at the model's geometry and formats (GitHub #306): a tool, NOT a CTest.
add_executable(ignis_kernel_flash_next_hc_bench bench_flash_next_hc.cu)
target_include_directories(ignis_kernel_flash_next_hc_bench
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_hc_bench PRIVATE ignis_kernel ignis_vendor)
# The indexer's decode score on the graph's grid, a fitted one and the strided one (GitHub #306,
# step 3): a tool, NOT a CTest.
add_executable(ignis_kernel_flash_next_score_bench bench_flash_next_score.cu)
target_include_directories(ignis_kernel_flash_next_score_bench
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_score_bench PRIVATE ignis_kernel ignis_vendor)
# The token embedding and the head (kernel/src/flash_next/embed_head.cu). GPU.
add_executable(ignis_kernel_flash_next_embed_head_test test_flash_next_embed_head.cu)
target_include_directories(ignis_kernel_flash_next_embed_head_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_flash_next_embed_head_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_embed_head_test COMMAND ignis_kernel_flash_next_embed_head_test)
# Slice S2 (kern): the GDN layer core (kernel/src/flash_next/gdn.cu) at the real geometry against
# an fp64 restatement of the checkpoint's Qwen4ExpTextGatedDeltaNet: prefill chunks, decode lanes
# eager and as a replayed graph, state carry and slot isolation. GPU.
# The QSA attention sublayer (kernel/src/flash_next/qsa.cu, qsa_dense.cu) on paged K/V of both
# formats: the norms and rope, the dense kernel against fp64, the whole sublayer in prefill and in
# graph-captured decode, hq-e8-2b within a bound derived from its measured codec error. GPU.
# GitHub #306: the forward's side branch (kernel/src/flash_next/side_branch.h) orders its work
# before the join, and a failed op on it still leaves a capture that ends once joined. GPU.
# GitHub #307 (spec flash-next/07 phase C): the verify round's GDN half -- the replay record and our fold of
# the first c records against c one-token rounds over the same columns, bit for bit. GPU.
# GitHub #329 item 5: a verify round's MoE past the decode route's one launch -- each row the bits
# of its own launch. GPU.
# ADR 0059 (spec rope-scaling/02): the two rope sites (qsa.cu prepare_kernel, indexer.cu rope_row)
# under a factor-4 YaRN table up to position 1,048,575, and the indexer's selection over it,
# against a host restatement. GPU.
# GitHub #564: the MTP head's combine takes the vision rows an image's placeholders took, bit for
# bit the entries of the ids those rows embed, from a VRAM and a mapped host page. GPU.
foreach(name IN ITEMS gdn qsa side_branch verify moe_rows rope mtp)
  add_executable(ignis_kernel_flash_next_${name}_test test_flash_next_${name}.cu)
  target_include_directories(ignis_kernel_flash_next_${name}_test
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_link_libraries(ignis_kernel_flash_next_${name}_test PRIVATE ignis_kernel ignis_vendor)
  add_test(NAME ignis_kernel_flash_next_${name}_test COMMAND ignis_kernel_flash_next_${name}_test)
endforeach()

# GitHub #301 (spec flash-next/03): OURS -- the expert residency miss path measured (copy
# engine against SM-driven copies from mapped pinned memory). A timing is a finding, not a
# pass/fail, so a tool and NOT registered with CTest. CUDA runtime only.
add_executable(ignis_expert_miss_path_bench bench_expert_miss_path.cu)
target_link_libraries(ignis_expert_miss_path_bench PRIVATE CUDA::cudart)

# GitHub #301 (spec flash-next/03): OURS -- expert residency's device side replayed against the
# CPU policy model's outcomes (fixtures/residency/trace_v1.txt, written by
# crates/core/tests/expert_residency_fixture.rs): outcomes, copies and slot-table invariants, once
# eagerly and once with the decode steps as captured CUDA graphs. Same no-SKIP_RETURN_CODE rule as
# every GPU test here (ADR 0006).
add_executable(ignis_kernel_residency_trace_test test_residency_trace.cu)
target_compile_definitions(ignis_kernel_residency_trace_test PRIVATE
  IGNIS_RESIDENCY_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/residency/trace_v1.txt")
target_link_libraries(ignis_kernel_residency_trace_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_residency_trace_test COMMAND ignis_kernel_residency_trace_test)
add_test(NAME ignis_kernel_residency_graph_test COMMAND ignis_kernel_residency_trace_test graph)
add_test(NAME ignis_kernel_residency_split_test COMMAND ignis_kernel_residency_trace_test split)
add_test(NAME ignis_kernel_residency_graph_split_test COMMAND ignis_kernel_residency_trace_test graph split)
# GitHub #324 (item 7): every upload a step reads lands 2 ms late on its own stream, so a step not
# ordered after its upload reads the old bytes every run instead of now and then. A guard on the
# test's uploads, not a reproduction of the flake: a synchronous copy plus the delay passed.
add_test(NAME ignis_kernel_residency_upload_order_test COMMAND ignis_kernel_residency_trace_test split)
set_tests_properties(ignis_kernel_residency_upload_order_test PROPERTIES
  ENVIRONMENT "IGNIS_RESIDENCY_UPLOAD_DELAY_US=2000")
# OURS -- a device-only residency (host_pool_bytes 0, ADR 0045's resident load): creation and
# admission rules, every record's bytes in its slot, and steps -- eager, with either lookahead,
# and captured -- that hit everything and copy nothing. GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_residency_device_only_test test_residency_device_only.cu)
target_link_libraries(ignis_kernel_residency_device_only_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_residency_device_only_test COMMAND ignis_kernel_residency_device_only_test)
# GitHub #329 item 12: OURS -- a pinned hot set beside a host pool: creation and admission rules,
# every pinned record on a slot past its class's LRU, and steps -- eager, split, routed and
# captured -- where a selected pinned key only ever hits and nothing evicts, copies or stages it.
# GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_residency_pinned_test test_residency_pinned.cu)
target_link_libraries(ignis_kernel_residency_pinned_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_residency_pinned_test COMMAND ignis_kernel_residency_pinned_test)
# GitHub #306 (decode fusion, step 5): OURS -- residency's routed demand step (the router's selection
# made in the resolve's launch) against the router and the demand step it replaces, bit for bit.
# GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_residency_routed_test test_residency_routed.cu)
target_link_libraries(ignis_kernel_residency_routed_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_residency_routed_test COMMAND ignis_kernel_residency_routed_test)

# GitHub #302 (spec flash-next/04, slice S4): OURS -- the n-gram embedding's device side
# (kernel/src/flash_next/ngram.cu) at the real geometry, against transformers'
# Qwen4ExpTextPLELayer recorded by fixtures/flash_next_ngram/record.py: one shot, chunked prefill
# and decode, three lanes, a call spanning two windows, a reused slot. `--host-only` runs the
# host model's check alone (no CUDA). GPU.
add_executable(ignis_kernel_flash_next_ngram_test test_flash_next_ngram.cu)
target_include_directories(ignis_kernel_flash_next_ngram_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(ignis_kernel_flash_next_ngram_test PRIVATE
  IGNIS_FLASH_NEXT_NGRAM_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flash_next_ngram")
target_link_libraries(ignis_kernel_flash_next_ngram_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_ngram_test COMMAND ignis_kernel_flash_next_ngram_test)
# The resolve's cost per layer step at Flash-Next's geometry (48 x 512, real pool capacities,
# 4 KiB records so PCIe is not what it times). A timing is a finding, not a pass/fail: a tool,
# NOT registered with CTest.
add_executable(ignis_residency_resolve_bench bench_residency_resolve.cu)
target_link_libraries(ignis_residency_resolve_bench PRIVATE ignis_kernel ignis_vendor)

# GitHub #302 (spec flash-next/04, slice S3): OURS -- the QSA indexer and the sparse attention.
# The indexer's stages (kernel/src/flash_next/indexer.cu) at the real geometry against the
# checkpoint's own Qwen4ExpTextQSAIndexer, recorded once on CPU by
# fixtures/flash_next_indexer/record_indexer.py: chunks that split blocks, two-lane decode, a
# one-shot replay, documented score ties, a graph-captured decode. The sparse attention
# (kernel/src/flash_next/qsa_sparse.cu) against fp64 attention masked to each row's selected
# tokens, over BF16 pages and over hq-e8-2b rows decoded with the vendored codec. GPU, same
# no-SKIP_RETURN_CODE rule (ADR 0006).
add_executable(ignis_kernel_flash_next_indexer_test test_flash_next_indexer.cu)
target_compile_definitions(ignis_kernel_flash_next_indexer_test PRIVATE
  IGNIS_FLASH_NEXT_INDEXER_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flash_next_indexer/indexer_ref.bin")
target_link_libraries(ignis_kernel_flash_next_indexer_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_indexer_test COMMAND ignis_kernel_flash_next_indexer_test)
# GitHub #558 (spec #555): OURS -- the attention's and the indexer's rotations over prompts with
# images (three-axis positions, decode at the rope delta, a block key at its first token's axes
# across calls through its AxisRecord) against the checkpoint's own transformers code, recorded once
# on CPU by fixtures/flash_next_mrope/record_mrope.py. GPU, same no-SKIP_RETURN_CODE rule (ADR 0006).
add_executable(ignis_kernel_flash_next_mrope_test test_flash_next_mrope.cu)
target_include_directories(ignis_kernel_flash_next_mrope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_definitions(ignis_kernel_flash_next_mrope_test PRIVATE
  IGNIS_FLASH_NEXT_MROPE_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/flash_next_mrope/mrope_ref.bin")
target_link_libraries(ignis_kernel_flash_next_mrope_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_flash_next_mrope_test COMMAND ignis_kernel_flash_next_mrope_test)
foreach(name IN ITEMS sparse_attention sparse_attention_hq)
  add_executable(ignis_kernel_flash_next_${name}_test test_flash_next_${name}.cu)
  target_link_libraries(ignis_kernel_flash_next_${name}_test PRIVATE ignis_kernel ignis_vendor)
  add_test(NAME ignis_kernel_flash_next_${name}_test COMMAND ignis_kernel_flash_next_${name}_test)
endforeach()
# Spec flash-next/04 acceptance 7: the hq route against the BF16 route on real Flash-Next KV rows.
target_compile_definitions(ignis_kernel_flash_next_sparse_attention_hq_test PRIVATE
  IGNIS_HQ_KV_FLASH_NEXT_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/hq_kv_rows_flash_next.bin")

# --- the image model's kernels (spec image-model/02, GitHub #319): OURS -----------------------
#
# The tensor-core instruction microbenchmark (acceptance 1): BF16 m16n8k16, FP8 kind::f8f6f4 and
# unit-scale kind::mxf8f6f4 rates on register-resident operands, plus the one-instruction
# semantics and accumulation probes the FP8 GEMM's error bound rests on. A timing is a finding,
# not a pass/fail: a tool, NOT registered with CTest.
add_executable(ignis_mma_peak_bench bench_mma_peak.cu)
target_include_directories(ignis_mma_peak_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_mma_peak_bench PRIVATE CUDA::cudart)
# GitHub #456: its FP8 forms are the sm_120a set's (kernel/CMakeLists.txt), and so are the
# image GEMM sweep's below.
set_target_properties(ignis_mma_peak_bench PROPERTIES CUDA_ARCHITECTURES "${IGNIS_SM120A_ARCHITECTURES}")

# GitHub #341: OURS -- the FP8 arms of the 27B's BF16 exception matrices (kernel/src/fp8_exception.h,
# a leaf-internal header) through the public attn_input_proj / linear_add dispatch. GPU (ADR 0006).
add_executable(ignis_kernel_fp8_exception_arms_test test_fp8_exception_arms.cu)
target_include_directories(ignis_kernel_fp8_exception_arms_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_fp8_exception_arms_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_fp8_exception_arms_test COMMAND ignis_kernel_fp8_exception_arms_test)

# GitHub #341: OURS -- the 27B's nine BF16 exception matrices at prefill width on the vendored
# BF16 routes against the FP8 W8A8 GEMM. A timing is a finding: a tool, NOT registered with CTest.
add_executable(ignis_fp8_exception_bench bench_fp8_exception.cu)
target_link_libraries(ignis_fp8_exception_bench PRIVATE ignis_kernel ignis_vendor)
# The FP8 W8A8 GEMM and the per-token activation quantization (acceptance 2 and 3,
# kernel/include/ignis_image_gemm.h) against our own fp64 references at the DiT's and the text
# encoder's linear shapes. GPU (ADR 0006: no SKIP_RETURN_CODE).
# image_gemm_sweep.cu: the bench sweep's tile configurations, held to the same mechanics check at
# the edge shapes; built here and into the bench, never into the leaf.
add_executable(ignis_kernel_image_gemm_test test_image_gemm.cu image_gemm_sweep.cu)
target_include_directories(ignis_kernel_image_gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_gemm_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_gemm_test COMMAND ignis_kernel_image_gemm_test)
set_target_properties(ignis_kernel_image_gemm_test PROPERTIES CUDA_ARCHITECTURES "${IGNIS_SM120A_ARCHITECTURES}")
# Its speed at the DiT's shapes (acceptance 2 "Speed") and the standalone quantization's cost
# (acceptance 3). A timing is a finding: a tool, NOT registered with CTest.
add_executable(ignis_image_gemm_bench bench_image_gemm.cu image_gemm_sweep.cu)
target_include_directories(ignis_image_gemm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_image_gemm_bench PRIVATE ignis_kernel ignis_vendor CUDA::nvml)
set_target_properties(ignis_image_gemm_bench PROPERTIES CUDA_ARCHITECTURES "${IGNIS_SM120A_ARCHITECTURES}")
# The 8-bit DiT attention (acceptance 4, kernel/include/ignis_image_attn.h; an adapted op): the
# quantizers against their definitions, every call the DiT makes against fp64 on the same
# quantized operands (a tight bound) and on the unquantized inputs (cosine per head). GPU
# (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_attn_test test_image_attn.cu)
target_include_directories(ignis_kernel_image_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_attn_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_attn_test COMMAND ignis_kernel_image_attn_test)
# The same checks on real Q, K and V from a diffusers run (the per-head cosine gated at 0.99). The dump
# lives outside the repo (--qkv-probe DIR, $IGNIS_IMAGE_QKV_PROBE, or the default under
# F:/ai/models): without it the case prints SKIP and passes, unless IGNIS_GPU_PROFILE=1, where a
# missing dump fails it (docs/agents/testing.md: a skip is never green in the profile).
add_test(NAME ignis_kernel_image_attn_qkv_probe_test COMMAND ignis_kernel_image_attn_test --cases qkv-probe)
# Its speed (acceptance 4 "Speed") as TOPS with the SM clock and board power, and the quantizers'
# cost. A timing is a finding: a tool, NOT registered with CTest.
add_executable(ignis_image_attn_bench bench_image_attn.cu)
target_include_directories(ignis_image_attn_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_image_attn_bench PRIVATE ignis_kernel ignis_vendor CUDA::nvml)
# The encoder attention (acceptance 5, kernel/include/ignis_image_encoder_attn.h; an adapted op,
# FlashAttention-2's forward): one causal GQA 32/8 sequence from 1 to 41,000 tokens against fp64
# that replays the kernel's softmax (a tight bound) and against plain fp64 attention. GPU (ADR
# 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_encoder_attn_test test_image_encoder_attn.cu)
target_include_directories(ignis_kernel_image_encoder_attn_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_encoder_attn_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_encoder_attn_test COMMAND ignis_kernel_image_encoder_attn_test)
# Its speed (acceptance 5 "TOPS") with the SM clock and board power, and the encoder's 36 layers.
# A timing is a finding: a tool, NOT registered with CTest.
add_executable(ignis_image_encoder_attn_bench bench_image_encoder_attn.cu)
target_include_directories(ignis_image_encoder_attn_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_image_encoder_attn_bench PRIVATE ignis_kernel ignis_vendor CUDA::nvml)
# The VAE ops (acceptance 6, kernel/include/ignis_image_vae.h): every convolution shape of the VAE
# at a 1024² and a 2048² image (and the 2752 x 1536 preset) against an fp64 conv2d, banded calls
# against the whole image bit for bit, the shortcuts against PyTorch's own outputs (the fixture,
# recorded by fixtures/image_vae/record.py), and the mid-block attention against fp64 that
# replays its softmax. GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_vae_test test_image_vae.cu)
target_include_directories(ignis_kernel_image_vae_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_vae_test PRIVATE ignis_kernel ignis_vendor)
target_compile_definitions(ignis_kernel_image_vae_test PRIVATE
  IGNIS_IMAGE_VAE_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/fixtures/image_vae/shortcuts.bin")
add_test(NAME ignis_kernel_image_vae_test COMMAND ignis_kernel_image_vae_test)
# Its speed (acceptance 6 "Speed"): the convolution's TFLOPS at the decoder's heavy shapes and a
# whole decode's convolutions, the attention's time, with the SM clock and board power. A timing is
# a finding: a tool, NOT registered with CTest.
add_executable(ignis_image_vae_bench bench_image_vae.cu)
target_include_directories(ignis_image_vae_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_image_vae_bench PRIVATE ignis_kernel ignis_vendor CUDA::nvml)
# The elementwise ops (acceptance 7, kernel/include/ignis_image_dit.h and the VAE's channel norm):
# the modulated LayerNorm with its per-token t / t=0 row, the zero-centred RMSNorm, the gated
# residual, the QK RMSNorm with RoPE, SiLU, GELU-tanh, SwiGLU and the VAE norm against fp64
# references that track the kernels' roundings, with the E4M3 outputs against the standalone
# quantization bit for bit; the RoPE positions and table against diffusers' definition restated.
# GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_elementwise_test test_image_elementwise.cu)
target_include_directories(ignis_kernel_image_elementwise_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_elementwise_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_elementwise_test COMMAND ignis_kernel_image_elementwise_test)
# The RoPE table of the oracle's t2i and edit layouts against diffusers' own pos_embed (rope.npy
# in spec 01's recordings). The oracle lives outside the repo (--oracle DIR, $IGNIS_IMAGE_ORACLE,
# or the default under F:/ai/models): without it the case prints SKIP and passes, as the attention's
# qkv-probe case does, unless IGNIS_GPU_PROFILE=1, where it fails.
add_test(NAME ignis_kernel_image_elementwise_rope_oracle_test
         COMMAND ignis_kernel_image_elementwise_test --cases rope-oracle)
# Their speed at a 1024² and a 2048² denoising step and the VAE decoder's stage sizes (GB/s), with
# the fused quantization against the separate one. A timing is a finding: a tool, NOT registered
# with CTest.
add_executable(ignis_image_elementwise_bench bench_image_elementwise.cu)
target_include_directories(ignis_image_elementwise_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_image_elementwise_bench PRIVATE ignis_kernel ignis_vendor)

# --- the image model's text encoder (spec image-model/03, GitHub #320): OURS ----------------------
#
# Its three ops (kernel/include/ignis_image_encoder.h) against fp64 references that track their
# roundings -- the residual add with the RMSNorm and its E4M3 output, the QK RMSNorm with the
# interleaved mrope on three different position rows (with two layouts as negative controls), the
# vision rows -- and the forward on synthetic FP8 weights: the vision hooks against the same
# additions made on the host between one-layer calls, bit for bit. The real model against the
# oracle is the Rust test crates/core/tests/image_encoder_gpu.rs. GPU (ADR 0006: no
# SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_encoder_test test_image_encoder.cu)
target_include_directories(ignis_kernel_image_encoder_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_encoder_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_encoder_test COMMAND ignis_kernel_image_encoder_test)
# The vision tower (kernel/include/ignis_image_vision.h: the 27B's tower program with the image
# model's parameters) on synthetic W8G32 weights: the merger and the three DeepStack mergers against
# fp64 that tracks the BF16 stores (with the norm placements swapped as negative controls), the
# DeepStack taps' blocks, two images in one call against each alone (bit for bit), the refusals.
# The real tower against transformers is the Rust test
# crates/server/tests/image_encoder_edit_gpu.rs. GPU (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_vision_test test_image_vision.cu)
target_include_directories(ignis_kernel_image_vision_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_vision_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_vision_test COMMAND ignis_kernel_image_vision_test)

# --- the image model's diffusion transformer (spec image-model/04, GitHub #321): OURS -------------
#
# The program's ops of its own (kernel/include/ignis_image_transformer.h) against fp64 references
# that track their roundings -- the gated residual with the modulated LayerNorm and its E4M3
# output, the BF16 linear, the fp32 quantization, the timestep embedding's chain -- and the program
# on synthetic FP8 weights: txt_in and img_in per row, the prefix and target passes split by block
# against whole (bit for bit), the target reading the prefix, the Euler update against the host.
# The real model against the oracle is the Rust test crates/core/tests/image_leaf_gpu.rs. GPU
# (ADR 0006: no SKIP_RETURN_CODE).
add_executable(ignis_kernel_image_transformer_test test_image_transformer.cu)
target_include_directories(ignis_kernel_image_transformer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_image_transformer_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_image_transformer_test COMMAND ignis_kernel_image_transformer_test)

# GitHub #339 (spec 27b-16gb/05a): OURS -- the 27B program binds the 16 GB artifact's trellis body.
# The binder through `ignis_model_plan_reservations` on a four-layer topology of the 27B's widths
# (the body by its head, one body never a mix, each code its K, strict shapes, suh/svh where the
# record puts them, the trellis workspace plan line); then the layer bodies' own plumbing
# (kernel/src/trellis_body.h, leaf-internal): the q_proj's per-head [q | gate] split, query_gate,
# the head and the two epilogues. GPU: the op's workspace query needs a prepared device (ADR 0006).
add_executable(ignis_kernel_model_load_trellis_test test_model_load_trellis.cpp)
target_include_directories(ignis_kernel_model_load_trellis_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_model_load_trellis_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_model_load_trellis_test COMMAND ignis_kernel_model_load_trellis_test)
add_executable(ignis_kernel_trellis_body_test test_trellis_body.cu)
target_include_directories(ignis_kernel_trellis_body_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(ignis_kernel_trellis_body_test PRIVATE ignis_kernel ignis_vendor)
add_test(NAME ignis_kernel_trellis_body_test COMMAND ignis_kernel_trellis_body_test)

# GitHub #456: the tests whose kernels only the sm_120a set holds (kernel/CMakeLists.txt). On a
# card below compute capability 12.0 they fail at their first launch, "no kernel image", which is
# the loud failure the set is built for; there run `ctest -LE sm120a-only`. W8 aborts at its
# [2048, 16384] instance (static shared memory past sm_86's 48 KiB), the MoE decode routes test at
# its staged arms, the cluster arms on the cluster kernel, the rest on FP4 W4A4 or FP8 MMA.
set(ignis_sm120a_only_tests
  ignis_kernel_w8_linear_tests
  ignis_kernel_nvfp4_linear_a4_tests
  ignis_kernel_attn_input_proj_tests
  ignis_kernel_nvfp4_linear_add_tests
  ignis_kernel_gdn_input_proj_tests
  ignis_kernel_nvfp4_linear_swiglu_tests
  ignis_kernel_moe_decode_routes_test
  ignis_kernel_moe_slot_trap_decode-clusters_test
  ignis_kernel_moe_slot_trap_decode-id-clusters_test
  ignis_kernel_moe_slot_trap_decode-nan-clusters_test
  ignis_kernel_moe_experts_clusters_test
  ignis_kernel_moe_block_clusters_test
  ignis_kernel_fp8_exception_arms_test
  ignis_kernel_image_gemm_test
  ignis_kernel_image_attn_test
  ignis_kernel_image_encoder_test
  ignis_kernel_image_transformer_test)
set_tests_properties(${ignis_sm120a_only_tests} PROPERTIES LABELS sm120a-only)
