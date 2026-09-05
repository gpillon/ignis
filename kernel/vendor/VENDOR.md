# The vendored reference subtree

Everything under `kernel/vendor/` except this file is a **verbatim copy** of a
file from the reference engine (ninfer) at one pinned commit. Nothing here is
written or edited by hand. That is the whole point of ADR 0010: a port claim
must be diffable, so "we ported the reference's kernel" means *this exact file
is the reference's file*, and a script proves it.

- Policy: `docs/adr/0010-vendored-reference-kernels.md`
- Attribution: `kernel/NOTICE` (the subtree is Apache-2.0; `LICENSE` is
  vendored alongside the code)
- Spec: `.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36)

## The manifest

`kernel/vendor/manifest.json` is the source of truth for what is vendored.

```json
{
  "reference": { "repo": "…", "branch": "…", "commit": "…", "default_path": "…" },
  "vendor_root": "kernel/vendor",
  "files": [
    { "path": "src/core/arena.h", "sha256": "…" },
    { "path": "src/core/arena.cu", "sha256": "…",
      "patch": { "diff": "patches/src/core/arena.cu.diff", "sha256": "…", "reason": "…" } }
  ]
}
```

- `reference.commit` is the **only** revision the hashes describe. Vendoring
  from any other checkout state is a hash mismatch, not a merge.
- `path` is shared by the reference and the leaf: `src/core/arena.h` in the
  reference is `kernel/vendor/src/core/arena.h` here. Vendored files keep the
  reference's namespaces and include paths (`#include "core/arena.h"`,
  `#include "ops/common/math.cuh"`), so a `diff` against the source is trivial
  and a reference update is a re-run of the script — never a re-port.
- `sha256` is the file's content hash **in the reference**. A vendored file
  must be byte-identical to it unless the entry records a `patch`.
- `.gitattributes` marks the subtree `-text` so git never rewrites the line
  endings the hashes describe.

## The script

`scripts/vendor-ninfer.ps1` wraps the `vendor-ninfer` binary of
`crates/vendor`. The logic lives in a Rust crate rather than in the script so
that `cargo test` covers it (`docs/agents/testing.md`).

| Command | What it does |
|---|---|
| `verify` | Every listed file exists under `kernel/vendor/` and hashes to what the manifest expects. With a reference checkout present, also checks the reference still carries the pinned content. **This is the byte-identical check.** |
| `sync` | Copies every listed file from the reference into the leaf. Verifies the reference against the pinned hashes *first* and copies nothing on a mismatch, so a wrong checkout cannot half-overwrite the subtree. Files with a recorded patch are kept unless `--force-patched`. |
| `repin` | Recomputes the hashes from the checkout on disk. Used when moving to a new reference commit — set `reference.commit` in the same change. |
| `record-patch PATH --reason TEXT` | Writes the local edit out as a diff under `patches/` and records its patched hash. |

Options: `--reference PATH` (defaults to `reference.default_path`, or
`$IGNIS_NINFER_REFERENCE`), `--manifest PATH`. Exit codes: `0` clean, `1` a
verification finding, `2` a usage or I/O error. `verify` works without a
reference checkout, so it runs from a bare clone.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/vendor-ninfer.ps1 verify
```

## The patch policy

A vendored file may not be edited. When the leaf genuinely needs a local
change, it is **recorded**, never silently applied:

1. edit the file under `kernel/vendor/`;
2. `scripts/vendor-ninfer.ps1 record-patch <path> --reason "<why>"` — this
   writes `patches/<path>.diff` and sets the entry's `patch.sha256` to the
   patched content's hash;
3. commit the diff with the file.

From then on `verify` expects the *patched* hash locally and the *reference's*
hash upstream, so both an unrecorded edit and upstream drift are caught. `sync`
will not overwrite a patched file unless it is told to.

Anything that is not a verbatim vendored file — including a file we edit
beyond a recorded patch — is our own implementation and carries no provenance
claim (ADR 0010). The leaf's own code lives in `kernel/src/`,
`kernel/include/` and `kernel/tests/`, never here.

## What is vendored today

P1-06 (GitHub #42) vendors the substrate the op families are built on:

- **core** — `dtype`, `tensor`, `arena`, `device`, `layout`, the weight
  descriptor, the PDL helper and the NVTX headers. Built as the static library
  `ignis_vendor` by `kernel/CMakeLists.txt`, with `kernel/vendor/src` as the
  include root.
- **ops common** — all of the reference's `ops/common`: `math`, `memory`,
  `mma`, `rowsplit_mma`, `rowsplit_grouped_mma`, `warp`, `bf16_vector`,
  `sampling_workspace`, `token_slices`. `rowsplit_grouped_mma.cuh` includes
  the q4/q5 rowsplit storage headers, so those two are vendored with it — the
  substrate has no dangling include even though this model uses neither
  quantization.
- **the op-test harness** — `tests/ops/op_check.h` and `tests/ops/op_tester.h`,
  the reference's error-record convention, used by
  `kernel/tests/ignis_kernel_op_tests` (CTest). `kernel/vendor/tests` is that
  executable's second include root, because the harness includes itself as
  `"ops/op_tester.h"`.

P1-07 (GitHub #43) adds the norm and glue op family — the leaf's first vendored
op family with a public API:

- **norms + glue** — `rmsnorm` (plain and gated, one kernel/launcher/wrapper
  file each), `l2norm`, `residual_add`, `silu_mul`, `sigmoid_mul`: each op's
  kernel (`src/ops/kernel`), launcher (`src/ops/launcher`), wrapper
  (`src/ops/wrapper`) and public header (`include/ninfer/ops`, the leaf's
  first vendored include root — `kernel/vendor/include`). The wrapper and
  test `.cpp` files are plain host C++ that declare a `cudaStream_t` and (in
  the tests) call the CUDA runtime API directly, so `kernel/CMakeLists.txt`
  now depends on `CUDAToolkit` for its include/link paths — the `.cu` files
  never needed this because nvcc supplies them itself.
- **their reference tests** — `tests/ops/test_rmsnorm.cpp`,
  `test_gated_rmsnorm.cpp`, `test_l2norm.cpp`, `test_residual_add.cpp`,
  `test_silu_mul.cpp`, `test_sigmoid_mul.cpp` and the shared
  `tests/ops/norm_test_common.h`, each built as its own CTest executable
  (`ignis_<op>_test`) since a reference test file brings its own `main()` and
  cannot share a binary with another. Cases already exercise real 27B widths
  (hidden 5120, GDN norm 128/head, MLP intermediate 17408).

P1-08 (GitHub #44) adds the second op family: **embedding** (dense BF16,
Q6G64_F16S, W8G32_F16S, FP8_E4M3FN_ROW_BF16S gather —
`ops/kernel/embed_gather.cuh`, `ops/launcher/embed_gather.{h,cu}`,
`ops/wrapper/embedding.cpp`, its `ninfer/ops/embedding.h` public header, and
the FP8 geometry validator `ops/linear/fp8/fp8_format.{h,cpp}` the wrapper
dispatches through) and **argmax** (`ops/kernel/argmax.cuh`,
`ops/launcher/argmax.{h,cu}`, `ops/wrapper/argmax.cpp`, `ninfer/ops/argmax.h`),
each with its reference op test (`tests/ops/test_embedding.cpp`,
`tests/ops/test_argmax.cpp`) built by the same `ignis_<op>_test` CTest loop as
P1-07 (`ignis_embedding_test`, `ignis_argmax_test`) — without the reference's
`SKIP_RETURN_CODE 77` (ADR 0006: a missing GPU fails here, never skips).

P1-09 (GitHub #45) vendors the NVFP4 linear op family: `nvfp4_codec.cuh`
(E2M1/E4M3 decode), `nvfp4_config.h` (geometries, schedules, the registered
`[N,K]` problems), `nvfp4_format.{h,cpp}` (the `blockscale-k16-m128x4-v1`
weight-layout validator), `nvfp4_dispatch.{h,cpp}` (A16 vs. W4A4 route
selection), `nvfp4_gemv.{cuh,cu}` / `nvfp4_small_t.{cuh,cu}` /
`nvfp4_launch.h` / `nvfp4_output.cuh` (the decode and small-T A16 kernels and
their launchers), and the large-T W4A4 route in full —
`nvfp4_w4a4_plan.h`, `nvfp4_w4a4_mma.cuh` / `nvfp4_w4a4.cu` (the tensor-core
MMA route) and `nvfp4_w4a4_tma.{cuh,cu}` / `nvfp4_w4a4_tma_launch.h` (the
warp-specialized SM120 TMA route, which needs the CUDA driver API directly for
`cuTensorMapEncodeTiled`; `ignis_vendor` links `CUDA::cuda_driver` for it). Two
small epilogue/output headers from *other* op families are vendored alongside
it because `nvfp4_w4a4_tma.cu` is the shared W4A4 GEMM kernel those families
reuse: `ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh` and
`ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh` — both self-contained
structs over the already-vendored `ops/common/memory.cuh`, with no other
dependency. The public header `include/ninfer/ops/linear.h` is vendored too,
under a new `kernel/vendor/include` root: unlike the reference's
`src/ops/linear/linear.cpp` (not vendored — it dispatches every weight qtype
at once, pulling in op families this ticket does not vendor), the header
itself has no such dependency, so it is byte-identical here and the leaf's own
thin dispatcher (`kernel/src/linear.cu`, not vendored) implements it, routing
`QType::NVFP4` to the vendored `detail::nvfp4_dispatch` and throwing a clear
error naming P1-10 (#46) for every other qtype. The reference's own NVFP4 a16
linear test — `tests/ops/linear/test_nvfp4_a16.cpp` plus its shared harness
`linear_test_common.{h,cpp}` and the packed-weight fixture generator
`tests/ops/quantized_weight.h` — is vendored and built as the CTest target
`ignis_kernel_nvfp4_linear_tests` (its own executable: the reference's test
file brings its own `main()`, so it cannot be another source in
`ignis_kernel_op_tests`). The large-T W4A4/TMA sources compile as part of
`ignis_vendor` but are not exercised by this test; their own reference test
(`test_nvfp4_a4.cpp`) is out of scope until G2.

P1-10 (GitHub #46) vendors the BF16 and W8G32 linear families:

- **BF16 linear** — `src/ops/linear/bf16/` in full (GEMV, small-T, MMA) plus
  the public `include/ninfer/ops/linear.h` (`LinearPolicy`,
  `linear_workspace_capacity_bytes`, the two `linear()` overloads) that every
  linear family's dispatch header depends on.
- **W8G32 linear** — `src/ops/linear/w8/` in full (GEMV, small-T, MMA and
  split-K variants, the rowsplit storage/output decoders): `w8_dispatch`'s
  launch table spans geometries beyond this model's own (it is the
  reference's one dispatcher for the whole W8 family), so the family is
  vendored whole rather than picking out only the shapes G1 needs.
- **their reference op tests** — `tests/ops/linear/test_bf16_a16.cpp` +
  `tests/ops/direct_bf16_weight.h`, and `tests/ops/linear/test_w8_a16.cpp` +
  `tests/ops/linear/linear_test_common.{h,cpp}` + `tests/ops/quantized_weight.h`,
  each its own CTest executable (`ignis_kernel_op_test_bf16_linear`,
  `ignis_kernel_op_test_w8_linear`) — a vendored test file owns a `main()`, so
  it cannot be appended as another source of `ignis_kernel_op_tests`. Neither
  sets `SKIP_RETURN_CODE 77` (the reference's own CMake does, for CPU-only
  machines): under ADR 0006 a missing/busy GPU must fail the run, not skip it,
  so the byte-identical vendored `main()`'s `return 77` surfaces to CTest as a
  plain nonzero exit.
- The top-level `ops::linear` / `ops::linear_workspace_capacity_bytes`
  dispatch across qtypes is **ours**, not vendored (ADR 0010: it is glue that
  composes per-family dispatchers, not an op) — `kernel/src/ops/linear.cu`.
  It currently switches on `BF16_CTRL` and `W8G32_F16S`; `NVFP4` (P1-09/#45)
  extends the same switch.

P1-16 (GitHub #52) adds the sequence-state pools:

- **paged KV pool** — `core/paged_kv_cache.{h,cpp}`: pages, entitlements,
  block-table rows, selective zeroing, multi-pool reserve/resize bundles.
- **linear-attention state pool** — `core/linear_attention_state.{h,cpp}`:
  per-slot recurrent fp32 state + BF16/FP32 conv taps, copy/zero/pack/unpack.
- **ring bits** — `core/kv_ring_bits.{h,cu}`: the hq-e8-2b residual-ring
  validity-word helper, vendored with its public header even though no
  current op enables the feature (kept dangling-include-free like q4/q5
  above).
- their reference tests, `tests/test_kv_cache.cpp` and
  `tests/test_state_store.cpp`, each its own CTest executable
  (`ignis_vendor_kv_cache_test`, `ignis_vendor_state_store_test`) since each
  is its own `main()`.
- the leaf's own `ignis_paged_kv_page_budget` (`kernel/include/ignis_paged_kv_budget.h`,
  `kernel/src/paged_kv_budget.cu`) — not vendored, ours: reports page count /
  bytes for a VRAM budget and plane geometry, routed through
  `plan_paged_kv_pool` so it can't drift from what the pool itself allocates.

The remaining op families (the fused projections, attention, GDN) arrive with
P1-11..P1-15, each adding its files to this manifest and its reference op
test to the leaf's test suite.

## Updating to a newer reference commit

1. move the reference checkout to the new commit;
2. `scripts/vendor-ninfer.ps1 repin` and set `reference.commit`;
3. `scripts/vendor-ninfer.ps1 sync` (re-apply and re-record any patch);
4. `kernel/build.ps1 -Test` — the leaf's op tests are the acceptance.
