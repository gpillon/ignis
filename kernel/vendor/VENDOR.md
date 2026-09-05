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

The remaining op families (NVFP4 / BF16 / W8G32 linear, the fused
projections, norms, attention, GDN) arrive with P1-07..P1-15, each adding its
files to this manifest and its reference op test to the leaf's test suite.

## Updating to a newer reference commit

1. move the reference checkout to the new commit;
2. `scripts/vendor-ninfer.ps1 repin` and set `reference.commit`;
3. `scripts/vendor-ninfer.ps1 sync` (re-apply and re-record any patch);
4. `kernel/build.ps1 -Test` — the leaf's op tests are the acceptance.
