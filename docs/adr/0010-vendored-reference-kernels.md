# ADR 0010 — Kernel policy: vendor the reference ops verbatim, under a manifest

## Status

Accepted (2026-09-05, project review). **Clarifies ADR 0005** — it makes
"port the proven CUDA for now, re-implement later" operational, and defines
what a port claim means.

Sources: `.scratch/REVIEW-2026-09-05.md` §3.2, §4.2, §5.4; spec
`.scratch/runtime/specs/01-device-resident-forward.md` (GitHub #36).

## Context

ADR 0005 says v1 uses "the proven ported CUDA kernels as a temporary starting
point," to be re-implemented later. What was actually shipped is neither a
port nor a re-implementation: hand-written scalar stand-ins carrying headers
that claim "faithful adaptation … ported 1:1" — where the ported part is the
E2M1 lookup table. Measured against the reference's equivalents (review §3.2):

| Op | ignis | reference | Gap |
|---|---|---|---|
| NVFP4 GEMV (decode) | one block per row, scalar E2M1/E4M3 decode, byte loads | warp-MMA rowsplit, vectorized tiles, shared-staged, PDL | ~10–30× |
| NVFP4 GEMM (prefill) | 16×16 tile, fp32 smem, scalar FMA | W4A4 tensor-core MMA + TMA, 52–59% of FP4 peak | ~50–100× |
| GQA decode attention | one block per head, `__syncthreads` block reduce **per key** | tensor-core split-KV, online softmax, at roofline @260k | ~100× at long context |
| GDN step | one thread per column, 1024+ threads, smem reduce | warp-tiled, register state, shuffles, `__launch_bounds__` | ~10×, plus an invalid launch at 27B geometry |

The stand-ins were also *wrong*, not merely slow: the storage-layout decoders
went with the kernels, so the scale plane and the weight divisor were the
first things lost. A reviewer trusting the provenance headers would not have
opened the reference's file.

Two ways forward:

- **(A) Vendor the reference ops verbatim** — the kernel, its launcher, its
  public wrapper and header, its storage-layout decoders, and its own op test,
  copied unchanged. Apache-2.0; `NOTICE` attribution already in the repo.
  The "the reference's dispatch layer is C++ state the ABI forbids" objection
  that motivated the rewrites was a misunderstanding of ADR 0001: dispatch
  stays *inside* the leaf, only the step crosses the boundary (ADR 0009).
- **(B) Write our own tensor-core kernels first** — months per op family to
  reach the reference's numbers (its W4A4 TMA GEMM alone is long-tuned). And
  where the reference is already at roofline (BF16 GEMV at 90–92% of
  sustained HBM read, int8 decode attention, GDN chunked prefill) there is
  nothing to win.

## Decision

**(A) now; (B) later, per family, gated by measurement.**

- **Every op in the program is a verbatim vendored copy of the reference's
  op** — kernel, launcher, wrapper, header, storage-layout decoder, and the
  reference's own op test — never a scalar re-implementation.
- Vendored files **keep the reference's namespaces, include paths and tensor
  conventions** (the feature-major convention: the token axis second), so a
  `diff` against the source stays trivial and a reference update is a re-run
  of the script.
- **Provenance is a manifest, not a header comment.** A copy-and-verify
  script maintains: the reference's pinned commit, the list of vendored files
  with content hashes, and any local patch recorded as a diff. A file whose
  hash does not match either the source or its recorded patch fails
  verification. `NOTICE` attribution is kept.
- **A port claim must be diffable.** Anything that is not a verbatim vendored
  file — including anything we edit beyond a recorded patch — is labelled our
  own implementation and carries no provenance claim.
- **The program layer is ours** (ADR 0009) and is *not* vendored: the
  per-layer op sequence, the sequence-state wiring, the prefill/decode loops
  and the step ABI.
- **Each vendored op brings its reference test** into a kernel-leaf test
  executable (CTest), run at real 27B geometry against the reference's own
  fp64 references and tolerances. An op ticket closes only with that test
  green on the GPU.
- **Our own kernels come after G4**, one op family at a time, each justified
  by a measurement and re-gated by the performance gate (ADR 0005 / 0007).
  Families already at roofline in the reference are not candidates.

## Consequences

- ADR 0005's kernel policy is unchanged in intent and now has a definition:
  "port" means *vendored verbatim under the manifest*.
- The scalar kernels (`nvfp4_gemm_*.cuh`, `gqa_attention_*.cuh`,
  `gdn_step.cuh`) and their host-pointer surfaces are deleted, not fixed
  (GitHub #39). Their f32 CPU references survive only where a vendored op
  test reuses them for tiny shapes.
- The kernel leaf grows a vendored subtree with the reference's core (dtype,
  tensor, arena, device, layout, weight descriptor, PDL, nvtx) and ops
  common headers underneath the op families — a larger leaf, built by the
  same CMake + Ninja + nvcc flow.
- Ops the current milestone does not exercise (W4A4 large-T, int8 / hq
  attention routes, GDN chunked prefill, MTP, DFlash2) are vendored and
  compiled but gated to their later phase; they do not drift because they
  are never hand-edited.
- A reference update is a mechanical re-run of the copy-and-verify script
  plus re-running the vendored op tests, instead of a re-port.
- Licensing: the vendored ops are Apache-2.0; attribution stays in
  `kernel/NOTICE`, and the manifest records exactly which files it covers.
