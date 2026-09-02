# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred. Updated by the coordinator
at each integration step; per-ticket details live in the ticket files under
`.scratch/<feature>/issues/`.

## GPU-verification items (ADR 0006: exclusive GPU testing)

On 2026-09-02 the user freed the RTX 5090 (stopping the reference
`ninfer-serve`, which had been holding ~30/32 GB at 98% util) and items 1–4
were re-run: all green. Item 5 remains blocked on the `ignis-bench` harness.
Per ADR 0006, GPU workloads only run while the GPU is free.

1. **Workspace `cargo test` (full)** — the FFI smoke tests in `crates/core`
   (`hello_smoke`, `vector_sum_smoke`) launch CUDA on the 5090.
   **Verified 2026-09-02 (free GPU)**: pass.
2. **Ticket #3 (kernel-port 03) — decode C ABI GPU launch tests**:
   `cargo test -p ignis-core --test decode_gpu -- --ignored` (NVFP4 GEMM +
   GQA attention decode with synthetic inputs, CPU-reference comparison).
   **Verified 2026-09-02**: both launch tests pass (2/2, not skipped).
3. **Ticket #4 (artifact 01) — `CudaDevice` VRAM materialization**:
   `IGNIS_TEST_CUDA=1 cargo test -p ignis-artifact --features cuda --
   real_nvfp4full_cuda_device` (bind + materialize all 1319 tensors to real
   VRAM, verify via d2h copy-back). **Verified 2026-09-02** (~19 GB H2D,
   11.6 s).
   - The test is gated by the `IGNIS_TEST_CUDA` env var, **not** `#[ignore]`;
     the earlier `-- --ignored` flag in this ledger was stale (it filtered
     out every test, running zero).
4. **Ticket #4 — full `CpuDevice` materialization of the 19 GB artifact**:
   `IGNIS_TEST_FULL_MATERIALIZE=1 cargo test -p ignis-artifact --
   real_nvfp4full_full_cpu_materialization` (multi-GB host alloc +
   whole-file read; CPU, no GPU needed). **Verified 2026-09-02** (13.7 s).
5. **Ticket #3 — 99% performance gate acceptance (ADR 0007)**: single-step
   decode logits within 99% of the reference on the canary suite, driven by
   `ignis-bench`. **Still blocked**: the GPU is no longer the blocker (free
   since 2026-09-02); the blocker is the `ignis-bench` harness, which is
   still a stub (bench-01/02, #19/#20 — not yet implemented).

## Hygiene / deferred (user's deliberate "zozzata" — do NOT fix unilaterally)

- The local `.git/config` remote URL has a token embedded (user's deliberate
  choice, local-only, never pushed). Cleanup deferred; revisit with the user
  before any push-related work.
- `~/.bash_profile` PATH additions (cargo bin, CUDA 13.1 bin + bin/x64, ninja
  at `F:/ai/q38/tools/ninja`) were handed to the user 2026-09-02; whether they
  applied is unconfirmed. Check once the user confirms their shell setup.

## Known-stale local state (fixed)

- `kernel-port/issues/02-artifact-reader.md` status backfilled 2026-09-02
  (reader done, commit f941ef3; GitHub #2 closed).