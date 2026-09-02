# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred. Updated by the coordinator
at each integration step; per-ticket details live in the ticket files under
`.scratch/<feature>/issues/`.

## GPU-verification items (ADR 0006: exclusive GPU testing)

The RTX 5090 (32 GB) is occupied by the reference `ninfer-serve`
(`F:\ai\q38\ninfer\build-ninja\apps\ninfer-serve.exe`, ~30/32 GB in use,
98% util, observed 2026-09-02). Per ADR 0006 we do not run GPU workloads
while it runs. Once the user frees the GPU (stop `ninfer-serve`), re-run:

1. **Workspace `cargo test` (full)** — the FFI smoke tests in `crates/core`
   (`hello_smoke`, `vector_sum_smoke`) launch CUDA on the 5090. Last green
   with a free GPU (ticket 01). Re-run after the GPU is free.
2. **Ticket #3 (kernel-port 03) — decode C ABI GPU launch tests**:
   `cargo test -p ignis-core --test decode_gpu -- --ignored` (NVFP4 GEMM +
   GQA attention decode with synthetic inputs, CPU-reference comparison).
3. **Ticket #4 (artifact 01) — `CudaDevice` VRAM materialization**:
   `cargo test -p ignis-artifact --features cuda -- --ignored` + set
   `IGNIS_TEST_CUDA=1` (bind + materialize a tensor to real VRAM, verify via
   d2h copy-back).
4. **Ticket #4 — full `CpuDevice` materialization of the 19 GB artifact**:
   `IGNIS_TEST_FULL_MATERIALIZE=1 cargo test -p ignis-artifact` (multi-GB
   host alloc + whole-file read; CPU, no GPU needed — can run any time).
5. **Ticket #3 — 99% performance gate acceptance (ADR 0007)**: single-step
   decode logits within 99% of the reference on the canary suite, driven by
   `ignis-bench`. Blocked on the bench harness (bench-01/02, #19/#20) and a
   free GPU. Tracked in `.scratch/kernel-port/issues/03-c-abi-surface.md`.

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