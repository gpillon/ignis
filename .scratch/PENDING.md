# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred. Updated by the coordinator
at each integration step; per-ticket details live in the ticket files under
`.scratch/<feature>/issues/`.

## Stopped subagents — WIP state (2026-09-02)

Three subagents (core, server, artifact) were launched in parallel and then
**stopped by the user** (to switch to the structured `/implement` workflow).
What each left behind:

- **Subagent A (artifact)** — stopped *before* producing any file. No partial
  work in `crates/artifact/` (only pre-existing files; `git status` clean for
  this crate). **Remaining: artifact-02 (frontend extraction, #7),
  artifact-03 (tensor checksum vs sidecars, #8).**
- **Subagent B (server)** — stopped *before* producing any file. No partial
  work in `crates/server/` (only the pre-existing `main.rs` stub; `git status`
  clean for this crate). **Remaining: server-01 (OpenAI HTTP, #14),
  server-02 (JSONL telemetry, #15).**
- **Subagent C (core)** — produced **`crates/core/src/kv.rs`** (core-01:
  paged KV + block table) and stopped there. `kv.rs` is complete + green
  (4 unit tests pass; one test bug was fixed and it was wired into `lib.rs`).
  **core-02 … core-07 were NOT started by this subagent.**

### Coordinator WIP — recovered (2026-09-02, this session)

The coordinator's in-progress files (the uncommitted WIP above) were
recovered through the `/implement` close-out: two-axis code review
(Standards + Spec, parallel subagents) → findings fixed → workspace
`cargo test` green → committed per ticket:

- `core 01: paged KV + block table (GitHub #9)` — 8a64a0d
- `core 02: GDN state (boundary-gated) (GitHub #11)` — 4e0d092
- `core 03: request state machine + basic admission (GitHub #12)` — 5d13202
  (the `request.rs` admission-test failure was fixed: the test pinned the
  physical lane deal-order, which is not part of the contract; it now
  asserts class-priority + FIFO properties, plus a single-lane FIFO
  scenario)
- `core 04 (start): scheduler contract + Compute seam (GitHub #13)` — 0966eef
  (contract only; the concrete N=8 scheduler + the deterministic Compute
  mock are the follow-up)
- `kernel-abi 01-03 (start): C ABI surface (GitHub #5, #6, #10)` — adb6ac9
  (surface only; CUDA implementations + the 99% gate deferred to the GPU,
  ADR 0006/0007)

Review-driven fixes folded into these commits: non-tautological geometry
tests in `ffi.rs` (independent literals), `KvPool::free` returns `bool`
(double-free / out-of-range surfaced, not swallowed), `Request::advance`
enforces the "Running ⇒ holds a lane" invariant, and `Scheduler::submit`
carries the `RequestClass` (ADR 0004).

**Next up:** core-04 remainder — the **concrete N=8 scheduler + the
`MockCompute`** are done and CPU-tested (this session): `ConcreteScheduler`
(N=8 resident lanes, batched prefill in one compute call, batched decode,
class-priority + FIFO lane deal, in-flight cap = N_DECODE_LANES until the
host tier, ADR 0004 basic admission) + a deterministic, recording
`MockCompute` behind the `Compute` seam (ADR 0006), with `DecodeParams`
carried on the prefill/decode jobs. What remains on core-04 is the
**GPU-saturation measurement** of batched prefill (a measure, not a
guarantee — ADR 0007 re-gates it via the 99% gate on the GPU, ADR 0006
exclusive-GPU rule), then core-05 … core-07, then server-01/02, then
artifact-02/03. Only the kernel-abi CUDA implementation + the 99% gate need
the GPU (ADR 0006/0007).

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
   `ignis-bench`. **Partially unblocked (2026-09-02):** the `ignis-bench`
   harness core now exists — trace loader, per-class metrics (tok-s, ttft
   percentiles), canary self-consistency (sane + greedy-deterministic),
   performance report + 99% gate, and a bounded-concurrency replay driver +
   CLI (all tested, workspace `cargo test` green). Still remaining:
   - the `HttpEndpoint` transport is a **stub** — it needs the `ignis-server`
     OpenAI endpoint (ticket #14) + an HTTP client dependency (bench-01, #19).
   - a **recorded reference baseline** (trace JSONL + a reference run) to
     compare against — ADR 0007 gates against the reference's *speed*, so a
     reference recording is required before the gate can be evaluated.

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