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

**Next up:** bench-02 (recorded reference baseline + 99% gate, GPU)
once a reference recording exists.

**Follow-up (tracked, pending on core):** server-02's interval line
stubs `prefilling` / `kv_used_pct` to 0 — the live counters are not on
core's public API (the `Scheduler` trait exposes no stats accessor, so
the engine's `Box<dyn Scheduler>` cannot reach `ConcreteScheduler`'s
`kv_used_pages` / `host_tier` / request-state counts). Once core ships
a `Scheduler::stats()` accessor (`waiting` / `prefilling` / `running` =
counts by `RequestState`; `kv_used_pct` = `kv_used_pages / capacity *
100`; `kv_evictions` = a cumulative counter at the evict/restore
paths), the server's ready-made `IntervalStatsProvider` seam wires the
live counters and the interval line becomes authoritative. A core
follow-up (the user's territory) — the server side is already wired
(`Engine::with_stats` + `IntervalStatsProvider`).

**Follow-up (tracked, uncommitted):** making a sidecar checksum mismatch
actually **fail the artifact load** — artifact-03's `checksum.rs`
(18043b3) delivers an inspectable `ChecksumReport` (per-object
`matched` / `mismatched` / `missing` + `flagged` / `global_flags`), but
nothing in the loader path calls `verify()` / `ChecksumReport::is_clean()`
yet. The call site needs both the `Reader` and the sidecar in the
engine/loader path (`crates/server` territory) — the natural next step
for the server actor, after server-02.

server-02 is **resolved** (2026-09-02, GitHub #15 closed): the JSONL
telemetry is committed (992aea0) — a `Telemetry` emitter in
`crates/server` (`telemetry.rs`) with injectable sink (null / memory /
stdout / file via `IGNIS_TELEMETRY`) + injectable clock (a `FixedClock`
keeps tests deterministic, ADR 0006), wired into the engine's
`submit()` / `step()`: the `interval` line once per `Engine::step()`
(driver tick), `request:admitted` on `Admitted`, `request:ttft` on the
request's first `Token`, `request:done` on `Done` (`n` = total tokens,
`tok_s` = n / elapsed_ms), `kv_evictions` bumped on `Evicted`; the sink
is a lock-protected buffer (non-blocking on the request path). 9 new
unit + 4 integration tests (line shapes, clock determinism, sink
injection, no-deadlock under 4-thread concurrent `submit`/`step`);
38/38 `ignis-server` green, clippy clean. `id` is the numeric
`RequestId` (the §5 example's string id is not in the ticket); request
lines uniformly carry `ms` / `n` / `tok_s` (the ticket's field set
wins over the §5 example). `class` / `sibling_prefix_reused_tok`
reserved for v1.1.

artifact-03 is **resolved** (2026-09-02, GitHub #8 closed): the tensor
checksum validation is committed (18043b3) — `Sidecar::load` (shared
fields of the `graft.json` / `conversion.json` shapes; an absent
`grafted_from` block is tolerated for the conversion shape) +
`verify(reader, sidecar) -> ChecksumReport` (global invariants +
per-parent checks; never panics, `is_clean()` is the load-failure
surface). Key finding: the v2 sidecars carry **no per-tensor digests**
(the contract requires none) — the per-tensor datum is the NVFP4
`local_nvfp4.parents` table (`weight_scale_divisor` float +
`relative_frobenius_error`), plus whole-file invariants
(`artifact.bytes`, `objects.count`); "checksum match" = file size +
object count hold, and each recorded parent's FP32 divisor word in the
container value-matches the sidecar's recorded number (compared as
promoted `f64` — no narrowing ULP shift). Gated real-artifact run: the
19.4 GB `qwen3_8_27b_nvfp4full-v2` container verifies clean (1,325
objects, 19,406,942,468 bytes, 34 parents all `divisor: null`, 281
NVFP4 tensors — 34 covered by the graft, 247 inherited). 45 lib + 4
integration `ignis-artifact` tests green.

bench-01 is **resolved** (2026-09-02, GitHub #19 closed): the
`HttpEndpoint` transport is committed (968a2c1) — a `reqwest` 0.21
blocking client (one shared `Client`, `Send + Sync` for the existing
`Endpoint` seam) driving `POST /v1/chat/completions` (SSE per-token
timing for streaming trace lines; non-streaming bodies where ttft ==
total) + a `GET /v1/models` readiness probe, with the "1 main + N
subagents" load reusing the existing bounded-concurrency `replay`
driver. CPU-only, in-process tests (axum mock engine on a random local
port): 37/37 `ignis-bench` green, asserting per-request ttft > 0 /
tok_s > 0 and the driver's concurrency bound. No recorded trace existed
in the repo, so the fixture is a documented synthetic
`main_plus_10.jsonl` (realistic load shape, **not** a reference
recording — the ADR 0007 99% gate still needs a recorded reference
run; that is bench-02, GPU-driven).
core-04 is **resolved** (2026-09-02, GitHub #13 closed): the concrete N=8
scheduler + `MockCompute` are committed (32cb738) and CPU-tested. What was
*deferred* on core-04 — the **GPU-saturation measurement** of batched
prefill (a measure, not a guarantee — ADR 0007 re-gates it via the 99% gate
on the GPU, ADR 0006 exclusive-GPU rule) — now sits with the bench harness
(bench-01/02, GitHub #19/#20); it runs once the harness `HttpEndpoint` +
a reference baseline are in place. Only the kernel-abi CUDA implementation
+ the 99% gate need the GPU (ADR 0006/0007).

core-05 is **resolved** (2026-09-02, GitHub #16 closed): the full
admission state machine (ADR 0004) is committed (e582621) — `admission.rs`
(pure Rust port of the reference policy: protection freeze, donor-prefix
selection, persistent-vs-temporal backfill, temporal-credit decay,
frontier distance, retained-lane victim policy) wired into
`ConcreteScheduler` as the lane-deal driver, plus the KV resource
dimension (per-request page reservation, over-reservation charged at
deal, `Oversized` rejection, hard-cap completion). 11 unit tests +
4 end-to-end scenarios (`admission_machine.rs`), CPU-tested (ADR 0006),
workspace `cargo test` green. The protection's **Drain** phase is
unreachable in v1's resource model (documented in the ticket +
`admission_machine.rs`); it ships for reference fidelity only.

core-06 is **resolved** (2026-09-02, GitHub #17 closed): the KV-RAM
host tier is committed (7115251) — a bounded `HostTier` in `host.rs`
(probation → protected two-tier eviction, GDN-boundary check, LRU
eviction) snapshots a lower-value GPU lane to host RAM so the evicted
(suspended) request can later *restore* instead of re-prefilling; the
`Running→Evicted→Running` (restore) and `Evicted→Admitted` (re-queue)
transitions in `request.rs` reset the re-queued request's GDN state so a
re-prefilled stream cannot accept a snapshot at a position it never
reached; `gdn.rs` `checkpoint` now records at the current position (>=) +
a new `advance` (mid-prefill progress, non-boundary); `concrete.rs`
wiring (`host_capacity_pages` knob, `try_evict_for_head`, `restore_pass`,
decode-phase GDN checkpoints, KV bookkeeping); `types.rs` `Evicted` /
`Restored` / `Requeued` schedule events. 8 unit tests + 2 end-to-end
scenarios (`host_tier.rs`), CPU-tested (ADR 0006), workspace `cargo test`
green. Admission tests pin `host_capacity_pages: 0` so the core-05
scenarios exercise the admission machine in isolation.

server-01 is **resolved** (2026-09-02, GitHub #14 closed): the
OpenAI-compatible HTTP surface is committed (84ada6d) — axum 0.8 +
tokio; `GET /v1/models`, `POST /v1/chat/completions` (SSE + non-
streaming), `POST /v1/responses` (non-streaming; `stream:true` → 400
in v1); the `Engine` drives the core `Scheduler` behind a `Mutex` with
per-request `SchedEvent` routing (review-caught bug fixed + regression-
tested: a `Protected` batch event early-returned, dropping later
`Token`/`Done` events). The chat template runs through the
`TemplateProvider` seam + built-in provider; the real artifact frontend
objects (tokenizer + chat template) arrive with artifact-02 (#7),
wired through the same seam. 21/21 `ignis-server` CPU tests (ADR 0006),
workspace `cargo test` green. Deferred: `/v1/responses` streaming
(out of v1 scope), `Compute` backend is `MockCompute` (kernel-leaf
adapter via the same scheduler-constructor injection).

artifact-02 is **resolved** (2026-09-02, GitHub #7 closed): the frontend
extraction is committed (d08759d) — the `frontend` module
(`FrontendSet::from_reader` loads all 6 frontend resources, missing or
ambiguous = load failure, ADR 0002; typed `Tokenizer` via HuggingFace
`tokenizers` 0.21; `ChatTemplate` via minijinja 2.24 + `json` with the
Qwen3.8-specific extensions registered: `raise_exception`, string
`.startswith`/`.endswith`). 39 unit tests + real-artifact verification
(`real_frontend.rs`, gated, CPU-only): all 6 resources present in the
19 GB `qwen3_8_27b_nvfp4full-v2` container, real BPE tokenizer
round-trips, the real Qwen3.8 template compiles + renders, and its
`raise` path fires end-to-end. 42/42 `ignis-artifact` tests green.

**Follow-up (resolved 2026-09-02, commit 217a0bd):** the last wiring
step of server-01's template seam is in — the `FrontendSet`-backed
`ArtifactTemplateProvider` in `crates/server` (`artifact_template.rs`):
`apply_chat_template` = the container's chat-template `render` + the
container's tokenizer `encode`, `render_tokens` = `decode`; a role that
does not parse to an artifact `Role` templates as `user` (infallible by
design — the request completes, logged, not panicked). `Server::with_artifact_template`
takes the `FrontendSet`; the entrypoint reads `IGNIS_ARTIFACT` and falls
back to the built-in placeholder when unset/unreadable (rendered
`content` is the token id-space, not natural text). 17 unit + 8
integration `ignis-server` tests green (3 new fixture-based
`artifact_template` tests: template+tokenizer determinism, tokenizer
decode round-trip, unknown-role fallback); the fixture's
`tokenizer.json` follows the `tokenizers` 0.21 schema (`unk_token`
lives inside the `model` object, not top-level).

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
   - the `HttpEndpoint` transport is a **stub** — the `ignis-server`
     OpenAI endpoint now exists (server-01, 84ada6d); what remains is
     wiring the bench `HttpEndpoint` against it + an HTTP client
     dependency (bench-01, #19).
   - a **recorded reference baseline** (trace JSONL + a reference run) to
     compare against — ADR 0007 gates against the reference's *speed*, so a
     reference recording is required before the gate can be evaluated.

## Hygiene / deferred

- `~/.bash_profile` PATH additions (cargo bin, CUDA 13.1 bin + bin/x64, ninja
  at `F:/ai/q38/tools/ninja`) were handed to the user 2026-09-02; they
  applied it.

## Known-stale local state (fixed)

- `kernel-port/issues/02-artifact-reader.md` status backfilled 2026-09-02
  (reader done, commit f941ef3; GitHub #2 closed).