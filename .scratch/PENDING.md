# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred or blocked on an external
dependency. Per-ticket details live in `.scratch/<feature>/specs/`; status and
blocking live on GitHub (AGENTS.md). This ledger records *capabilities that do
not work yet*, never tickets that closed.

Reset 2026-09-05 by the project review (`.scratch/REVIEW-2026-09-05.md`): the
engine has never produced a real completion, and the work that claimed to have
delivered one is superseded. The phase / gate plan is `.scratch/ROADMAP.md`.

## Open

- **Two `openai_http_gpu.rs` tests looked noticeably slower than the other
  two in a 2026-09-07 serialized GPU profile run (post-#75) — not yet
  investigated.** With `--test-threads=1` (#75) the whole 4-test file took
  41.72s; `a_non_streaming_completion_returns_coherent_text_with_finish_reason_and_usage`
  and `a_streaming_completion_emits_token_deltas_then_a_finish_reason_chunk`
  stood out against `a_streaming_completions_first_chunk_arrives_before_generation_completes`
  and `a_thinking_disabled_request_returns_a_real_answer_with_no_reasoning`
  (per-test timing isn't broken out by the harness, so this is an observed,
  not measured, gap). Worth checking whether it's just full-32-token
  generation cost on those two prompts, or a per-test setup/teardown cost
  stacking on top of #71's fix. Owner: unassigned.
  Blocker: GPU exclusivity (ADR 0006).

- **The G2 verdict's two legs were not interleaved (ADR 0015, GitHub #88).**
  The gate passed at 0.878 (8K) and 0.851 (32K) against a 1.5 threshold, but
  the reference leg was measured 2026-09-08 23:59 and the ignis leg
  2026-09-09 02:24 — each with the GPU exclusively its own, same session id,
  same harness, every sample cold, yet not in one sitting as ADR 0015 asks.
  The margin is far outside what two hours of drift can move, so this is a
  recorded deviation and not a re-open. Anyone re-running the cells for G3
  should take both legs back to back and overwrite the record. Owner:
  unassigned. Blocker: GPU exclusivity (ADR 0006) and the owner's ninfer.

## Blocked (external)

- **GPU availability (ADR 0006).** Every gate run and every GPU-profile test
  requires the RTX 5090 free — the owner's ninfer (the coding agent's own LLM
  backend) must be stopped first, and restarted after. A destructive
  shared-state operation on the owner's working environment: never done
  autonomously. Re-check before scheduling GPU work.

- **GitHub #80's G4 gate run is outstanding.** Issue #80 (structured logging
  Phase 3: hot-path guarantees, bounded queues, shutdown flush) added the
  two-channel bounded logging queue in `crates/logging`, wired shutdown flush
  into `ignis-server`/`vendor-ninfer`, and did a manual/grep-based hot-path
  audit + a static lint (`crates/logging/src/hotpath_lint.rs`) — all
  non-GPU work, done and tested (`cargo test --workspace` green). What is
  **not** done: the acceptance criterion itself, `ignis-bench gate` (G4,
  ≥99% of reference performance) run with the Phase 1-3 logging system wired
  into `ignis-server`, per ADR 0007/0011 and the issue's own "Gate" note.
  This was deliberately not run by the implementing agent (GPU is a shared
  resource with the owner's own ninfer, ADR 0006 — stopping it is a manual,
  owner-triggered action, not something an autonomous agent does). Run
  `scripts/gpu-profile.ps1` against this branch before closing #80. Owner:
  repo owner. Blocker: GPU exclusivity (ADR 0006).
  **Update 2026-09-09:** `scripts/gpu-profile.ps1` (kernel op-tests +
  `cargo test --workspace --features cuda -- --ignored` under
  `IGNIS_GPU_PROFILE=1`) was run against this branch on a free GPU and
  passed clean, 0 failures — confirms no GPU-gated-test regression from
  the logging queue/shutdown-flush changes. This is **not** the G4
  acceptance criterion itself (the `ignis-bench gate` trace-replay
  throughput run, see the "99% performance gate" entry below) — that
  instrument has no runnable baseline in this repo yet and remains
  blocked behind G1→G2→G3→G4, unchanged by this run.

- **GitHub #81's G4 gate run is outstanding.** Issue #81 (structured logging
  Phase 4: internal request-tracing spans, `request.id` as `trace_id`, ADR
  0012) wired the `ignis.admission`/`ignis.prefill`/`ignis.decode.round`/
  `ignis.completion` span tree into the live `crates/core` scheduler
  (`concrete.rs`) plus `tower-http`'s `TraceLayer` as the HTTP-ingress root
  span (`crates/server/src/api.rs`) — all non-GPU work, done and tested
  (`cargo test --workspace` green; `crates/core/tests/tracing_spans.rs` and
  `crates/server/tests/tracing_root_span.rs` cover trace_id propagation and
  round-not-token span granularity). A careful manual read of every span's
  open/close point (documented in `docs/design/tracing-spans.md`) confirms
  each one sits in code that already runs once per request per prefill
  chunk / decode round (an existing per-request bookkeeping loop, not the
  batched `Compute::prefill_step`/`decode_step` call itself, and never
  inside a per-token loop — there is no per-token loop in Rust to begin
  with; the leaf is device-resident, ADR 0009). What is **not** done: the
  acceptance criterion's own re-run of `ignis-bench gate` (G4, ≥99% of
  reference performance) with this span instrumentation wired into the live
  prefill/decode path, per the issue's explicit "GPU test" note and Phase
  3's standing rule that any hot-path-adjacent change needs the performance
  gate, not just a design argument. This was deliberately not run by the
  implementing agent (GPU is a shared resource with the owner's own ninfer,
  ADR 0006 — stopping it is a manual, owner-triggered action). Run
  `scripts/gpu-profile.ps1` against this branch before closing #81. Owner:
  repo owner. Blocker: GPU exclusivity (ADR 0006).
  **Update 2026-09-09:** `scripts/gpu-profile.ps1` was run against this
  branch (which includes both #80's and #81's changes) on a free GPU and
  passed clean, 0 failures, including a live TTFT run against
  `ignis-server` — confirms no GPU-gated-test regression from the span
  instrumentation. This is **not** the G4 acceptance criterion itself
  (the `ignis-bench gate` trace-replay throughput run); that instrument
  has no runnable baseline in this repo yet and remains blocked behind
  G1→G2→G3→G4, unchanged by this run.

- **The 99% performance gate (ADR 0007; GitHub #20 / #24, bench specs 02/03).**
  Parked behind **G4**, not open work — the roadmap folds #20 / #24 into the
  G4 master (#65); the trace-replay gate needs the hq-e8-2b profile on both
  sides, which is G4 work. (G3 has its own, narrower 99% check: decode
  throughput at C=1 and C=4.) The `ignis-bench` harness is code complete —
  replay, per-class metrics, canary self-consistency, the gate check, the
  composed gate artifact, and `ignis-bench record` (the capture
  proxy that turns a live agent session into a valid trace). What it lacks is
  a subject: an engine that computes the model (G1) and a serving loop worth
  measuring (G3), plus the recorded reference side (a real trace + a reference
  run through the same harness — procedure in `bench/traces/README.md`) and a
  live "1 main + ~10 subagents" session. Owner: bench actor.
  Blocker: G1 → G2 → G3 → G4, then the operational run.
