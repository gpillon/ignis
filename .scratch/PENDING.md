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

- **An intermittent wrong-output event in the GQA layer oracle (GitHub #96).**
  `gqa_layer_gpu::gqa_layers_match_f64_reference` fails roughly two runs in
  five: one token of a four-token decode sequence lands at a relative L2 of
  0.32-0.37 against the f64 reference while every other token in the same run
  is bit-identical to a clean run at ~0.004. Deterministic when it does not
  fire, so this is a wrong-output event, not drift. `A4_LAYER_TOLERANCE = 0.32`
  was very likely calibrated on the same event in #85, which would mean the
  layer oracles carry ~80x more slack than the kernels need and would not
  catch a real precision regression. This is what keeps the GPU profile from
  being green in one run, and so what keeps #88 (and #63) open. Owner:
  unassigned. Blocker: GPU exclusivity (ADR 0006).

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
