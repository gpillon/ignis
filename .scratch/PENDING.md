# Pending / revisit ledger (ignis)

Cross-cutting items that are intentionally deferred or blocked on an external
dependency. Per-ticket details live in `.scratch/<feature>/specs/`; status and
blocking live on GitHub (AGENTS.md). This ledger records *capabilities that do
not work yet*, never tickets that closed.

Reset 2026-09-05 by the project review (`.scratch/REVIEW-2026-09-05.md`): the
engine has never produced a real completion, and the work that claimed to have
delivered one is superseded. The phase / gate plan is `.scratch/ROADMAP.md`.

## Open

- **G1 — a correct, device-resident forward pass (GitHub #36, spec
  `.scratch/runtime/specs/01-device-resident-forward.md`, ADR 0009/0010).**
  ignis computes the model and produces coherent greedy completions as of the
  RMSNorm `unit_offset` fix (#67), and the full GPU profile
  (`scripts/gpu-profile.ps1`) now runs green end to end (#73/#74/#75 fixed
  2026-09-07) — but the gate is still **not** green: first-32-token agreement
  with the canary oracle is 52%, against a ≥ 95% floor (#72 below). Until the
  G1 gate is recorded green on a free RTX 5090 — coherent greedy completions
  on the canary suite, per-layer output within bf16 tolerance of the f64
  layer reference, ≥ 95% first-32-token agreement with the canary oracle, EOS
  honored, reproducible across loads — **nothing downstream of it is
  meaningful**, including every performance number. Everything the review
  deleted (the host-resident forward, the toy decode graphs, the scalar
  kernels and their host-pointer surfaces) is gone; do not resurrect it.
  Owner: the runtime work, tickets #37–#62.
  Blocker: #72, plus GPU exclusivity for the gate run (ADR 0006).

- **Two canaries diverge from the oracle at token 0 (GitHub #72; blocks G1).**
  With the RMSNorm convention fixed (#67) and the chat template's thinking
  disabled to match how the oracle was recorded, `rust-hello` scores 21/21 and
  `math-greedy` 32/32 — exact agreement with the reference. But `rust-sort`
  and `explain-reverse` score 0/32 and 0/17, diverging at the very first
  generated token, which puts the suite at 52% against the ≥ 95% G1 floor.
  Both produce correct, fluent answers (`` `v` is set to `[1, 2, 3]`. `` and a
  correct one-sentence description of `Vec::reverse`), just different wording
  from the oracle's — so this is not the #67 class of failure. The working
  theory is an argmax near-tie flipped by a residual numeric difference; once
  flipped, the trajectory diverges entirely. Confirming that needs logits,
  which the server does not expose: the next boundary is top-k IDs and values
  after the final head, for the same tokenized prompt, on both engines.
  `enable_thinking` (#68) has since landed, which should unblock reproducing
  the oracle's exact prompt through the HTTP surface — not yet re-checked.
  Owner: unassigned. Blocker: GPU exclusivity (ADR 0006).

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
