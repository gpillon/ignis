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
  2026-09-07). The canary-agreement criterion is now **met**: teacher-forced
  next-token agreement is 99/102 = 97.1% against the ≥ 95% floor (#76,
  ADR 0014). Until the G1 gate is recorded green on a free RTX 5090 —
  coherent greedy completions on the canary suite, per-layer output within
  bf16 tolerance of the f64 layer reference, ≥ 95% teacher-forced next-token
  agreement with the canary oracle, EOS honored, reproducible across loads —
  **nothing downstream of it is meaningful**, including every performance
  number. Everything the review
  deleted (the host-resident forward, the toy decode graphs, the scalar
  kernels and their host-pointer surfaces) is gone; do not resurrect it.
  Owner: the runtime work, tickets #37–#62.
  Blocker: GPU exclusivity for the gate run (ADR 0006). #72 (diagnosed) and
  #76 (metric decided, ADR 0014) no longer block; what remains for #62 is
  recording the verdict — the f64 layer checks and the reproducibility run —
  on a free GPU.

- **`rust-sort` position 23: a genuine (small) logit disagreement with the
  reference — diagnostic follow-up, not a gate blocker (GitHub #76).** Under
  the teacher-forced G1 metric (ADR 0014) the canary suite scores 99/102 =
  97.1%. Three positions mismatch. Two are the exact BF16 logit ties
  diagnosed in #72 (`rust-sort` position 0: tokens 63 and 5836 both at 19.5;
  `explain-reverse` position 0: tokens 760 and 2064 both at 22.875) — ignis's
  argmax is the reference's own vendored kernel with the same lowest-token-id
  tie-break, so there is nothing to "fix" there and nothing is waived. The
  third is a real one: at `rust-sort` position 23, given the oracle's own
  prefix, ignis prefers token 198 at logit 20.25 while the oracle's token 25
  sits at 19.25 on ignis's logits — a gap of 1.0, roughly eight units in the
  last place at bf16. Small, isolated, and well inside the accepted floor,
  but it is the one position where the two forward passes genuinely disagree
  rather than coin-flip. Worth a look if a future numeric change is
  suspected. Owner: unassigned. Blocker: GPU exclusivity (ADR 0006).

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
