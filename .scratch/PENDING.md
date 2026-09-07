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
  RMSNorm `unit_offset` fix (#67), but the gate is **not** green: first-32-token
  agreement with the canary oracle is 52%, against a ≥ 95% floor (see #72
  below), and the 2026-09-07 gate run (#62) found the server e2e leg of the
  GPU profile red plus a test-wiring gap that hid three GPU tests from every
  profile run (#70, #71, #73). Until the G1 gate is recorded green on a free
  RTX 5090 — coherent greedy completions on the canary suite, per-layer output
  within bf16 tolerance of the f64 layer reference, ≥ 95% first-32-token
  agreement with the canary oracle, EOS honored, reproducible across loads —
  **nothing downstream of it is meaningful**, including every performance
  number. Everything the review deleted (the host-resident forward, the toy
  decode graphs, the scalar kernels and their host-pointer surfaces) is gone;
  do not resurrect it. Owner: the runtime work, tickets #37–#62.
  Blocker: #70, #71, #72, #73, plus GPU exclusivity for the gate run (ADR 0006).

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

- **The GPU profile's fail-never-skip rule now runs green for op/layer/program
  tests, but the server e2e leg is red and three tests are invisible to the
  profile entirely (GitHub #38, ADR 0006; found running #62 on 2026-09-07).**
  A full `scripts/gpu-profile.ps1` run on a free RTX 5090 came back:
  kernel build green; `gqa_layers_match_f64_reference`,
  `full_program_prefill_and_greedy_decode_are_deterministic`, the `cuda_leaf`
  decode test, and `seq_alloc_gpu`'s 3 tests all green. `openai_http_gpu.rs`
  (P1-25 server e2e) failed 2 of 3 under default (concurrent) `cargo test`
  parallelism with a GPU OOM panic; re-run serialized
  (`--test-threads=1`) it no longer OOMs but still climbs to ~31/32 GB VRAM
  and takes 668s instead of ~160s (#71 — the test harness never tears down a
  prior test's scheduler/driver task), and the same 2 tests instead fail on
  empty/malformed content because they don't disable `enable_thinking` and
  the model spends the whole 32-token budget on the reasoning channel (#70).
  Separately, `model_load_gpu.rs` (P1-17), `gdn_layer_gpu.rs` (P1-22), and
  `step_degenerate_gpu.rs` (P1-18) all lack `#[ignore]`, so
  `cargo test --features cuda -- --ignored` — the profile script's exact
  invocation — filters them **out** every time; they have never actually run
  under the profile (#73). Owner: whoever next has a free GPU; fix #73 first
  (cheap), then re-run the full profile to get first real data on those three,
  then #70/#71 to get server e2e green. Blocker: GPU exclusivity (ADR 0006).

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
