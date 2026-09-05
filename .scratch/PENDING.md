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
  ignis does not yet compute the model. The forward pass is being rebuilt as a
  device-resident program in the kernel leaf behind a step-level C ABI, on
  vendored reference ops. Until the G1 gate is recorded green on a free RTX
  5090 — coherent greedy completions on the canary suite, per-layer output
  within bf16 tolerance of the f64 layer reference, ≥ 95% first-32-token
  agreement with the canary oracle, EOS honored, reproducible across loads —
  **nothing downstream of it is meaningful**, including every performance
  number. Everything the review deleted (the host-resident forward, the toy
  decode graphs, the scalar kernels and their host-pointer surfaces) is gone;
  do not resurrect it. Owner: the runtime work, tickets #37–#62.
  Blocker: the work itself, plus GPU exclusivity for the gate run (ADR 0006).

- **Canary oracle fixture recording (GitHub #41, spec 01 — human
  prerequisite).** The G1 agreement check needs a recorded fixture: 3 canary
  prompts × 32 greedy tokens, recorded against the *reference* engine (ninfer)
  with exact argmax over its HTTP API, on the same artifact, tokenized with the
  artifact's tokenizer. The tooling (recorder + comparer + fixture format) is
  agent work (#40); the recording itself is **operational** — it needs the
  reference stack running and the GPU (ADR 0006: one engine at a time, so the
  recording and any ignis GPU run are sequential). Owner: human.
  Blocker: a live reference stack + GPU exclusivity.

- **The GPU profile's fail-never-skip rule has no caller yet (GitHub #38,
  ADR 0006).** The rule and its harness are in place: `ignis_core::gpu_profile`
  decides skip-vs-fail, `scripts/gpu-preflight.ps1` records a pass in a marker
  file, `active()` refuses `IGNIS_GPU_PROFILE=1` without a recent one, and
  `scripts/gpu-profile.ps1` preflights, runs the GPU work and consumes the
  marker. What is *not* demonstrated is the rule doing its job for a real GPU
  test: #39 deleted the last ten `*_gpu.rs` tests, so the helper has no callers
  outside its own unit tests, and those drive an injected rc, not a kernel. The
  first vendored-op / step-ABI GPU test to land (P1-07 onward) must call
  `check_rc` / `check_compute_err` / `skip_or_fail` rather than returning early
  on its own — that is when #38's "the rule must survive" is actually verified
  end to end, and nothing but review enforces it until then.
  Owner: whoever lands the first GPU test after the vendored ops.
  Blocker: no GPU-gated test exists to carry it (P1-07+).

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
