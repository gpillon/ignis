//! ignis-bench: trace-replay harness + canary suite runner.
//!
//! Re-sends a recorded "1 main agent + N subagents" load trace (JSONL) against
//! a running engine and produces the **performance report** (tok-s, ttft vs
//! reference) + a **self-consistency check** (ADR 0007: ≥ 99% of the
//! reference's *speed* — a performance gate, not a token-parity gate).
//!
//! The I/O is isolated behind the `client::Endpoint` trait, so the core
//! logic (trace, metrics, canary, report) is fully testable with **no**
//! running server; the real HTTP endpoint (`client::HttpEndpoint`, a
//! `reqwest` blocking client) drives the running `ignis-server`
//! (`POST /v1/chat/completions` — streaming + non-streaming — and
//! `GET /v1/models`) (see `.scratch/bench/specs/01-trace-replay.md`).
//!
//! The capture proxy (`record`) is the recording side (spec 03): a
//! transparent OpenAI endpoint in front of a target engine that records a
//! live agent session into a load trace the replay harness re-sends — the
//! gate-run's capture piece.
//!
//! The canary oracle (`oracle`, P1-04) is the G1 correctness tooling: a
//! recorder captures the reference engine's greedy canary completions as a
//! fixture (tokenized with the artifact's tokenizer), and the suite is
//! scored against it two different ways (ADR 0014):
//!
//! - **Teacher-forced next-token agreement — the G1 floor.** Each position
//!   is scored against the *oracle's own* prefix, so one divergence cannot
//!   cascade. `oracle::score_teacher_forced` + `oracle::G1_AGREEMENT_FLOOR`;
//!   the GPU driver is `crates/server/tests/oracle_teacher_forced_gpu.rs`.
//! - **Free-running agreement — diagnostic only.** The candidate generates
//!   its own continuation, so this measures continuation similarity rather
//!   than whether the forward pass is grossly broken.
//!   `oracle::compare_fixtures`, reported by `ignis-bench oracle compare`.

pub mod canary;
pub mod client;
pub mod gate;
pub mod metrics;
pub mod oracle;
pub mod record;
pub mod report;
pub mod trace;