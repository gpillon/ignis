//! ignis-bench: trace-replay harness + canary suite runner.
//!
//! Re-sends a recorded "1 main agent + N subagents" load trace (JSONL) against
//! a running engine and produces the **performance report** (tok-s, ttft vs
//! reference) + a **self-consistency check** (ADR 0007: ≥ 99% of the
//! reference's *speed* — a performance gate, not a token-parity gate).
//!
//! The I/O is isolated behind the `client::Endpoint` trait, so the core logic
//! (trace, metrics, canary, report) is fully testable with **no** running
//! server; the real HTTP endpoint is a thin follow-on (see
//! `.scratch/bench/issues/01-trace-replay.md`, blocked by #14).

pub mod canary;
pub mod client;
pub mod metrics;
pub mod report;
pub mod trace;