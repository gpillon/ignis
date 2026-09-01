//! ignis-bench: trace-replay harness + canary suite runner.
//!
//! Replays a recorded "1 main agent + N subagents" load trace (JSONL, recorded
//! against the reference stack with this same harness) against a running ignis
//! server and produces the parity + divergence report (ADR 0003).

fn main() {
    // v1: trace replay + canary suite + divergence report (docs/design/ignis-v1.md §4).
    eprintln!("ignis-bench: not implemented yet (see docs/design/ignis-v1.md §4)");
}