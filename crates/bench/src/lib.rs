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
//! `GET /v1/models`) (see GitHub #19).
//!
//! The capture proxy (`record`) is the recording side (spec 03): a
//! transparent OpenAI endpoint in front of a target engine that records a
//! live agent session into a load trace the replay harness re-sends — the
//! gate-run's capture piece.
//!
//! ## The measuring rule
//!
//! `ttft`, `g3`, `g4` and `g5` measure one engine through its
//! OpenAI-compatible endpoint, and `g2`, `g3_gate`, `g4_gate` and `g5_gate`
//! compare two engines' records, or two builds' of ignis. A comparison holds
//! only by this rule, and a check refuses what its records show broken (two
//! sessions, a missing cell, a sample that is not cold, one launch where it
//! takes the launches) rather than leave it to the operator:
//!
//! - **Both sides in one session (live/live).** The two sides are measured in the same
//!   measurement session, on the same machine, by this harness, and their
//!   records carry that session's id. A number published elsewhere, or
//!   recorded on another day, driver, thermal state or build, or by another
//!   harness, whose clock starts elsewhere, is not the other side of a
//!   comparison, and a committed record is a fixture for regression, never
//!   one. A result holds for its session: after a change both sides are
//!   measured again.
//! - **Cold-prefix samples.** Every sample, the warmup included, sends its
//!   own prompt of the exact length, differing from the first content token,
//!   and proves it was cold by the engine's own count of the prompt tokens it
//!   computed. Repeating one prompt would time a prefix cache from the second
//!   sample on, and flatter whichever engine has one. A sample that is not
//!   provably cold is void and fails its cell. A measurement of a cache is
//!   the one place this item gives way; the others still apply.
//! - **Pooled over independent launches.** Each side is launched at least
//!   twice, the server stopped and started again between the launches, and
//!   the statistic is taken over all of a side's launches, not the best nor
//!   one chosen. One process launch picks its own throughput band, up to 17%
//!   from another launch of the same build, and repetition inside a launch
//!   cannot cancel a bias that lasts the whole launch
//!   (`docs/findings/2026-09-10-one-process-launch-picks-its-own-throughput-band.md`).
//!   `g4_gate` and `g5_gate` take every launch of each side and refuse
//!   fewer than two; `ttft::pool_launches` pools TTFT records for `g2`;
//!   `g3_gate` takes one record a side, so two launches a side are two
//!   checks, read together.
//! - **Each engine as it is served.** An engine is measured in the profile
//!   it is served in, since a profile nobody serves is a hypothetical, and
//!   what is unequal between the two sides is recorded beside the result,
//!   never corrected for.
//!
//! The thresholds the checks print (`g2`'s 1.5, the others' 99%) are those
//! of the phase gates against the reference engine, which decide nothing
//! since ADR 0082: an artifact is admitted by the KLD against BF16, and a
//! speed claim about a change compares ignis with ignis by this same rule.
//!
//! ## The instruments
//!
//! The G2 measurement instrument (`ttft` + `g2`, P2-05) was the phase-2
//! gate's tooling: `ttft` measures time to first token at an
//! exact prompt length on **cold prefixes** against any OpenAI-compatible
//! endpoint — so both engines are measured by the same
//! instrument — and `g2` turns two such records into the G2 verdict,
//! refusing one outright when the evidence does not support it (records
//! from different sessions, a missing cell, a void sample).
//!
//! The canary oracle (`oracle`, P1-04) is the G1 correctness tooling: a
//! recorder captures the reference engine's greedy canary completions as a
//! fixture (tokenized with the artifact's tokenizer), and the suite is
//! scored against it two different ways:
//!
//! - **Teacher-forced next-token agreement — the G1 floor.** Each position
//!   is scored against the *oracle's own* prefix, so one divergence cannot
//!   cascade. `oracle::score_teacher_forced` + `oracle::G1_AGREEMENT_FLOOR`;
//!   the GPU driver is `crates/server/tests/oracle_teacher_forced_gpu.rs`.
//! - **Free-running agreement — diagnostic only.** The candidate generates
//!   its own continuation, so this measures continuation similarity rather
//!   than whether the forward pass is grossly broken.
//!   `oracle::compare_fixtures`, reported by `ignis-bench oracle compare`.
//!
//! The G3 measurement instrument (`g3` + `g3_gate`, P3-07) was the phase-3
//! gate's tooling (spec 03): `g3` measures the C=1 / C=4 / ITL
//! cells (single-sequence and aggregate throughput, and inter-token
//! latency under a concurrent prefill) over HTTP/SSE against any
//! OpenAI-compatible endpoint, reusing `ttft`'s exact-length prompt
//! generator and cold-prefix rule; `g3_gate` turns two such records into
//! the G3 verdict, with the same refusals as `g2`.
//!
//! The G4 measurement instrument (`g4` + `g4_gate`, P4-01) was the phase-4
//! gate's tooling (spec 04): `g4` replays the trace-replay
//! load and measures the needle-retrieval correctness floor, with every run
//! record carrying its measurement session and the SHA-256 of the load
//! trace it replayed; `g4_gate` pools at least two independent process
//! launches per engine into one verdict — the per-class / aggregate pooled
//! throughput ratio, and the needle floor, never folded into a ratio.
//!
//! The G5 measurement instrument (`g5` + `g5_gate`, P5-07) was the phase-5
//! gate's tooling: `g5` reuses `g3`'s C=1 throughput
//! cell (and its committed-token counter) at three prompt depths — 24K /
//! 98K / 196K — instead of a new instrument; `g5_gate` pools at least two
//! independent process launches per engine into one ratio per depth,
//! reusing `g4_gate`'s pooling rule. `equivalence` is the same phase's
//! correctness oracle: the canary suite run against a spec-on and a
//! spec-off endpoint, compared token-for-token for exact equality (not a
//! tolerance floor like `oracle` — any divergence is the bug).
//!
//! Flash-Next's acceptance scorers (`flash_next`, spec flash-next/04
//! acceptance 4-6 and 8) judge an engine's teacher-forced logits against the
//! converter's reference recordings: the top-64 KLD per domain, the MMLU-Pro
//! proxy and the G1 run on the expected-argmax column, through one seam
//! (`flash_next::SpanLogits`).

pub mod canary;
pub mod client;
pub mod equivalence;
pub mod flash_next;
pub mod g2;
pub mod g3;
pub mod g3_gate;
pub mod g4;
pub mod g4_gate;
pub mod g5;
pub mod g5_gate;
pub mod gate;
pub mod metrics;
pub mod oracle;
pub mod record;
pub mod report;
pub mod sampling;
pub mod small_card;
#[cfg(test)]
pub(crate) mod testutil;
pub mod time;
pub mod trace;
pub mod ttft;