//! Trace format + loader for the "1 main agent + N subagents" load trace.
//!
//! A trace is a JSONL file: one recorded request per line. The harness re-sends
//! the whole trace against a running engine (`client.rs`) and compares the
//! measured per-request metrics against the reference baseline recorded with
//! the *same* harness, so the comparison is apples-to-apples (ADR 0005: the
//! reference is a speed reference only; ADR 0007: the 99% gate is a
//! performance gate, not token parity).
//!
//! The load shape is a "1 main agent + ~N subagents" concurrent coding
//! workload (docs/design/ignis-v1.md §4): the main agent and the subagents
//! share a large, near-identical prefix (sibling-prefix reuse), and their
//! arrivals are staggered over time (`t_arrive_ms`) so the scheduler sees a
//! realistic concurrency profile.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Request class inside a "1 main + N subagents" load. The gate (ADR 0007)
/// checks ttft / tok-s *per class*, so main and subagent requests are tracked
/// separately and must not be averaged together.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RequestClass {
    /// The single main agent request (long-running, high context).
    Main,
    /// A subagent request (shorter, fired in fan-out).
    Sub,
}

fn default_class() -> RequestClass {
    RequestClass::Sub
}

fn default_max_tokens() -> u32 {
    1024
}

/// A single recorded request in a load trace (one JSONL line).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceLine {
    /// Stable request id. Used to match a replay against the reference
    /// baseline (the reference run must emit the same id).
    pub id: String,

    /// Class: main agent vs subagent (per-class gate, ADR 0007).
    #[serde(default = "default_class")]
    pub class: RequestClass,

    /// Milliseconds from the trace start at which this request arrives.
    /// Drives the concurrency profile of the replay (a burst of subagents at
    /// the same offset models the fan-out).
    pub t_arrive_ms: u64,

    /// The prompt content to re-send. It is the *actual* prompt so the engine
    /// sees the same input the reference did (apples-to-apples).
    pub prompt: String,

    /// Maximum number of tokens to generate for this request.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    /// Whether the original request was streaming. Streaming is what makes
    /// ttft (time-to-first-token) measurable; non-streaming requests report
    /// ttft == total_ms.
    #[serde(default)]
    pub stream: bool,
}

impl TraceLine {
    /// Build a replay request for the driver from this line.
    pub fn request(&self) -> super::client::Request {
        super::client::Request {
            id: self.id.clone(),
            class: self.class,
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens,
            stream: self.stream,
        }
    }
}

/// A recorded load trace: the ordered set of requests to replay.
#[derive(Debug, Clone, Default)]
pub struct Trace {
    lines: Vec<TraceLine>,
}

impl Trace {
    /// Parse a trace from a JSONL string. Each non-empty line must be valid
    /// JSON for a `TraceLine`. Lines may be in any order; the driver
    /// re-sorts by arrival regardless (defensive).
    pub fn from_jsonl(text: &str) -> Result<Self, String> {
        let mut lines = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parsed: TraceLine = serde_json::from_str(line)
                .map_err(|e| format!("line {n}: invalid JSON: {e}"))?;
            if parsed.id.is_empty() {
                return Err(format!("line {n}: missing request id"));
            }
            lines.push(parsed);
        }
        if lines.is_empty() {
            return Err("trace has no requests".into());
        }
        let trace = Self { lines };
        trace.validate()?;
        Ok(trace)
    }

    /// Load a trace from a JSONL file on disk.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_jsonl(&text)
    }

    /// Check the internal invariants: unique ids, and (for a realistic load)
    /// at most one main-agent request. Returns a human-readable error on
    /// violation.
    fn validate(&self) -> Result<(), String> {
        use std::collections::BTreeSet;
        let mut seen = BTreeSet::new();
        for line in &self.lines {
            if !seen.insert(line.id.as_str()) {
                return Err(format!("duplicate request id: {}", line.id));
            }
        }
        let mains = self
            .lines
            .iter()
            .filter(|l| l.class == RequestClass::Main)
            .count();
        if mains > 1 {
            return Err(format!("trace has {mains} main requests; expected at most 1"));
        }
        Ok(())
    }

    /// All requests, in arrival order (stable for ties).
    pub fn lines_sorted_by_arrival(&self) -> Vec<&TraceLine> {
        let mut sorted: Vec<&TraceLine> = self.lines.iter().collect();
        sorted.sort_by_key(|l| l.t_arrive_ms);
        sorted
    }

    /// The number of requests in the trace.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// `true` when the trace has no requests.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The raw lines, in file order.
    pub fn lines(&self) -> &[TraceLine] {
        &self.lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(id: &str, class: RequestClass, t: u64) -> TraceLine {
        TraceLine {
            id: id.into(),
            class,
            t_arrive_ms: t,
            prompt: format!("prompt {id}"),
            max_tokens: 64,
            stream: true,
        }
    }

    #[test]
    fn parses_a_main_plus_subagent_trace() {
        let jsonl = [
            r#"{"id":"main-1","class":"main","t_arrive_ms":0,"prompt":"P","max_tokens":512,"stream":true}"#,
            r#"{"id":"sub-1","class":"sub","t_arrive_ms":120,"prompt":"Q","stream":true}"#,
            r#"{"id":"sub-2","class":"sub","t_arrive_ms":120,"prompt":"Q","stream":true}"#,
        ]
        .join("\n");
        let trace = Trace::from_jsonl(&jsonl).expect("valid trace");
        assert_eq!(trace.len(), 3);
        assert_eq!(trace.lines()[0].class, RequestClass::Main);
        assert_eq!(trace.lines()[1].class, RequestClass::Sub);
        // Defaults fill in when omitted.
        assert_eq!(trace.lines()[1].max_tokens, 1024);
        assert!(trace.lines()[0].stream);
    }

    #[test]
    fn arrival_order_is_stable_for_ties() {
        let jsonl = [
            r#"{"id":"b","class":"sub","t_arrive_ms":50,"prompt":"x"}"#,
            r#"{"id":"a","class":"sub","t_arrive_ms":50,"prompt":"x"}"#,
            r#"{"id":"c","class":"sub","t_arrive_ms":10,"prompt":"x"}"#,
        ]
        .join("\n");
        let trace = Trace::from_jsonl(&jsonl).expect("valid trace");
        let order: Vec<&str> = trace
            .lines_sorted_by_arrival()
            .iter()
            .map(|l| l.id.as_str())
            .collect();
        // c (10) first; b and a tie at 50 and keep file order (stable sort).
        assert_eq!(order, vec!["c", "b", "a"]);
    }

    #[test]
    fn duplicate_id_is_rejected() {
        let jsonl = [
            r#"{"id":"r1","class":"sub","t_arrive_ms":0,"prompt":"x"}"#,
            r#"{"id":"r1","class":"sub","t_arrive_ms":9,"prompt":"y"}"#,
        ]
        .join("\n");
        assert!(Trace::from_jsonl(&jsonl).is_err());
    }

    #[test]
    fn more_than_one_main_is_rejected() {
        let a = line("m1", RequestClass::Main, 0);
        let b = line("m2", RequestClass::Main, 1);
        // Two mains is not a "1 main + N subagents" load.
        let trace = Trace { lines: vec![a, b] };
        assert!(trace.validate().is_err());
    }

    #[test]
    fn empty_trace_is_an_error() {
        assert!(Trace::from_jsonl("   \n  ").is_err());
    }

    #[test]
    fn a_trace_builds_requests_for_the_driver() {
        let t = line("r1", RequestClass::Sub, 0);
        let req = t.request();
        assert_eq!(req.id, "r1");
        assert_eq!(req.class, RequestClass::Sub);
        assert_eq!(req.max_tokens, 64);
        assert!(req.stream);
    }
}