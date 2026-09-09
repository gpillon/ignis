//! The line sink each layer writes rendered records through — injectable so
//! tests capture output in memory instead of going through real stdout
//! (mirrors `ignis_server::telemetry::TelemetrySink`'s shape, a deliberately
//! separate, unmigrated concern — see the issue's "Named `logging`, not
//! `telemetry`" note).

use std::sync::Mutex;

/// Receives one rendered line per event (no trailing newline; the sink owns
/// framing).
pub trait LineSink: Send + Sync {
    fn write_line(&self, line: &str);
}

/// The production sink: one line per event on stdout.
pub struct StdoutSink;

impl LineSink for StdoutSink {
    fn write_line(&self, line: &str) {
        println!("{line}");
    }
}

/// A one-shot-command sink: one line per event on stderr, reserving stdout
/// for command-result output (GitHub #79 — `vendor-ninfer` and other
/// one-shot binaries route diagnostics here so `| jq .` on stdout is never
/// interleaved with log lines; `ignis serve` keeps using [`StdoutSink`]
/// since its entire event stream — not just command results — belongs on
/// stdout).
pub struct StderrSink;

impl LineSink for StderrSink {
    fn write_line(&self, line: &str) {
        eprintln!("{line}");
    }
}

/// An in-memory sink: captures emitted lines, in order, so tests can assert
/// on them without a real stdout.
#[derive(Default)]
pub struct MemorySink {
    lines: Mutex<Vec<String>>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// A snapshot of the lines emitted so far, in order.
    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().unwrap().clone()
    }
}

impl LineSink for MemorySink {
    fn write_line(&self, line: &str) {
        self.lines.lock().unwrap().push(line.to_string());
    }
}
