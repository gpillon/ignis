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

#[cfg(test)]
mod tests {
    use super::*;

    // `StdoutSink` wraps `println!`, which stable Rust gives no portable way
    // to capture and assert on from a unit test (no stdout-redirect seam,
    // and this isn't worth adding one for a one-line wrapper) — so this
    // covers what a unit test safely can: it never panics on ordinary,
    // empty, or embedded-newline input, and it satisfies `LineSink` as a
    // trait object exactly like every other sink here.
    #[test]
    fn stdout_sink_writes_without_panicking() {
        let sink: Box<dyn LineSink> = Box::new(StdoutSink);
        sink.write_line("a plain line");
        sink.write_line("");
        sink.write_line("a line with an\nembedded newline");
    }

    #[test]
    fn stdout_sink_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StdoutSink>();
    }

    #[test]
    fn memory_sink_records_lines_in_order() {
        let sink = MemorySink::new();
        sink.write_line("first");
        sink.write_line("second");
        assert_eq!(sink.lines(), vec!["first".to_string(), "second".to_string()]);
    }
}
