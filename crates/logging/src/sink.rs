//! The line sink each layer writes rendered records through — injectable so
//! tests capture output in memory instead of going through real stdout.
//!
//! `ignis_server::telemetry`'s `kind:"interval"` JSONL stream (GitHub #77)
//! is a separate concern from this crate's event model (ADR 0011/0017: the
//! scheduler interval counters stay metrics-shaped, not a `tracing::Event`,
//! and the future Prometheus projection reads the telemetry consumer
//! directly, never rendered log lines) — but GitHub #108 moved its sink
//! *implementation* onto [`LineSink`]/[`FileSink`]/[`NullSink`] here, since
//! the two had been carrying identical `fn write_line(&str)` shapes since
//! #78. Only the line-writing plumbing is shared; the two event streams
//! never merge.

use std::sync::Mutex;

/// Receives one rendered line per event (no trailing newline; the sink owns
/// framing).
pub trait LineSink: Send + Sync {
    fn write_line(&self, line: &str);

    /// Same as [`write_line`](LineSink::write_line), but also given the
    /// event's severity — the seam [`crate::queue::QueuedSink`] (GitHub #80)
    /// needs to route DEBUG/TRACE into its drop-oldest channel and
    /// INFO/WARN/ERROR into its bounded-blocking one. Every other sink here
    /// (`StdoutSink`, `StderrSink`, `MemorySink`, `FileSink`, `NullSink`)
    /// doesn't care about priority, so the default just forwards to
    /// `write_line` — only a priority-aware sink needs to override this.
    fn write_line_at(&self, _level: tracing::Level, line: &str) {
        self.write_line(line);
    }
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

/// A no-op sink: every line is discarded. The default for a consumer that
/// tracks state but has nowhere configured to write (e.g. telemetry with no
/// `--telemetry` sink selected and metrics disabled).
pub struct NullSink;

impl LineSink for NullSink {
    fn write_line(&self, _line: &str) {}
}

/// A file sink: appends one line per call to an opened (buffered) file. The
/// file is held in a `Mutex` so concurrent writers serialize into a single,
/// short, non-inverting lock.
pub struct FileSink {
    file: Mutex<std::fs::File>,
}

impl FileSink {
    /// Open (creating if absent) a sink at `path`, appending to an existing
    /// file rather than truncating it — matching a long-running process's
    /// expectation that restarting it does not erase prior output.
    pub fn open(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl LineSink for FileSink {
    fn write_line(&self, line: &str) {
        use std::io::Write;
        let mut file = self.file.lock().unwrap();
        let _ = writeln!(file, "{line}");
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

    #[test]
    fn null_sink_discards_every_line_without_panicking() {
        let sink: Box<dyn LineSink> = Box::new(NullSink);
        sink.write_line("anything");
        sink.write_line("");
    }

    #[test]
    fn file_sink_appends_lines_across_opens() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ignis-logging-sink-test-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        {
            let sink = FileSink::open(&path).unwrap();
            sink.write_line("first");
            sink.write_line("second");
        }
        // Re-opening an existing file appends rather than truncating it —
        // matching a long-running server restarted against the same path.
        {
            let sink = FileSink::open(&path).unwrap();
            sink.write_line("third");
        }

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines, vec!["first", "second", "third"]);
        let _ = std::fs::remove_file(&path);
    }
}
