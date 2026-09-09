//! The two-channel bounded queue that decouples event *creation* (a
//! `tracing` call site, potentially on the prefill/decode path) from
//! physical I/O (`LineSink::write_line`, potentially slow: a full disk, a
//! stalled pipe) — GitHub #80, spec §27/28, ADR 0011's "lean on the
//! `tracing` ecosystem, not a hand-rolled ad-hoc channel" default, applied
//! here as a small purpose-built queue rather than `tracing-appender`'s
//! `non_blocking` writer: that writer only has one backpressure policy per
//! instance (its `lossy` flag drops the *newest* line, not the oldest), so
//! getting the spec's two distinct policies — drop-oldest for DEBUG/TRACE,
//! never-drop-blocks-briefly for INFO/WARN/ERROR — in one coherent unit
//! meant one small `Mutex`+`Condvar` queue instead of two independently
//! configured `tracing-appender` writers with no shared draining order.
//!
//! - [`QueuedSink`] is the [`LineSink`] every layer writes through in
//!   production: `write_line_at` routes DEBUG/TRACE into a bounded ring
//!   buffer that evicts its oldest entry on overflow (never blocks, never
//!   grows past its bound), and INFO/WARN/ERROR into a bounded queue whose
//!   `push` blocks until there is room rather than dropping (acceptable —
//!   these are rare by construction, never emitted from the per-token inner
//!   loop, spec §27 / user story 6).
//! - One background thread drains both queues (priority first) and performs
//!   the real, possibly-slow `inner.write_line` call — no `tracing` call
//!   site ever touches physical I/O directly.
//! - [`QueuedSink::flush`] is the shutdown seam (spec §28): wait up to a
//!   caller-supplied timeout for every INFO/WARN/ERROR line enqueued
//!   *before* the call to have been handed to the inner sink — never
//!   indefinitely (a stalled sink must not hang shutdown).
//! - [`QueueWorkerGuard`] owns the background thread's lifetime the same
//!   way `tracing_appender::non_blocking`'s own `WorkerGuard` does: dropping
//!   it signals the worker to stop, but never blocks waiting for it to
//!   actually exit — `Drop` must never introduce indefinite blocking either.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use tracing::Level;

use crate::sink::LineSink;

/// Queue capacities (line counts, not bytes) — small, fixed, and generous
/// enough that an ordinary burst does not visibly degrade before either
/// policy kicks in, while still bounding worst-case memory (spec §27:
/// "uncontrolled memory growth MUST NOT be possible").
#[derive(Debug, Clone, Copy)]
pub struct QueueConfig {
    pub debug_trace_capacity: usize,
    pub priority_capacity: usize,
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self { debug_trace_capacity: 2048, priority_capacity: 512 }
    }
}

/// Both queues plus the bookkeeping `flush` needs, behind one lock. A single
/// lock (rather than one per queue) keeps the design small; nothing here is
/// held across I/O — the lock only ever guards `VecDeque` push/pop and
/// counter updates, never the `inner.write_line` call itself.
struct State {
    debug_trace: VecDeque<String>,
    debug_trace_dropped: u64,
    priority: VecDeque<String>,
    priority_enqueued: u64,
    priority_written: u64,
}

/// The shared queue core: reachable from the producer side ([`QueuedSink`]),
/// the background worker thread, and the shutdown guard
/// ([`QueueWorkerGuard`]) — all three hold an `Arc` of this.
struct SharedQueue {
    debug_trace_capacity: usize,
    priority_capacity: usize,
    state: Mutex<State>,
    /// Notified on every push (either queue), every pop, and every
    /// `mark_written` — one condvar for all three is deliberate: every
    /// waiter re-checks its own predicate in a loop, so a wakeup that turns
    /// out irrelevant just costs one cheap re-check, never a correctness
    /// bug (spurious wakeups are already something every `Condvar::wait`
    /// caller must tolerate).
    cv: Condvar,
    stop: AtomicBool,
}

impl SharedQueue {
    fn new(config: QueueConfig) -> Self {
        Self {
            debug_trace_capacity: config.debug_trace_capacity.max(1),
            priority_capacity: config.priority_capacity.max(1),
            state: Mutex::new(State {
                debug_trace: VecDeque::new(),
                debug_trace_dropped: 0,
                priority: VecDeque::new(),
                priority_enqueued: 0,
                priority_written: 0,
            }),
            cv: Condvar::new(),
            stop: AtomicBool::new(false),
        }
    }

    /// Never blocks: evicts the oldest queued line first if already at
    /// capacity, then appends `line`.
    fn push_debug_trace(&self, line: String) {
        let mut state = self.state.lock().unwrap();
        if state.debug_trace.len() >= self.debug_trace_capacity {
            state.debug_trace.pop_front();
            state.debug_trace_dropped += 1;
        }
        state.debug_trace.push_back(line);
        self.cv.notify_all();
    }

    /// Blocks until there is room, then appends `line` — never drops. If
    /// shutdown has already been signaled and the queue is still full (a
    /// narrow race: the worker thread stopped draining right as a last
    /// INFO/WARN/ERROR event was emitted), pushes anyway rather than
    /// dropping or hanging forever: exceeding the bound by a little in that
    /// one narrow window is preferable to silently losing an ERROR.
    fn push_priority(&self, line: String) {
        let mut state = self.state.lock().unwrap();
        while state.priority.len() >= self.priority_capacity && !self.stop.load(Ordering::Acquire) {
            let (guard, _timeout) =
                self.cv.wait_timeout(state, Duration::from_millis(50)).unwrap();
            state = guard;
        }
        state.priority.push_back(line);
        state.priority_enqueued += 1;
        self.cv.notify_all();
    }

    fn try_pop_priority(&self) -> Option<String> {
        let mut state = self.state.lock().unwrap();
        let line = state.priority.pop_front();
        if line.is_some() {
            self.cv.notify_all();
        }
        line
    }

    fn try_pop_debug_trace(&self) -> Option<String> {
        let mut state = self.state.lock().unwrap();
        state.debug_trace.pop_front()
    }

    fn mark_priority_written(&self) {
        let mut state = self.state.lock().unwrap();
        state.priority_written += 1;
        self.cv.notify_all();
    }

    /// Wait (bounded by `timeout`) for at least `watermark` priority lines
    /// to have been handed to the inner sink. Returns whether the watermark
    /// was reached — `false` means the timeout elapsed first (a stalled
    /// sink), never a hang.
    fn wait_until_written(&self, watermark: u64, timeout: Duration) -> bool {
        let state = self.state.lock().unwrap();
        let (state, timeout_result) = self
            .cv
            .wait_timeout_while(state, timeout, |s| s.priority_written < watermark)
            .unwrap();
        !timeout_result.timed_out() || state.priority_written >= watermark
    }

    fn priority_enqueued(&self) -> u64 {
        self.state.lock().unwrap().priority_enqueued
    }

    fn priority_len(&self) -> usize {
        self.state.lock().unwrap().priority.len()
    }

    fn debug_trace_len(&self) -> usize {
        self.state.lock().unwrap().debug_trace.len()
    }

    fn debug_trace_dropped(&self) -> u64 {
        self.state.lock().unwrap().debug_trace_dropped
    }

    fn both_empty(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.debug_trace.is_empty() && state.priority.is_empty()
    }
}

/// The background drain loop: priority lines first (so an ERROR is never
/// stuck behind a backlog of DEBUG noise), then debug/trace lines, then wait
/// for more work — until `stop` is set and both queues have been drained.
fn worker_loop(shared: Arc<SharedQueue>, inner: Arc<dyn LineSink>) {
    loop {
        let mut drained_any = false;
        while let Some(line) = shared.try_pop_priority() {
            inner.write_line(&line);
            shared.mark_priority_written();
            drained_any = true;
        }
        while let Some(line) = shared.try_pop_debug_trace() {
            inner.write_line(&line);
            drained_any = true;
        }
        if drained_any {
            continue;
        }
        if shared.stop.load(Ordering::Acquire) && shared.both_empty() {
            break;
        }
        let state = shared.state.lock().unwrap();
        let _ = shared.cv.wait_timeout(state, Duration::from_millis(50));
    }
}

/// The [`LineSink`] every layer writes through in production (GitHub #80):
/// wraps a real sink, routes by severity into one of the two channels
/// above, and returns to the caller immediately — the background thread
/// (spawned in [`QueuedSink::new`]) is the only thing that ever calls
/// `inner.write_line`.
pub struct QueuedSink {
    shared: Arc<SharedQueue>,
}

impl QueuedSink {
    /// Spawn the background worker and return the sink plus the guard that
    /// owns the worker's lifetime (drop it to signal shutdown — see
    /// [`QueueWorkerGuard`]).
    pub fn new(inner: Arc<dyn LineSink>, config: QueueConfig) -> (Arc<QueuedSink>, QueueWorkerGuard) {
        let shared = Arc::new(SharedQueue::new(config));
        let worker_shared = shared.clone();
        std::thread::Builder::new()
            .name("ignis-logging-writer".to_owned())
            .spawn(move || worker_loop(worker_shared, inner))
            .expect("spawning the logging writer thread");
        (Arc::new(QueuedSink { shared: shared.clone() }), QueueWorkerGuard { shared })
    }

    /// Wait up to `timeout` for every INFO/WARN/ERROR line enqueued before
    /// this call to have been handed to the inner sink. Returns `true` if
    /// they all were, `false` if `timeout` elapsed first (a stalled sink —
    /// this never blocks past `timeout`, satisfying spec §28's "MUST NOT
    /// introduce indefinite shutdown blocking"). DEBUG/TRACE lines are not
    /// waited on: they are lossy by design, so a pending one at shutdown is
    /// exactly the kind of data this channel already permits losing.
    pub fn flush(&self, timeout: Duration) -> bool {
        let watermark = self.shared.priority_enqueued();
        self.shared.wait_until_written(watermark, timeout)
    }

    /// How many DEBUG/TRACE lines have been evicted (dropped) so far —
    /// observability of the lossy channel's own pressure, not part of the
    /// hot path (called by tests / future metrics, never per-event).
    pub fn debug_trace_dropped(&self) -> u64 {
        self.shared.debug_trace_dropped()
    }

    pub fn debug_trace_len(&self) -> usize {
        self.shared.debug_trace_len()
    }

    pub fn priority_len(&self) -> usize {
        self.shared.priority_len()
    }
}

impl LineSink for QueuedSink {
    fn write_line(&self, line: &str) {
        // No level given: treat as important rather than risk dropping it —
        // only [`write_line_at`](LineSink::write_line_at) (what both layers
        // actually call) makes the DEBUG/TRACE-vs-priority distinction.
        self.shared.push_priority(line.to_owned());
    }

    fn write_line_at(&self, level: Level, line: &str) {
        match level {
            Level::TRACE | Level::DEBUG => self.shared.push_debug_trace(line.to_owned()),
            Level::INFO | Level::WARN | Level::ERROR => self.shared.push_priority(line.to_owned()),
        }
    }
}

/// Owns the background writer thread's lifetime. Dropping it signals the
/// worker to stop after draining what is already queued — but never blocks
/// waiting for that to happen: `Drop` must never introduce the indefinite
/// blocking spec §28 forbids. Call [`QueuedSink::flush`] first if the
/// pending lines must be confirmed written before the process exits (the
/// shutdown pattern this crate documents).
pub struct QueueWorkerGuard {
    shared: Arc<SharedQueue>,
}

impl Drop for QueueWorkerGuard {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering as AtOrd};
    use std::time::{Duration, Instant};

    use super::*;
    use crate::sink::MemorySink;

    /// A [`LineSink`] whose `write_line` blocks until released — the "stuck
    /// sink" fixture the shutdown-timeout tests need. `release()` is called
    /// from the test to unblock it after the assertion under test has run.
    struct StallingSink {
        gate: Mutex<bool>,
        cv: Condvar,
        received: Mutex<Vec<String>>,
    }

    impl StallingSink {
        fn new() -> Arc<Self> {
            Arc::new(Self { gate: Mutex::new(false), cv: Condvar::new(), received: Mutex::new(Vec::new()) })
        }

        fn release(&self) {
            *self.gate.lock().unwrap() = true;
            self.cv.notify_all();
        }
    }

    impl LineSink for StallingSink {
        fn write_line(&self, line: &str) {
            let mut released = self.gate.lock().unwrap();
            while !*released {
                released = self.cv.wait(released).unwrap();
            }
            self.received.lock().unwrap().push(line.to_owned());
        }
    }

    #[test]
    fn debug_trace_channel_drops_oldest_rather_than_growing_past_its_bound() {
        let sink = StallingSink::new();
        // Held stalled for the whole test: this test only cares about the
        // producer-side ring-buffer behaviour, never about what the worker
        // thread does with a drained line.
        let (queued, _guard) = QueuedSink::new(sink.clone(), QueueConfig { debug_trace_capacity: 4, priority_capacity: 64 });

        for i in 0..100 {
            queued.write_line_at(Level::DEBUG, &format!("line-{i}"));
        }

        // Never grows past its bound, no matter how large the burst.
        assert!(queued.debug_trace_len() <= 4, "len={}", queued.debug_trace_len());
        // And it did in fact drop rather than block: 100 pushes into a
        // 4-capacity ring buffer with a stalled drain means at least 96
        // evictions (draining may have raced a couple through first, so
        // this asserts a floor, not an exact count).
        assert!(queued.debug_trace_dropped() >= 90, "dropped={}", queued.debug_trace_dropped());

        sink.release();
    }

    #[test]
    fn debug_trace_burst_never_blocks_the_caller() {
        let sink = StallingSink::new(); // never released — drain is permanently stuck
        let (queued, _guard) = QueuedSink::new(sink, QueueConfig { debug_trace_capacity: 2, priority_capacity: 64 });

        let start = Instant::now();
        for i in 0..10_000 {
            queued.write_line_at(Level::TRACE, &format!("line-{i}"));
        }
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(2), "10k debug/trace pushes took {elapsed:?}, expected near-instant");
    }

    #[test]
    fn priority_channel_never_silently_drops_a_burst_within_its_bound() {
        let sink = Arc::new(MemorySink::new());
        let (queued, _guard) =
            QueuedSink::new(sink.clone(), QueueConfig { debug_trace_capacity: 64, priority_capacity: 32 });

        for i in 0..32 {
            queued.write_line_at(Level::ERROR, &format!("error-{i}"));
        }
        assert!(queued.flush(Duration::from_secs(2)), "flush should complete well within its budget");
        assert_eq!(sink.lines().len(), 32, "every enqueued ERROR line must reach the sink");
    }

    #[test]
    fn priority_channel_blocks_rather_than_drops_when_full() {
        // A capacity-1 priority queue with a slow (but not stalled) drain:
        // pushing a burst larger than capacity must deliver every line, not
        // drop the overflow — proving the "blocks briefly rather than
        // drops" policy actually blocks instead of silently discarding.
        struct SlowSink {
            count: AtomicUsize,
        }
        impl LineSink for SlowSink {
            fn write_line(&self, _line: &str) {
                std::thread::sleep(Duration::from_millis(2));
                self.count.fetch_add(1, AtOrd::Relaxed);
            }
        }
        let sink = Arc::new(SlowSink { count: AtomicUsize::new(0) });
        let (queued, _guard) =
            QueuedSink::new(sink.clone(), QueueConfig { debug_trace_capacity: 64, priority_capacity: 1 });

        for i in 0..50 {
            queued.write_line_at(Level::WARN, &format!("warn-{i}"));
        }
        assert!(queued.flush(Duration::from_secs(5)), "flush should complete within its budget");
        assert_eq!(sink.count.load(AtOrd::Relaxed), 50, "no WARN line may be dropped, even under sustained backpressure");
    }

    #[test]
    fn shutdown_flush_emits_all_pending_priority_events_within_the_timeout() {
        let sink = Arc::new(MemorySink::new());
        let (queued, _guard) = QueuedSink::new(sink.clone(), QueueConfig::default());

        for i in 0..10 {
            queued.write_line_at(Level::INFO, &format!("ignis.test.event-{i}"));
        }
        let flushed = queued.flush(Duration::from_secs(1));
        assert!(flushed, "flush must report success for a healthy sink");
        assert_eq!(sink.lines().len(), 10);
    }

    #[test]
    fn shutdown_does_not_hang_when_the_sink_is_stalled_past_the_timeout() {
        let sink = StallingSink::new(); // never released
        let (queued, _guard) = QueuedSink::new(sink.clone(), QueueConfig::default());

        queued.write_line_at(Level::ERROR, "ignis.test.stuck");

        let start = Instant::now();
        let flushed = queued.flush(Duration::from_millis(150));
        let elapsed = start.elapsed();

        assert!(!flushed, "flush must report it did not complete against a stalled sink");
        assert!(elapsed < Duration::from_millis(500), "flush must not block substantially past its own timeout, took {elapsed:?}");

        sink.release(); // let the worker thread unstick before the test process moves on
    }

    #[test]
    fn dropping_the_worker_guard_never_blocks() {
        let sink = StallingSink::new(); // permanently stuck drain
        let (queued, guard) = QueuedSink::new(sink.clone(), QueueConfig::default());
        queued.write_line_at(Level::INFO, "ignis.test.pending");

        let start = Instant::now();
        drop(guard);
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_millis(100), "dropping the guard must return immediately, took {elapsed:?}");

        sink.release();
    }

    #[test]
    fn write_line_without_a_level_is_treated_as_priority() {
        let sink = Arc::new(MemorySink::new());
        let (queued, _guard) = QueuedSink::new(sink.clone(), QueueConfig::default());
        LineSink::write_line(&*queued, "no level given");
        assert!(queued.flush(Duration::from_secs(1)));
        assert_eq!(sink.lines(), vec!["no level given".to_string()]);
    }
}
