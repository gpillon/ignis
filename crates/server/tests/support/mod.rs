//! Shared test-only helpers (GitHub #69), included via `#[path]` by
//! individual integration test binaries (`tests/*.rs`) — this file is not
//! itself a test binary (it lives under a subdirectory of `tests/`, so
//! cargo does not compile it as one).

/// A few deterministic scheduling turns (never a sleep — ADR 0006) to let a
/// spawned task run up to its next await point — e.g. far enough into
/// `Engine::submit` to have enqueued its command, before the test proceeds
/// to release a gate the model thread is held on.
pub async fn nudge() {
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }
}
