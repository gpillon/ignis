//! GitHub #89 / ADR 0017, structurally: no metrics dependency or
//! metrics-aware code on the inference path. The scheduler (`ignis-core`),
//! the runtime, the kernel leaf and the server's model-thread loop must not
//! so much as name metrics — the projection lives entirely on the
//! asynchronous telemetry side. Behavioral equality of the model thread's
//! fact traffic is `engine.rs`'s
//! `metrics_leave_the_model_thread_fact_traffic_unchanged`.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Every file under `dir`, recursively.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}

/// The lines of `text` naming metrics, with their line numbers.
fn metrics_mentions(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .filter(|(_, line)| {
            let line = line.to_ascii_lowercase();
            // `metrics`, not `metric`: "asymmetric" is not a metrics mention.
            line.contains("metrics") || line.contains("prometheus")
        })
        .map(|(n, line)| (n + 1, line.to_owned()))
        .collect()
}

#[test]
fn the_scheduler_runtime_and_kernel_leaf_never_name_metrics() {
    let root = repo_root();
    let mut scanned = 0;
    let mut offenders = Vec::new();
    for dir in ["crates/core", "crates/runtime", "kernel/src", "kernel/include"] {
        for path in files_under(&root.join(dir)) {
            // Build output is not source.
            if path.components().any(|c| c.as_os_str() == "target") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue; // not text
            };
            scanned += 1;
            for (line, content) in metrics_mentions(&text) {
                offenders.push(format!("{}:{line}: {content}", path.display()));
            }
        }
    }
    assert!(scanned > 50, "scanned only {scanned} files — wrong root?");
    assert!(offenders.is_empty(), "metrics-aware code on the inference path:\n{}", offenders.join("\n"));
}

/// GitHub #90 / ADR 0017: a scrape reads fixed atomics and formats them —
/// nothing else. The projection's module names no lock, no channel, no
/// waker and no engine handle, so however many scrapers there are, and
/// however slow, a scrape has no way to send a command to, wait for, lock,
/// wake or backpressure the model thread or the telemetry consumer.
#[test]
fn a_scrape_has_no_way_to_reach_the_model_thread_or_the_telemetry_consumer() {
    let source = std::fs::read_to_string(repo_root().join("crates/server/src/metrics.rs")).unwrap();
    // The code, not its unit tests or the doc comments that explain what it
    // avoids.
    let production = &source[..source.find("#[cfg(test)]").unwrap_or(source.len())];
    // Whole identifiers, so a longer name that merely contains one (`awake`)
    // is not an offender.
    let identifiers: std::collections::BTreeSet<&str> = production
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .flat_map(|line| line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')))
        .filter(|word| !word.is_empty())
        .collect();
    let forbidden = [
        "Mutex", "RwLock", "Condvar", "Notify", "Barrier", "mpsc", "oneshot", "watch", "broadcast",
        "channel", "unbounded_channel", "Engine", "engine", "TelemetryFact", "facts", "lock", "await",
        "block_on", "spawn", "spawn_blocking", "Waker", "wake", "ArcSwap", "SystemTime", "Instant",
    ];
    let offenders: Vec<&str> = forbidden.into_iter().filter(|name| identifiers.contains(name)).collect();
    assert!(offenders.is_empty(), "metrics.rs can reach past its atomics: {offenders:?}");
    assert!(identifiers.contains("AtomicU64"), "metrics.rs no longer holds fixed atomics; update this test");
}

#[test]
fn the_model_thread_loop_never_names_metrics() {
    let source = std::fs::read_to_string(repo_root().join("crates/server/src/engine.rs")).unwrap();
    // From the model thread's loop through its event router: everything
    // the model thread runs, up to the telemetry consumer.
    let start = source.find("fn model_thread_loop(").expect("model_thread_loop");
    let end = source.find("async fn telemetry_task(").expect("telemetry_task");
    assert!(start < end, "engine.rs's layout changed; update this test");
    let model_thread = &source[start..end];
    for name in ["fn handle_command(", "fn route_events("] {
        assert!(model_thread.contains(name), "{name} moved out of the scanned span");
    }
    assert!(
        metrics_mentions(model_thread).is_empty(),
        "metrics-aware code in the model thread: {:?}",
        metrics_mentions(model_thread)
    );
}
