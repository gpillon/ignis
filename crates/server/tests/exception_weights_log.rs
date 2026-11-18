//! GitHub #682 (ADR 0103): a 27B load says which exception weights it holds,
//! the setting they resolved from and where that setting came from, through
//! the seam the GPU load derives its shape at (`load::load_shape`), from a
//! real resolution with no config file but the test's own.
//!
//! Its own test binary: a log capture beside tests that reach the same
//! callsite with no subscriber can see nothing (tracing caches a callsite's
//! interest across threads).

use std::sync::Arc;

use ignis_core::compute::ModelFamily;
use ignis_server::config::profile::Card;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/card.rs"]
mod card;

/// The options of a start with `flags` on `card`, fitted to the 27B, with no
/// environment and only the config file a flag names.
fn the_27b(flags: &[&str], card: Card) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let files = ignis_server::config::file::RealFiles;
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve_on(&args, |_| None, &files, &|| card.clone()).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(ModelFamily::Qwen38_27b).expect("the 27B")
}

/// The one `ignis.runtime.exception_weights` event a load of `config` logs
/// as it derives its shape.
fn exception_weights_event(config: &ignis_server::config::Config) -> serde_json::Value {
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let shape = tracing::subscriber::with_default(subscriber, || ignis_server::load::load_shape(config, None));
    let events: Vec<serde_json::Value> = sink
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.runtime.exception_weights")
        .collect();
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0]["attributes"]["exception_weights"], shape.exception_weights.as_str(), "the shape's: {}", events[0]);
    events[0].clone()
}

/// A config file of the test's own naming the 27B's exception weights.
fn config_file(weights: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-exception-weights-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ignis.config.yaml");
    std::fs::write(&path, format!("model:\n  qwen38:\n    exception_weights: {weights}\n")).unwrap();
    path
}

#[test]
fn the_load_logs_the_exception_weights_it_holds_and_the_origin_of_their_setting() {
    let file = config_file("bf16");
    let file = file.to_str().unwrap();
    let rtx_5090 = || Card::Read(card::rtx_5090());
    for (flags, card, held, setting, origin) in [
        (&["--profile", "rtx5090"][..], rtx_5090(), "fp8", "auto", "default"),
        (&["--profile", "rtx5090"], Card::Read(card::rtx_4090()), "bf16", "auto", "default"),
        (&["--profile", "rtx5090"], Card::Absent, "bf16", "auto", "default"),
        (&["--profile", "rtx5090", "--model-exception-weights", "bf16"], rtx_5090(), "bf16", "bf16", "flag"),
        (&["--profile", "rtx5090", "--model-exception-weights", "fp8"], rtx_5090(), "fp8", "fp8", "flag"),
        (&["--profile", "rtx5090", "--config", file], rtx_5090(), "bf16", "bf16", "file"),
    ] {
        let event = exception_weights_event(&the_27b(flags, card));
        assert_eq!(event["body"], format!("exception_weights {setting} -> {held} ({origin})"), "{flags:?}: {event}");
        let attributes = &event["attributes"];
        assert_eq!((&attributes["exception_weights"], &attributes["setting"], &attributes["origin"]), (&held.into(), &setting.into(), &origin.into()), "{flags:?}: {event}");
    }
}
