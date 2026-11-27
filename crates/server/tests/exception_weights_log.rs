//! GitHub #682 (ADR 0065): a 27B load says which exception weights it holds,
//! the setting they resolved from and where that setting came from, and on a
//! trellis body, which has no BF16 exception tensor, that it holds none:
//! through the shape the GPU load derives (`load::load_shape`) and the line it
//! logs for its artifact's body (`load::log_exception_weights`), from a real
//! resolution with no config file but the test's own.
//!
//! Its own test binary: a log capture beside tests that reach the same
//! callsite with no subscriber can see nothing (tracing caches a callsite's
//! interest across threads).

use std::sync::Arc;

use ignis_artifact::Body27b;
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

/// The one `ignis.runtime.exception_weights` event a load of `config` from
/// an artifact of `body` logs.
fn exception_weights_event(config: &ignis_server::config::Config, body: Body27b) -> serde_json::Value {
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let shape = tracing::subscriber::with_default(subscriber, || {
        let shape = ignis_server::load::load_shape(config, None);
        ignis_server::load::log_exception_weights(config, &shape, body);
        shape
    });
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

/// A config file of the test's own, `name`, holding `text`.
fn config_file(name: &str, text: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ignis-exception-weights-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn the_load_logs_the_exception_weights_it_holds_and_the_origin_of_their_setting() {
    let file = config_file("named.yaml", "model:\n  qwen38:\n    exception_weights: bf16\n");
    let file = file.to_str().unwrap();
    let profile = config_file("profile.yaml", "profiles:\n  mine:\n    model:\n      qwen38:\n        exception_weights: bf16\n");
    let profile = profile.to_str().unwrap();
    let rtx_5090 = || Card::Read(card::rtx_5090());
    let rtx_5080 = || Card::Read(card::card(16, "NVIDIA GeForce RTX 5080", 12, 0));
    let (nvfp4, trellis) = (Body27b::Nvfp4, Body27b::Trellis);
    for (flags, card, body, held, setting, origin) in [
        (&["--profile", "rtx5090"][..], rtx_5090(), nvfp4, "fp8", "auto", "default"),
        (&["--profile", "rtx5090"], Card::Read(card::rtx_4090()), nvfp4, "bf16", "auto", "default"),
        (&["--profile", "rtx5090"], Card::Absent, nvfp4, "bf16", "auto", "default"),
        (&["--profile", "rtx5090", "--model-exception-weights", "bf16"], rtx_5090(), nvfp4, "bf16", "bf16", "flag"),
        (&["--profile", "rtx5090", "--model-exception-weights", "fp8"], rtx_5090(), nvfp4, "fp8", "fp8", "flag"),
        (&["--profile", "rtx5090", "--config", file], rtx_5090(), nvfp4, "bf16", "bf16", "file"),
        (&["--profile", "mine", "--config", profile], rtx_5090(), nvfp4, "bf16", "bf16", "profile"),
        // The 16 GB class's trellis artifact on an RTX 50 card: `auto` still
        // resolves to fp8, and the line says nothing is held in it.
        (&["--profile", "rtx-16gb"], rtx_5080(), trellis, "fp8", "auto", "default"),
    ] {
        let event = exception_weights_event(&the_27b(flags, card), body);
        let none = if body == trellis { ": the body has no BF16 exception tensor, so none is held" } else { "" };
        assert_eq!(event["body"], format!("exception_weights {setting} -> {held} ({origin}){none}"), "{flags:?}: {event}");
        let attributes = &event["attributes"];
        assert_eq!((&attributes["exception_weights"], &attributes["setting"], &attributes["origin"]), (&held.into(), &setting.into(), &origin.into()), "{flags:?}: {event}");
        assert_eq!(attributes["exception_tensors"], body == nvfp4, "{flags:?}: {event}");
    }
}
