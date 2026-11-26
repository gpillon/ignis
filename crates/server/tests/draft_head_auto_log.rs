//! ADR 0063, spec 27b-16gb/04 AC 5 (GitHub #336): a 27B load whose
//! drafter's head is left to `auto` says what `auto` resolved to, through the
//! seam the GPU load derives its shape at (`load::load_shape`) from a real
//! resolution; a load that names its head, or leaves it unset, says nothing.
//!
//! Its own test binary: a log capture beside tests that reach the same
//! callsite with no subscriber can see nothing (tracing caches a callsite's
//! interest across threads).

use std::sync::Arc;

use ignis_core::compute::ModelFamily;
use tracing_subscriber::layer::SubscriberExt;

/// The options of a start with `flags`, fitted to the 27B.
fn the_27b(flags: &[&str]) -> ignis_server::config::Config {
    fitted(flags, ModelFamily::Qwen38_27b)
}

/// The options of a start with `flags`, fitted to `family`.
fn fitted(flags: &[&str], family: ModelFamily) -> ignis_server::config::Config {
    let args: Vec<String> = flags.iter().map(|flag| flag.to_string()).collect();
    let ignis_server::config::ConfigOutcome::Config(config) =
        ignis_server::config::resolve(&args, |_| None).expect("resolve")
    else {
        panic!("expected a runnable config");
    };
    config.for_family(family).expect("the family's start")
}

/// The `ignis.runtime.draft_head` events a load of `config` logs as it
/// derives its shape.
fn draft_head_events(config: &ignis_server::config::Config) -> Vec<serde_json::Value> {
    let sink = Arc::new(ignis_logging::MemorySink::new());
    let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
    let shape = tracing::subscriber::with_default(subscriber, || ignis_server::load::load_shape(config, None));
    assert_eq!(shape, ignis_server::runtime::EngineShape::from(config), "the resolution's shape, vision off");
    sink.lines()
        .iter()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.runtime.draft_head")
        .collect()
}

#[test]
fn a_head_left_to_auto_says_it_is_the_full_head_and_a_named_or_unset_one_says_nothing() {
    // `rtx50-16gb` leaves the head to `auto`; a flag can name `auto` too.
    let named_auto = ["--profile", "rtx5090", "--spec-backend", "dflash2", "--spec-draft-tokens", "7", "--spec-draft-head", "auto"];
    for flags in [&["--profile", "rtx50-16gb"][..], &named_auto] {
        let events = draft_head_events(&the_27b(flags));
        assert_eq!(events.len(), 1, "{flags:?}: {events:?}");
        let event = &events[0];
        assert_eq!(event["body"], "draft_head auto -> full", "{event}");
        assert_eq!(event["attributes"]["setting"], "auto", "{event}");
        assert_eq!(event["attributes"]["draft_head"], "full", "{event}");
    }
    for flags in [
        // The profile's speculation is the 27B's section's, and so is a head
        // named over it.
        &["--profile", "rtx50-16gb", "--qwen38-spec-draft-head", "full"][..],
        &["--profile", "rtx50-16gb", "--qwen38-spec-draft-head", "shortlist"],
        &["--profile", "rtx5090", "--spec-backend", "dflash2", "--spec-draft-tokens", "7"],
        // No drafter, no head to say.
        &["--profile", "rtx5090"],
    ] {
        assert_eq!(draft_head_events(&the_27b(flags)), Vec::<serde_json::Value>::new(), "{flags:?}");
    }
    // GitHub #729: an MTP head, on either model, ignores the head setting and
    // says nothing of it.
    let mtp = ["--profile", "rtx5090", "--spec-backend", "mtp", "--spec-draft-head", "auto"];
    for family in [ModelFamily::Qwen38_27b, ModelFamily::FlashNext] {
        assert_eq!(draft_head_events(&fitted(&mtp, family)), Vec::<serde_json::Value>::new(), "{family:?}'s MTP head");
    }
}
