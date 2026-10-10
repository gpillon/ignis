//! The runtime model switch on the card (spec model-switch/01, GitHub #305):
//! one process serves the 27B, switches to Qwen3.8-Flash-Next and back
//! through its own HTTP surface (`POST /v1/models/switch`), and each
//! direction's wall time — the request to `GET /v1/models` reporting the new
//! id `serving` — is printed: the measured replacement for the 14-15 s /
//! 8-10 s estimates in `docs/specs/flash-next/phase2-model-switch-notes.md`.
//!
//! The direct proof that the teardown ran is the 27B's KV-RAM arena, the
//! process-wide one `ignis_core::seq::host_pool_stats` reads (Flash-Next
//! pins an arena of its own instance, which that does not see): pinned at
//! the options' size while the 27B serves, gone (0) once the switch to
//! Flash-Next has dropped the 27B, and pinned again — a create that refuses
//! while one exists — after the switch back.
//!
//! A measurement, not a gate: it prints its numbers rather than asserting a
//! threshold, like the G2/G4 instruments. Run under `scripts/gpu-profile.ps1`
//! only (`IGNIS_GPU_PROFILE=1`): it needs both artifacts and the card
//! exclusively. Machine-local paths: the 27B at [`ARTIFACT_27B`] (or
//! `IGNIS_ARTIFACT_27B`), Flash-Next under [`FLASH_NEXT_DIR`] (or
//! `IGNIS_FLASH_NEXT_DIR`).

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::gpu_profile;

#[path = "support/mod.rs"]
mod support;

use support::live_server::SwitchingLiveServer;

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL_27B: &str = "qwen3.8-27b";
const MODEL_FLASH_NEXT: &str = "qwen3.8-flash-next";

/// The start options both models load with: the defaults, at a context both
/// serve, so the measurement is of the switch rather than of a large pool.
///
/// Flash-Next's host plan (expert pool, n-gram hot rows, prompt reuse's
/// retained slots and KV-RAM arena) is what the host plan check
/// (`crates/core/src/residency/plan.rs`) sizes against free physical memory;
/// the expert pool alone (~35 GiB) is not a knob, and the KV-RAM arena is
/// what this test's own assertions pin on the 27B side (`host_pool_stats`),
/// so neither is touched. `--ngram-hot-bytes` is Flash-Next-only and this
/// same config starts the 27B first (a switch, not a start, is what drops a
/// target-incompatible flag -- `config::fit_to_family`), so it cannot be
/// named here at all. Prompt reuse's retained slots are not part of what is
/// measured here (it is the switch's wall time and the 27B's arena, not
/// Flash-Next's reuse), so they are zeroed to leave the host plan's margin
/// check a little more room on a machine running other things at the same
/// time.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = ["--max-context", "16384", "--retained-host", "0"].map(String::from).to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("flags without --help are a config"),
    }
}

#[test]
#[ignore = "GPU profile only: the real 27B and Flash-Next artifacts, and the card exclusively"]
fn the_27b_and_flash_next_switch_back_and_forth_on_one_process() {
    let _ = tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).try_init();
    let artifact_27b = std::env::var("IGNIS_ARTIFACT_27B").unwrap_or_else(|_| ARTIFACT_27B.to_owned());
    let flash_next_dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let flash_next = flash_next_dir.join(ARTIFACT_FILE_NAME);
    if !flash_next.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", flash_next.display())) {
        return;
    }
    let flash_next = flash_next.to_string_lossy().into_owned();
    let options = options();
    let arena = options.host_pool_bytes;

    let started = std::time::Instant::now();
    let Some(live) = SwitchingLiveServer::start(options, &artifact_27b, MODEL_27B) else {
        return;
    };
    println!("27B cold start to serving: {:?}", started.elapsed());
    live.completes(MODEL_27B);
    let at_start = ignis_core::seq::host_pool_stats();
    println!("27B arena after its start (capacity, used): {at_start:?}");

    let to_flash_next = live.switch_to(&flash_next, MODEL_FLASH_NEXT);
    let after_to = ignis_core::seq::host_pool_stats();
    println!("27B -> Flash-Next: {to_flash_next:?}; 27B arena (capacity, used) {after_to:?}");
    live.completes(MODEL_FLASH_NEXT);

    let back = live.switch_to(&artifact_27b, MODEL_27B);
    let after_back = ignis_core::seq::host_pool_stats();
    println!("Flash-Next -> 27B: {back:?}; 27B arena (capacity, used) {after_back:?}");
    live.completes(MODEL_27B);

    // `used` is printed, not asserted: what a load places in its arena is
    // the leaf's business. The capacity is the teardown's proof.
    assert_eq!(at_start.0, arena, "the 27B pinned the options' arena");
    assert_eq!(after_to.0, 0, "the switch to Flash-Next dropped the 27B and its arena with it");
    assert_eq!(after_back.0, arena, "the switch back pinned the 27B's arena again");
}
