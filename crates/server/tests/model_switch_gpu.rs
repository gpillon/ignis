//! The runtime model switch on the card (spec model-switch/01, GitHub #305):
//! one process serves the 27B, switches to Qwen3.8-Flash-Next and back
//! through its own HTTP surface (`POST /v1/models/switch`), and each
//! direction's wall time — the request to `GET /v1/models` reporting the new
//! id `serving` — is printed: the measured replacement for the 14-15 s /
//! 8-10 s estimates in `docs/specs/flash-next/phase2-model-switch-notes.md`.
//!
//! The direct proof that each old model was torn down before the next one
//! loaded is that the next load succeeded at all — its pinned KV-RAM arena's
//! create refuses while one exists — and that the arena standing after each
//! switch is the one the new load pinned, of the size the start options
//! name (`ignis_core::seq::host_pool_stats`).
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
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = ["--max-context", "16384"].map(String::from).to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        _ => unreachable!("flags without --help are a config"),
    }
}

#[test]
#[ignore = "GPU profile only: the real 27B and Flash-Next artifacts, and the card exclusively"]
fn the_27b_and_flash_next_switch_back_and_forth_on_one_process() {
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
    println!("host pool after the 27B's start (capacity, used): {:?}", ignis_core::seq::host_pool_stats());

    let to_flash_next = live.switch_to(&flash_next, MODEL_FLASH_NEXT);
    let after_to = ignis_core::seq::host_pool_stats();
    println!("27B -> Flash-Next: {to_flash_next:?}; host pool (capacity, used) {after_to:?}");
    live.completes(MODEL_FLASH_NEXT);

    let back = live.switch_to(&artifact_27b, MODEL_27B);
    let after_back = ignis_core::seq::host_pool_stats();
    println!("Flash-Next -> 27B: {back:?}; host pool (capacity, used) {after_back:?}");
    live.completes(MODEL_27B);

    // The new load pinned its own arena, of the options' size: its create
    // succeeded, so the old arena was destroyed first. `used` is printed
    // above, not asserted: what a fresh load places there is the leaf's.
    for (direction, (capacity, _)) in [("to Flash-Next", after_to), ("back to the 27B", after_back)] {
        assert_eq!(capacity, arena, "{direction}: the arena standing is the new load's");
    }
}
