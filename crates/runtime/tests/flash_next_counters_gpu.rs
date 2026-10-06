//! The counters a Flash-Next leaf publishes (GitHub #301, #302): after
//! every step, its cell holds residency's counts and occupancy, read off the
//! device, and the n-gram table's rows -- what the server's exposition
//! renders. A prefill moves the prefill counts and none of decode's, a
//! decode round the reverse; the n-gram rows are the tokens times the
//! table's heads; no class holds more slots than it has.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::flash_next::EngineOptions;
use ignis_core::flash_next_counters::FlashNextCounters;
use ignis_core::gpu_profile;
use ignis_core::residency::{KClass, Phase};
use ignis_core::DecodeParams;
use ignis_runtime::{DecodeLane, FlashNextLeaf, StepLeaf};

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const PROMPT: usize = 300;
const DECODED: usize = 4;

/// Selected projections of `phase`, hits and misses, over every class.
fn selected(reading: &FlashNextCounters, phase: Phase) -> u64 {
    let r = &reading.residency;
    (0..KClass::COUNT).map(|c| r.hits[c][phase.index()] + r.misses[c][phase.index()]).sum()
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn the_leaf_publishes_its_counts_after_every_step() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display()));
        return;
    }
    let options = EngineOptions { max_context_tokens: 4096, ..EngineOptions::default() };
    let leaf = match FlashNextLeaf::open(&path, options) {
        Ok(leaf) => leaf,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("open the Flash-Next leaf: {e}"));
            return;
        }
    };
    let heads = (leaf.ngram_table().token_bytes() / leaf.ngram_table().row_bytes()) as u64;
    let cell = leaf.counters();

    // At open: the pools' capacity, and no step counted yet.
    let opened = cell.read();
    assert!(opened.slots_capacity.iter().any(|&c| c > 0), "{opened:?}");
    assert_eq!((selected(&opened, Phase::Prefill), selected(&opened, Phase::Decode)), (0, 0));
    assert!(opened.slots_in_use.is_some(), "residency reads out its occupancy");

    let model = leaf.load_model().expect("load the model");
    let mut seq = leaf.allocate_sequence(&model, 4096).expect("a free lane");
    let prompt: Vec<u32> = (0..PROMPT as u32).map(|i| 1000 + (i * 7919) % 60_000).collect();
    leaf.prefill(&model, &mut seq, &prompt, 0, DecodeParams::default(), &[], None, None).expect("prefill");

    let prefilled = cell.read();
    assert!(selected(&prefilled, Phase::Prefill) > 0, "{prefilled:?}");
    assert_eq!(selected(&prefilled, Phase::Decode), 0, "a prefill counts no decode selection");
    assert_eq!(prefilled.ngram.rows - opened.ngram.rows, PROMPT as u64 * heads);
    assert!(prefilled.ngram.hot_rows <= prefilled.ngram.rows);

    let lane = DecodeLane { params: DecodeParams::default(), remaining_tokens: 1, stop_ids: &[], permitted: &[] };
    for round in 0..DECODED {
        leaf.decode(&model, &mut [&mut seq], std::slice::from_ref(&lane))
            .unwrap_or_else(|code| panic!("decode round {round}: leaf code {code}"));
    }
    let decoded = cell.read();
    assert!(selected(&decoded, Phase::Decode) > 0, "{decoded:?}");
    assert_eq!(
        selected(&decoded, Phase::Prefill),
        selected(&prefilled, Phase::Prefill),
        "a decode round counts no prefill selection"
    );
    assert_eq!(decoded.ngram.rows - prefilled.ngram.rows, DECODED as u64 * heads);
    let in_use = decoded.slots_in_use.expect("occupancy");
    for class in KClass::ALL {
        let c = class.index();
        assert!(in_use[c] <= decoded.slots_capacity[c], "{}: {} of {}", class.as_str(), in_use[c], decoded.slots_capacity[c]);
    }
    assert!(in_use.iter().any(|&n| n > 0), "the steps' misses filled slots: {in_use:?}");
    println!("counters after {PROMPT} prefilled and {DECODED} decoded: {decoded:?}");

    leaf.release_sequence(&model, seq);
    leaf.release_model(model);
}
