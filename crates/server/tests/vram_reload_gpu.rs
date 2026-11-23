//! A reload plans the KV pool its cold start planned (GitHub #324). A load
//! after another in the same process reads NVML's free memory with the CUDA
//! context, and the state the kernels set up once per process, still held,
//! and the VRAM plan charges both again as its `cuda_context` and `residual`
//! lines: before #324 the 27B came back from a switch, or from an idle
//! unload, with 8.2% fewer pages.
//!
//! The 27B is loaded through the production load path (`load::prepare_model`,
//! `load::load_model`), shut down, and loaded again in the same process; the
//! second pool must be the first within 1%, which leaves room for the
//! desktop's own drift between the two readings. And what the reload adds
//! back must not be more than the process holds once the cold load is torn
//! down, within [`HELD_SLACK_BYTES`]: an add-back above it plans more than the
//! card has, and 1% (about 125 pages, 74 MB) would let most of an over-count
//! through. That is read from the process's own WDDM dedicated usage, the
//! counter the held figures came from: NVML's free memory near a full card
//! moves with what WDDM pages in and out for other processes, 84 MB within one
//! reload on 2026-10-15. Its own test binary: the
//! first load must make the process's context, which it measures (GitHub
//! #604), and the plan charges that figure as its `cuda_context` line. Where
//! the driver reports the process's own memory (NVML's per-process figure,
//! native Linux) the reload adds back that figure, capped at what the plan
//! charges again; where it does not (Windows, WSL) it adds back what the plan
//! charges again. The test prints the measured context, the add-back and where
//! it came from, and both loads' pages, on every OS. Run under
//! `scripts/gpu-profile.ps1` only (`IGNIS_GPU_PROFILE=1`). Machine-local path:
//! the 27B at [`ARTIFACT_27B`] (or `IGNIS_ARTIFACT_27B`).
//!
//! GitHub #687: then, in the same process, a two-shape load (DFlash2 at 15
//! drafts) and its reload. Its extended round shape leaves state of its own
//! after the teardown, which the reload adds back: within 1% of the pages
//! again, an add-back no more than the process holds, and, on Windows, no
//! more of what the process holds left un-added than the 7-draft reload left,
//! within [`UNDER_SLACK_BYTES`], so the two-shape reload loses only what the
//! 7-draft one does.
//!
//! GitHub #714 (spec #711): then the same two-shape load with its mixed
//! rounds' graphs asked ([`MIXED_ROUNDS`]`=1`: the twelve mixes at the default
//! 8 lanes, under the shape policy `aggregate` every load here names,
//! [`options`]), and its reload, held as the two-shape ones are; and, on Windows,
//! while each two-shape load is in, what the process holds past its plan: the
//! mixed load no more past its plan than the two-shape load without them,
//! within [`UNDER_SLACK_BYTES`], so the graphs are a line of the plan and do
//! not come out of the headroom.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::CudaDevice;
use ignis_core::gpu_profile;

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL_27B: &str = "qwen3.8-27b";

/// The two-shape load's draft window (GitHub #687).
const TWO_SHAPE_DRAFT_TOKENS: u32 = 15;

/// The switch that asks for the mixed rounds' graphs, read at the plan, the
/// load and its capture (GitHub #714).
const MIXED_ROUNDS: &str = "IGNIS_MIXED_ROUNDS";

/// The one-shape load's draft window, named since the `rtx5090` profile's
/// default became 15 (GitHub #696): the load the reload was first held at.
const SEVEN_DRAFT_TOKENS: u32 = 7;

/// The defaults at a context that fits beside the desktop, with idle unload
/// off (ADR 0049): the reload is the test's. The pool takes the rest of the
/// budget at any context. `draft_tokens` names the DFlash2 window, and the
/// shape policy is named `aggregate`, since the `rtx5090` profile's became
/// `mixed` (GitHub #722): the two-shape load captures no mixed graph unless
/// [`MIXED_ROUNDS`] asks.
fn options(draft_tokens: u32) -> ignis_server::config::Config {
    let mut args: Vec<String> = ["--model-max-context", "16384", "--model-idle-unload", "0", "--spec-backend", "dflash2", "--spec-shape-policy", "aggregate", "--spec-draft-tokens"]
        .map(String::from)
        .to_vec();
    args.push(draft_tokens.to_string());
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => {
            ignis_server::config::Config { model: MODEL_27B.to_owned(), model_named: true, ..config }
        }
        _ => unreachable!("flags without --help are a config"),
    }
}

/// How much more than the process holds the add-back may be (GitHub #324):
/// the 27B's per-process state moves with the kernels a load runs: this
/// test's shape left 2 MiB less of it than the Makefile's, which the constant
/// was, on 2026-10-15, and with the copy-free decode graphs it reads 2 MiB
/// apart from one run to the next (GitHub #680).
#[cfg(windows)]
const HELD_SLACK_BYTES: u64 = 16 << 20;

/// How much more of what the process holds the two-shape reload may leave
/// un-added than the 7-draft reload left (GitHub #687): the 2 MiB a 7-draft
/// load's reading moved from one run to the next (GitHub #680), and by which
/// a process whose first load is at 15 drafts holds more than one that loaded
/// 7 drafts first. Without its extended shape's state the add-back left 18 MiB
/// more.
#[cfg(windows)]
const UNDER_SLACK_BYTES: u64 = 4 << 20;

/// One load of `artifact` at `draft_tokens` ([`options`]), torn down before
/// it returns: its KV pool's pages and the free memory NVML reads once it is
/// in; `None` when the GPU profile says to skip. Its plan's `cuda_context`
/// line is printed, and (GitHub #714) its `mixed_graphs` line and, on
/// Windows, what the process holds past the plan while it is in.
fn one_load(artifact: &Path, draft_tokens: u32) -> Option<(u32, u64)> {
    one_load_held(artifact, draft_tokens).map(|(pages, free, _)| (pages, free))
}

/// [`one_load`], and on Windows what the process holds past the load's plan
/// while it is in (GitHub #714): its WDDM dedicated usage less every line of
/// the plan and its KV pool's pages, a signed figure.
fn one_load_held(artifact: &Path, draft_tokens: u32) -> Option<(u32, u64, Option<i64>)> {
    let loaded = match ignis_server::load::prepare_model(&options(draft_tokens), artifact).and_then(ignis_server::load::load_model) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load {}: {e}", artifact.display()));
            return None;
        }
    };
    let reservations = loaded.reservations.as_ref().expect("a text load reserves its pool");
    println!(
        "27B load: cuda_context {} B, budget {} B, {} KV pages",
        reservations.lines.cuda_context, reservations.budget_bytes, reservations.kv_pool_pages
    );
    let pages = reservations.kv_pool_pages;
    let (free_after_load, _) = CudaDevice::nvml_memory(0).expect("NVML once the load is in");
    let planned = reservations.lines.total() + u64::from(pages) * reservations.kv_page_bytes;
    #[cfg(windows)]
    let past_plan = {
        let held = process_dedicated_bytes();
        let past = held as i64 - planned as i64;
        println!(
            "27B load at {draft_tokens} drafts: mixed_graphs {} B; the process holds {held} B in, {past} B past the plan's {planned} B",
            reservations.lines.mixed_graphs
        );
        Some(past)
    };
    #[cfg(not(windows))]
    let past_plan = {
        println!("27B load at {draft_tokens} drafts: mixed_graphs {} B, plan {planned} B", reservations.lines.mixed_graphs);
        None
    };
    loaded.model.shut_down();
    drop(loaded);
    Some((pages, free_after_load, past_plan))
}

/// This process's dedicated device memory, as WDDM counts it
/// (`\GPU Process Memory(pid_*)\Dedicated Usage`, Task Manager's figure).
#[cfg(windows)]
fn process_dedicated_bytes() -> u64 {
    let counter = format!(
        r"(Get-Counter '\GPU Process Memory(pid_{}_*)\Dedicated Usage').CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object {{ [uint64]$_.Sum }}",
        std::process::id()
    );
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Command", &counter])
        .output()
        .expect("powershell reads the WDDM counter");
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().parse().unwrap_or_else(|_| panic!("the WDDM counter reads as bytes, not {text:?}"))
}

/// What a reload adds back once `torn_down` is, both printed, and on Windows
/// what the process then holds, of which the add-back is no more, within
/// [`HELD_SLACK_BYTES`]: `(added back, held)`.
fn added_back_and_held(context: ignis_runtime::ContextCharge, reading: &str, torn_down: &str) -> (u64, Option<u64>) {
    let process_bytes = CudaDevice::nvml_process_bytes(0);
    let held = ignis_runtime::held_before_load(
        ignis_core::compute::ModelFamily::Qwen38_27b,
        Some(context),
        ignis_runtime::qwen27b_state(),
        process_bytes,
    );
    println!(
        "after {torn_down}: context measured at {} B ({reading}); per-process figure {process_bytes:?}; a reload adds back {} B ({})",
        context.bytes,
        held.bytes,
        held.source.as_str()
    );
    #[cfg(windows)]
    let held_bytes = {
        let added_back = held.bytes;
        let held = process_dedicated_bytes();
        println!("after {torn_down}'s teardown: the process holds {held} B, a reload adds back {added_back} B");
        assert!(
            added_back <= held + HELD_SLACK_BYTES,
            "a reload adds back {added_back} B against the {held} B the process holds after {torn_down}: more than {HELD_SLACK_BYTES} B over"
        );
        Some(held)
    };
    #[cfg(not(windows))]
    let held_bytes = None;
    (held.bytes, held_bytes)
}

#[test]
#[ignore = "GPU profile only: the real 27B artifact, and the card exclusively"]
fn a_reload_plans_the_kv_pool_its_cold_start_planned() {
    let artifact = std::env::var("IGNIS_ARTIFACT_27B").unwrap_or_else(|_| ARTIFACT_27B.to_owned());
    let artifact = Path::new(&artifact);
    if !artifact.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {}", artifact.display())) {
        return;
    }
    assert!(!CudaDevice::context_created(), "the cold load must make this process's context");
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().expect("runtime");
    let _guard = runtime.enter();

    let Some((cold, cold_free)) = one_load(artifact, SEVEN_DRAFT_TOKENS) else {
        return;
    };
    assert!(CudaDevice::context_created(), "the cold load made the context, and the teardown kept it");
    assert!(ignis_runtime::qwen27b_loaded(), "the cold load left the 27B's per-process state behind");
    let reading = CudaDevice::context_reading().expect("the cold load made the context");
    let context = ignis_runtime::ContextCharge::of(reading);
    assert_eq!(context.source, ignis_runtime::ContextSource::Measured, "the context was not measured: {reading}");
    let (seven_added_back, seven_held) = added_back_and_held(context, &reading.to_string(), "the cold load");
    let (reload, reload_free) = one_load(artifact, SEVEN_DRAFT_TOKENS).expect("the reload loads what the cold start loaded");
    println!("27B KV pool: cold {cold} pages, reload {reload} pages; NVML free once in: cold {cold_free} B, reload {reload_free} B");
    assert!(
        cold.abs_diff(reload) <= cold / 100,
        "the reload planned {reload} pages against the cold start's {cold}: more than 1% apart"
    );

    // GitHub #687: a two-shape load, planned on what the 7-draft loads left,
    // then torn down, and its reload.
    let (two_shape, _, two_shape_past) =
        one_load_held(artifact, TWO_SHAPE_DRAFT_TOKENS).expect("a two-shape load after the 7-draft ones");
    let (two_shape_added_back, two_shape_held) = added_back_and_held(context, &reading.to_string(), "the two-shape load");
    let (two_shape_reload, _) = one_load(artifact, TWO_SHAPE_DRAFT_TOKENS).expect("the two-shape load's reload");
    println!(
        "27B KV pool at {TWO_SHAPE_DRAFT_TOKENS} drafts: {two_shape} pages after the 7-draft loads, reload {two_shape_reload} pages"
    );
    assert!(
        two_shape.abs_diff(two_shape_reload) <= two_shape / 100,
        "the two-shape reload planned {two_shape_reload} pages against the two-shape load's {two_shape}: more than 1% apart"
    );
    if let (Some(seven_held), Some(two_shape_held)) = (seven_held, two_shape_held) {
        let seven_under = seven_held.saturating_sub(seven_added_back);
        let two_shape_under = two_shape_held.saturating_sub(two_shape_added_back);
        println!("left un-added: {seven_under} B after the 7-draft load, {two_shape_under} B after the two-shape load");
        #[cfg(windows)]
        assert!(
            two_shape_under <= seven_under + UNDER_SLACK_BYTES,
            "the two-shape reload leaves {two_shape_under} B the process holds un-added, the 7-draft reload {seven_under} B: \
             more than {UNDER_SLACK_BYTES} B more"
        );
    }

    // GitHub #714: the two-shape load with its mixed graphs, then torn down,
    // and its reload.
    // SAFETY (test-only): this binary's one test reads and writes the
    // environment on its own thread, where the leaf reads it too.
    unsafe { std::env::set_var(MIXED_ROUNDS, "1") };
    let (mixed, _, mixed_past) = one_load_held(artifact, TWO_SHAPE_DRAFT_TOKENS).expect("a mixed load after the two-shape ones");
    let (mixed_added_back, mixed_held) = added_back_and_held(context, &reading.to_string(), "the mixed load");
    let (mixed_reload, _) = one_load(artifact, TWO_SHAPE_DRAFT_TOKENS).expect("the mixed load's reload");
    // SAFETY: as above.
    unsafe { std::env::remove_var(MIXED_ROUNDS) };
    println!("27B KV pool at {TWO_SHAPE_DRAFT_TOKENS} drafts with the mixed graphs: {mixed} pages, reload {mixed_reload} pages");
    assert!(
        mixed.abs_diff(mixed_reload) <= mixed / 100,
        "the mixed reload planned {mixed_reload} pages against the mixed load's {mixed}: more than 1% apart"
    );
    if let (Some(two_shape_held), Some(mixed_held)) = (two_shape_held, mixed_held) {
        let two_shape_under = two_shape_held.saturating_sub(two_shape_added_back);
        let mixed_under = mixed_held.saturating_sub(mixed_added_back);
        println!(
            "left un-added: {two_shape_under} B after the two-shape load, {mixed_under} B after the mixed load; \
             the mixed load's teardown left {} B more than the two-shape load's",
            mixed_held as i64 - two_shape_held as i64
        );
        #[cfg(windows)]
        assert!(
            mixed_under <= two_shape_under + UNDER_SLACK_BYTES,
            "the mixed reload leaves {mixed_under} B the process holds un-added, the two-shape reload {two_shape_under} B: \
             more than {UNDER_SLACK_BYTES} B more"
        );
    }
    if let (Some(two_shape_past), Some(mixed_past)) = (two_shape_past, mixed_past) {
        println!("past the plan while in: {two_shape_past} B at the two-shape load, {mixed_past} B at the mixed load");
        assert!(
            mixed_past <= two_shape_past + UNDER_SLACK_BYTES as i64,
            "the mixed load holds {mixed_past} B past its plan, the two-shape load {two_shape_past} B: its graphs came out of the headroom"
        );
    }
}
