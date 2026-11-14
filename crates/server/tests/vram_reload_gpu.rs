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

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::CudaDevice;
use ignis_core::gpu_profile;

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL_27B: &str = "qwen3.8-27b";

/// The defaults at a context that fits beside the desktop, with idle unload
/// off (ADR 0049): the reload is the test's. The pool takes the rest of the
/// budget at any context.
fn options() -> ignis_server::config::Config {
    let args: Vec<String> = ["--model-max-context", "16384", "--model-idle-unload", "0"].map(String::from).to_vec();
    match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => {
            ignis_server::config::Config { model: MODEL_27B.to_owned(), model_named: true, ..config }
        }
        _ => unreachable!("flags without --help are a config"),
    }
}

/// How much more than the process holds the add-back may be (GitHub #324):
/// the 27B's per-process state moves with the kernels a load runs, and this
/// test's shape leaves 2 MiB less of it than the Makefile's, which the
/// constant is.
#[cfg(windows)]
const HELD_SLACK_BYTES: u64 = 16 << 20;

/// One load of `artifact`, torn down before it returns: its KV pool's pages
/// and the free memory NVML reads once it is in; `None` when the GPU profile
/// says to skip. Its plan's `cuda_context` line is printed.
fn one_load(artifact: &Path) -> Option<(u32, u64)> {
    let loaded = match ignis_server::load::prepare_model(&options(), artifact).and_then(ignis_server::load::load_model) {
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
    loaded.model.shut_down();
    drop(loaded);
    Some((pages, free_after_load))
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

    let Some((cold, cold_free)) = one_load(artifact) else {
        return;
    };
    assert!(CudaDevice::context_created(), "the cold load made the context, and the teardown kept it");
    assert!(ignis_runtime::qwen27b_loaded(), "the cold load left the 27B's per-process state behind");
    let reading = CudaDevice::context_reading().expect("the cold load made the context");
    let context = ignis_runtime::ContextCharge::of(reading);
    assert_eq!(context.source, ignis_runtime::ContextSource::Measured, "the context was not measured: {reading}");
    let process_bytes = CudaDevice::nvml_process_bytes(0);
    let held = ignis_runtime::held_before_load(ignis_core::compute::ModelFamily::Qwen38_27b, Some(context), true, process_bytes);
    println!(
        "context measured at {} B ({reading}); per-process figure {process_bytes:?}; a reload adds back {} B ({})",
        context.bytes,
        held.bytes,
        held.source.as_str()
    );
    #[cfg(windows)]
    {
        let added_back = held.bytes;
        let held = process_dedicated_bytes();
        println!("after the cold load's teardown: the process holds {held} B, a reload adds back {added_back} B");
        assert!(
            added_back <= held + HELD_SLACK_BYTES,
            "a reload adds back {added_back} B against the {held} B the process holds: more than {HELD_SLACK_BYTES} B over"
        );
    }
    let (reload, reload_free) = one_load(artifact).expect("the reload loads what the cold start loaded");
    println!("27B KV pool: cold {cold} pages, reload {reload} pages; NVML free once in: cold {cold_free} B, reload {reload_free} B");
    assert!(
        cold.abs_diff(reload) <= cold / 100,
        "the reload planned {reload} pages against the cold start's {cold}: more than 1% apart"
    );
}
