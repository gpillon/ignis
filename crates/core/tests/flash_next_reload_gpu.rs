//! A dropped Flash-Next engine gives back the device memory it took (spec
//! flash-next/04 AC12, spec flash-next/03 AC8): the model switch of phase 2
//! reloads in one process, and so do the GPU tests that load the engine
//! one after another (`flash_next_forward_gpu` runs four loads).
//!
//! - **Load, run, drop, three times.** Free device memory after each drop
//!   comes back within [`TOLERANCE`] of what was free before the first load,
//!   and the drops agree with each other within [`TOLERANCE`]. Each load runs
//!   a short prompt, so its kernels and their lazily loaded modules are in.
//!
//! Machine-local: `F:/ai/models/Qwen3.8-Flash-Next-ignis/` (or
//! `IGNIS_FLASH_NEXT_DIR`). Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a
//! failure. Needs ~38 GB of free RAM for the pinned expert pool.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{CudaDevice, Device};
use ignis_core::flash_next::{EngineOptions, FlashNextEngine};
use ignis_core::gpu_profile;

const MODEL_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
/// What a drop may leave behind, or another process may take meanwhile:
/// a load holds tens of GB, a leak of any of its buffers is far above it.
const TOLERANCE: u64 = 256 << 20;
const LOADS: usize = 3;

fn model_dir() -> PathBuf {
    std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(MODEL_DIR), PathBuf::from)
}

fn mib(bytes: i64) -> f64 {
    bytes as f64 / f64::from(1 << 20)
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn a_dropped_engine_gives_back_its_device_memory() {
    let dir = model_dir();
    if !dir.join(ARTIFACT_FILE_NAME).exists() {
        gpu_profile::skip_or_fail(&format!("no Flash-Next artifact in {}", dir.display()));
        return;
    }
    // The probe holds the context the loads share, so every reading
    // counts the same things.
    let probe = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA device: {e}"));
            return;
        }
    };
    let free = || probe.free_bytes().expect("cudaMemGetInfo") as i64;
    let baseline = free();
    let mut after_drops = Vec::new();
    for load in 1..=LOADS {
        let before = free();
        let options = EngineOptions { max_context_tokens: 8192, ..EngineOptions::default() };
        let mut engine = match FlashNextEngine::load(&dir, options) {
            Ok(engine) => engine,
            Err(e) => {
                gpu_profile::skip_or_fail(&format!("load {load} of the Flash-Next engine: {e}"));
                return;
            }
        };
        engine.last_logits(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap_or_else(|e| panic!("load {load}: a prompt: {e}"));
        let loaded = free();
        drop(engine);
        let dropped = free();
        eprintln!(
            "load {load}: held {:.0} MiB, the drop gave back {:.0} MiB, {:.0} MiB below the baseline after it",
            mib(before - loaded),
            mib(dropped - loaded),
            mib(baseline - dropped)
        );
        after_drops.push(dropped);
    }
    for (i, &dropped) in after_drops.iter().enumerate() {
        assert!(
            baseline - dropped <= TOLERANCE as i64,
            "after drop {} free memory is {:.0} MiB below the baseline (tolerance {:.0} MiB): the drop path leaks",
            i + 1,
            mib(baseline - dropped),
            mib(TOLERANCE as i64)
        );
        assert!(
            (dropped - after_drops[0]).abs() <= TOLERANCE as i64,
            "drop {} and drop 1 differ by {:.0} MiB: each load leaks",
            i + 1,
            mib(dropped - after_drops[0])
        );
    }
}
