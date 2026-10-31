//! GPU coverage for `CudaDevice::nvml_memory` (GitHub #210): the free memory
//! a VRAM budget is derived from counts what this process holds, its CUDA
//! context included, the way nvidia-smi and Task Manager do -- and the
//! context outlives the device that made it (GitHub #324).
//!
//! NVML counts a fresh allocation late (GitHub #424): on the RTX 5090 under
//! WDDM, the free reading right after `cudaMalloc` of 1 GiB missed its last
//! 64 to 448 MiB in 12 of 40 allocations, and a later reading had all of it
//! (27 and 63 ms later, where a probe timed it). So the allocation's reading
//! waits for the count, for at most [`COUNTED_WITHIN`]; its CPU tests feed it
//! injected readings.

use std::time::Duration;

#[cfg(feature = "cuda")]
use ignis_artifact::{CudaDevice, Device};
#[cfg(feature = "cuda")]
use ignis_core::gpu_profile;

const GIB: u64 = 1024 * 1024 * 1024;

/// How long a reading waits for NVML to count an allocation in full: some
/// thirty times the longest wait seen (GitHub #424).
const COUNTED_WITHIN: Duration = Duration::from_secs(2);

/// The free memory `read` reports once it counts `bytes` allocated with
/// `free_before` free, or its last reading if it has not counted them within
/// `within`.
fn free_once_counted(mut read: impl FnMut() -> u64, free_before: u64, bytes: u64, within: Duration) -> u64 {
    let deadline = std::time::Instant::now() + within;
    loop {
        let free = read();
        if free_before.saturating_sub(free) >= bytes || std::time::Instant::now() >= deadline {
            return free;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Readings in turn, the last one repeated: NVML as a test injects it.
fn readings(values: &[u64]) -> impl FnMut() -> u64 + '_ {
    let mut next = 0;
    move || {
        let value = values[next.min(values.len() - 1)];
        next += 1;
        value
    }
}

#[test]
fn an_allocation_counted_late_is_read_once_it_is_counted() {
    // GitHub #424's run: 1 GiB taken from 20 GiB free, 960 MiB of it counted
    // at the first reading after the allocation and the rest one reading later.
    let free_before = 20 * GIB;
    let free = free_once_counted(
        readings(&[free_before - (960 << 20), free_before - GIB]),
        free_before,
        GIB,
        COUNTED_WITHIN,
    );
    assert_eq!(free, free_before - GIB);
}

#[test]
fn an_allocation_never_counted_is_read_as_it_is_once_the_wait_runs_out() {
    // A free reading that does not count the allocation must still fail the
    // GPU test: the wait ends and hands back what NVML said.
    let free_before = 20 * GIB;
    let started = std::time::Instant::now();
    let free = free_once_counted(readings(&[free_before]), free_before, GIB, Duration::from_millis(20));
    assert_eq!(free, free_before);
    assert!(started.elapsed() >= Duration::from_millis(20), "the wait ended early: {:?}", started.elapsed());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn nvml_free_memory_counts_the_context_and_every_allocation() {
    let (free_before, total) = match CudaDevice::nvml_memory(0) {
        Ok(memory) => memory,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("NVML unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    assert!(free_before > 0 && free_before <= total, "{free_before} of {total}");

    // This binary's one GPU test: nothing has made the context yet.
    assert!(!CudaDevice::context_created(), "a context before the first device");
    let mut device = CudaDevice::create(0).expect("CUDA device");
    assert!(CudaDevice::context_created(), "the first device made the context");
    let (free_with_context, _) = CudaDevice::nvml_memory(0).expect("NVML after the context");
    // The context holds hundreds of MiB; `cudaMemGetInfo` would not see it.
    assert!(
        free_with_context < free_before,
        "creating the context left NVML free memory at {free_with_context} (was {free_before})"
    );

    let buffer = device.allocate(GIB).expect("1 GiB");
    let free_with_buffer = free_once_counted(
        || CudaDevice::nvml_memory(0).expect("NVML after the allocation").0,
        free_with_context,
        GIB,
        COUNTED_WITHIN,
    );
    assert!(
        free_with_context.saturating_sub(free_with_buffer) >= GIB,
        "a 1 GiB allocation moved NVML free memory by {} bytes within {COUNTED_WITHIN:?}",
        free_with_context.saturating_sub(free_with_buffer)
    );
    device.deallocate(buffer).expect("free");

    // GitHub #324: dropping the device keeps the context, so the next load's
    // free reading does not count it -- what `context_created` tells a plan.
    drop(device);
    let (free_after_drop, _) = CudaDevice::nvml_memory(0).expect("NVML after the drop");
    assert!(
        free_before.saturating_sub(free_after_drop) >= 256 << 20,
        "the dropped device gave its context back: NVML free {free_after_drop}, {free_before} before the context"
    );
    assert!(CudaDevice::context_created(), "the context outlives the device");
}
