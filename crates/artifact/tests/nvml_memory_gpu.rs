//! GPU coverage for `CudaDevice::nvml_memory` (GitHub #210): the free memory
//! a VRAM budget is derived from counts what this process holds, its CUDA
//! context included, the way nvidia-smi and Task Manager do.

#![cfg(feature = "cuda")]

use ignis_artifact::{CudaDevice, Device};
use ignis_core::gpu_profile;

const GIB: u64 = 1024 * 1024 * 1024;

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

    let mut device = CudaDevice::create(0).expect("CUDA device");
    let (free_with_context, _) = CudaDevice::nvml_memory(0).expect("NVML after the context");
    // The context holds hundreds of MiB; `cudaMemGetInfo` would not see it.
    assert!(
        free_with_context < free_before,
        "creating the context left NVML free memory at {free_with_context} (was {free_before})"
    );

    let buffer = device.allocate(GIB).expect("1 GiB");
    let (free_with_buffer, _) = CudaDevice::nvml_memory(0).expect("NVML after the allocation");
    assert!(
        free_with_context - free_with_buffer >= GIB,
        "a 1 GiB allocation moved NVML free memory by {} bytes",
        free_with_context - free_with_buffer
    );
    device.deallocate(buffer).expect("free");
}
