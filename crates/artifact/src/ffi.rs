//! FFI declarations for the kernel leaf's device surface (ADR 0001).
//!
//! 1:1 with `kernel/include/ignis_device.h` — flat C: explicit pointers +
//! sizes, `i32` return codes (0 = success, -1 = error), no C++ types
//! across the boundary. Linked by this crate's `build.rs` when the `cuda`
//! feature is on (the default build stays pure Rust).

#![cfg(feature = "cuda")]

use std::os::raw::c_void;

/// Opaque device context (load stream + blocking-sync event; see
/// `kernel/src/device.cu`).
///
/// FFI-safe: `#[repr(C)]` + non-zero-sized so `*mut IgnisDevice` is a valid
/// C pointer across the boundary (ADR 0001 flat C ABI).
#[repr(C)]
pub struct IgnisDevice([u8; 1]);

unsafe extern "C" {
    /// Create a device context (non-blocking load stream + blocking-sync
    /// event). Returns NULL on CUDA error (driver missing, bad device id).
    pub fn ignis_device_create(device_id: i32) -> *mut IgnisDevice;

    /// Allocate `bytes` of device memory; `out_ptr` receives the device
    /// pointer. Returns 0 on success, -1 on CUDA error.
    pub fn ignis_device_alloc(
        d: *mut IgnisDevice,
        bytes: u64,
        out_ptr: *mut *mut c_void,
    ) -> i32;

    /// Enqueue a host->device copy on the load stream (asynchronous: call
    /// [`ignis_device_sync`] before reading the destination).
    pub fn ignis_device_copy_h2d(
        d: *mut IgnisDevice,
        dst: *mut c_void,
        src: *const c_void,
        bytes: u64,
    ) -> i32;

    /// Enqueue a device->host copy on the load stream (asynchronous).
    pub fn ignis_device_copy_d2h(
        d: *mut IgnisDevice,
        dst: *mut c_void,
        src: *const c_void,
        bytes: u64,
    ) -> i32;

    /// Block until the load stream is idle (the blocking-sync event sleeps
    /// instead of spinning).
    pub fn ignis_device_sync(d: *mut IgnisDevice) -> i32;

    /// GitHub #341: enqueue the FP8 row quantizer on the load stream (a BF16
    /// weight on the device to its row-scale payload).
    pub fn ignis_device_fp8_quantize_rows(
        d: *mut IgnisDevice,
        w: *const c_void,
        rows: u32,
        cols: u32,
        payload: *mut c_void,
    ) -> i32;

    /// Free / total device memory in bytes (cudaMemGetInfo).
    pub fn ignis_device_mem_info(
        d: *mut IgnisDevice,
        free_bytes: *mut u64,
        total_bytes: *mut u64,
    ) -> i32;

    /// The CUDA version the installed driver supports (`cudaDriverGetVersion`),
    /// 0 when the machine has no driver (GitHub #329). No context, no device.
    pub fn ignis_device_driver_version(version: *mut i32) -> i32;

    /// The CUDA version this build's runtime is (`cudaRuntimeGetVersion`),
    /// the figure the driver's version is checked against. No context.
    pub fn ignis_device_runtime_version(version: *mut i32) -> i32;

    /// Free / total memory of CUDA device `device_id` as NVML reports it,
    /// read without creating a CUDA context (GitHub #210).
    pub fn ignis_device_nvml_mem_info(
        device_id: i32,
        free_bytes: *mut u64,
        total_bytes: *mut u64,
    ) -> i32;

    /// The name (NVML's, NUL-terminated into `name`) and compute capability
    /// (the CUDA runtime's) of CUDA device `device_id`, read without
    /// creating a CUDA context (GitHub #457).
    pub fn ignis_device_identity(
        device_id: i32,
        name: *mut std::os::raw::c_char,
        name_bytes: u32,
        major: *mut i32,
        minor: *mut i32,
    ) -> i32;

    /// The CUDA architectures this leaf's kernels were compiled for, as the
    /// build named them (`120a`, `86;120a`): a static NUL-terminated string.
    pub fn ignis_build_cuda_architectures() -> *const std::os::raw::c_char;

    /// Free a device allocation returned by [`ignis_device_alloc`]. NULL
    /// device or pointer is a no-op.
    pub fn ignis_device_free(d: *mut IgnisDevice, ptr: *mut c_void);

    /// Allocate `bytes` of mapped pinned host memory (ADR 0062:
    /// `cudaHostAllocMapped | cudaHostAllocPortable`): `out_host` receives the
    /// page-aligned host pointer, `out_device` the pointer a kernel reads it
    /// through. Returns 0 on success, -1 on a CUDA error or a bad argument.
    pub fn ignis_device_host_alloc_mapped(
        d: *mut IgnisDevice,
        bytes: u64,
        out_host: *mut *mut c_void,
        out_device: *mut *mut c_void,
    ) -> i32;

    /// Free a host allocation returned by [`ignis_device_host_alloc_mapped`]
    /// (its host pointer). NULL device or pointer is a no-op.
    pub fn ignis_device_host_free(d: *mut IgnisDevice, host: *mut c_void);

    /// Destroy the context (drains the load stream first). NULL is a no-op.
    pub fn ignis_device_destroy(d: *mut IgnisDevice);
}

/// 1:1 with `struct ignis_paged_kv_plane` (`kernel/include/ignis_paged_kv_budget.h`).
///
/// `dtype` mirrors `ninfer::DType` (`kernel/vendor/src/core/dtype.h`): 0
/// BF16, 1 FP32, 2 I32, 3 U8, 4 I64, 5 I8, 6 FP16, 7 FP8_E4M3FN.
#[repr(C)]
pub struct IgnisPagedKvPlane {
    pub dtype: i32,
    pub leading_extent: i32,
    pub head_extent: i32,
}

unsafe extern "C" {
    /// Host-only: reports the largest page count fitting `vram_budget_bytes`
    /// for a pool built from `planes` (`plane_count` entries), and the bytes
    /// that count of pages consumes. No CUDA device needed (pure arithmetic
    /// over the vendored planner). Returns 0 on success, -1 on a bad
    /// argument.
    pub fn ignis_paged_kv_page_budget(
        planes: *const IgnisPagedKvPlane,
        plane_count: i32,
        vram_budget_bytes: u64,
        out_page_count: *mut u32,
        out_page_bytes: *mut u64,
    ) -> i32;
}