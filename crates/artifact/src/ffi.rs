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

    /// Free / total device memory in bytes (cudaMemGetInfo).
    pub fn ignis_device_mem_info(
        d: *mut IgnisDevice,
        free_bytes: *mut u64,
        total_bytes: *mut u64,
    ) -> i32;

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