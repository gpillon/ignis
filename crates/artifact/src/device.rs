//! Device abstraction for materialization (the minimal testable surface).
//!
//! The materializer's contract: one arena allocation, host->device /
//! device->host copies at explicit offsets, synchronization, and optional
//! memory stats.
//!
//! - [`CpuDevice`] (always available): a `Vec<u8>`-backed mock — the ADR 0006
//!   stand-in while the RTX 5090 is held by the reference runner.
//! - [`CudaDevice`] (feature `cuda`): links the kernel leaf's flat C device
//!   surface (`kernel/include/ignis_device.h`, ADR 0001) and uploads to real
//!   VRAM.
//!
//! A [`DeviceBuffer`] points into memory owned by the `Device` that produced
//! it: keep the device alive for as long as the [`crate::MaterializedArtifact`]
//! (and its typed views) are in use.

use crate::{fail, Result};

#[cfg(feature = "cuda")]
use std::os::raw::c_void;

/// An opaque device allocation (the materializer's single arena chunk).
///
/// Never dereferenced by this crate: the base points into the producing
/// device's memory (a host `Vec` for [`CpuDevice`], a `cudaMalloc` chunk
/// for `CudaDevice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceBuffer {
    base: *mut u8,
    bytes: u64,
}

impl DeviceBuffer {
    pub(crate) fn new(base: *mut u8, bytes: u64) -> Self {
        Self { base, bytes }
    }

    /// Base pointer of this allocation (device or host memory, depending on
    /// the `Device` implementation).
    pub fn base_ptr(&self) -> *const u8 {
        self.base as *const u8
    }

    /// Exact length of this allocation in bytes.
    pub fn len(&self) -> u64 {
        self.bytes
    }

    /// Pointer at `offset` inside this allocation (the materializer
    /// bounds-checks against [`DeviceBuffer::len`]).
    pub(crate) fn offset_ptr(&self, offset: u64) -> *const u8 {
        (unsafe { self.base.add(offset as usize) }) as *const u8
    }

    /// Mutable pointer at `offset` inside this allocation.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn offset_mut_ptr(&self, offset: u64) -> *mut u8 {
        unsafe { self.base.add(offset as usize) }
    }
}

/// A device memory allocator + copy engine (the materializer's contract).
pub trait Device {
    /// Allocate a `bytes`-byte region (the materializer's arena).
    fn allocate(&mut self, bytes: u64) -> Result<DeviceBuffer>;

    /// Copy `src` into `dst` at `dst_offset` (host -> device; may be
    /// asynchronous — call [`Device::synchronize`] before use).
    fn copy_h2d(&mut self, dst: &DeviceBuffer, dst_offset: u64, src: &[u8]) -> Result<()>;

    /// Copy `dst` from `src` at `src_offset` (device -> host; verification).
    fn copy_d2h(&self, src: &DeviceBuffer, src_offset: u64, dst: &mut [u8]) -> Result<()>;

    /// Block until all pending copies complete.
    fn synchronize(&mut self) -> Result<()>;

    /// Free memory in bytes (None: the implementation cannot report it).
    fn free_bytes(&self) -> Option<u64> {
        None
    }

    /// Total memory in bytes (None: the implementation cannot report it).
    fn total_bytes(&self) -> Option<u64> {
        None
    }
}

/// A host-memory mock device (backed by `Vec<u8>` arenas).
///
/// No GPU required — all placement, stats, and verification tests run
/// against it (ADR 0006: exclusive GPU; the RTX 5090 is held by the
/// reference `ninfer-serve`).
#[derive(Default)]
pub struct CpuDevice {
    storage: Vec<Vec<u8>>,
    used: u64,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bytes allocated across all arenas.
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    fn storage_at(&self, base: *const u8) -> Option<&Vec<u8>> {
        self.storage.iter().find(|v| v.as_ptr() == base)
    }

    fn storage_at_mut(&mut self, base: *const u8) -> Option<&mut Vec<u8>> {
        self.storage.iter_mut().find(|v| v.as_ptr() == base)
    }

    fn bounds(&self, buf: &DeviceBuffer, offset: u64, len: usize) -> Result<(usize, usize)> {
        let start = usize::try_from(offset)
            .map_err(|_| fail("device copy offset overflows usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| fail("device copy range overflows usize"))?;
        if end > buf.len() as usize {
            return Err(fail("device copy extends beyond the allocation"));
        }
        Ok((start, end))
    }
}

impl Device for CpuDevice {
    fn allocate(&mut self, bytes: u64) -> Result<DeviceBuffer> {
        let len = usize::try_from(bytes)
            .map_err(|_| fail("device allocation size overflows usize"))?;
        let mut buffer = vec![0u8; len];
        let base = buffer.as_mut_ptr();
        self.storage.push(buffer);
        self.used = self
            .used
            .checked_add(bytes)
            .ok_or_else(|| fail("device allocation count overflows u64"))?;
        Ok(DeviceBuffer::new(base, bytes))
    }

    fn copy_h2d(&mut self, dst: &DeviceBuffer, dst_offset: u64, src: &[u8]) -> Result<()> {
        let (start, end) = self.bounds(dst, dst_offset, src.len())?;
        let storage = self
            .storage_at_mut(dst.base_ptr())
            .ok_or_else(|| fail("device buffer was not allocated by this device"))?;
        storage[start..end].copy_from_slice(src);
        Ok(())
    }

    fn copy_d2h(&self, src: &DeviceBuffer, src_offset: u64, dst: &mut [u8]) -> Result<()> {
        let (start, end) = self.bounds(src, src_offset, dst.len())?;
        let storage = self
            .storage_at(src.base_ptr())
            .ok_or_else(|| fail("device buffer was not allocated by this device"))?;
        dst.copy_from_slice(&storage[start..end]);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CUDA device (feature `cuda`): the kernel leaf's flat C surface (ADR 0001),
// declared in `crate::ffi` (1:1 with `kernel/include/ignis_device.h`).
// ---------------------------------------------------------------------------

/// A CUDA device context (a dedicated non-blocking load stream + a
/// blocking-sync event — the reference `DeviceContext` pattern; the event
/// sleeps instead of spinning on the stream).
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    handle: *mut crate::ffi::IgnisDevice,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    /// Create a CUDA device context.
    ///
    /// Fails (the ADR 0006 skip path) if the driver is missing or
    /// `device_id` is invalid.
    pub fn create(device_id: i32) -> Result<CudaDevice> {
        let handle = unsafe { crate::ffi::ignis_device_create(device_id) };
        if handle.is_null() {
            return Err(fail(format!(
                "ignis_device_create({device_id}) failed; is the CUDA driver available?"
            )));
        }
        Ok(CudaDevice { handle })
    }
}

#[cfg(feature = "cuda")]
impl Device for CudaDevice {
    fn allocate(&mut self, bytes: u64) -> Result<DeviceBuffer> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { crate::ffi::ignis_device_alloc(self.handle, bytes, &mut ptr) };
        if rc != 0 || ptr.is_null() {
            return Err(fail(format!(
                "ignis_device_alloc({bytes}) failed (rc={rc})"
            )));
        }
        Ok(DeviceBuffer::new(ptr as *mut u8, bytes))
    }

    fn copy_h2d(&mut self, dst: &DeviceBuffer, dst_offset: u64, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        let rc = unsafe {
            crate::ffi::ignis_device_copy_h2d(
                self.handle,
                dst.offset_mut_ptr(dst_offset) as *mut c_void,
                src.as_ptr() as *const c_void,
                src.len() as u64,
            )
        };
        if rc != 0 {
            return Err(fail("ignis_device_copy_h2d failed"));
        }
        Ok(())
    }

    fn copy_d2h(&self, src: &DeviceBuffer, src_offset: u64, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        let rc = unsafe {
            crate::ffi::ignis_device_copy_d2h(
                self.handle,
                dst.as_mut_ptr() as *mut c_void,
                src.offset_ptr(src_offset) as *const c_void,
                dst.len() as u64,
            )
        };
        if rc != 0 {
            return Err(fail("ignis_device_copy_d2h failed"));
        }
        Ok(())
    }

    fn synchronize(&mut self) -> Result<()> {
        let rc = unsafe { crate::ffi::ignis_device_sync(self.handle) };
        if rc != 0 {
            return Err(fail("ignis_device_sync failed"));
        }
        Ok(())
    }

    fn free_bytes(&self) -> Option<u64> {
        let mut free = 0u64;
        let mut total = 0u64;
        if unsafe { crate::ffi::ignis_device_mem_info(self.handle, &mut free, &mut total) } != 0 {
            return None;
        }
        Some(free)
    }

    fn total_bytes(&self) -> Option<u64> {
        let mut free = 0u64;
        let mut total = 0u64;
        if unsafe { crate::ffi::ignis_device_mem_info(self.handle, &mut free, &mut total) } != 0 {
            return None;
        }
        Some(total)
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaDevice {
    fn drop(&mut self) {
        unsafe { crate::ffi::ignis_device_destroy(self.handle) };
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU mock only — no GPU, ADR 0006)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_round_trip() {
        let mut device = CpuDevice::new();
        let arena = device.allocate(1024).expect("allocate");
        let payload = [7u8, 8, 9, 10, 11];
        device.copy_h2d(&arena, 8, &payload).expect("h2d");
        let mut back = [0u8; 5];
        device.copy_d2h(&arena, 8, &mut back).expect("d2h");
        assert_eq!(back, payload);
        assert_eq!(device.used_bytes(), 1024);
    }

    #[test]
    fn cpu_device_rejects_out_of_range_copies() {
        let mut device = CpuDevice::new();
        let arena = device.allocate(16).expect("allocate");
        // start + len past the end: error (no panic).
        assert!(device.copy_h2d(&arena, 12, &[1, 2, 3, 4, 5]).is_err());
        let mut back = [0u8; 16];
        assert!(device.copy_d2h(&arena, 12, &mut back).is_err());
        assert!(device.copy_d2h(&arena, 0, &mut back).is_ok());
    }

    #[test]
    fn foreign_buffer_rejected() {
        // A buffer not allocated by this device is rejected, not accepted.
        let mut device = CpuDevice::new();
        let mut foreign = vec![0u8; 8];
        let base = foreign.as_mut_ptr();
        let buffer = DeviceBuffer::new(base, 8);
        assert!(device.copy_h2d(&buffer, 0, &[1u8, 2]).is_err());
        let _ = foreign; // keep the foreign buffer alive (pointee)
    }

    #[test]
    fn cpu_device_reports_no_memory_stats() {
        let device = CpuDevice::new();
        assert!(device.free_bytes().is_none());
        assert!(device.total_bytes().is_none());
    }
}