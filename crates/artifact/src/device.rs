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

/// A mapped pinned host allocation (ADR 0062): page-locked host memory the
/// device reads in place through [`HostMappedBuffer::device_ptr`]. The host
/// writes it through [`HostMappedBuffer::host_ptr`], once, at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostMappedBuffer {
    host: *mut u8,
    device: *const u8,
    bytes: u64,
}

impl HostMappedBuffer {
    pub(crate) fn new(host: *mut u8, device: *const u8, bytes: u64) -> Self {
        Self { host, device, bytes }
    }

    /// The host side, [`crate::DIRECT_IO_ALIGNMENT`]-aligned.
    pub fn host_ptr(&self) -> *mut u8 {
        self.host
    }

    /// The pointer a kernel reads it through.
    pub fn device_ptr(&self) -> *const u8 {
        self.device
    }

    /// Exact length of this allocation in bytes.
    pub fn len(&self) -> u64 {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// The whole allocation, writable from the host.
    ///
    /// # Safety
    /// The buffer must still be allocated, and nothing else may read or write
    /// it while the slice lives: no kernel in flight reading it.
    pub unsafe fn host_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.host, self.bytes as usize) }
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

    /// Release a buffer this device produced (`buffer` must come from this
    /// same device's own [`Device::allocate`]). Default: a no-op —
    /// [`CpuDevice`]'s arenas are plain `Vec`s it already owns, freed when
    /// the device itself drops; only [`CudaDevice`] needs an explicit
    /// `cudaFree` for its raw allocation.
    fn deallocate(&mut self, _buffer: DeviceBuffer) -> Result<()> {
        Ok(())
    }

    /// Allocate `bytes` of mapped pinned host memory (ADR 0062), its host
    /// side [`crate::DIRECT_IO_ALIGNMENT`]-aligned.
    fn allocate_host_mapped(&mut self, bytes: u64) -> Result<HostMappedBuffer>;

    /// GitHub #341: the BF16 weight `[rows][cols]` at `src` + `src_offset` to
    /// its FP8 row-scale payload at `dst` + `dst_offset` ([`crate::fp8`]), in
    /// order after the copies before it (call [`Device::synchronize`] before
    /// reading it).
    fn quantize_fp8_rows(
        &mut self,
        src: &DeviceBuffer,
        src_offset: u64,
        rows: u32,
        cols: u32,
        dst: &DeviceBuffer,
        dst_offset: u64,
    ) -> Result<()>;

    /// Release a mapped host buffer this device produced.
    fn free_host_mapped(&mut self, buffer: HostMappedBuffer) -> Result<()>;

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
    /// The "mapped" host buffers and the bytes each was asked for: aligned
    /// heap memory the mock's "device" reads through the same address.
    host_mapped: Vec<(crate::AlignedBuffer, u64)>,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total bytes allocated across all arenas.
    pub fn used_bytes(&self) -> u64 {
        self.used
    }

    /// Bytes of the mapped host buffers still allocated.
    pub fn host_mapped_bytes(&self) -> u64 {
        self.host_mapped.iter().map(|(_, bytes)| bytes).sum()
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

    fn allocate_host_mapped(&mut self, bytes: u64) -> Result<HostMappedBuffer> {
        let len = usize::try_from(bytes).map_err(|_| fail("host-mapped allocation size overflows usize"))?;
        let mut buffer = crate::AlignedBuffer::new(len)?;
        let host = buffer.as_mut_slice().as_mut_ptr();
        self.host_mapped.push((buffer, bytes));
        Ok(HostMappedBuffer::new(host, host, bytes))
    }

    fn free_host_mapped(&mut self, buffer: HostMappedBuffer) -> Result<()> {
        let index = self
            .host_mapped
            .iter()
            .position(|(b, _)| b.as_slice().as_ptr() == buffer.host_ptr() as *const u8)
            .ok_or_else(|| fail("host-mapped buffer was not allocated by this device"))?;
        self.host_mapped.swap_remove(index);
        Ok(())
    }

    fn quantize_fp8_rows(
        &mut self,
        src: &DeviceBuffer,
        src_offset: u64,
        rows: u32,
        cols: u32,
        dst: &DeviceBuffer,
        dst_offset: u64,
    ) -> Result<()> {
        let words = rows as usize * cols as usize;
        let mut bf16 = vec![0u8; words * 2];
        self.copy_d2h(src, src_offset, &mut bf16)?;
        let w: Vec<u16> = bf16.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
        let mut payload = vec![0u8; crate::fp8::payload_bytes(rows, cols) as usize];
        crate::fp8::quantize_rows(&w, rows, cols, &mut payload);
        self.copy_h2d(dst, dst_offset, &payload)
    }
}

// ---------------------------------------------------------------------------
// CUDA device (feature `cuda`): the kernel leaf's flat C surface (ADR 0001),
// declared in `crate::ffi` (1:1 with `kernel/include/ignis_device.h`).
// ---------------------------------------------------------------------------

/// The NVIDIA driver's own library, which the CUDA runtime loads: what a
/// machine without the driver lacks (GitHub #329).
const DRIVER_LIBRARY: &str = if cfg!(windows) { "nvcuda.dll" } else { "libcuda.so.1" };

/// NVML's library, which also ships with the driver.
#[cfg(feature = "cuda")]
const NVML_LIBRARY: &str = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };

/// What the leaf's `ignis_device_driver_version` answered -- its return code
/// and the version it read -- means for a GPU path (GitHub #329). The CUDA
/// build links no driver library, so it starts on a machine without the
/// driver (`model download`, `--help`); a GPU load there is refused naming the
/// driver, rather than by a runtime error from whichever call came first.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn driver_check(rc: i32, version: i32, runtime: i32) -> Result<()> {
    if rc != 0 || version <= 0 {
        return Err(fail(format!(
            "no NVIDIA driver ({DRIVER_LIBRARY}) on this machine: a GPU load needs the host's NVIDIA \
             driver (in a container, the NVIDIA container runtime provides it on a GPU node)"
        )));
    }
    // CUDA minor-version compatibility lets a 13.0 driver run a 13.1 runtime,
    // so only a lower major is too old.
    if version / 1000 < runtime / 1000 {
        let cuda = |v: i32| format!("{}.{}", v / 1000, (v % 1000) / 10);
        return Err(fail(format!(
            "the NVIDIA driver on this machine supports CUDA {} but this build's CUDA runtime is {}: update the NVIDIA driver to one that supports CUDA {} or newer",
            cuda(version),
            cuda(runtime),
            runtime / 1000
        )));
    }
    Ok(())
}

/// How often NVML's free memory is read once the CUDA context is made
/// (GitHub #604).
const CONTEXT_READ_EVERY: std::time::Duration = std::time::Duration::from_millis(10);
/// How close the readings after the context must stay to one another, and
/// for how long, before the lowest of them is taken: NVML can count a fresh
/// allocation late, in 64 MiB pieces, by up to 63 ms (the finding
/// 2026-10-31, `nvml-counts-a-fresh-allocation-late`).
const CONTEXT_STILL_WITHIN_BYTES: u64 = 1 << 20;
const CONTEXT_STILL_FOR: std::time::Duration = std::time::Duration::from_millis(250);
/// When the readings stop, held still or not: the lowest seen is taken then.
const CONTEXT_READ_WITHIN: std::time::Duration = std::time::Duration::from_secs(2);

/// What NVML's free memory said this process's CUDA context took when its
/// first device made it (GitHub #604), as [`measure_context`] read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextReading {
    /// The bytes the free memory fell by: the reading before the context,
    /// less the one taken after it.
    Measured(u64),
    /// A reading before or after the context failed.
    Unreadable,
    /// The free memory after the context was no lower than before: another
    /// process freed as much inside the window, or more.
    NotLower { before: u64, after: u64 },
}

impl std::fmt::Display for ContextReading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextReading::Measured(bytes) => write!(f, "{bytes} bytes measured"),
            ContextReading::Unreadable => f.write_str("NVML's free memory could not be read either side of the context"),
            ContextReading::NotLower { before, after } => write!(
                f,
                "NVML read {after} bytes free after the context against {before} before it: another process freed memory meanwhile"
            ),
        }
    }
}

/// The time a context measurement reads by: the wall clock on the card, a
/// test's own in the CPU tests.
pub trait MeasureClock {
    /// The time since the measurement began.
    fn elapsed(&self) -> std::time::Duration;
    /// Waits `pause` before the next reading.
    fn pause(&mut self, pause: std::time::Duration);
}

/// The wall clock, from when it is started.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct WallClock(std::time::Instant);

impl MeasureClock for WallClock {
    fn elapsed(&self) -> std::time::Duration {
        self.0.elapsed()
    }
    fn pause(&mut self, pause: std::time::Duration) {
        std::thread::sleep(pause);
    }
}

/// What the CUDA context took (GitHub #604): `before`, NVML's free memory
/// read just before the context was made, less `read`'s free memory after
/// it. `read` is asked every [`CONTEXT_READ_EVERY`], and its readings are
/// taken once they have held within [`CONTEXT_STILL_WITHIN_BYTES`] of one
/// another for [`CONTEXT_STILL_FOR`], the lowest of them; at
/// [`CONTEXT_READ_WITHIN`] the lowest reading seen is taken, the largest
/// difference: an over-charge, never short. A reading that fails is
/// [`ContextReading::Unreadable`]; a difference of zero or less,
/// [`ContextReading::NotLower`].
pub fn measure_context(
    before: Option<u64>, mut read: impl FnMut() -> Option<u64>, clock: &mut impl MeasureClock,
) -> ContextReading {
    let Some(before) = before else {
        return ContextReading::Unreadable;
    };
    let mut lowest = u64::MAX;
    // Since when the readings have held still, and their lowest and highest.
    let mut still: Option<(std::time::Duration, u64, u64)> = None;
    let mut reads = 0u32;
    let after = loop {
        let Some(free) = read() else {
            return ContextReading::Unreadable;
        };
        let now = clock.elapsed();
        lowest = lowest.min(free);
        let (since, low, high) = match still {
            Some((since, low, high)) if high.max(free) - low.min(free) <= CONTEXT_STILL_WITHIN_BYTES => {
                (since, low.min(free), high.max(free))
            }
            _ => (now, free, free),
        };
        still = Some((since, low, high));
        if now - since >= CONTEXT_STILL_FOR {
            break low;
        }
        if now >= CONTEXT_READ_WITHIN {
            break lowest;
        }
        reads += 1;
        clock.pause((CONTEXT_READ_EVERY * reads).saturating_sub(clock.elapsed()));
    };
    match before.checked_sub(after) {
        Some(bytes) if bytes > 0 => ContextReading::Measured(bytes),
        _ => ContextReading::NotLower { before, after },
    }
}

/// A CUDA device context (a dedicated non-blocking load stream + a
/// blocking-sync event — the reference `DeviceContext` pattern; the event
/// sleeps instead of spinning on the stream).
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    handle: *mut crate::ffi::IgnisDevice,
    /// The ordinal it was created on.
    device_id: i32,
}

/// What this process's CUDA context took, set when the first [`CudaDevice`]
/// made it (GitHub #324, #604) and kept for the process's life, as the
/// context is.
#[cfg(feature = "cuda")]
static CONTEXT: std::sync::OnceLock<ContextReading> = std::sync::OnceLock::new();

/// Held while a device is made, so the first device's measurement is the
/// only one and no second device is made inside its window.
#[cfg(feature = "cuda")]
static MAKING_A_DEVICE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "cuda")]
impl CudaDevice {
    /// Create a CUDA device context.
    ///
    /// The first device makes the process's context, and measures it from
    /// NVML's free memory either side ([`measure_context`], GitHub #604):
    /// what the context takes on this card and this OS, which
    /// [`CudaDevice::context_reading`] then answers. That first create takes
    /// 250 ms more, 2 s at most.
    ///
    /// Fails (the ADR 0006 skip path) if the driver is missing or
    /// `device_id` is invalid.
    pub fn create(device_id: i32) -> Result<CudaDevice> {
        Self::require_driver()?;
        let _making = MAKING_A_DEVICE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let before = CONTEXT.get().is_none().then(|| Self::nvml_free(device_id));
        let handle = unsafe { crate::ffi::ignis_device_create(device_id) };
        if handle.is_null() {
            return Err(fail(format!(
                "ignis_device_create({device_id}) failed; is the CUDA driver available?"
            )));
        }
        if let Some(before) = before {
            let reading = measure_context(before, || Self::nvml_free(device_id), &mut WallClock(std::time::Instant::now()));
            let _ = CONTEXT.set(reading);
        }
        Ok(CudaDevice { handle, device_id })
    }

    /// Whether this process has created its CUDA context (GitHub #324).
    /// Dropping a device frees its stream and event, never the context
    /// (`ignis_device_destroy` resets nothing), so from the first
    /// [`CudaDevice::create`] on the process holds it until it exits: a free
    /// reading by [`CudaDevice::nvml_memory`] no longer counts it.
    pub fn context_created() -> bool {
        CONTEXT.get().is_some()
    }

    /// What the process's CUDA context took when the first device made it
    /// (GitHub #604); `None` before the context exists.
    pub fn context_reading() -> Option<ContextReading> {
        CONTEXT.get().copied()
    }

    /// This process's device memory on device `device_id` as NVML's
    /// per-process accounting reports it (GitHub #604): what a later load can
    /// add back. `None` where the driver reports no figure for this process:
    /// a PID absent from its list (under WSL the list is empty), a figure it
    /// marks not available (every figure under WDDM), a driver without the
    /// query, or a call that fails. Creates no context.
    pub fn nvml_process_bytes(device_id: i32) -> Option<u64> {
        Self::require_driver().ok()?;
        let mut bytes = 0u64;
        let rc = unsafe { crate::ffi::ignis_device_nvml_process_bytes(device_id, std::process::id(), &mut bytes) };
        (rc == 0).then_some(bytes)
    }

    /// NVML's free memory on `device_id`, `None` when it cannot be read (the
    /// leaf says why on stderr): one reading of a context measurement.
    fn nvml_free(device_id: i32) -> Option<u64> {
        let (mut free, mut total) = (0u64, 0u64);
        let rc = unsafe { crate::ffi::ignis_device_nvml_mem_info(device_id, &mut free, &mut total) };
        (rc == 0).then_some(free)
    }

    /// The device's ordinal: what [`CudaDevice::nvml_memory`] reads for it.
    pub fn ordinal(&self) -> i32 {
        self.device_id
    }

    /// Free and total memory of device `device_id` as NVML reports it -- what
    /// nvidia-smi and Task Manager show -- read without creating a CUDA
    /// context (GitHub #210). Unlike [`Device::free_bytes`], it is meaningful
    /// before this process holds anything: on Windows `cudaMemGetInfo` does
    /// not count the caller's own context and reads about 400 MiB more free
    /// memory than the adapter has.
    pub fn nvml_memory(device_id: i32) -> Result<(u64, u64)> {
        Self::require_driver()?;
        let mut free = 0u64;
        let mut total = 0u64;
        let rc = unsafe { crate::ffi::ignis_device_nvml_mem_info(device_id, &mut free, &mut total) };
        if rc != 0 {
            return Err(fail(format!(
                "ignis_device_nvml_mem_info({device_id}) failed; is NVML ({NVML_LIBRARY}, shipped with the driver) available?"
            )));
        }
        Ok((free, total))
    }

    /// The name (NVML's) and compute capability (the CUDA runtime's, as
    /// `(major, minor)`) of device `device_id`, read without creating a CUDA
    /// context (GitHub #457): what the start checks against
    /// [`built_cuda_architectures`].
    pub fn identity(device_id: i32) -> Result<(String, (u32, u32))> {
        Self::require_driver()?;
        let mut name = [0 as std::os::raw::c_char; 256];
        let (mut major, mut minor) = (0i32, 0i32);
        let rc = unsafe {
            crate::ffi::ignis_device_identity(device_id, name.as_mut_ptr(), name.len() as u32, &mut major, &mut minor)
        };
        if rc != 0 || major <= 0 || minor < 0 {
            return Err(fail(format!(
                "ignis_device_identity({device_id}) failed; is NVML ({NVML_LIBRARY}, shipped with the driver) available?"
            )));
        }
        let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        Ok((name, (major as u32, minor as u32)))
    }

    /// Asked first by everything here that reaches the GPU -- a load reaches
    /// [`CudaDevice::nvml_memory`] or [`CudaDevice::create`] before anything
    /// else -- so a machine without the driver is named as such (GitHub #329).
    fn require_driver() -> Result<()> {
        let mut version = 0i32;
        let mut runtime = 0i32;
        let rc = unsafe { crate::ffi::ignis_device_driver_version(&mut version) };
        let runtime_rc = unsafe { crate::ffi::ignis_device_runtime_version(&mut runtime) };
        driver_check(if runtime_rc == 0 { rc } else { runtime_rc }, version, runtime)
    }
}

/// The CUDA architectures the kernel leaf linked into this binary was
/// compiled for, as its build named them (`120a`, `86;120a`): the set a card's
/// compute capability is checked against at start (GitHub #457). Needs no
/// driver and no device.
#[cfg(feature = "cuda")]
pub fn built_cuda_architectures() -> String {
    let list = unsafe { crate::ffi::ignis_build_cuda_architectures() };
    unsafe { std::ffi::CStr::from_ptr(list) }.to_string_lossy().into_owned()
}

#[cfg(feature = "cuda")]
impl Device for CudaDevice {
    fn quantize_fp8_rows(
        &mut self,
        src: &DeviceBuffer,
        src_offset: u64,
        rows: u32,
        cols: u32,
        dst: &DeviceBuffer,
        dst_offset: u64,
    ) -> Result<()> {
        let rc = unsafe {
            crate::ffi::ignis_device_fp8_quantize_rows(
                self.handle,
                src.offset_ptr(src_offset) as *const c_void,
                rows,
                cols,
                dst.offset_mut_ptr(dst_offset) as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(fail(format!("ignis_device_fp8_quantize_rows([{rows}][{cols}]) failed")));
        }
        Ok(())
    }

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

    fn deallocate(&mut self, buffer: DeviceBuffer) -> Result<()> {
        unsafe { crate::ffi::ignis_device_free(self.handle, buffer.base_ptr() as *mut c_void) };
        Ok(())
    }

    fn allocate_host_mapped(&mut self, bytes: u64) -> Result<HostMappedBuffer> {
        let mut host: *mut c_void = std::ptr::null_mut();
        let mut device: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { crate::ffi::ignis_device_host_alloc_mapped(self.handle, bytes, &mut host, &mut device) };
        if rc != 0 || host.is_null() || device.is_null() {
            return Err(fail(format!(
                "ignis_device_host_alloc_mapped({bytes}) failed (rc={rc}): the host cannot pin that much memory"
            )));
        }
        Ok(HostMappedBuffer::new(host as *mut u8, device as *const u8, bytes))
    }

    fn free_host_mapped(&mut self, buffer: HostMappedBuffer) -> Result<()> {
        unsafe { crate::ffi::ignis_device_host_free(self.handle, buffer.host_ptr() as *mut c_void) };
        Ok(())
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

    /// ADR 0062: a mapped host buffer is writable through its host pointer
    /// and read through its device pointer; on the mock they are one, the
    /// address is direct-I/O aligned, and none of it counts as device memory.
    #[test]
    fn cpu_device_host_mapped_buffer_reads_where_it_is_written() {
        let mut device = CpuDevice::new();
        let mut mapped = device.allocate_host_mapped(5000).expect("host mapped");
        assert_eq!(mapped.len(), 5000);
        assert_eq!(mapped.host_ptr() as usize % 4096, 0, "a direct read lands in it");
        assert_eq!(mapped.device_ptr(), mapped.host_ptr() as *const u8);
        let host = unsafe { mapped.host_slice_mut() };
        host[4999] = 7;
        assert_eq!(unsafe { *mapped.device_ptr().add(4999) }, 7);
        assert_eq!(device.used_bytes(), 0, "host memory, not the device's");
        assert_eq!(device.host_mapped_bytes(), 5000);
        device.free_host_mapped(mapped).expect("free");
        assert_eq!(device.host_mapped_bytes(), 0);
        assert!(device.free_host_mapped(mapped).is_err(), "a buffer is freed once");
    }

    #[test]
    fn cpu_device_reports_no_memory_stats() {
        let device = CpuDevice::new();
        assert!(device.free_bytes().is_none());
        assert!(device.total_bytes().is_none());
    }

    #[test]
    fn a_machine_without_the_driver_is_named_before_any_gpu_work() {
        // GitHub #329: with no driver library to load, the runtime reports
        // driver version 0 -- or the query itself fails.
        for (rc, version) in [(0, 0), (-1, 0)] {
            let message = driver_check(rc, version, 13010).unwrap_err().to_string();
            assert!(message.starts_with(&format!("no NVIDIA driver ({DRIVER_LIBRARY}) on this machine")), "{message}");
        }
        assert!(driver_check(0, 13010, 13010).is_ok());
    }

    #[test]
    fn a_driver_older_than_the_runtime_is_named_with_both_versions() {
        // A CUDA 12.x driver under a 13.1 runtime: the major is below.
        let message = driver_check(0, 12040, 13010).unwrap_err().to_string();
        assert!(message.contains("12.4") && message.contains("13.1"), "{message}");
        assert!(message.contains("update the NVIDIA driver"), "{message}");
        assert!(!message.contains("  "), "one space between words: {message}");
        // Minor-version compatibility: same major, older minor is accepted.
        assert!(driver_check(0, 13000, 13010).is_ok());
        // A newer driver is accepted.
        assert!(driver_check(0, 13020, 13010).is_ok());
        assert!(driver_check(0, 14000, 13010).is_ok());
    }
}