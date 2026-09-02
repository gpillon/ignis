//! The materializer: place a binder's [`MaterializationPlan`] on a
//! [`Device`] (ADR 0002).
//!
//! Rust port of the reference `materializer` (`ninfer`
//! `materializer.h`/`.cpp`): device objects are read from the container
//! via 4096-aligned direct I/O (bypassing the page cache), streamed
//! through a bounded pool of reusable 4096-aligned staging slots (peak host
//! memory = a few slots, not the sum of every object — the reference's
//! streaming design), and uploaded to the plan's device arena at the
//! reader-computed geometry. Host-retained resources keep their bytes in RAM
//! (the reference `take_resource_bytes` pattern).
//!
//! The device arena is a single allocation (the reference `DeviceArena`
//! sub-allocation pattern): one `allocate(capacity)`, each device object
//! lives at its plan offset inside it.

use std::alloc::{Layout, alloc_zeroed, dealloc};

use crate::binder::MaterializationPlan;
use crate::device::{Device, DeviceBuffer};
use crate::{
    block_scale_geometry, checked_add, fail, row_scale_geometry, row_split_geometry, Reader,
    Result, StorageLayout, TensorDescriptor, DIRECT_IO_ALIGNMENT,
};

/// Materialization statistics (mirrors the reference `MaterializationStats`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MaterializationStats {
    /// Bytes read from the container (aligned direct reads + retained
    /// resources).
    pub file_bytes: u64,
    /// Host->device bytes copied (sum of device-object payload bytes).
    pub h2d_bytes: u64,
    /// Device arena size (the plan's `device_capacity_bytes`).
    pub device_capacity_bytes: u64,
    /// Bytes of host-retained resources (decremented by
    /// [`MaterializedArtifact::take_resource_bytes`]).
    pub retained_resource_bytes: u64,
    /// Largest single-object staging span (streaming: the max, not the sum).
    /// The bounded pool holds up to [`STAGING_SLOTS`] such slots, so the
    /// peak host-staging memory is at most [`STAGING_SLOTS`] × this value
    /// (fewer when the plan has fewer than [`STAGING_SLOTS`] device objects).
    pub peak_staging_bytes: u64,
    /// Device tensors in the plan.
    pub tensor_count: usize,
    /// Host-retained resources in the plan.
    pub resource_count: usize,
    /// Wall-clock upload time (seconds).
    pub upload_seconds: f64,
}

/// A typed, non-owning view of a device-placed tensor.
///
/// `base` points into the device arena (the kernel entry point); the plane
/// offsets are relative to `base` and come from the reader's storage-layout
/// geometry (ADR 0002: geometry is the container's, not the model's).
#[derive(Debug)]
pub struct TensorView {
    /// The object's directory name.
    pub name: String,
    pub format: crate::NumericFormat,
    pub layout: StorageLayout,
    pub shape: Vec<u64>,
    /// Pointer to the tensor payload on the device (first byte of the
    /// layout's payload).
    pub base: *const u8,
    /// Exact payload length (the layout's encoded size).
    pub bytes: u64,
    /// Offset of this tensor inside the device arena (for d2h
    /// verification).
    pub offset_in_arena: u64,
    /// Row-split high-plane offset relative to `base` (row-split layouts).
    pub high_plane_offset: Option<u64>,
    /// Scale-plane offset relative to `base` (row-split / blockscale /
    /// row-scale layouts).
    pub scale_plane_offset: Option<u64>,
    /// Trailing FP32 weight-divisor offset relative to `base` (blockscale
    /// layouts only).
    pub weight_divisor_offset: Option<u64>,
}

impl TensorView {
    /// Pointer to the row-split high plane (row-split layouts).
    pub fn high_plane(&self) -> Option<*const u8> {
        self.high_plane_offset.map(|o| unsafe { self.base.add(o as usize) })
    }

    /// Pointer to the scale plane (row-split / blockscale / row-scale
    /// layouts).
    pub fn scale_plane(&self) -> Option<*const u8> {
        self.scale_plane_offset.map(|o| unsafe { self.base.add(o as usize) })
    }

    /// Pointer to the trailing FP32 weight divisor (blockscale layouts).
    pub fn weight_divisor(&self) -> Option<*const f32> {
        self.weight_divisor_offset
            .map(|o| unsafe { self.base.add(o as usize) as *const f32 })
    }
}

#[derive(Debug)]
enum MaterializedObject {
    Tensor(TensorView),
    Resource(Vec<u8>),
    /// Consumed but placed nowhere (`validate_only`).
    Unplaced,
}

/// A materialized artifact: one device arena (suballocated per plan
/// offset) + host-retained resources + statistics.
///
/// The producing [`Device`] must outlive this artifact (the arena is the
/// device's allocation).
#[derive(Debug)]
pub struct MaterializedArtifact {
    arena: Option<DeviceBuffer>,
    objects: Vec<MaterializedObject>,
    stats: MaterializationStats,
}

impl MaterializedArtifact {
    /// The typed view of a device-placed tensor (error if the handle names
    /// a host-retained or validate-only object).
    pub fn device_view(&self, handle: crate::binder::ObjectHandle) -> Result<&TensorView> {
        match self.objects.get(handle.index) {
            Some(MaterializedObject::Tensor(view)) => Ok(view),
            Some(_) => Err(fail(
                "object handle does not name a materialized tensor",
            )),
            None => Err(fail("object handle is out of range")),
        }
    }

    /// The host-retained resource bytes (e.g. the frontend objects).
    pub fn resource_bytes(&self, handle: crate::binder::ObjectHandle) -> Result<&[u8]> {
        match self.objects.get(handle.index) {
            Some(MaterializedObject::Resource(bytes)) => Ok(bytes),
            Some(_) => Err(fail(
                "object handle does not name a materialized resource",
            )),
            None => Err(fail("object handle is out of range")),
        }
    }

    /// Take ownership of the host-retained resource bytes (decrementing
    /// `stats().retained_resource_bytes`; the handle stops naming a
    /// resource).
    pub fn take_resource_bytes(&mut self, handle: crate::binder::ObjectHandle) -> Result<Vec<u8>> {
        let taken = match self.objects.get_mut(handle.index) {
            Some(MaterializedObject::Resource(bytes)) => std::mem::take(bytes),
            Some(_) => {
                return Err(fail(
                    "object handle does not name a materialized resource",
                ))
            }
            None => return Err(fail("object handle is out of range")),
        };
        self.stats.retained_resource_bytes =
            self.stats
                .retained_resource_bytes
                .saturating_sub(taken.len() as u64);
        Ok(taken)
    }

    /// The device arena (the single allocation the views point into), if
    /// the plan placed any device objects.
    pub fn arena(&self) -> Option<&DeviceBuffer> {
        self.arena.as_ref()
    }

    /// The load statistics.
    pub fn stats(&self) -> &MaterializationStats {
        &self.stats
    }
}

/// Place `plan` on `device`: allocate the device arena, upload the device
/// objects (direct I/O -> staging -> device), retain the host resources.
///
/// `progress` (optional) is called per device object as
/// `(object name, h2d bytes so far, h2d bytes total)`.
pub fn materialize<D: Device>(
    reader: &Reader,
    plan: &MaterializationPlan,
    device: &mut D,
    mut progress: Option<&mut dyn FnMut(&str, u64, u64)>,
) -> Result<MaterializedArtifact> {
    let start = std::time::Instant::now();
    let capacity = plan.device_capacity_bytes;

    // One arena allocation (the reference DeviceArena sub-allocation
    // pattern). A plan with no device objects carries no arena.
    let arena = if capacity == 0 {
        None
    } else {
        Some(device.allocate(capacity)?)
    };

    let object_count = reader.objects().len();
    let mut objects: Vec<MaterializedObject> = (0..object_count)
        .map(|_| MaterializedObject::Unplaced)
        .collect();

    let mut stats = MaterializationStats {
        device_capacity_bytes: capacity,
        tensor_count: plan.device_objects.len(),
        resource_count: plan.host_objects.len(),
        ..Default::default()
    };

    // --- host-retained resources (reference `take_resource_bytes` pattern)
    for host in &plan.host_objects {
        let object = reader
            .objects()
            .get(host.handle.index)
            .ok_or_else(|| fail("host placement handle is out of range"))?;
        let span = reader.payload_at(object)?;
        let bytes = span.data.to_vec();
        stats.retained_resource_bytes = checked_add(
            stats.retained_resource_bytes,
            bytes.len() as u64,
            "retained resource bytes overflow u64",
        )?;
        stats.file_bytes =
            checked_add(stats.file_bytes, bytes.len() as u64, "artifact read bytes overflow u64")?;
        objects[host.handle.index] = MaterializedObject::Resource(bytes);
    }

    // --- device objects: direct I/O -> bounded staging pool -> device ------
    // The reference materializer streams through a small pool of pinned
    // staging slots (peak host memory = a few slots, not the sum of every
    // object). We mirror that: N reusable 4096-aligned slots, assigned
    // round-robin. The CUDA H2D copies are asynchronous on the load stream,
    // so a slot's bytes must stay valid until its copy completes. We keep at
    // most N copies in flight; when N are pending we drain once
    // (`synchronize`) so every slot is free before reuse. The pool lives
    // until the final `synchronize`, so no in-flight copy ever reads freed
    // memory.
    let mut total_h2d = 0u64;
    for p in &plan.device_objects {
        total_h2d = checked_add(total_h2d, p.bytes, "h2d total overflow u64")?;
    }
    let mut h2d_done = 0u64;

    // Pre-size the bounded slot pool to the largest aligned direct-read span
    // (every span is 4096-aligned, so any object fits without growth).
    let mut max_staging_len = 0usize;
    for dev in &plan.device_objects {
        let object = reader
            .objects()
            .get(dev.handle.index)
            .ok_or_else(|| fail("device placement handle is out of range"))?;
        let span = reader.payload_at(object)?;
        let source_end =
            checked_add(span.absolute_offset, dev.bytes, "artifact tensor source range overflow u64")?;
        let read_begin = span.absolute_offset & !(DIRECT_IO_ALIGNMENT - 1);
        let read_end = (source_end + (DIRECT_IO_ALIGNMENT - 1)) & !(DIRECT_IO_ALIGNMENT - 1);
        max_staging_len = max_staging_len.max((read_end - read_begin) as usize);
    }
    let staging_slots = STAGING_SLOTS.min(plan.device_objects.len());
    let mut slots: Vec<AlignedStaging> = (0..staging_slots)
        .map(|_| AlignedStaging::new(max_staging_len.max(DIRECT_IO_ALIGNMENT as usize)))
        .collect::<Result<Vec<_>>>()?;
    let mut pending = 0usize;

    for (k, dev) in plan.device_objects.iter().enumerate() {
        let object = reader
            .objects()
            .get(dev.handle.index)
            .ok_or_else(|| fail("device placement handle is out of range"))?;
        if object.bytes() != dev.bytes {
            return Err(fail(format!(
                "materialization plan does not match artifact payload: {}",
                object.name()
            )));
        }
        if dev.offset % dev.alignment != 0 {
            return Err(fail(format!(
                "materialization plan offset is misaligned: {}",
                object.name()
            )));
        }
        if dev.offset + dev.bytes > capacity {
            return Err(fail(format!(
                "materialization plan offset extends beyond device capacity: {}",
                object.name()
            )));
        }
        let tensor = match object {
            crate::Object::Tensor(t) => t,
            crate::Object::Resource(_) => {
                return Err(fail(format!(
                    "resource cannot be materialized as a device tensor: {}",
                    object.name()
                )))
            }
        };
        let view_base = match &arena {
            Some(a) => a.offset_ptr(dev.offset),
            None => {
                return Err(fail(format!(
                    "materialization plan has device objects but no arena capacity: {}",
                    object.name()
                )))
            }
        };

        // 4096-aligned direct-I/O span (the reference read-span logic):
        // align the source start down, the source end up, read the whole
        // span in one aligned read (bypasses the page cache).
        let span = reader.payload_at(object)?;
        let source_end =
            checked_add(span.absolute_offset, dev.bytes, "artifact tensor source range overflow u64")?;
        let read_begin = span.absolute_offset & !(DIRECT_IO_ALIGNMENT - 1);
        let read_end = (source_end + DIRECT_IO_ALIGNMENT - 1) & !(DIRECT_IO_ALIGNMENT - 1);
        let staging_len = (read_end - read_begin) as usize;

        // Round-robin slot; once N copies are pending, drain so every slot
        // is free (its in-flight copy done) before reuse.
        let s = k % staging_slots;
        if pending == staging_slots {
            device.synchronize()?;
            pending = 0;
        }
        let buf = slots[s].as_slice();
        let read = reader.read_direct(read_begin, &mut buf[0..staging_len])?;
        let head = (span.absolute_offset - read_begin) as usize;
        let needed = checked_add(head as u64, dev.bytes, "staging head overflow u64")? as usize;
        if read < needed {
            return Err(fail(format!(
                "direct artifact read ended before the planned tensor range: {}",
                object.name()
            )));
        }
        let payload = &buf[head..needed];

        if let Some(a) = &arena {
            device.copy_h2d(a, dev.offset, payload)?;
        }
        pending += 1;
        h2d_done = checked_add(h2d_done, dev.bytes, "copied byte count overflows u64")?;
        stats.h2d_bytes = h2d_done;
        stats.file_bytes =
            checked_add(stats.file_bytes, read as u64, "artifact read bytes overflow u64")?;
        stats.peak_staging_bytes = stats.peak_staging_bytes.max(staging_len as u64);

        let view = tensor_view(tensor, dev.offset, view_base, dev.bytes)?;
        objects[dev.handle.index] = MaterializedObject::Tensor(view);

        if let Some(callback) = &mut progress {
            callback(object.name(), h2d_done, total_h2d);
        }
    }

    // The pool lives until here: the final drain means no in-flight copy is
    // still reading a slot when the buffers are dropped (at end of scope).
    device.synchronize()?;
    stats.upload_seconds = start.elapsed().as_secs_f64();

    Ok(MaterializedArtifact {
        arena,
        objects,
        stats,
    })
}

/// Number of reusable staging slots (the reference `Materializer` streams
/// through four pinned slots; peak host staging = `STAGING_SLOTS` x the
/// largest aligned span, not the sum of every object).
const STAGING_SLOTS: usize = 4;

/// A 4096-aligned staging buffer for direct I/O (page-aligned allocation:
/// the reader's direct path requires 4096-aligned offsets, lengths, *and*
/// addresses).
struct AlignedStaging {
    ptr: std::ptr::NonNull<u8>,
    layout: Layout,
}

impl AlignedStaging {
    fn new(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len, DIRECT_IO_ALIGNMENT as usize).map_err(|e| {
            fail(format!(
                "staging length {len} is not a valid 4096-aligned layout: {e}"
            ))
        })?;
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(fail("staging buffer allocation failed"));
        }
        Ok(Self {
            ptr: std::ptr::NonNull::new(ptr).unwrap(),
            layout,
        })
    }

    fn as_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.layout.size()) }
    }
}

impl Drop for AlignedStaging {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, self.layout) };
    }
}

/// Build the typed view for a device object (plane offsets from the
/// reader-computed layout geometry).
fn tensor_view(
    tensor: &TensorDescriptor,
    offset: u64,
    base: *const u8,
    bytes: u64,
) -> Result<TensorView> {
    let (high, scale, wdiv) = match tensor.layout {
        StorageLayout::ContiguousLeV1 => (None, None, None),
        StorageLayout::RowSplitK128V1 => {
            let g = row_split_geometry(tensor.format, &tensor.shape)?;
            (
                Some(g.high_plane_offset),
                Some(g.scale_plane_offset),
                None,
            )
        }
        StorageLayout::BlockScaleK16M128x4V1 => {
            let g = block_scale_geometry(tensor.format, &tensor.shape)?;
            (
                None,
                Some(g.scale_plane_offset),
                Some(g.weight_divisor_offset),
            )
        }
        StorageLayout::RowScaleV1 => {
            let g = row_scale_geometry(tensor.format, &tensor.shape)?;
            (None, Some(g.scale_plane_offset), None)
        }
    };
    Ok(TensorView {
        name: tensor.name.clone(),
        format: tensor.format,
        layout: tensor.layout,
        shape: tensor.shape.clone(),
        base,
        bytes,
        offset_in_arena: offset,
        high_plane_offset: high,
        scale_plane_offset: scale,
        weight_divisor_offset: wdiv,
    })
}

// ---------------------------------------------------------------------------
// Tests (CPU mock only — no GPU, ADR 0006)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuDevice;
    use crate::{fixture, Binder, NumericFormat, ResourceEncoding, StorageLayout};

    /// Bind all five fixture objects (four device, one host) and finish.
    fn bind_all(reader: &Reader) -> (crate::binder::MaterializationPlan, Vec<crate::binder::ObjectHandle>) {
        let mut binder = Binder::new(reader);
        let resource = binder
            .require_resource("frontend/tokenizer.json", ResourceEncoding::RawBytesV1)
            .expect("resource");
        binder.retain_on_host(resource).expect("retain");

        let handles = vec![
            binder
                .require_tensor("w/bf16", NumericFormat::Bf16, StorageLayout::ContiguousLeV1, &[4, 8])
                .expect("bf16"),
            binder
                .require_tensor("w/nvfp4", NumericFormat::Nvfp4, StorageLayout::BlockScaleK16M128x4V1, &[128, 64])
                .expect("nvfp4"),
            binder
                .require_tensor("w/q4", NumericFormat::Q4G64F16S, StorageLayout::RowSplitK128V1, &[128, 128])
                .expect("q4"),
            binder
                .require_tensor("w/fp8", NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1, &[128, 64])
                .expect("fp8"),
        ];
        for h in &handles {
            binder.materialize_on_device(*h).expect("device placement");
        }
        let plan = binder.finish().expect("plan (all objects consumed)");
        (plan, handles)
    }

    #[test]
    fn aligned_staging_is_4096_aligned() {
        let mut s = AlignedStaging::new(4096 * 2).expect("staging");
        assert_eq!(s.ptr.as_ptr() as usize % 4096, 0);
        assert_eq!(s.as_slice().len(), 4096 * 2);
    }

    #[test]
    fn cpu_materialize_round_trip() {
        let fixture =
            fixture::write_fixture(&fixture::all_layout_objects(), &fixture::all_layout_payload(), "cpu").expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let (plan, handles) = bind_all(&reader);

        // Plan geometry: alignment-aware arena layout (256 B tensor align).
        assert_eq!(plan.object_count, 5);
        assert_eq!(plan.device_objects.len(), 4);
        assert_eq!(plan.host_objects.len(), 1);
        assert_eq!(plan.device_capacity_bytes, 22272);
        let offsets: Vec<u64> = plan
            .device_objects
            .iter()
            .map(|p| p.offset)
            .collect();
        assert_eq!(offsets, vec![0, 256, 5120, 13824]);

        let mut device = CpuDevice::new();
        let mut artifact = materialize(&reader, &plan, &mut device, None).expect("materialize");

        // One arena allocation of exactly the plan's capacity.
        let arena = artifact.arena().expect("arena");
        assert_eq!(arena.len(), 22272);

        // Every device object's payload lands at its plan offset with the
        // stored content (d2h verification through the CPU mock).
        for (dev, handle) in plan.device_objects.iter().zip(handles.iter()) {
            let view = artifact.device_view(*handle).expect("view");
            assert_eq!(view.offset_in_arena, dev.offset);
            assert_eq!(view.bytes, dev.bytes);
            let mut back = vec![0u8; dev.bytes as usize];
            device
                .copy_d2h(arena, dev.offset, &mut back)
                .expect("d2h");
            let fill_byte = match dev.offset {
                0 => 0x02,
                256 => 0x03,
                5120 => 0x04,
                _ => 0x05,
            };
            assert!(
                back.iter().all(|&b| b == fill_byte),
                "device payload differs at offset {}",
                dev.offset
            );
        }

        // Host-retained resource round-trip.
        let resource_handle = reader
            .index_of("frontend/tokenizer.json")
            .expect("resource index");
        let resource_handle = crate::binder::ObjectHandle {
            index: resource_handle,
        };
        let resource_bytes = artifact
            .resource_bytes(resource_handle)
            .expect("resource bytes");
        assert_eq!(resource_bytes.len(), 4096);
        assert!(resource_bytes.iter().all(|&b| b == 0x01));
        // take_resource_bytes moves the bytes (reference take pattern) and
        // decrements the retained stat.
        let taken = artifact.take_resource_bytes(resource_handle).expect("take");
        assert_eq!(taken.len(), 4096);
        assert_eq!(artifact.stats().retained_resource_bytes, 0);

        // Stats.
        let stats = artifact.stats();
        assert_eq!(stats.device_capacity_bytes, 22272);
        assert_eq!(stats.h2d_bytes, 64 + 4612 + 8704 + 8448);
        assert_eq!(stats.tensor_count, 4);
        assert_eq!(stats.resource_count, 1);
        assert!(stats.upload_seconds >= 0.0);

        // The fixture file is dropped (TempArtifact), then the reader.
        drop(reader);
    }

    #[test]
    fn cpu_materialize_stats_and_staging() {
        let fixture =
            fixture::write_fixture(&fixture::all_layout_objects(), &fixture::all_layout_payload(), "stats")
                .expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let (plan, _) = bind_all(&reader);
        let mut device = CpuDevice::new();
        let artifact = materialize(&reader, &plan, &mut device, None).expect("materialize");

        // Aligned direct-read spans (offsets relative to the payload
        // start, which is 4096-aligned).
        let payload_start = reader.payload_offset();
        assert!(payload_start % 4096 == 0);
        // Host-retained resources are read via the mapping (full bytes).
        let file_len = std::fs::metadata(&fixture.path).expect("meta").len() as u64;
        let mut expected_file = 0u64;
        let mut peak = 0u64;
        for host in &plan.host_objects {
            expected_file += reader.objects()[host.handle.index].bytes();
        }
        // Device objects: 4096-aligned direct-read spans. A span extending
        // past EOF reads a short (capped) count, so `file_bytes` reflects
        // the actual bytes read, not the padded span size.
        for dev in &plan.device_objects {
            let object = &reader.objects()[dev.handle.index];
            let abs = object.offset() + payload_start;
            let begin = abs & !4095;
            let end = (abs + object.bytes() + 4095) & !4095;
            expected_file += end.min(file_len) - begin; // actual bytes read (EOF-capped)
            peak = peak.max(end - begin); // staging span size (max, not sum)
        }
        let stats = artifact.stats();
        assert_eq!(stats.file_bytes, expected_file);
        assert_eq!(stats.peak_staging_bytes, peak);
        drop(reader);
    }

    #[test]
    fn plan_payload_mismatch_rejected() {
        let fixture =
            fixture::write_fixture(&fixture::all_layout_objects(), &fixture::all_layout_payload(), "mismatch")
            .expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let (plan, handles) = bind_all(&reader);
        let mut device = CpuDevice::new();

        // Corrupt one placement: claim fewer bytes than the descriptor.
        let mut bad_plan = plan.clone();
        bad_plan.device_objects[0].bytes = 32;
        let err = materialize(&reader, &bad_plan, &mut device, None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not match artifact payload"),
            "{err}"
        );
        drop(reader);
        let _ = handles;
    }

    #[test]
    fn plan_capacity_shortfall_rejected() {
        let fixture =
            fixture::write_fixture(&fixture::all_layout_objects(), &fixture::all_layout_payload(), "short")
                .expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let (plan, _) = bind_all(&reader);
        let mut device = CpuDevice::new();

        // Shrink the declared arena: the last placement no longer fits.
        let mut bad_plan = plan.clone();
        bad_plan.device_capacity_bytes = 1000;
        let err = materialize(&reader, &bad_plan, &mut device, None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("extends beyond device capacity"),
            "{err}"
        );
        drop(reader);
    }

    #[test]
    fn plan_misaligned_offset_rejected() {
        let fixture =
            fixture::write_fixture(&fixture::all_layout_objects(), &fixture::all_layout_payload(), "misaligned")
                .expect("fixture");
        let reader = Reader::open(&fixture.path).expect("open fixture");
        let (plan, _) = bind_all(&reader);
        let mut device = CpuDevice::new();

        let mut bad_plan = plan.clone();
        bad_plan.device_objects[0].offset = 257;
        let err = materialize(&reader, &bad_plan, &mut device, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("misaligned"), "{err}");
        drop(reader);
    }
}