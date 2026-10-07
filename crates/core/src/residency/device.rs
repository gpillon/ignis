//! Expert residency's device side (spec flash-next/03): the flat C ABI of
//! `kernel/include/ignis_residency.h`, 1:1, and an owning handle over it.
//!
//! The leaf object holds the pinned host expert pool, the eight class pools,
//! the staging ring and every table; [`DeviceResidency`] frees all of it on
//! drop, so the loaded model owns it and nothing is process-wide. Its steps
//! are the CPU policy model's ([`ResidencyModel::step`](super::ResidencyModel::step)),
//! exactly: the leaf's residency test replays the model's committed outcomes,
//! and `crates/core/tests/expert_residency_gpu.rs` compares the two on random
//! traces.
//!
//! Our own implementation, with no port claim (ADR 0010 / ADR 0043).

#![cfg(feature = "cuda")]

use std::ffi::CStr;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use super::class::{ExpertCatalog, KClass, Projection, ProjectionId};
use super::counters::{ResidencyCounters, ResidencyMirror, MIRROR_BYTES};
use super::policy::{Admission, Phase};
use crate::moe::MoeSlot;

/// `prefetch_budget_bytes` meaning "no budget".
pub const NO_BUDGET: u64 = u64::MAX;
/// The status of a step whose ids name an expert outside the catalog.
pub const STATUS_INVALID: u32 = 0x100;
/// Bit 31 of a report entry: the projection went to the staging ring.
pub const STAGING_BIT: u32 = 0x8000_0000;
/// The lists of a step's report, in [`DeviceReport::lists`] order.
pub const LISTS: usize = 6;

/// 1:1 with `struct ignis_residency_desc`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyDesc {
    pub layers: u32,
    pub experts: u32,
    pub capacity: [u32; KClass::COUNT],
    pub record_bytes: [u64; KClass::COUNT],
    pub max_tokens: u32,
    pub lookahead_width: u32,
    pub prefill_lookahead_width: u32,
    /// A one-row decode step's prefetch budget, or [`NO_BUDGET`].
    pub prefetch_budget_bytes: u64,
    /// Added to it for each further row of a decode step's lookahead
    /// ([`PrefetchBudget`](super::PrefetchBudget)'s `per_row_bytes`).
    pub prefetch_budget_row_bytes: u64,
    pub staging_half_bytes: u64,
    pub host_pool_bytes: u64,
    pub copy_blocks: u32,
    pub report: u32,
}

/// 1:1 with `struct ignis_residency_plan`: the device bytes a residency
/// reserves at load, each part rounded up to 256 bytes (ADR 0030).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyPlanBytes {
    pub pools: u64,
    pub staging: u64,
    pub tables: u64,
    pub total: u64,
}

/// 1:1 with `struct ignis_residency_counters`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RawCounters {
    hits: [[u64; 2]; KClass::COUNT],
    misses: [[u64; 2]; KClass::COUNT],
    prefetch_issued: u64,
    prefetch_used: u64,
    bytes_moved: [u64; 2],
    stall_nanos: [u64; 2],
}

/// 1:1 with `struct ignis_residency_report`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RawReport {
    status: u32,
    count: [u32; LISTS],
    reserved: u32,
    bytes_moved: u64,
}

/// 1:1 with `struct ignis_residency_layout`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ResidencyLayout {
    pub pool: [*const c_void; KClass::COUNT],
    pub pool_bytes: [u64; KClass::COUNT],
    pub ring: *const c_void,
    pub ring_bytes: u64,
}

#[repr(C)]
struct RawResidency {
    _private: [u8; 0],
}

mod ffi {
    use super::*;
    use std::os::raw::c_char;

    unsafe extern "C" {
        pub fn ignis_residency_plan_bytes(desc: *const ResidencyDesc, plan: *mut ResidencyPlanBytes) -> i32;
        pub fn ignis_residency_create(
            desc: *const ResidencyDesc,
            k2: *const u8,
            pool_offsets: *const u64,
            out: *mut *mut RawResidency,
        ) -> i32;
        pub fn ignis_residency_free(r: *mut RawResidency);
        pub fn ignis_residency_host_pool(r: *mut RawResidency) -> *mut c_void;
        pub fn ignis_residency_slot_table(r: *mut RawResidency, layer: u32) -> *const MoeSlot;
        pub fn ignis_residency_warm_start(r: *mut RawResidency, keys: *const u32, n: u32, admitted: *mut u32) -> i32;
        pub fn ignis_residency_step(
            r: *mut RawResidency,
            layer: u32,
            phase: u32,
            ids: *const i32,
            tokens: u32,
            lookahead_logits: *const f32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_residency_step_ranked(
            r: *mut RawResidency,
            layer: u32,
            phase: u32,
            ids: *const i32,
            tokens: u32,
            lookahead: *const i32,
            rows: u32,
            stride: u32,
            stream: *mut c_void,
        ) -> i32;
        pub fn ignis_residency_join(r: *mut RawResidency, stream: *mut c_void) -> i32;
        pub fn ignis_residency_read_counters(r: *mut RawResidency, out: *mut RawCounters) -> i32;
        pub fn ignis_residency_read_occupancy(r: *mut RawResidency, out: *mut u32) -> i32;
        pub fn ignis_residency_set_mirror(r: *mut RawResidency, host: *mut c_void) -> i32;
        pub fn ignis_residency_last_report(
            r: *mut RawResidency,
            layer: u32,
            head: *mut RawReport,
            entries: *mut u32,
            capacity: u32,
        ) -> i32;
        pub fn ignis_residency_get_layout(r: *mut RawResidency, out: *mut ResidencyLayout) -> i32;
        pub fn ignis_residency_last_error() -> *const c_char;
    }
}

/// The leaf's message for the most recent failed residency call on this thread.
pub fn last_error() -> String {
    unsafe { CStr::from_ptr(ffi::ignis_residency_last_error()) }.to_string_lossy().into_owned()
}

fn check(rc: i32) -> Result<(), String> {
    if rc == 0 { Ok(()) } else { Err(last_error()) }
}

/// The plan lines a residency of `desc` reserves: the leaf's own numbers.
pub fn plan_bytes(desc: &ResidencyDesc) -> Result<ResidencyPlanBytes, String> {
    let mut plan = ResidencyPlanBytes::default();
    check(unsafe { ffi::ignis_residency_plan_bytes(desc, &mut plan) })?;
    Ok(plan)
}

/// A projection's key: `(layer * experts + expert) * 2 + projection`, its
/// index in the concatenated slot tables.
pub fn key(experts: u16, id: ProjectionId) -> u32 {
    (u32::from(id.layer) * u32::from(experts) + u32::from(id.expert)) * 2
        + u32::from(id.projection == Projection::Down)
}

/// The projection a key names.
pub fn projection_of(experts: u16, key: u32) -> ProjectionId {
    let projection = if key & 1 == 1 { Projection::Down } else { Projection::GateUp };
    let e = key / 2;
    ProjectionId::new((e / u32::from(experts)) as u16, (e % u32::from(experts)) as u16, projection)
}

/// The K map and the records' host-pool offsets in key order, from a
/// catalog whose records are packed back to back (the binder's expert pool).
pub fn packed_layout(catalog: &ExpertCatalog) -> (Vec<u8>, Vec<u64>, u64) {
    let mut k2 = Vec::new();
    let mut offsets = Vec::new();
    let mut at = 0u64;
    for layer in 0..catalog.layers() {
        for expert in 0..catalog.experts() {
            for projection in Projection::ALL {
                let id = ProjectionId::new(layer, expert, projection);
                k2.push(catalog.class_of(id).k.half_bits() as u8);
                offsets.push(at);
                at += catalog.bytes(id);
            }
        }
    }
    (k2, offsets, at)
}

/// One step's outcome as the leaf reports it (a residency created with
/// `report`), lists in key order like
/// [`StepOutcome`](super::StepOutcome)'s.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeviceReport {
    pub status: u32,
    pub hits: Vec<ProjectionId>,
    pub prefetch_hits: Vec<ProjectionId>,
    pub misses: Vec<(ProjectionId, Admission)>,
    pub evictions: Vec<ProjectionId>,
    pub prefetches: Vec<(ProjectionId, Admission)>,
    pub prefetch_dropped: Vec<ProjectionId>,
    pub bytes_moved: u64,
}

/// The leaf's residency object, freed on drop.
pub struct DeviceResidency {
    raw: NonNull<RawResidency>,
    desc: ResidencyDesc,
    /// Kept past the free, which unregisters it, so its readers never see
    /// freed memory.
    mirror: Option<Arc<ResidencyMirror>>,
}

// The mirror is `struct ignis_residency_mirror`: the counters, then the
// slots in use.
const _: () = assert!(MIRROR_BYTES == std::mem::size_of::<RawCounters>() + KClass::COUNT * 4);

// The handle is used from one thread at a time; the leaf has no thread
// affinity beyond the CUDA context.
unsafe impl Send for DeviceResidency {}

impl DeviceResidency {
    /// Creates the residency: every pool empty, every slot-table entry
    /// ABSENT, and the pinned host pool allocated for the caller to fill.
    pub fn new(desc: &ResidencyDesc, k2: &[u8], pool_offsets: &[u64]) -> Result<Self, String> {
        let keys = desc.layers as usize * desc.experts as usize * 2;
        if k2.len() != keys || pool_offsets.len() != keys {
            return Err(format!("residency: {keys} keys, {} K values and {} offsets", k2.len(), pool_offsets.len()));
        }
        let mut raw = std::ptr::null_mut();
        check(unsafe { ffi::ignis_residency_create(desc, k2.as_ptr(), pool_offsets.as_ptr(), &mut raw) })?;
        Ok(Self {
            raw: NonNull::new(raw).ok_or_else(|| "residency: created nothing".to_owned())?,
            desc: *desc,
            mirror: None,
        })
    }

    /// Mirror the counts and the slots in use into `mirror`, for a host
    /// reader to read with no CUDA call; before the first step.
    pub fn mirror(&mut self, mirror: Arc<ResidencyMirror>) -> Result<(), String> {
        if self.mirror.is_some() {
            return Err("residency: already mirrored".to_owned());
        }
        let status = unsafe { ffi::ignis_residency_set_mirror(self.raw.as_ptr(), mirror.as_mut_ptr()) };
        // Kept whatever the outcome: a call that failed may have registered it.
        self.mirror = Some(mirror);
        check(status)
    }

    pub fn desc(&self) -> &ResidencyDesc {
        &self.desc
    }

    /// The leaf object, for a Flash-Next load to borrow
    /// (`ignis_model_load_options::residency`): the load must be freed
    /// before this residency is (GitHub #302).
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw.as_ptr().cast()
    }

    /// The pinned host expert pool, for the loader to fill.
    pub fn host_pool_mut(&mut self) -> &mut [u8] {
        let base = unsafe { ffi::ignis_residency_host_pool(self.raw.as_ptr()) } as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(base, self.desc.host_pool_bytes as usize) }
    }

    /// Layer `layer`'s device slot table, what the expert ops take.
    pub fn slot_table(&self, layer: u32) -> *const MoeSlot {
        unsafe { ffi::ignis_residency_slot_table(self.raw.as_ptr(), layer) }
    }

    /// The optional warm start, hottest first; how many were admitted.
    pub fn warm_start(&mut self, hottest_first: &[ProjectionId]) -> Result<usize, String> {
        let experts = self.desc.experts as u16;
        // A projection outside the catalog is skipped, as the policy skips it; its key
        // would otherwise alias another layer's.
        let keys: Vec<u32> = hottest_first
            .iter()
            .filter(|id| u32::from(id.layer) < self.desc.layers && id.expert < experts)
            .map(|&id| key(experts, id))
            .collect();
        let mut admitted = 0u32;
        check(unsafe {
            ffi::ignis_residency_warm_start(self.raw.as_ptr(), keys.as_ptr(), keys.len() as u32, &mut admitted)
        })?;
        Ok(admitted as usize)
    }

    /// One layer step, the lookahead given as the next router's fp32 logits.
    ///
    /// # Safety
    /// `ids` is a device `[tokens][10]` int32 buffer and `lookahead_logits`
    /// null or a device `[tokens][experts]` fp32 buffer, both live until the
    /// stream has run the step.
    pub unsafe fn step(
        &mut self,
        layer: u32,
        phase: Phase,
        ids: *const i32,
        tokens: u32,
        lookahead_logits: *const f32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_residency_step(self.raw.as_ptr(), layer, phase.index() as u32, ids, tokens, lookahead_logits, stream)
        })
    }

    /// One layer step with the lookahead already ranked.
    ///
    /// # Safety
    /// `ids` is a device `[tokens][10]` int32 buffer and `lookahead` null or a
    /// device `[rows][stride]` int32 buffer, both live until the stream has
    /// run the step.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_ranked(
        &mut self,
        layer: u32,
        phase: Phase,
        ids: *const i32,
        tokens: u32,
        lookahead: *const i32,
        rows: u32,
        stride: u32,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_residency_step_ranked(
                self.raw.as_ptr(),
                layer,
                phase.index() as u32,
                ids,
                tokens,
                lookahead,
                rows,
                stride,
                stream,
            )
        })
    }

    /// Joins the last step's prefetch copies into `stream`.
    pub fn join(&mut self, stream: *mut c_void) -> Result<(), String> {
        check(unsafe { ffi::ignis_residency_join(self.raw.as_ptr(), stream) })
    }

    /// What the device counted; waits for its work.
    pub fn counters(&self) -> Result<ResidencyCounters, String> {
        let mut raw = RawCounters::default();
        check(unsafe { ffi::ignis_residency_read_counters(self.raw.as_ptr(), &mut raw) })?;
        Ok(ResidencyCounters {
            hits: raw.hits,
            misses: raw.misses,
            prefetch_issued: raw.prefetch_issued,
            prefetch_used: raw.prefetch_used,
            bytes_moved: raw.bytes_moved,
            stall_nanos: raw.stall_nanos,
        })
    }

    /// Slots holding a projection, per class in `KClass::index` order (the
    /// policy model's `occupancy`); waits for the device.
    pub fn slots_in_use(&self) -> Result<[u32; KClass::COUNT], String> {
        let mut out = [0u32; KClass::COUNT];
        check(unsafe { ffi::ignis_residency_read_occupancy(self.raw.as_ptr(), out.as_mut_ptr()) })?;
        Ok(out)
    }

    /// The outcome of the last step of `layer`, every list sorted into key
    /// order; waits for the device.
    pub fn last_report(&self, layer: u32) -> Result<DeviceReport, String> {
        let capacity = 4 * self.desc.experts;
        let mut head = RawReport::default();
        let mut entries = vec![0u32; LISTS * capacity as usize];
        check(unsafe {
            ffi::ignis_residency_last_report(self.raw.as_ptr(), layer, &mut head, entries.as_mut_ptr(), capacity)
        })?;
        let experts = self.desc.experts as u16;
        let list = |l: usize| -> Vec<u32> {
            let start = l * capacity as usize;
            let mut v = entries[start..start + head.count[l] as usize].to_vec();
            v.sort_unstable_by_key(|k| k & !STAGING_BIT);
            v
        };
        let ids = |l: usize| list(l).into_iter().map(|k| projection_of(experts, k)).collect::<Vec<_>>();
        let admitted = |l: usize| {
            list(l)
                .into_iter()
                .map(|k| {
                    let admission = if k & STAGING_BIT != 0 { Admission::Staging } else { Admission::Slot };
                    (projection_of(experts, k & !STAGING_BIT), admission)
                })
                .collect::<Vec<_>>()
        };
        if head.status != 0 {
            return Ok(DeviceReport { status: head.status, ..Default::default() });
        }
        Ok(DeviceReport {
            status: 0,
            hits: ids(0),
            prefetch_hits: ids(1),
            misses: admitted(2),
            evictions: ids(3),
            prefetches: admitted(4),
            prefetch_dropped: ids(5),
            bytes_moved: head.bytes_moved,
        })
    }

    /// Where the pools and the ring sit.
    pub fn layout(&self) -> Result<ResidencyLayout, String> {
        let mut out = ResidencyLayout {
            pool: [std::ptr::null(); KClass::COUNT],
            pool_bytes: [0; KClass::COUNT],
            ring: std::ptr::null(),
            ring_bytes: 0,
        };
        check(unsafe { ffi::ignis_residency_get_layout(self.raw.as_ptr(), &mut out) })?;
        Ok(out)
    }
}

impl Drop for DeviceResidency {
    fn drop(&mut self) {
        unsafe { ffi::ignis_residency_free(self.raw.as_ptr()) }
    }
}
