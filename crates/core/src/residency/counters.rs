//! What expert residency counts (spec flash-next/03): hits and misses per
//! class and phase, prefetches issued and used, bytes moved, the time the
//! expert kernels waited on their demand copies, from the first step on. Facts only: the server's
//! exposition names and renders them (ADR 0017), and nothing here knows it
//! exists.

use std::alloc::Layout;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::class::KClass;

/// Everything residency counts, from the first step on. `[class][phase]`
/// arrays are indexed like [`KClass::ALL`] and
/// [`Phase::ALL`](super::policy::Phase::ALL).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyCounters {
    /// Selected projections already resident, or staged for their layer.
    pub hits: [[u64; 2]; KClass::COUNT],
    /// Selected projections copied in by their own step.
    pub misses: [[u64; 2]; KClass::COUNT],
    /// Projections copied ahead for the next layer.
    pub prefetch_issued: u64,
    /// Prefetched projections at their first use, whenever it comes.
    pub prefetch_used: u64,
    /// Bytes copied host-to-device, misses and prefetches, by phase.
    pub bytes_moved: [u64; 2],
    /// Time the expert kernels waited on their steps' demand copies, by
    /// phase: each copy from its first block's start to its last block's
    /// end (a step with no miss copies nothing and adds nothing). The wait
    /// for a previous step's unfinished prefetch is not in it. The device
    /// times it; the CPU policy model has no clock and leaves it at zero.
    pub stall_nanos: [u64; 2],
}

/// The counts and the slots in use as the device mirrors them into host
/// memory (`struct ignis_residency_mirror`): a page of its own, page-locked
/// and mapped by the residency it is handed to, which writes the totals
/// there at the last layer of every step, and each demand copy its phase's
/// stall. A reader on any thread reads it
/// with no CUDA call and no wait; it outlives the residency, which only
/// stops writing it, so a reader never holds device resources.
pub struct ResidencyMirror {
    cells: NonNull<MirrorCells>,
}

/// 1:1 with `struct ignis_residency_mirror`, each word an atomic: the device
/// stores every one whole.
#[repr(C)]
struct MirrorCells {
    hits: [[AtomicU64; 2]; KClass::COUNT],
    misses: [[AtomicU64; 2]; KClass::COUNT],
    prefetch_issued: AtomicU64,
    prefetch_used: AtomicU64,
    bytes_moved: [AtomicU64; 2],
    stall_nanos: [AtomicU64; 2],
    in_use: [AtomicU32; KClass::COUNT],
}

/// The size of `struct ignis_residency_mirror`.
pub const MIRROR_BYTES: usize = 38 * 8 + KClass::COUNT * 4;
const _: () = assert!(std::mem::size_of::<MirrorCells>() == MIRROR_BYTES);

/// One page: what the residency page-locks.
const MIRROR_PAGE: usize = 4096;

// Atomics only, and the page is never freed while a handle exists.
unsafe impl Send for ResidencyMirror {}
unsafe impl Sync for ResidencyMirror {}

impl Default for ResidencyMirror {
    fn default() -> Self {
        Self::new()
    }
}

impl ResidencyMirror {
    /// A zeroed page.
    pub fn new() -> Self {
        let layout = Self::layout();
        // Safety: a non-zero size; all-zero bytes are valid atomics.
        let page = unsafe { std::alloc::alloc_zeroed(layout) };
        let cells = NonNull::new(page.cast::<MirrorCells>()).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { cells }
    }

    fn layout() -> Layout {
        Layout::from_size_align(MIRROR_PAGE, MIRROR_PAGE).expect("a page")
    }

    fn cells(&self) -> &MirrorCells {
        // Safety: allocated and zeroed in `new`, freed only in `drop`.
        unsafe { self.cells.as_ref() }
    }

    /// The address the residency registers and writes.
    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.cells.as_ptr().cast()
    }

    /// The counts, series by series: each only grows.
    pub fn counters(&self) -> ResidencyCounters {
        let c = self.cells();
        let load = |cell: &AtomicU64| cell.load(Ordering::Relaxed);
        ResidencyCounters {
            hits: c.hits.each_ref().map(|phases| phases.each_ref().map(load)),
            misses: c.misses.each_ref().map(|phases| phases.each_ref().map(load)),
            prefetch_issued: load(&c.prefetch_issued),
            prefetch_used: load(&c.prefetch_used),
            bytes_moved: c.bytes_moved.each_ref().map(load),
            stall_nanos: c.stall_nanos.each_ref().map(load),
        }
    }

    /// Slots holding a projection, per class in [`KClass::index`] order.
    pub fn slots_in_use(&self) -> [u32; KClass::COUNT] {
        self.cells().in_use.each_ref().map(|cell| cell.load(Ordering::Relaxed))
    }

    /// Write `counters` and `in_use` as the device does: for a host-side
    /// writer (the tests, a CPU residency).
    pub fn store(&self, counters: &ResidencyCounters, in_use: &[u32; KClass::COUNT]) {
        let c = self.cells();
        let store = |cell: &AtomicU64, value: u64| cell.store(value, Ordering::Relaxed);
        for class in 0..KClass::COUNT {
            for phase in 0..2 {
                store(&c.hits[class][phase], counters.hits[class][phase]);
                store(&c.misses[class][phase], counters.misses[class][phase]);
            }
            c.in_use[class].store(in_use[class], Ordering::Relaxed);
        }
        store(&c.prefetch_issued, counters.prefetch_issued);
        store(&c.prefetch_used, counters.prefetch_used);
        for phase in 0..2 {
            store(&c.bytes_moved[phase], counters.bytes_moved[phase]);
        }
        for phase in 0..2 {
            store(&c.stall_nanos[phase], counters.stall_nanos[phase]);
        }
    }
}

impl Drop for ResidencyMirror {
    fn drop(&mut self) {
        // Safety: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.cells.as_ptr().cast(), Self::layout()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mirror_reads_back_what_was_stored_and_starts_at_zero() {
        let mirror = ResidencyMirror::new();
        assert_eq!(mirror.counters(), ResidencyCounters::default());
        assert_eq!(mirror.slots_in_use(), [0; KClass::COUNT]);
        assert_eq!(mirror.as_mut_ptr() as usize % MIRROR_PAGE, 0, "a page of its own");

        let mut counters = ResidencyCounters::default();
        counters.hits[2][0] = 9;
        counters.misses[7][1] = 4;
        counters.prefetch_issued = 3;
        counters.prefetch_used = 2;
        counters.bytes_moved = [10, 20];
        counters.stall_nanos = [1, 30];
        let in_use = [1, 2, 3, 4, 5, 6, 7, 8];
        mirror.store(&counters, &in_use);
        assert_eq!(mirror.counters(), counters);
        assert_eq!(mirror.slots_in_use(), in_use);
    }
}
