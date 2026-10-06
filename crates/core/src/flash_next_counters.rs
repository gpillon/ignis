//! What a Flash-Next load counts beside the scheduler (GitHub #301, #302):
//! its expert residency ([`ResidencyCounters`], the K-class pools' slots)
//! and its n-gram table ([`NgramCounters`]). The leaf, which alone may read
//! the device, publishes them into a [`FlashNextCounterCell`] after every
//! step, whoever reads it; a reader on another thread reads the cell when it
//! likes. Facts only, as [`ResidencyCounters`]: nothing here knows who reads.
//!
//! Atomics, no lock: a reader never makes a step wait, and a step never
//! makes a reader wait. A read therefore takes each series on its own, not
//! one consistent cut of all of them. The cell holds numbers only, so a
//! reader that outlives the model holds none of the model's resources.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::ngram_table::NgramCounters;
use crate::residency::{KClass, ResidencyCounters};

/// One reading of a Flash-Next load's counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlashNextCounters {
    pub residency: ResidencyCounters,
    /// The VRAM expert cache's slots per K class, reserved at load.
    pub slots_capacity: [u32; KClass::COUNT],
    /// Of them, the slots holding a projection; `None` while residency
    /// exports no occupancy.
    pub slots_in_use: Option<[u32; KClass::COUNT]>,
    pub ngram: NgramCounters,
}

/// [`FlashNextCounters`] as atomics, written by the leaf after every step
/// and read by whoever holds the cell.
#[derive(Debug)]
pub struct FlashNextCounterCell {
    hits: [[AtomicU64; 2]; KClass::COUNT],
    misses: [[AtomicU64; 2]; KClass::COUNT],
    prefetch_issued: AtomicU64,
    prefetch_used: AtomicU64,
    bytes_moved: [AtomicU64; 2],
    stall_nanos: AtomicU64,
    slots_capacity: [AtomicU64; KClass::COUNT],
    /// `NO_OCCUPANCY` for a reading without it.
    slots_in_use: [AtomicU64; KClass::COUNT],
    /// The rows the file served, not all rows: each series a reader sees
    /// then only grows, where a total read beside the hot rows could, mid
    /// publish, put fewer rows than before on the file.
    ngram_file_rows: AtomicU64,
    ngram_hot_rows: AtomicU64,
    ngram_reads: AtomicU64,
    ngram_read_bytes: AtomicU64,
}

/// What `slots_in_use` holds for a reading without occupancy: no `u32` count
/// reaches it.
const NO_OCCUPANCY: u64 = u64::MAX;

impl Default for FlashNextCounterCell {
    /// Every count zero, no occupancy.
    fn default() -> Self {
        Self {
            hits: Default::default(),
            misses: Default::default(),
            prefetch_issued: AtomicU64::new(0),
            prefetch_used: AtomicU64::new(0),
            bytes_moved: Default::default(),
            stall_nanos: AtomicU64::new(0),
            slots_capacity: Default::default(),
            slots_in_use: std::array::from_fn(|_| AtomicU64::new(NO_OCCUPANCY)),
            ngram_file_rows: AtomicU64::new(0),
            ngram_hot_rows: AtomicU64::new(0),
            ngram_reads: AtomicU64::new(0),
            ngram_read_bytes: AtomicU64::new(0),
        }
    }
}

impl FlashNextCounterCell {
    /// Replace every series with `counters`'.
    pub fn publish(&self, counters: &FlashNextCounters) {
        let store = |cell: &AtomicU64, value: u64| cell.store(value, Ordering::Relaxed);
        let r = &counters.residency;
        for class in 0..KClass::COUNT {
            for phase in 0..2 {
                store(&self.hits[class][phase], r.hits[class][phase]);
                store(&self.misses[class][phase], r.misses[class][phase]);
            }
            store(&self.slots_capacity[class], u64::from(counters.slots_capacity[class]));
            store(
                &self.slots_in_use[class],
                counters.slots_in_use.map_or(NO_OCCUPANCY, |in_use| u64::from(in_use[class])),
            );
        }
        store(&self.prefetch_issued, r.prefetch_issued);
        store(&self.prefetch_used, r.prefetch_used);
        for phase in 0..2 {
            store(&self.bytes_moved[phase], r.bytes_moved[phase]);
        }
        store(&self.stall_nanos, r.stall_nanos);
        let n = &counters.ngram;
        store(&self.ngram_file_rows, n.rows.saturating_sub(n.hot_rows));
        store(&self.ngram_hot_rows, n.hot_rows);
        store(&self.ngram_reads, n.reads);
        store(&self.ngram_read_bytes, n.read_bytes);
    }

    /// The latest reading, series by series.
    pub fn read(&self) -> FlashNextCounters {
        let load = |cell: &AtomicU64| cell.load(Ordering::Relaxed);
        let in_use = self.slots_in_use.each_ref().map(load);
        let hot_rows = load(&self.ngram_hot_rows);
        FlashNextCounters {
            residency: ResidencyCounters {
                hits: self.hits.each_ref().map(|phases| phases.each_ref().map(load)),
                misses: self.misses.each_ref().map(|phases| phases.each_ref().map(load)),
                prefetch_issued: load(&self.prefetch_issued),
                prefetch_used: load(&self.prefetch_used),
                bytes_moved: self.bytes_moved.each_ref().map(load),
                stall_nanos: load(&self.stall_nanos),
            },
            slots_capacity: self.slots_capacity.each_ref().map(|c| load(c) as u32),
            slots_in_use: (!in_use.contains(&NO_OCCUPANCY)).then(|| in_use.map(|c| c as u32)),
            ngram: NgramCounters {
                rows: load(&self.ngram_file_rows) + hot_rows,
                hot_rows,
                reads: load(&self.ngram_reads),
                read_bytes: load(&self.ngram_read_bytes),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residency::{KBits, Projection};

    #[test]
    fn a_published_snapshot_reads_back_whole_and_a_new_cell_reads_zeros() {
        let cell = FlashNextCounterCell::default();
        assert_eq!(cell.read(), FlashNextCounters::default());

        let down_k3 = KClass::new(Projection::Down, KBits::K3);
        let mut counters = FlashNextCounters::default();
        counters.residency.hits[down_k3.index()][1] = 11;
        counters.residency.misses[0][0] = 2;
        counters.residency.prefetch_issued = 5;
        counters.residency.prefetch_used = 4;
        counters.residency.bytes_moved = [700, 900];
        counters.residency.stall_nanos = 3;
        counters.slots_capacity[down_k3.index()] = 40;
        counters.slots_in_use = Some([1, 2, 3, 4, 5, 6, 7, 8]);
        counters.ngram = NgramCounters { rows: 160, hot_rows: 150, reads: 3, read_bytes: 12_288 };
        cell.publish(&counters);
        assert_eq!(cell.read(), counters);

        // A later publish replaces every series, the occupancy's absence too.
        let later = FlashNextCounters { slots_in_use: None, ..FlashNextCounters::default() };
        cell.publish(&later);
        assert_eq!(cell.read(), later);
    }
}
