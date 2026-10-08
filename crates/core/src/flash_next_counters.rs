//! What a Flash-Next load counts beside the scheduler (GitHub #301, #302):
//! its expert residency, as the device mirrors it into host memory
//! ([`ResidencyMirror`]), the K-class pools' slots, and its n-gram table's
//! rows ([`NgramCounts`]). A [`FlashNextCounterSource`] reads them on any
//! thread with no CUDA call and no wait. The leaf builds it at open and does
//! nothing for it after: a step costs the same whoever reads, or whether
//! anyone does.
//!
//! Numbers only: a reader that outlives the model holds none of the model's
//! resources, and reads the last totals the model wrote.

use std::sync::Arc;

use crate::ngram_table::{NgramCounters, NgramCounts};
use crate::residency::{KClass, ResidencyCounters, ResidencyMirror};

/// One reading of a Flash-Next load's counters, each series on its own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlashNextCounters {
    pub residency: ResidencyCounters,
    /// The VRAM expert cache's slots per K class, reserved at load.
    pub slots_capacity: [u32; KClass::COUNT],
    /// Of them, the slots holding a projection.
    pub slots_in_use: [u32; KClass::COUNT],
    pub ngram: NgramCounters,
}

/// Where a reader reads a Flash-Next load's counters.
pub struct FlashNextCounterSource {
    residency: Arc<ResidencyMirror>,
    slots_capacity: [u32; KClass::COUNT],
    ngram: Arc<NgramCounts>,
}

impl std::fmt::Debug for FlashNextCounterSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlashNextCounterSource").field("slots_capacity", &self.slots_capacity).finish_non_exhaustive()
    }
}

impl FlashNextCounterSource {
    /// The mirror residency writes, the pools' capacity (the host knows it
    /// from the plan), and the n-gram table's counts.
    pub fn new(residency: Arc<ResidencyMirror>, slots_capacity: [u32; KClass::COUNT], ngram: Arc<NgramCounts>) -> Self {
        Self { residency, slots_capacity, ngram }
    }

    /// What the device and the table last wrote.
    pub fn read(&self) -> FlashNextCounters {
        FlashNextCounters {
            residency: self.residency.counters(),
            slots_capacity: self.slots_capacity,
            slots_in_use: self.residency.slots_in_use(),
            ngram: self.ngram.read(),
        }
    }

    /// The wall time the table's prefill gathers have taken, in nanoseconds
    /// (spec vram-budget/03 AC 25).
    pub fn ngram_prefill_gather_nanos(&self) -> u64 {
        self.ngram.prefill_gather_nanos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_reads_the_mirror_the_capacity_and_the_table_counts() {
        let mirror = Arc::new(ResidencyMirror::new());
        let ngram = Arc::new(NgramCounts::default());
        let capacity = [40, 0, 0, 12, 40, 0, 6, 0];
        let source = FlashNextCounterSource::new(Arc::clone(&mirror), capacity, Arc::clone(&ngram));
        assert_eq!(
            source.read(),
            FlashNextCounters { slots_capacity: capacity, ..FlashNextCounters::default() },
            "before any step: the capacity, and zeros"
        );

        let mut residency = ResidencyCounters::default();
        residency.hits[3][0] = 11;
        residency.bytes_moved = [700, 900];
        mirror.store(&residency, &[40, 0, 0, 5, 1, 0, 0, 0]);
        ngram.record(150, 10, 3, 12_288);
        let reading = source.read();
        assert_eq!(reading.residency, residency);
        assert_eq!(reading.slots_in_use, [40, 0, 0, 5, 1, 0, 0, 0]);
        assert_eq!(
            reading.ngram,
            NgramCounters { rows: 160, hot_rows: 150, file_rows: 10, reads: 3, read_bytes: 12_288 }
        );
    }
}
