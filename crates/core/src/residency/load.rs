//! Filling the pinned host expert pool at load (spec flash-next/03,
//! acceptance 1): every expert projection, read from the artifact into the
//! pool at the offset the Flash-Next binder's expert index lays out, with
//! plain buffered reads (so a later load finds the file in the page cache).
//! The pool itself is the leaf residency's (`ignis_residency_host_pool`);
//! this module only reads into a byte slice, so it is pinned on the CPU.
//!
//! One handle reading the whole pool sequentially holds this drive to
//! ~1.3 GiB/s; splitting the layers across a few handles reading in
//! parallel reaches its ~2.8 GiB/s ceiling (measured on the real artifact,
//! `docs/findings/2026-10-09-flash-next-expert-pool-parallel-read.md`).
//! [`fill_expert_pool`] opens [`READ_WORKERS`] handles on `path` and gives
//! each a disjoint, round-robined set of layers.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use ignis_artifact::flash_next::{ExpertIndex, Projection as ArtifactProjection};

use super::class::{ExpertCatalog, KBits, KClass, Projection};

/// How many file handles [`fill_expert_pool`] reads with at once: past this
/// the drive is already saturated (measured 4-6, no further gain, 8+
/// regresses from contention).
const READ_WORKERS: usize = 4;

/// The pool as the leaf residency takes it: each projection's K (as `k2`)
/// and pool offset in key order, and the pool's size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolLayout {
    pub k2: Vec<u8>,
    pub offsets: Vec<u64>,
    pub bytes: u64,
}

/// The pool layout of a bound artifact's expert index.
pub fn pool_layout(index: &ExpertIndex) -> PoolLayout {
    let (mut k2, mut offsets, mut bytes) = (Vec::new(), Vec::new(), 0u64);
    for layer in 0..index.layers() {
        for expert in 0..index.experts_per_layer() {
            for projection in ArtifactProjection::ALL {
                let record = index.get(layer, expert, projection).expect("every projection is indexed");
                k2.push(record.k.k2());
                offsets.push(record.pool_offset);
                bytes = bytes.max(record.pool_offset + record.bytes);
            }
        }
    }
    PoolLayout { k2, offsets, bytes }
}

/// The residency catalog of a bound artifact: its K map and each class's
/// record bytes, read off the index (every record of a class has one size).
pub fn catalog(index: &ExpertIndex) -> Result<ExpertCatalog, String> {
    let mut slot_bytes = [0u64; KClass::COUNT];
    let mut map = Vec::new();
    for layer in 0..index.layers() {
        for expert in 0..index.experts_per_layer() {
            let mut ks = [KBits::K2; 2];
            for (i, projection) in ArtifactProjection::ALL.into_iter().enumerate() {
                let record = index.get(layer, expert, projection).expect("every projection is indexed");
                let k = KBits::from_half_bits(u32::from(record.k.k2()))
                    .ok_or_else(|| format!("k2 {} is not a K class", record.k.k2()))?;
                let class = KClass::new(if i == 0 { Projection::GateUp } else { Projection::Down }, k);
                let at = &mut slot_bytes[class.index()];
                if *at != 0 && *at != record.bytes {
                    return Err(format!("{} records of {} and {} bytes", class.as_str(), at, record.bytes));
                }
                *at = record.bytes;
                ks[i] = k;
            }
            map.push((ks[0], ks[1]));
        }
    }
    // A class no projection uses still needs a size: one that copies nothing
    // wrong if it were ever asked (it never is).
    for (i, bytes) in slot_bytes.iter_mut().enumerate() {
        if *bytes == 0 {
            *bytes = 16 * (i as u64 + 1);
        }
    }
    ExpertCatalog::new(index.layers() as u16, index.experts_per_layer() as u16, map, slot_bytes)
        .map_err(|e| e.to_string())
}

/// Each layer's `[lo, hi)` byte range in the pool: the min and max of its
/// records' `pool_offset`/`pool_offset + bytes`, indexed by layer. Ranges
/// must be ascending and non-overlapping (gaps between them are fine --
/// nobody reads into them) so each layer's region can be carved off the
/// pool as its own disjoint `&mut [u8]`.
fn layer_pool_ranges(index: &ExpertIndex, pool_len: usize) -> std::io::Result<Vec<(u64, u64)>> {
    let mut ranges = Vec::with_capacity(index.layers());
    for layer in 0..index.layers() {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for expert in 0..index.experts_per_layer() {
            for projection in ArtifactProjection::ALL {
                let r = index.get(layer, expert, projection).expect("indexed");
                lo = lo.min(r.pool_offset);
                hi = hi.max(r.pool_offset + r.bytes);
            }
        }
        if hi as usize > pool_len {
            return Err(std::io::Error::other(format!("layer {layer} ends at {hi}, past the {pool_len}-byte pool")));
        }
        ranges.push((lo, hi));
    }
    for i in 1..ranges.len() {
        if ranges[i].0 < ranges[i - 1].1 {
            return Err(std::io::Error::other(format!("layer {i}'s pool range overlaps layer {}'s", i - 1)));
        }
    }
    Ok(ranges)
}

/// Reads one layer's projections from `file` into `slice`, `slice[0]`
/// standing for pool offset `lo` (the layer's own range, carved from the
/// full pool by [`fill_expert_pool`]): one read for the whole layer where
/// its records sit back to back in both pool and file (the converter's
/// layout), one read per record otherwise.
fn fill_layer<F: Read + Seek>(
    file: &mut F,
    index: &ExpertIndex,
    layer: usize,
    lo: u64,
    slice: &mut [u8],
) -> std::io::Result<u64> {
    let mut read = 0u64;
    let first = index.get(layer, 0, ArtifactProjection::GateUp).expect("a layer has experts");
    let contiguous_in_pool = index.layer_range(layer).filter(|&(start, len)| {
        (0..index.experts_per_layer()).all(|e| {
            ArtifactProjection::ALL.into_iter().all(|p| {
                let r = index.get(layer, e, p).expect("indexed");
                r.pool_offset - first.pool_offset == r.file_offset - start
            })
        }) && first.pool_offset + len <= lo + slice.len() as u64
    });
    match contiguous_in_pool {
        Some((start, len)) => {
            file.seek(SeekFrom::Start(start))?;
            let at = (first.pool_offset - lo) as usize;
            file.read_exact(&mut slice[at..at + len as usize])?;
            read += len;
        }
        None => {
            for expert in 0..index.experts_per_layer() {
                for projection in ArtifactProjection::ALL {
                    let r = index.get(layer, expert, projection).expect("indexed");
                    let at = (r.pool_offset - lo) as usize;
                    let end = at + r.bytes as usize;
                    if end > slice.len() {
                        return Err(std::io::Error::other(format!(
                            "record of layer {layer} expert {expert} ends at {end}, past its {}-byte layer slice",
                            slice.len()
                        )));
                    }
                    file.seek(SeekFrom::Start(r.file_offset))?;
                    file.read_exact(&mut slice[at..end])?;
                    read += r.bytes;
                }
            }
        }
    }
    Ok(read)
}

/// Reads every expert projection of the artifact at `path` into `pool` at
/// its pool offset, [`READ_WORKERS`] layers at a time in parallel (one
/// handle per worker, round-robined over the layers). Returns the bytes
/// read.
pub fn fill_expert_pool(path: &Path, index: &ExpertIndex, pool: &mut [u8]) -> std::io::Result<u64> {
    let ranges = layer_pool_ranges(index, pool.len())?;
    let workers = READ_WORKERS.max(1);
    let mut buckets: Vec<Vec<(usize, u64, &mut [u8])>> = (0..workers).map(|_| Vec::new()).collect();
    let mut rest = pool;
    let mut cursor = 0u64;
    for (layer, &(lo, hi)) in ranges.iter().enumerate() {
        let (_gap, remainder) = rest.split_at_mut((lo - cursor) as usize);
        let (slice, remainder) = remainder.split_at_mut((hi - lo) as usize);
        buckets[layer % workers].push((layer, lo, slice));
        rest = remainder;
        cursor = hi;
    }
    let total = AtomicU64::new(0);
    thread::scope(|scope| -> std::io::Result<()> {
        let mut handles = Vec::new();
        for bucket in buckets {
            if bucket.is_empty() {
                continue;
            }
            let total = &total;
            handles.push(scope.spawn(move || -> std::io::Result<()> {
                let mut file = File::open(path)?;
                for (layer, lo, slice) in bucket {
                    let read = fill_layer(&mut file, index, layer, lo, slice)?;
                    total.fetch_add(read, Ordering::Relaxed);
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().expect("expert pool read worker panicked")?;
        }
        Ok(())
    })?;
    Ok(total.load(Ordering::Relaxed))
}
