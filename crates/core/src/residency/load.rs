//! Filling the pinned host expert pool at load (spec flash-next/03,
//! acceptance 1): every expert projection, read from the artifact into the
//! pool at the offset the Flash-Next binder's expert index lays out, layer by
//! layer, with plain buffered reads (so a later load finds the file in the
//! page cache). The pool itself is the leaf residency's
//! (`ignis_residency_host_pool`); this module only reads into a byte slice,
//! so it is pinned on the CPU.

use std::io::{Read, Seek, SeekFrom};

use ignis_artifact::flash_next::{ExpertIndex, Projection as ArtifactProjection};

use super::class::{ExpertCatalog, KBits, KClass, Projection};

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

/// Reads every expert projection of `file` into `pool` at its pool offset,
/// one read per layer where the records sit back to back in both (the
/// converter's layout), one per record otherwise. Returns the bytes read.
pub fn fill_expert_pool<F: Read + Seek>(file: &mut F, index: &ExpertIndex, pool: &mut [u8]) -> std::io::Result<u64> {
    let mut read = 0u64;
    for layer in 0..index.layers() {
        let first = index.get(layer, 0, ArtifactProjection::GateUp).expect("a layer has experts");
        let contiguous_in_pool = index.layer_range(layer).filter(|&(start, len)| {
            (0..index.experts_per_layer()).all(|e| {
                ArtifactProjection::ALL.into_iter().all(|p| {
                    let r = index.get(layer, e, p).expect("indexed");
                    r.pool_offset - first.pool_offset == r.file_offset - start
                })
            }) && first.pool_offset + len <= pool.len() as u64
        });
        match contiguous_in_pool {
            Some((start, len)) => {
                file.seek(SeekFrom::Start(start))?;
                let at = first.pool_offset as usize;
                file.read_exact(&mut pool[at..at + len as usize])?;
                read += len;
            }
            None => {
                for expert in 0..index.experts_per_layer() {
                    for projection in ArtifactProjection::ALL {
                        let r = index.get(layer, expert, projection).expect("indexed");
                        let at = r.pool_offset as usize;
                        let end = at + r.bytes as usize;
                        if end > pool.len() {
                            return Err(std::io::Error::other(format!(
                                "record of layer {layer} expert {expert} ends at {end}, past the {}-byte pool",
                                pool.len()
                            )));
                        }
                        file.seek(SeekFrom::Start(r.file_offset))?;
                        file.read_exact(&mut pool[at..end])?;
                        read += r.bytes;
                    }
                }
            }
        }
    }
    Ok(read)
}
