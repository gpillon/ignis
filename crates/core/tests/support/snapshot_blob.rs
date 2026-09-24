//! The version-5 snapshot blob layout (`kernel/include/ignis_seq_sections.h`),
//! for the GPU tests that read what a snapshot carries (P5-03, GitHub #152;
//! P5-05, GitHub #155; GitHub #194; GitHub #257). The layout is leaf internal
//! (ADR 0024): a test asserts `snapshot_format_version() == 5` before trusting
//! these offsets, so a format change fails that assertion rather than
//! misreading bytes.

// Each including test reads the sections it needs.
#![allow(dead_code)]

pub const HEADER_BYTES: usize = 128;
pub const RECORD_BYTES: usize = 24;
pub const SECTION_PROGRESS: i32 = 4;
pub const SECTION_DFLASH_WINDOW: i32 = 5;
/// The kind the drafter window's rewrite checkpoint had until format 5,
/// which stopped carrying it: never listed any more, and never reused.
pub const SECTION_RETIRED_DFLASH_CHECKPOINT: i32 = 6;
/// An hq-e8-2b pool's residual window (GitHub #257): every GQA layer's K side
/// plane, then every layer's V side plane, then the 16 ring words.
pub const SECTION_HQ_RESIDUAL: i32 = 7;

/// The progress image's offset of the multimodal rope delta (GitHub #194).
pub const PROGRESS_ROPE_DELTA: usize = 12;

/// The progress image's offset of the drafter window's frontier.
pub const PROGRESS_DRAFTER_FRONTIER: usize = 16;

pub fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub fn read_i32(bytes: &[u8], at: usize) -> i32 {
    i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// `(offset, bytes)` of the blob's section of `kind`, if it lists one.
pub fn find_section(blob: &[u8], kind: i32) -> Option<(usize, usize)> {
    assert_eq!(read_u32(blob, 12) as usize, HEADER_BYTES, "snapshot header bytes");
    assert_eq!(read_u32(blob, 16) as usize, RECORD_BYTES, "snapshot record bytes");
    let count = read_u32(blob, 20) as usize;
    (0..count).map(|i| HEADER_BYTES + i * RECORD_BYTES).find_map(|at| {
        (read_u32(blob, at) as i32 == kind)
            .then(|| (read_u64(blob, at + 8) as usize, read_u64(blob, at + 16) as usize))
    })
}

/// `(offset, bytes)` of the blob's section of `kind`.
pub fn section(blob: &[u8], kind: i32) -> (usize, usize) {
    find_section(blob, kind).unwrap_or_else(|| panic!("the snapshot lists no section of kind {kind}"))
}

/// The hq residual window section of `blob` (GitHub #257): every layer's K and
/// V side planes, then the 16 ring words.
pub fn hq_window(blob: &[u8]) -> &[u8] {
    let (at, bytes) = section(blob, SECTION_HQ_RESIDUAL);
    &blob[at..at + bytes]
}

/// The ring validity words the hq window section ends with, as the device
/// holds them.
pub fn hq_ring_words(blob: &[u8]) -> ignis_core::hq_ring::RingWords {
    let window = hq_window(blob);
    let ring = &window[window.len() - ignis_core::hq_ring::RING_WORDS * 4..];
    std::array::from_fn(|i| read_u32(ring, i * 4))
}
