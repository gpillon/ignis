//! The version-2 snapshot blob layout (`kernel/include/ignis_seq_sections.h`),
//! for the GPU tests that read what a snapshot carries (P5-03, GitHub #152;
//! P5-05, GitHub #155). The layout is leaf internal (ADR 0024): a test
//! asserts `snapshot_format_version() == 2` before trusting these offsets,
//! so a format change fails that assertion rather than misreading bytes.

// Each including test reads the sections it needs.
#![allow(dead_code)]

pub const HEADER_BYTES: usize = 128;
pub const RECORD_BYTES: usize = 24;
pub const SECTION_PROGRESS: i32 = 4;
pub const SECTION_DFLASH_WINDOW: i32 = 5;
pub const SECTION_DFLASH_CHECKPOINT: i32 = 6;

/// The progress image's offset of the drafter window's frontier.
pub const PROGRESS_DRAFTER_FRONTIER: usize = 16;

pub fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

pub fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

/// `(offset, bytes)` of the blob's section of `kind`.
pub fn section(blob: &[u8], kind: i32) -> (usize, usize) {
    assert_eq!(read_u32(blob, 12) as usize, HEADER_BYTES, "snapshot header bytes");
    assert_eq!(read_u32(blob, 16) as usize, RECORD_BYTES, "snapshot record bytes");
    let count = read_u32(blob, 20) as usize;
    for i in 0..count {
        let at = HEADER_BYTES + i * RECORD_BYTES;
        if read_u32(blob, at) as i32 == kind {
            return (read_u64(blob, at + 8) as usize, read_u64(blob, at + 16) as usize);
        }
    }
    panic!("the snapshot lists no section of kind {kind}");
}
