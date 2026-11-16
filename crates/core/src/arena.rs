//! The leaf's scratch arena as the VRAM plan counts it: every allocation
//! starts on a block of [`ALIGN`] bytes (`kArenaAlign`,
//! `DeviceArena::alloc_bytes`, `kernel/src/model.cu`), so a tensor takes its
//! bytes rounded up to whole blocks. One copy of the rounding for the plan's
//! lines that hold the leaf's layouts to the byte (the drafters'
//! [`crate::speculation`], the MTP head's [`crate::mtp_head`], the lanes'
//! [`crate::lane_lines`]).

/// The arena's allocation alignment, in bytes.
pub(crate) const ALIGN: u64 = 256;

/// `bytes` rounded up to whole blocks.
pub(crate) fn aligned(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// A BF16 tensor of `elements` elements, as the arena allocates it.
pub(crate) fn bf16(elements: u64) -> u64 {
    aligned(elements * 2)
}

/// A tensor of `elements` four-byte elements (I32 or F32), as the arena
/// allocates it.
pub(crate) fn words(elements: u64) -> u64 {
    aligned(elements * 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tensor takes whole 256-byte blocks: one byte past a block takes the
    /// next one, and nothing is a block of nothing.
    #[test]
    fn a_tensor_takes_whole_blocks() {
        assert_eq!((aligned(0), aligned(1), aligned(256), aligned(257)), (0, 256, 256, 512));
        assert_eq!((bf16(128), bf16(129), words(64), words(65)), (256, 512, 256, 512));
    }
}
