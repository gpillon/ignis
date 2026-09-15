//! Vision as a load option (GitHub #177, spec
//! `.scratch/vision/specs/01-image-input.md`).
//!
//! Like speculation, vision is engine residency, chosen at load and frozen for
//! the life of that load: with a [`Vision`], the `vision/*` objects are bound
//! in their stored formats and the leaf reserves the encoder workspace and one
//! item's output transient for the envelope, before the sequence pool exists.
//! `None` is today's engine — nothing vision-related is bound or allocated.

/// The default vision envelope, in merged vision tokens per request (131,072
/// raw patches): the reference's own frontend limit.
pub const DEFAULT_VISION_MAX_TOKENS: u32 = 32_768;

/// The widest envelope a load accepts (`IGNIS_VISION_MAX_TOKENS_LIMIT`,
/// `kernel/include/ignis_model.h`).
pub const VISION_MAX_TOKENS_LIMIT: u32 = 1 << 20;

/// The artifact's `vision/*` object count: three patch/position globals, 27
/// blocks of twelve, and the merger's six.
pub const VISION_OBJECTS: usize = 3 + 27 * 12 + 6;

/// The target's hidden width, which the encoder's merged output carries.
const HIDDEN: u64 = 5120;

/// The leaf's vision allocation alignment (`kVisionWorkspaceAlignment`).
const VISION_ALIGN: u64 = 256;

/// The vision envelope a load reserves for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vision {
    max_tokens: u32,
}

/// A vision envelope outside `1..=VISION_MAX_TOKENS_LIMIT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionEnvelopeOutOfRange(pub u32);

impl std::fmt::Display for VisionEnvelopeOutOfRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "vision max tokens must be in 1..={VISION_MAX_TOKENS_LIMIT}, got {}",
            self.0
        )
    }
}

impl std::error::Error for VisionEnvelopeOutOfRange {}

impl Default for Vision {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_VISION_MAX_TOKENS,
        }
    }
}

impl Vision {
    /// An envelope of `max_tokens` merged vision tokens per request.
    pub fn new(max_tokens: u32) -> Result<Self, VisionEnvelopeOutOfRange> {
        if max_tokens == 0 || max_tokens > VISION_MAX_TOKENS_LIMIT {
            return Err(VisionEnvelopeOutOfRange(max_tokens));
        }
        Ok(Self { max_tokens })
    }

    /// The configured envelope, in merged vision tokens.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// The envelope the leaf actually reserves for: no request can carry more
    /// vision tokens than its context holds.
    pub fn envelope_tokens(&self, max_context_tokens: u32) -> u32 {
        self.max_tokens.min(max_context_tokens)
    }

    /// One item's `[5120, V]` BF16 encoder output at the envelope, rounded to
    /// the leaf's alignment (the output transient half of the reservation).
    pub fn output_transient_bytes(&self, max_context_tokens: u32) -> u64 {
        let bytes = HIDDEN * u64::from(self.envelope_tokens(max_context_tokens)) * 2;
        bytes.div_ceil(VISION_ALIGN) * VISION_ALIGN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_envelope_is_the_references() {
        assert_eq!(Vision::default().max_tokens(), 32_768);
    }

    #[test]
    fn an_envelope_outside_the_range_is_refused() {
        assert_eq!(Vision::new(0), Err(VisionEnvelopeOutOfRange(0)));
        assert!(Vision::new(VISION_MAX_TOKENS_LIMIT + 1).is_err());
        assert_eq!(Vision::new(1).unwrap().max_tokens(), 1);
        assert_eq!(
            Vision::new(VISION_MAX_TOKENS_LIMIT).unwrap().max_tokens(),
            VISION_MAX_TOKENS_LIMIT
        );
    }

    #[test]
    fn the_envelope_is_capped_by_the_context() {
        let vision = Vision::default();
        assert_eq!(vision.envelope_tokens(40_960), 32_768);
        assert_eq!(vision.envelope_tokens(1024), 1024);
    }

    #[test]
    fn the_output_transient_is_one_items_bf16_hidden_columns() {
        // 5120 x 32768 x 2 bytes, already a multiple of 256.
        assert_eq!(Vision::default().output_transient_bytes(262_144), 335_544_320);
        assert_eq!(Vision::new(1).unwrap().output_transient_bytes(4096), 10_240);
    }

    #[test]
    fn the_vision_object_count_is_the_artifacts() {
        assert_eq!(VISION_OBJECTS, 333);
    }
}
