//! Speculation as a load option (P5-02, GitHub #150, spec 05).
//!
//! Speculative decoding is engine residency, chosen at load and frozen for
//! the life of that load, as in the reference: a backend and a draft window.
//! `None` (no [`Speculation`]) is today's engine — the drafter's weights are
//! not bound and no pool carries its window.
//!
//! The drafter's per-slot window and its prefill scratch are sized here as
//! well as in the leaf (`kernel/src/seq.cu`, `kernel/src/model.cu`), on
//! purpose: the GPU load test compares the leaf's reported VRAM against this
//! arithmetic, so the two cannot drift silently.

/// The widest draft window DFlash2 accepts (the reference's `--draft-tokens`
/// range for the 27B DFlash2 module, spec 05: `1..7`).
pub const MAX_DRAFT_TOKENS: u32 = 7;

/// The drafter's layers, KV heads, head width and sliding window (the
/// reference's `qwen3.8-27b-artifact.md` §15.1).
pub const DFLASH2_LAYERS: u64 = 5;
pub const DFLASH2_KV_HEADS: u64 = 8;
pub const DFLASH2_HEAD_DIM: u64 = 128;
pub const DFLASH2_WINDOW_TOKENS: u64 = 2048;

/// The target layers whose outputs feed the drafter (`[5, 19, 33, 47, 61]`),
/// concatenated into its `feature_projection` input.
pub const DFLASH2_FEATURE_TAPS: u64 = 5;

/// The drafter's query width (32 heads x 128).
pub const DFLASH2_QUERY_SIZE: u64 = 4096;

/// The target's hidden width, which the drafter shares.
const HIDDEN: u64 = 5120;

/// The leaf's scratch-arena allocation alignment (`DeviceArena::alloc_bytes`).
const ARENA_ALIGN: u64 = 256;

/// A speculative backend a load can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeBackend {
    /// The 5-layer sliding-window DFlash2 drafter (`CONTEXT.md`).
    Dflash2,
}

impl SpeculativeBackend {
    /// Parse the operator's spelling (`--spec`), naming the accepted values
    /// on a refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "dflash2" => Ok(Self::Dflash2),
            other => Err(format!("unknown speculative backend `{other}` (expected dflash2)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dflash2 => "dflash2",
        }
    }

    /// `enum ignis_speculative_backend` (`kernel/include/ignis_model.h`);
    /// 0 is "no speculation" and is never produced by a backend.
    pub fn abi_code(&self) -> i32 {
        match self {
            Self::Dflash2 => 1,
        }
    }
}

/// A validated speculation option: a backend and its draft window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    backend: SpeculativeBackend,
    draft_tokens: u32,
}

impl Speculation {
    /// `draft_tokens` must be in `1..=MAX_DRAFT_TOKENS`; the refusal names the
    /// range and the value.
    pub fn new(backend: SpeculativeBackend, draft_tokens: u32) -> Result<Self, String> {
        if !(1..=MAX_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(format!(
                "draft tokens must be in 1..{MAX_DRAFT_TOKENS}, got {draft_tokens}"
            ));
        }
        Ok(Self {
            backend,
            draft_tokens,
        })
    }

    pub fn backend(&self) -> SpeculativeBackend {
        self.backend
    }

    pub fn draft_tokens(&self) -> u32 {
        self.draft_tokens
    }

    /// The device bytes the drafter's per-sequence state takes in a pool of
    /// `slot_count` slots: BF16, layers × window × KV heads × head width ×
    /// (K + V) per slot — 40 MiB — twice with the rewrite checkpoint.
    /// Independent of the draft window. The state lives in the sequence pool
    /// (P5-03, GitHub #152), one lane per slot, because snapshot, restore and
    /// prefix clone carry it with the rest of a sequence.
    pub fn window_pool_bytes(&self, slot_count: u32) -> u64 {
        match self.backend {
            SpeculativeBackend::Dflash2 => {
                let bf16 = 2;
                let per_slot = DFLASH2_LAYERS
                    * DFLASH2_WINDOW_TOKENS
                    * DFLASH2_KV_HEADS
                    * DFLASH2_HEAD_DIM
                    * 2
                    * bf16;
                let with_checkpoint = 2 * per_slot;
                u64::from(slot_count) * with_checkpoint
            }
        }
    }

    /// What a `prefill_chunk_tokens`-wide chunk adds to the leaf's prefill
    /// scratch under this backend (`dflash2_prefill_scratch_bytes`,
    /// `kernel/src/model.cu`): the feature taps, their projection and
    /// normalization, and one drafter layer's query/key/value parent with the
    /// key and value rows kept from it — each at most one window of columns
    /// wide, each rounded to the arena's alignment.
    pub fn prefill_scratch_bytes(&self, prefill_chunk_tokens: u32) -> u64 {
        match self.backend {
            SpeculativeBackend::Dflash2 => {
                let bf16 = |elements: u64| (elements * 2).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let i32 = |elements: u64| (elements * 4).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let columns = u64::from(prefill_chunk_tokens).min(DFLASH2_WINDOW_TOKENS);
                let kv_width = DFLASH2_KV_HEADS * DFLASH2_HEAD_DIM;
                bf16(DFLASH2_FEATURE_TAPS * HIDDEN * columns)
                    + i32(columns)
                    + i32(1)
                    + i32(1)
                    + bf16(HIDDEN * columns)
                    + bf16(HIDDEN * columns)
                    + bf16((DFLASH2_QUERY_SIZE + 2 * kv_width) * columns)
                    + 3 * bf16(kv_width * columns)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_window_in_range_is_accepted() {
        for n in 1..=7 {
            let spec = Speculation::new(SpeculativeBackend::Dflash2, n).expect("in range");
            assert_eq!(spec.draft_tokens(), n);
        }
    }

    #[test]
    fn a_window_outside_the_range_is_refused_naming_the_range() {
        for n in [0, 8, 15, u32::MAX] {
            let err = Speculation::new(SpeculativeBackend::Dflash2, n).expect_err("out of range");
            assert!(err.contains("1..7"), "{err}");
            assert!(err.contains(&n.to_string()), "{err}");
        }
    }

    #[test]
    fn only_dflash2_parses() {
        assert_eq!(SpeculativeBackend::parse("dflash2"), Ok(SpeculativeBackend::Dflash2));
        for bad in ["mtp", "dflash", "DFLASH2", ""] {
            let err = SpeculativeBackend::parse(bad).expect_err("not a backend");
            assert!(err.contains("dflash2"), "{err}");
        }
    }

    #[test]
    fn the_dflash2_window_pool_is_80_mib_per_slot() {
        // Spec 05: 40 MiB per sequence, x2 with the checkpoint, 640 MiB for
        // the server's eight slots.
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(spec.window_pool_bytes(1), 80 * 1024 * 1024);
        assert_eq!(spec.window_pool_bytes(crate::N_DECODE_LANES as u32), 8 * 80 * 1024 * 1024);
        let narrow = Speculation::new(SpeculativeBackend::Dflash2, 1).unwrap();
        assert_eq!(narrow.window_pool_bytes(8), spec.window_pool_bytes(8), "the window does not size the pool");
    }

    #[test]
    fn the_dflash2_prefill_scratch_is_bounded_by_the_window() {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        // 128 columns, worked by hand: taps 6,553,600 + positions 512 + two
        // scalars 256 each + projected and context 1,310,720 each + qkv
        // 1,572,864 + key_raw, key, value 262,144 each.
        assert_eq!(spec.prefill_scratch_bytes(128), 11_535_360);
        // A chunk wider than the window taps at most the window.
        assert_eq!(spec.prefill_scratch_bytes(4096), spec.prefill_scratch_bytes(2048));
        assert!(spec.prefill_scratch_bytes(1024) < spec.prefill_scratch_bytes(2048));
    }
}
