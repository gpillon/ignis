//! Speculation as a load option (P5-02, GitHub #150, spec 05).
//!
//! Speculative decoding is engine residency, chosen at load and frozen for
//! the life of that load, as in the reference: a backend and a draft window.
//! `None` (no [`Speculation`]) is today's engine — the drafter's weights are
//! not bound and its window pool is not allocated.
//!
//! The drafter's per-lane window pool is sized here as well as in the leaf
//! (`kernel/src/model.cu`), on purpose: the GPU load test compares the leaf's
//! reported VRAM against this arithmetic, so the two cannot drift silently.

/// The widest draft window DFlash2 accepts (the reference's `--draft-tokens`
/// range for the 27B DFlash2 module, spec 05: `1..7`).
pub const MAX_DRAFT_TOKENS: u32 = 7;

/// The drafter's layers, KV heads, head width and sliding window (the
/// reference's `qwen3.8-27b-artifact.md` §15.1).
pub const DFLASH2_LAYERS: u64 = 5;
pub const DFLASH2_KV_HEADS: u64 = 8;
pub const DFLASH2_HEAD_DIM: u64 = 128;
pub const DFLASH2_WINDOW_TOKENS: u64 = 2048;

/// The lanes a drafter window pool is allocated for: one per decode lane
/// (`IGNIS_DECODE_MAX_BATCH`, [`crate::N_DECODE_LANES`]).
pub const DFLASH2_WINDOW_LANES: u64 = crate::N_DECODE_LANES as u64;

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

    /// The device bytes the drafter's window pool takes: BF16, layers ×
    /// window × KV heads × head width × (K + V) per lane — 40 MiB — twice
    /// with the rewrite checkpoint, for every decode lane. Independent of the
    /// draft window.
    pub fn window_pool_bytes(&self) -> u64 {
        match self.backend {
            SpeculativeBackend::Dflash2 => {
                let bf16 = 2;
                let per_lane = DFLASH2_LAYERS
                    * DFLASH2_WINDOW_TOKENS
                    * DFLASH2_KV_HEADS
                    * DFLASH2_HEAD_DIM
                    * 2
                    * bf16;
                let with_checkpoint = 2 * per_lane;
                DFLASH2_WINDOW_LANES * with_checkpoint
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
    fn the_dflash2_window_pool_is_80_mib_per_lane_for_eight_lanes() {
        // Spec 05: 40 MiB per lane, x2 with the checkpoint, 640 MiB for eight.
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(spec.window_pool_bytes(), 8 * 80 * 1024 * 1024);
        let narrow = Speculation::new(SpeculativeBackend::Dflash2, 1).unwrap();
        assert_eq!(narrow.window_pool_bytes(), spec.window_pool_bytes(), "the window does not size the pool");
    }
}
