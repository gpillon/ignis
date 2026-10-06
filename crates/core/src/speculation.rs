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

/// The target's MLP width and vocabulary, which the drafter's forward shares
/// (its MLP, and the target's output head over its draft columns).
const FFN_INTERMEDIATE: u64 = 17408;
const VOCAB: u64 = 248_320;

/// The drafter's dynamic-conv coefficient rows (320 groups x 2 taps x 2
/// sides), its selector rank and the candidates per draft column it scores.
const DFLASH2_CONV_PROJ_ROWS: u64 = 1280;
const DFLASH2_SELECTOR_RANK: u64 = 256;
const DFLASH2_SELECTOR_TOP_K: u64 = 16;

/// The row-split partials of the drafter's top-k (`ignis_dflash2_topk_workspace_bytes`,
/// `kernel/src/dflash2_topk.cu`): the vocabulary is cut into splits of this
/// many rows, at most this many splits, each keeping one `(value, row)` pair
/// per candidate. Nonzero only because [`DFLASH2_SELECTOR_TOP_K`] is the one k
/// that kernel specializes; any other k forwards to the vendored op and takes
/// none. Rows per split (`kRowsPerSplit`).
const DFLASH2_TOPK_ROWS_PER_SPLIT: u64 = 2048;
/// The cap on splits (`kMaxSplits`).
const DFLASH2_TOPK_MAX_SPLITS: u64 = 1024;
/// One candidate: an FP32 value and an I32 row (`Entry`).
const DFLASH2_TOPK_ENTRY_BYTES: u64 = 8;

/// The leaf's fixed allowance for the vendored workspaces inside the
/// drafter's forward (`kDflash2RoundWorkspaceBytes`, `kernel/src/model.cu`).
const DFLASH2_ROUND_WORKSPACE_BYTES: u64 = 32 * 1024 * 1024;

/// The leaf's scratch-arena allocation alignment (`DeviceArena::alloc_bytes`).
const ARENA_ALIGN: u64 = 256;

/// A speculative backend a load can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeculativeBackend {
    /// The 5-layer sliding-window DFlash2 drafter (`CONTEXT.md`).
    Dflash2,
    /// The verify substrate alone (P5-04, GitHub #153): the verify round,
    /// its ReplaySSM records and its graphs at the window, with no drafter
    /// bound. The drafts come per call through
    /// [`crate::step::decode_program_verify`] -- the seam a test's fake
    /// drafter fills. Not an operator spelling: [`Self::parse`] refuses it.
    VerifyOnly,
    /// Qwen3.8-Flash-Next's own draft head, the checkpoint's MTP layer
    /// (spec flash-next/07, GitHub #307), bound from its companion container.
    /// Flash-Next's backend only: the 27B keeps DFlash2.
    Mtp,
}

impl SpeculativeBackend {
    /// Parse the operator's spelling (`--spec`), naming the accepted values
    /// on a refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "dflash2" => Ok(Self::Dflash2),
            "mtp" => Ok(Self::Mtp),
            other => Err(format!("unknown speculative backend `{other}` (expected dflash2 or mtp)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dflash2 => "dflash2",
            Self::VerifyOnly => "verify-only",
            Self::Mtp => "mtp",
        }
    }

    /// `enum ignis_speculative_backend` (`kernel/include/ignis_model.h`);
    /// 0 is "no speculation" and is never produced by a backend.
    pub fn abi_code(&self) -> i32 {
        match self {
            Self::Dflash2 => 1,
            Self::VerifyOnly => 2,
            Self::Mtp => 3,
        }
    }
}

/// The most rows a Flash-Next verify round carries (`lanes * (k + 1)`): the
/// MoE decode route's, the GDN replay record's and the QSA listed decode's
/// bound (`verify::kMaxRows`, `kernel/src/flash_next/verify.h`).
pub const FLASH_NEXT_VERIFY_ROWS: u32 = 8;

/// The draft tokens a Flash-Next MTP load verifies per lane at most when
/// the operator names none (`--spec mtp` alone, or the default on with the
/// companion present): phase A's best one-lane projection at the measured
/// column cost (spec flash-next/07), cut by the row budget as lanes join.
pub const FLASH_NEXT_DEFAULT_DRAFT_TOKENS: u32 = 2;

/// The window a Flash-Next round of `lanes` lanes verifies at (spec
/// flash-next/07): the load's draft tokens, cut so that `lanes * (k + 1)`
/// stays within the row budget (0, or past [`FLASH_NEXT_VERIFY_ROWS`]: that
/// bound). 0 means that width runs a one-token round. The leaf's
/// `verify::window_for` is the same rule; a round asks it for this window.
pub fn flash_next_window(draft_tokens: u32, row_budget: u32, lanes: u32) -> u32 {
    if draft_tokens == 0 || lanes == 0 {
        return 0;
    }
    let budget = if row_budget == 0 || row_budget > FLASH_NEXT_VERIFY_ROWS { FLASH_NEXT_VERIFY_ROWS } else { row_budget };
    let columns = budget / lanes;
    if columns <= 1 {
        0
    } else {
        draft_tokens.min(columns - 1)
    }
}

/// A Flash-Next load's speculation (spec flash-next/07): its backend
/// ([`SpeculativeBackend::Mtp`], or [`SpeculativeBackend::VerifyOnly`] for a
/// test's fake drafter), the most drafts a lane verifies per round, and the
/// row budget that cuts them as lanes join ([`flash_next_window`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashNextSpeculation {
    backend: SpeculativeBackend,
    draft_tokens: u32,
    row_budget: u32,
}

impl FlashNextSpeculation {
    /// `draft_tokens` in `1..=MAX_DRAFT_TOKENS`, a row budget of 0 (the
    /// bound) or 2..=bound; DFlash2 is the 27B's.
    pub fn new(backend: SpeculativeBackend, draft_tokens: u32, row_budget: u32) -> Result<Self, String> {
        if backend == SpeculativeBackend::Dflash2 {
            return Err("DFlash2 is the 27B's drafter; Qwen3.8-Flash-Next drafts with mtp".to_string());
        }
        if !(1..=MAX_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(format!("draft tokens must be in 1..{MAX_DRAFT_TOKENS}, got {draft_tokens}"));
        }
        if row_budget > FLASH_NEXT_VERIFY_ROWS || row_budget == 1 {
            return Err(format!(
                "a draft row budget of {row_budget}: 0 (the decode route's {FLASH_NEXT_VERIFY_ROWS}) or 2..{FLASH_NEXT_VERIFY_ROWS}"
            ));
        }
        Ok(Self { backend, draft_tokens, row_budget })
    }

    pub fn backend(&self) -> SpeculativeBackend {
        self.backend
    }

    pub fn draft_tokens(&self) -> u32 {
        self.draft_tokens
    }

    pub fn row_budget(&self) -> u32 {
        self.row_budget
    }

    /// The window a round of `lanes` lanes verifies at.
    pub fn window(&self, lanes: u32) -> u32 {
        flash_next_window(self.draft_tokens, self.row_budget, lanes)
    }
}

/// The head the drafter scores its draft columns with.
///
/// The artifact carries two: the target's own output head (W8, all 248,320
/// rows, 1.27 GB streamed per round) and a shortlist proposal head
/// (`text/draft_head`, Q4 over the 131,072 most frequent tokens, 356 MB, with
/// `text/draft_head_token_ids` mapping its rows back to token ids). The verify
/// round always scores with the full head, so what a round accepts is still
/// the target's choice: a token outside the shortlist is still produced, it
/// is just never drafted. The text moves only where every verify round's
/// does -- at a near tie, since different drafts group positions into
/// different rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProposalHead {
    /// The target's output head (the only one before the shortlist existed).
    #[default]
    Full,
    /// The artifact's Q4 shortlist head: bound beside the full head, so it
    /// costs its 356 MB of device memory.
    Shortlist,
}

impl ProposalHead {
    /// Parse the operator's spelling (`--draft-head`), naming the accepted
    /// values on a refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "full" => Ok(Self::Full),
            "shortlist" => Ok(Self::Shortlist),
            other => Err(format!("unknown draft head `{other}` (expected full or shortlist)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Shortlist => "shortlist",
        }
    }
}

/// A validated speculation option: a backend, its draft window and the
/// drafter's proposal head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    backend: SpeculativeBackend,
    draft_tokens: u32,
    proposal_head: ProposalHead,
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
            proposal_head: ProposalHead::Full,
        })
    }

    /// The same option with the drafter scoring through `proposal_head`.
    pub fn with_proposal_head(self, proposal_head: ProposalHead) -> Self {
        Self { proposal_head, ..self }
    }

    pub fn proposal_head(&self) -> ProposalHead {
        self.proposal_head
    }

    pub fn backend(&self) -> SpeculativeBackend {
        self.backend
    }

    pub fn draft_tokens(&self) -> u32 {
        self.draft_tokens
    }

    /// The device bytes the drafter's per-sequence state takes in a pool of
    /// `slot_count` slots: BF16, layers × window × KV heads × head width ×
    /// (K + V) per slot — 40 MiB. The window only: the reference also keeps a
    /// rewrite checkpoint of it, which ignis carried for every slot and never
    /// read, so it is not reserved. Independent of the draft window. The
    /// state lives in the sequence pool
    /// (P5-03, GitHub #152), one lane per slot, because snapshot, restore and
    /// prefix clone carry it with the rest of a sequence. Zero for
    /// [`SpeculativeBackend::VerifyOnly`], which binds no drafter.
    pub fn window_pool_bytes(&self, slot_count: u32) -> u64 {
        match self.backend {
            SpeculativeBackend::VerifyOnly | SpeculativeBackend::Mtp => 0,
            SpeculativeBackend::Dflash2 => {
                let bf16 = 2;
                let per_slot = DFLASH2_LAYERS
                    * DFLASH2_WINDOW_TOKENS
                    * DFLASH2_KV_HEADS
                    * DFLASH2_HEAD_DIM
                    * 2
                    * bf16;
                u64::from(slot_count) * per_slot
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
            // No drafter bound, nothing tapped (P5-04, GitHub #153).
            SpeculativeBackend::VerifyOnly | SpeculativeBackend::Mtp => 0,
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

    /// What the drafter's verify round adds to the load under this backend
    /// (P5-05, GitHub #155; `kernel/src/model.cu`): the feature taps of every
    /// verify column of the widest round, one append count per lane, and the
    /// round scratch -- the larger of the drafter's forward activations and
    /// the round's context append, each rounded to the arena's alignment
    /// (the top-k's row-split partials excepted, added as the leaf adds
    /// them), plus the fixed allowance for the vendored workspaces the
    /// forward calls into.
    pub fn round_scratch_bytes(&self) -> u64 {
        match self.backend {
            // The fake drafter proposes from the host: nothing to reserve
            // (Flash-Next's MTP sizes its own, spec flash-next/07).
            SpeculativeBackend::VerifyOnly | SpeculativeBackend::Mtp => 0,
            SpeculativeBackend::Dflash2 => {
                let bf16 = |elements: u64| (elements * 2).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let wide = |elements: u64| (elements * 4).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let lanes = crate::N_DECODE_LANES as u64;
                let columns = u64::from(self.draft_tokens + 1) * lanes;
                let drafts = u64::from(self.draft_tokens) * lanes;
                let kv_width = DFLASH2_KV_HEADS * DFLASH2_HEAD_DIM;
                let hidden_columns = bf16(HIDDEN * columns);
                // Added unrounded, as the leaf adds it (GitHub #259: this
                // term was missing since the top-k was replaced, 5dcfade).
                let topk_splits = VOCAB.div_ceil(DFLASH2_TOPK_ROWS_PER_SPLIT).min(DFLASH2_TOPK_MAX_SPLITS);
                let topk_partials =
                    topk_splits * drafts * DFLASH2_SELECTOR_TOP_K * DFLASH2_TOPK_ENTRY_BYTES;

                let forward = 2 * wide(columns)
                    + hidden_columns
                    + 2 * hidden_columns
                    + bf16(DFLASH2_CONV_PROJ_ROWS * columns)
                    + bf16((DFLASH2_QUERY_SIZE + 2 * kv_width) * columns)
                    + 3 * bf16(DFLASH2_QUERY_SIZE * columns)
                    + 3 * bf16(kv_width * columns)
                    + 2 * hidden_columns
                    + 2 * hidden_columns
                    + bf16(DFLASH2_CONV_PROJ_ROWS * columns)
                    + bf16(FFN_INTERMEDIATE * columns)
                    + 2 * hidden_columns
                    + 2 * bf16(HIDDEN * drafts)
                    + bf16(VOCAB * drafts)
                    + wide(DFLASH2_SELECTOR_TOP_K * drafts)
                    + bf16(DFLASH2_SELECTOR_TOP_K * drafts)
                    + topk_partials
                    + 2 * wide(DFLASH2_SELECTOR_TOP_K * drafts)
                    + bf16(DFLASH2_SELECTOR_RANK * drafts)
                    + wide(DFLASH2_SELECTOR_RANK * drafts)
                    + wide(DFLASH2_SELECTOR_TOP_K * DFLASH2_SELECTOR_TOP_K * drafts);
                let append = 2 * hidden_columns
                    + bf16((DFLASH2_QUERY_SIZE + 2 * kv_width) * columns)
                    + 3 * bf16(kv_width * columns);

                let features = DFLASH2_FEATURE_TAPS * HIDDEN * columns * 2;
                let append_counts = lanes * 4;
                features + append_counts + forward.max(append) + DFLASH2_ROUND_WORKSPACE_BYTES
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
    fn only_dflash2_and_mtp_parse() {
        assert_eq!(SpeculativeBackend::parse("dflash2"), Ok(SpeculativeBackend::Dflash2));
        assert_eq!(SpeculativeBackend::parse("mtp"), Ok(SpeculativeBackend::Mtp));
        assert_eq!(SpeculativeBackend::Mtp.abi_code(), 3);
        assert_eq!(SpeculativeBackend::Mtp.as_str(), "mtp");
        // The verify-only backend is an internal seam (P5-04, GitHub #153),
        // never an operator spelling.
        for bad in ["MTP", "dflash", "DFLASH2", "", "verify-only"] {
            let err = SpeculativeBackend::parse(bad).expect_err("not a backend");
            assert!(err.contains("dflash2") && err.contains("mtp"), "{err}");
        }
    }

    /// Spec flash-next/07: k adapts to the row budget as lanes join -- the
    /// decode route's 8 rows give a 2-draft load k = 2 at one and two lanes
    /// and k = 1 at three (phase A's widths), and no draft past four lanes.
    #[test]
    fn the_flash_next_window_fits_the_row_budget() {
        let widths = |draft, budget| (1..=8).map(|lanes| flash_next_window(draft, budget, lanes)).collect::<Vec<_>>();
        assert_eq!(widths(2, 0), [2, 2, 1, 1, 0, 0, 0, 0]);
        assert_eq!(widths(3, 8), [3, 3, 1, 1, 0, 0, 0, 0]);
        assert_eq!(widths(7, 0), [7, 3, 1, 1, 0, 0, 0, 0]);
        // A budget of 6 rows: one lane k = 5 at most, two lanes 2, three 1.
        assert_eq!(widths(7, 6), [5, 2, 1, 0, 0, 0, 0, 0]);
        assert_eq!(widths(1, 2), [1, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(flash_next_window(0, 8, 1), 0, "no draft tokens, no window");
        // Every window fits the bound.
        for draft in 1..=MAX_DRAFT_TOKENS {
            for budget in 0..=FLASH_NEXT_VERIFY_ROWS {
                for lanes in 1..=8 {
                    let k = flash_next_window(draft, budget, lanes);
                    assert!(k <= draft && (k == 0 || lanes * (k + 1) <= FLASH_NEXT_VERIFY_ROWS), "{draft} {budget} {lanes}");
                }
            }
        }
    }

    #[test]
    fn flash_next_speculation_refuses_dflash2_and_out_of_range_options() {
        let spec = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 3, 6).expect("valid");
        assert_eq!((spec.window(1), spec.window(2), spec.window(3)), (3, 2, 1));
        let err = FlashNextSpeculation::new(SpeculativeBackend::Dflash2, 3, 0).expect_err("the 27B's drafter");
        assert!(err.contains("DFlash2") && err.contains("mtp"), "{err}");
        for draft in [0, 8] {
            let err = FlashNextSpeculation::new(SpeculativeBackend::Mtp, draft, 0).expect_err("out of range");
            assert!(err.contains("1..7"), "{err}");
        }
        for budget in [1, 9] {
            let err = FlashNextSpeculation::new(SpeculativeBackend::VerifyOnly, 2, budget).expect_err("bad budget");
            assert!(err.contains(&budget.to_string()) && err.contains("2..8"), "{err}");
        }
    }

    #[test]
    fn verify_only_takes_the_same_window_rule_and_no_drafter_state() {
        let spec = Speculation::new(SpeculativeBackend::VerifyOnly, 7).expect("in range");
        assert_eq!(spec.draft_tokens(), 7);
        assert_eq!(spec.backend().abi_code(), 2);
        assert_eq!(spec.window_pool_bytes(8), 0, "no drafter, no window pool");
        assert_eq!(spec.prefill_scratch_bytes(128), 0, "no drafter, no prefill taps");
        assert_eq!(spec.round_scratch_bytes(), 0, "no drafter, no round scratch");
        let err = Speculation::new(SpeculativeBackend::VerifyOnly, 8).expect_err("out of range");
        assert!(err.contains("1..7"), "{err}");
    }

    #[test]
    fn the_dflash2_window_pool_is_40_mib_per_slot() {
        // Spec 05: 5 layers x 2048 tokens x 8 KV heads x 128 x (K + V) in
        // BF16 is 40 MiB per sequence, 320 MiB for the server's eight lanes.
        // The window alone: the rewrite checkpoint that once doubled it was
        // never read, and is gone.
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(spec.window_pool_bytes(1), 40 * 1024 * 1024);
        assert_eq!(spec.window_pool_bytes(crate::N_DECODE_LANES as u32), 8 * 40 * 1024 * 1024);
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

    #[test]
    fn the_dflash2_round_scratch_is_the_widest_rounds_forward_plus_the_allowance() {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        // Window 7, eight lanes: 64 columns, 56 draft columns, worked by hand.
        // Forward 41,196,288: ids and positions 512, residual 655,360, the
        // attention block 5,537,792 (normed, conv 1,310,720; dynamic 163,840;
        // qkv 786,432; query_raw, query, attention 1,572,864; key_raw, value,
        // key 393,216; projected, conv 1,310,720), the MLP block 5,013,504
        // (normed, conv_hidden 1,310,720; dynamic 163,840; intermediate
        // 2,228,224; projected, conv 1,310,720), the head 29,989,120 (packed,
        // proposal 1,146,880; logits 27,811,840; ids 3,584 + values 1,792;
        // top-k partials 874,496 (122 row splits x 56 x 16 x 8 bytes);
        // unary 3,584 + predecessors 3,584; hidden_proj 28,672 + its FP32
        // 57,344; scores 57,344). The append (2,490,368) is smaller. Taps
        // 3,276,800, counts 32, allowance 33,554,432.
        assert_eq!(spec.round_scratch_bytes(), 78_027_552);
        // A narrower window reserves less; the window pool does not change.
        let narrow = Speculation::new(SpeculativeBackend::Dflash2, 3).unwrap();
        assert!(narrow.round_scratch_bytes() < spec.round_scratch_bytes());
    }
}
