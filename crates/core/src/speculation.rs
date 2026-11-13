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

/// The widest draft window DFlash2 accepts (the reference's `--spec-draft-tokens`
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
    /// The 5-layer sliding-window DFlash2 drafter (`GLOSSARY.md`).
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
    /// Parse the operator's spelling (`--spec-backend`), naming the accepted values
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

    /// Whether the backend binds DFlash2's drafter and so reserves its
    /// window, prefill taps and round scratch; the verify-only backend
    /// proposes from the host and Flash-Next's MTP sizes its own (spec
    /// flash-next/07).
    pub fn binds_dflash2(&self) -> bool {
        matches!(self, Self::Dflash2)
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

/// The rows a Flash-Next verify round carries (`lanes * (k + 1)`) when the
/// operator names no row budget (0): the decode route's one-launch width
/// (`verify::kDefaultRows`, `kernel/src/flash_next/verify.h`).
pub const FLASH_NEXT_DEFAULT_VERIFY_ROWS: u32 = 8;

/// The widest row budget a Flash-Next load takes (`verify::kMaxRows`): past
/// the default's 8 rows a round's MoE runs the decode route in launches of
/// 8 and its linears on tensor cores (GitHub #329 item 5).
pub const FLASH_NEXT_MAX_VERIFY_ROWS: u32 = 32;

/// The draft tokens a Flash-Next MTP load verifies per lane at most when
/// the operator names none (`--spec-backend mtp` alone): phase A's best one-lane
/// projection at the measured column cost (spec flash-next/07), cut by the
/// row budget as lanes join.
pub const FLASH_NEXT_DEFAULT_DRAFT_TOKENS: u32 = 2;

/// The window a Flash-Next round of `lanes` lanes verifies at (spec
/// flash-next/07): the load's draft tokens, cut so that `lanes * (k + 1)`
/// stays within the row budget (0: [`FLASH_NEXT_DEFAULT_VERIFY_ROWS`]; past
/// [`FLASH_NEXT_MAX_VERIFY_ROWS`]: that maximum). 0 means that width runs a
/// one-token round. The leaf's `verify::window_for` is the same rule; a
/// round asks it for this window.
pub fn flash_next_window(draft_tokens: u32, row_budget: u32, lanes: u32) -> u32 {
    if draft_tokens == 0 || lanes == 0 {
        return 0;
    }
    let budget =
        if row_budget == 0 { FLASH_NEXT_DEFAULT_VERIFY_ROWS } else { row_budget.min(FLASH_NEXT_MAX_VERIFY_ROWS) };
    let columns = budget / lanes;
    if columns <= 1 {
        0
    } else {
        draft_tokens.min(columns - 1)
    }
}

/// The rows one Flash-Next decode step carries at most at `decode_lanes`:
/// the widest verify round, `w * (k + 1)` over 1..=`decode_lanes` lanes, or
/// the lanes themselves when a plain step of them is wider. The leaf's
/// `verify::decode_rows`, which sizes its MoE decode route by it.
pub fn flash_next_decode_rows(draft_tokens: u32, row_budget: u32, decode_lanes: u32) -> u32 {
    (1..=decode_lanes)
        .map(|w| match flash_next_window(draft_tokens, row_budget, w) {
            0 => 0,
            k => w * (k + 1),
        })
        .fold(decode_lanes, u32::max)
}

/// The window a round runs at when its tightest lane has `room` positions left
/// before its sequence's capacity (or the load's context): `window` while the
/// round's writes fit -- `2 * window` positions under MTP, `window + 1`
/// otherwise (the kernel's `written_positions`) -- else 0, a plain step.
/// Never a window in between: unlike the 27B, which clamps a lane's draft
/// extent inside its round, the leaf runs a width at its one window (its
/// captured graphs' shape) and refuses any other (GitHub #329).
pub fn flash_next_window_within(window: u32, mtp: bool, room: u64) -> u32 {
    let written = if mtp { 2 * u64::from(window) } else { u64::from(window) + 1 };
    if written <= room {
        window
    } else {
        0
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
    /// default's [`FLASH_NEXT_DEFAULT_VERIFY_ROWS`]) or
    /// 2..=[`FLASH_NEXT_MAX_VERIFY_ROWS`]; DFlash2 is the 27B's.
    pub fn new(backend: SpeculativeBackend, draft_tokens: u32, row_budget: u32) -> Result<Self, String> {
        if backend == SpeculativeBackend::Dflash2 {
            return Err("DFlash2 is the 27B's drafter; Qwen3.8-Flash-Next drafts with mtp".to_string());
        }
        if !(1..=MAX_DRAFT_TOKENS).contains(&draft_tokens) {
            return Err(format!("draft tokens must be in 1..{MAX_DRAFT_TOKENS}, got {draft_tokens}"));
        }
        if row_budget > FLASH_NEXT_MAX_VERIFY_ROWS || row_budget == 1 {
            return Err(format!(
                "a draft row budget of {row_budget}: 0 (= {FLASH_NEXT_DEFAULT_VERIFY_ROWS}) or 2..{FLASH_NEXT_MAX_VERIFY_ROWS}"
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

    /// The rows a decode step of a `decode_lanes` load carries at most
    /// ([`flash_next_decode_rows`]).
    pub fn decode_rows(&self, decode_lanes: u32) -> u32 {
        flash_next_decode_rows(self.draft_tokens, self.row_budget, decode_lanes)
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
/// (`text/draft_head`, Q4 over the 131,072 most frequent tokens, ~357 MB, with
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
    /// costs its ~357 MB of device memory (+357,040,380 bytes in the plan).
    Shortlist,
}

impl ProposalHead {
    /// Parse the operator's spelling (`--spec-draft-head`), naming the accepted
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

/// The drafter's proposal head as configured (`--spec-draft-head`, ADR
/// 0063): one of the two heads, or `auto`, which the load's shape resolves --
/// as `auto` rope scaling is (ADR 0057) -- rather than a third head.
///
/// `auto` is [`ProposalHead::Full`] on every card class. The shortlist is
/// bound beside the full head, so it frees no bytes: it costs its ~357 MB.
/// And the one card that measured it found it slower (-7.4% tokens/round,
/// -5.5% tok/s on the 5090, finding 2026-09-24-upstream-quick-wins-ab). The
/// rule is revisited when a real 16 GB card measures the trade (spec
/// 27b-16gb/04, phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalHeadSetting {
    Auto,
    Fixed(ProposalHead),
}

impl ProposalHeadSetting {
    /// `auto`, or what [`ProposalHead::parse`] takes, naming all three on a
    /// refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "auto" => Ok(Self::Auto),
            other => ProposalHead::parse(other)
                .map(Self::Fixed)
                .map_err(|_| format!("unknown draft head `{other}` (expected full, shortlist or auto)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fixed(head) => head.as_str(),
        }
    }

    /// Whether the head is left to [`Self::resolve`] rather than spelled out.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// The head a load scores its drafts with: a spelled-out head is itself,
    /// `auto` the full head.
    pub fn resolve(&self) -> ProposalHead {
        match self {
            Self::Auto => ProposalHead::Full,
            Self::Fixed(head) => *head,
        }
    }
}

/// How a verify round's sampling lanes draft and accept (GitHub #576, spec
/// #574, ADR 0090). A greedy lane drafts the selector walk's draft and takes
/// it where the target's argmax agrees under either rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DraftRule {
    /// The fixed draft, the rule of earlier releases: the selector walk's
    /// draft, taken with the target's probability of it (the vendored
    /// accept). A [`Speculation`]'s rule until one is named; the server's
    /// `spec.draft_rule` defaults to the sampled draft (GitHub #578).
    #[default]
    Fixed,
    /// The sampled draft: a sampling lane draws each draft from the drafter's
    /// distribution q over its candidates ([`crate::sampled_draft`]), and the
    /// accept takes it with probability min(1, p/q), drawing from
    /// max(p − q, 0) on a refusal. The text keeps the target's distribution;
    /// a seed's text is not the fixed rule's.
    Sampled,
}

impl DraftRule {
    /// The operator's spellings (`--spec-draft-rule`).
    pub const NAMES: [&'static str; 2] = ["fixed", "sampled"];

    /// Parse the operator's spelling, naming both on a refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "fixed" => Ok(Self::Fixed),
            "sampled" => Ok(Self::Sampled),
            other => Err(format!("unknown draft rule `{other}` (expected fixed or sampled)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Sampled => "sampled",
        }
    }

    /// `enum ignis_draft_rule` (`kernel/include/ignis_model.h`).
    pub fn abi_code(&self) -> u32 {
        match self {
            Self::Fixed => 0,
            Self::Sampled => 1,
        }
    }
}

/// The drafter's own sampling under [`DraftRule::Sampled`], in thousandths as
/// the configuration states them: its temperature as a factor of the lane's
/// (a lane at temperature T drafts at factor × T) and its top-p over its
/// candidates. Speed settings only: whatever they are, the text keeps the
/// target's distribution (ADR 0090).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterSampling {
    pub temperature_factor_milli: u32,
    pub top_p_milli: u32,
}

impl Default for DrafterSampling {
    /// The study's optimum at request temperature 1 (spec #574): 0.9 × T,
    /// top-p 0.95.
    fn default() -> Self {
        Self {
            temperature_factor_milli: 900,
            top_p_milli: 950,
        }
    }
}

impl DrafterSampling {
    /// The temperature factor as the leaf takes it.
    pub fn temperature_factor(&self) -> f32 {
        self.temperature_factor_milli as f32 / 1000.0
    }

    /// The top-p as the leaf takes it.
    pub fn top_p(&self) -> f32 {
        self.top_p_milli as f32 / 1000.0
    }
}

/// A validated speculation option: a backend, its draft window, the
/// drafter's proposal head and the rounds' draft rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    backend: SpeculativeBackend,
    draft_tokens: u32,
    proposal_head: ProposalHead,
    draft_rule: DraftRule,
    drafter_sampling: DrafterSampling,
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
            draft_rule: DraftRule::Fixed,
            drafter_sampling: DrafterSampling::default(),
        })
    }

    /// The same option with the drafter scoring through `proposal_head`.
    pub fn with_proposal_head(self, proposal_head: ProposalHead) -> Self {
        Self { proposal_head, ..self }
    }

    pub fn proposal_head(&self) -> ProposalHead {
        self.proposal_head
    }

    /// The same option with its rounds under `draft_rule`, the drafter
    /// sampling by `drafter_sampling` where the rule draws.
    pub fn with_draft_rule(self, draft_rule: DraftRule, drafter_sampling: DrafterSampling) -> Self {
        Self {
            draft_rule,
            drafter_sampling,
            ..self
        }
    }

    pub fn draft_rule(&self) -> DraftRule {
        self.draft_rule
    }

    pub fn drafter_sampling(&self) -> DrafterSampling {
        self.drafter_sampling
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
        if !self.backend.binds_dflash2() {
            return 0;
        }
        let bf16 = 2;
        let per_slot = DFLASH2_LAYERS
            * DFLASH2_WINDOW_TOKENS
            * DFLASH2_KV_HEADS
            * DFLASH2_HEAD_DIM
            * 2
            * bf16;
        u64::from(slot_count) * per_slot
    }

    /// What a `prefill_chunk_tokens`-wide chunk adds to the leaf's prefill
    /// scratch under this backend (`dflash2_prefill_scratch_bytes`,
    /// `kernel/src/model.cu`): the feature taps, their projection and
    /// normalization, and one drafter layer's query/key/value parent with the
    /// key and value rows kept from it — each at most one window of columns
    /// wide, each rounded to the arena's alignment.
    pub fn prefill_scratch_bytes(&self, prefill_chunk_tokens: u32) -> u64 {
        // No drafter bound, nothing tapped (P5-04, GitHub #153).
        if !self.backend.binds_dflash2() {
            return 0;
        }
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

    /// What the drafter's verify round adds to a load of `decode_lanes`
    /// lanes under this backend (P5-05, GitHub #155; `kernel/src/model.cu`):
    /// the feature taps of every verify column of the widest round -- one of
    /// `decode_lanes` lanes since GitHub #336, where it was always 8 -- one
    /// append count per lane, and the round scratch -- the larger of the
    /// drafter's forward activations and the round's context append, each
    /// rounded to the arena's alignment (the top-k's row-split partials
    /// excepted, added as the leaf adds them), plus the fixed allowance for
    /// the vendored workspaces the forward calls into. Under
    /// [`DraftRule::Sampled`], also the drafter's distribution kept from the
    /// draw to the accept: each draft column's candidate ids and their
    /// probabilities (GitHub #576, `dflash2_draw_bytes`).
    pub fn round_scratch_bytes(&self, decode_lanes: u32) -> u64 {
        match self.backend {
            // The fake drafter proposes from the host: nothing to reserve
            // (Flash-Next's MTP sizes its own, spec flash-next/07).
            SpeculativeBackend::VerifyOnly | SpeculativeBackend::Mtp => 0,
            SpeculativeBackend::Dflash2 => {
                let bf16 = |elements: u64| (elements * 2).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let wide = |elements: u64| (elements * 4).div_ceil(ARENA_ALIGN) * ARENA_ALIGN;
                let lanes = u64::from(decode_lanes);
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
                let draws = match self.draft_rule {
                    DraftRule::Fixed => 0,
                    DraftRule::Sampled => 2 * DFLASH2_SELECTOR_TOP_K * drafts * 4,
                };
                features + append_counts + draws + forward.max(append) + DFLASH2_ROUND_WORKSPACE_BYTES
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR 0063: `auto` is a third spelling of the setting, not a third
    /// head, and resolves to the full head; the two heads stay themselves.
    #[test]
    fn a_head_setting_is_a_head_or_auto_and_auto_is_the_full_head() {
        for (raw, setting, head) in [
            ("auto", ProposalHeadSetting::Auto, ProposalHead::Full),
            ("full", ProposalHeadSetting::Fixed(ProposalHead::Full), ProposalHead::Full),
            ("shortlist", ProposalHeadSetting::Fixed(ProposalHead::Shortlist), ProposalHead::Shortlist),
        ] {
            assert_eq!(ProposalHeadSetting::parse(raw), Ok(setting), "{raw}");
            assert_eq!(setting.as_str(), raw);
            assert_eq!(setting.resolve(), head, "{raw}");
            assert_eq!(setting.is_auto(), raw == "auto");
        }
        let err = ProposalHeadSetting::parse("medium").unwrap_err();
        assert!(err.contains("`medium`") && err.contains("full, shortlist or auto"), "{err}");
    }

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
    fn only_dflash2_reserves_drafter_state() {
        for backend in [SpeculativeBackend::VerifyOnly, SpeculativeBackend::Mtp] {
            assert!(!backend.binds_dflash2());
            let spec = Speculation::new(backend, 3).expect("in range");
            assert_eq!((spec.window_pool_bytes(4), spec.prefill_scratch_bytes(1024), spec.round_scratch_bytes(4)), (0, 0, 0));
        }
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 3).expect("in range");
        assert!(SpeculativeBackend::Dflash2.binds_dflash2());
        assert!(spec.window_pool_bytes(4) > 0 && spec.prefill_scratch_bytes(1024) > 0 && spec.round_scratch_bytes(4) > 0);
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

    /// GitHub #329: the leaf runs a width at its one window (its captured
    /// graphs' shape) or not at all, so a lane short of room for the whole
    /// window takes a plain step -- a narrower window is refused mid-request
    /// (`speculative_window 2 is not the window a round of 1 lanes runs`).
    #[test]
    fn a_round_without_room_for_its_window_is_a_plain_step() {
        // MTP writes 2k positions, verify-only k + 1.
        assert_eq!(flash_next_window_within(3, true, 100), 3);
        assert_eq!(flash_next_window_within(3, true, 6), 3, "exactly 2k fits");
        for room in 0..=5 {
            assert_eq!(flash_next_window_within(3, true, room), 0, "room {room}: a plain step, never a narrower window");
        }
        assert_eq!(flash_next_window_within(3, false, 4), 3, "exactly k + 1 fits");
        for room in 0..=3 {
            assert_eq!(flash_next_window_within(3, false, room), 0, "room {room}");
        }
        for mtp in [false, true] {
            for window in 0..=7 {
                for room in 0..=20u64 {
                    let k = flash_next_window_within(window, mtp, room);
                    let written = u64::from(if mtp { 2 * window } else { window + 1 });
                    let expected = if written <= room { window } else { 0 };
                    assert_eq!(k, expected, "window={window} mtp={mtp} room={room}");
                }
            }
        }
    }

    /// Spec flash-next/07: k adapts to the row budget as lanes join -- the
    /// default's 8 rows give a 2-draft load k = 2 at one and two lanes
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
        // GitHub #329 item 5: a budget past the default's lets five to eight
        // lanes draft; 0 stays the default's 8 rows.
        assert_eq!(widths(3, 16), [3, 3, 3, 3, 2, 1, 1, 1]);
        assert_eq!(widths(2, 16), [2, 2, 2, 2, 2, 1, 1, 1]);
        assert_eq!(widths(1, 16), [1; 8]);
        assert_eq!(widths(3, 24), [3, 3, 3, 3, 3, 3, 2, 2]);
        assert_eq!(widths(7, FLASH_NEXT_MAX_VERIFY_ROWS), [7, 7, 7, 7, 5, 4, 3, 3]);
        for draft in 1..=MAX_DRAFT_TOKENS {
            assert_eq!(widths(draft, 0), widths(draft, FLASH_NEXT_DEFAULT_VERIFY_ROWS), "0 is the default budget");
        }
        // Every window fits its budget, and none goes past the maximum.
        for draft in 1..=MAX_DRAFT_TOKENS {
            for budget in 0..=FLASH_NEXT_MAX_VERIFY_ROWS + 1 {
                let rows =
                    if budget == 0 { FLASH_NEXT_DEFAULT_VERIFY_ROWS } else { budget.min(FLASH_NEXT_MAX_VERIFY_ROWS) };
                for lanes in 1..=8 {
                    let k = flash_next_window(draft, budget, lanes);
                    assert!(k <= draft && (k == 0 || lanes * (k + 1) <= rows), "{draft} {budget} {lanes}");
                }
            }
        }
    }

    /// The rows a decode step carries at most, as the leaf sizes its MoE
    /// decode route (`verify::decode_rows`): the widest verify round over
    /// one lane up to the load's lanes, never fewer than the lanes.
    #[test]
    fn the_widest_decode_step_is_the_leaf_s_decode_rows() {
        let rows = |draft, budget, lanes| flash_next_decode_rows(draft, budget, lanes);
        // A 2-draft load at the decode route's 8 rows: 1 x 3, 2 x 3, 3 x 2.
        assert_eq!((rows(2, 0, 1), rows(2, 0, 2), rows(2, 0, 3)), (3, 6, 6));
        // The rtxpro6000's 3 drafts over 8 lanes: the budget caps it, and
        // from five lanes a plain step of every lane is the widest.
        assert_eq!((rows(3, 0, 1), rows(3, 0, 3), rows(3, 0, 4), rows(3, 0, 8)), (4, 8, 8, 8));
        assert_eq!(rows(7, 6, 3), 6, "a 6-row budget");
        assert_eq!(rows(1, 2, 3), 3, "one lane's 2 rows, fewer than the lanes");
        // GitHub #329 item 5: past the default's 8 rows the widest step is
        // the budget's, 8 lanes of 3 drafts at 16 and 32 rows.
        assert_eq!((rows(3, 16, 8), rows(3, 32, 8)), (16, 32));
        let spec = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 2, 0).expect("valid");
        assert_eq!(spec.decode_rows(3), 6);
        for draft in 1..=MAX_DRAFT_TOKENS {
            for budget in 0..=FLASH_NEXT_MAX_VERIFY_ROWS {
                for lanes in 1..=8 {
                    let widest = (1..=lanes)
                        .map(|w| w * (flash_next_window(draft, budget, w) + 1))
                        .max()
                        .unwrap()
                        .max(lanes);
                    assert_eq!(rows(draft, budget, lanes), widest, "{draft} {budget} {lanes}");
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
        for budget in [1, FLASH_NEXT_MAX_VERIFY_ROWS + 1] {
            let err = FlashNextSpeculation::new(SpeculativeBackend::VerifyOnly, 2, budget).expect_err("bad budget");
            assert!(err.contains(&budget.to_string()) && err.contains("0 (= 8)") && err.contains("2..32"), "{err}");
        }
        for budget in [9, 16, 24, FLASH_NEXT_MAX_VERIFY_ROWS] {
            let spec = FlashNextSpeculation::new(SpeculativeBackend::Mtp, 3, budget).expect("a budget up to the maximum");
            assert_eq!(spec.window(8), flash_next_window(3, budget, 8), "{budget}");
        }
    }

    #[test]
    fn verify_only_takes_the_same_window_rule_and_no_drafter_state() {
        let spec = Speculation::new(SpeculativeBackend::VerifyOnly, 7).expect("in range");
        assert_eq!(spec.draft_tokens(), 7);
        assert_eq!(spec.backend().abi_code(), 2);
        assert_eq!(spec.window_pool_bytes(8), 0, "no drafter, no window pool");
        assert_eq!(spec.prefill_scratch_bytes(128), 0, "no drafter, no prefill taps");
        assert_eq!(spec.round_scratch_bytes(8), 0, "no drafter, no round scratch");
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
        assert_eq!(spec.round_scratch_bytes(8), 78_027_552);
        // A narrower window reserves less; the window pool does not change.
        let narrow = Speculation::new(SpeculativeBackend::Dflash2, 3).unwrap();
        assert!(narrow.round_scratch_bytes(8) < spec.round_scratch_bytes(8));
    }

    /// GitHub #336: the round is as wide as the load's decode lanes, so its
    /// scratch is too -- eight lanes is the figure above to the byte.
    #[test]
    fn the_dflash2_round_scratch_follows_the_loads_lanes() {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        // One lane, worked as above: 8 columns and 7 draft columns, forward
        // 5,150,208 (the append 311,296 is smaller), taps 409,600, one count
        // of 4 bytes and the allowance 33,554,432.
        assert_eq!(spec.round_scratch_bytes(1), 5_150_208 + 409_600 + 4 + 33_554_432);
        assert_eq!(spec.round_scratch_bytes(2), 44_673_544);
        // Each lane adds 5,559,300 bytes, less what the arena's 256-byte
        // rounding of its smallest buffers (the ids, the top-k candidates)
        // had already reserved.
        for lanes in 2..=8 {
            let step = spec.round_scratch_bytes(lanes) - spec.round_scratch_bytes(lanes - 1);
            assert!((5_559_300 - 1024..=5_559_300).contains(&step), "lane {lanes} adds {step}");
        }
        assert_eq!(spec.round_scratch_bytes(crate::N_DECODE_LANES as u32), 78_027_552);
    }

    /// GitHub #576 (ADR 0090): the fixed draft is a speculation's rule until
    /// one is named (the server names its own default), the drafter
    /// samples at 0.9 x T and top-p 0.95, and both spellings cross the ABI
    /// as `enum ignis_draft_rule`.
    #[test]
    fn the_draft_rule_is_fixed_unless_the_sampled_draft_is_named() {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        assert_eq!(spec.draft_rule(), DraftRule::Fixed);
        assert_eq!(spec.drafter_sampling(), DrafterSampling { temperature_factor_milli: 900, top_p_milli: 950 });
        assert_eq!((spec.drafter_sampling().temperature_factor(), spec.drafter_sampling().top_p()), (0.9, 0.95));
        for (raw, rule, code) in [("fixed", DraftRule::Fixed, 0), ("sampled", DraftRule::Sampled, 1)] {
            assert_eq!(DraftRule::parse(raw), Ok(rule));
            assert_eq!((rule.as_str(), rule.abi_code()), (raw, code));
        }
        assert_eq!(DraftRule::NAMES, ["fixed", "sampled"]);
        let err = DraftRule::parse("coupled").unwrap_err();
        assert!(err.contains("`coupled`") && err.contains("fixed or sampled"), "{err}");
        let drafter = DrafterSampling { temperature_factor_milli: 750, top_p_milli: 1000 };
        let sampled = spec.with_draft_rule(DraftRule::Sampled, drafter);
        assert_eq!((sampled.draft_rule(), sampled.drafter_sampling()), (DraftRule::Sampled, drafter));
        assert_eq!((sampled.backend(), sampled.draft_tokens(), sampled.proposal_head()), (spec.backend(), 7, spec.proposal_head()));
    }

    /// GitHub #576: under the sampled draft the drafter's round line also
    /// holds q from the draw to the accept, the 16 candidates' ids and
    /// probabilities of every draft column (`dflash2_draw_bytes`); no drafter,
    /// nothing.
    #[test]
    fn the_sampled_draft_keeps_the_drafters_distribution_on_the_round_line() {
        let fixed = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        let sampled = fixed.with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
        // 2 buffers x 16 candidates x 7 columns x 4 bytes = 896 per lane.
        for lanes in 1..=8u32 {
            assert_eq!(sampled.round_scratch_bytes(lanes) - fixed.round_scratch_bytes(lanes), 896 * u64::from(lanes));
        }
        for backend in [SpeculativeBackend::VerifyOnly, SpeculativeBackend::Mtp] {
            let spec = Speculation::new(backend, 7).unwrap().with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
            assert_eq!(spec.round_scratch_bytes(8), 0, "{}: no drafter, no draws", backend.as_str());
        }
    }
}
