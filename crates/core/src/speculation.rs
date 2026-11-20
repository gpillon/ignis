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

/// The widest draft window DFlash2 accepts: GitHub #687 (spec #684), 15, two
/// round shapes past [`BASE_SHAPE_DRAFT_TOKENS`] ([`Speculation::round_shapes`]),
/// where the reference's `--spec-draft-tokens` range for the 27B DFlash2
/// module (spec 05) was `1..7`. The leaf's own is
/// `IGNIS_DFLASH2_MAX_DRAFT_TOKENS` (`kernel/include/ignis_model.h`), which a
/// build-time check holds inside the vendored ops' domains
/// (`kernel/src/draft_window.h`): 15 is their edge.
pub const DFLASH2_MAX_DRAFT_TOKENS: u32 = 15;

/// GitHub #687: the base shape's draft window, 8 columns a lane at every lane
/// count: a DFlash2 load's past it, and the one shape of every load at it or
/// fewer (`IGNIS_BASE_SHAPE_DRAFT_TOKENS`, `kernel/include/ignis_model.h`).
pub const BASE_SHAPE_DRAFT_TOKENS: u32 = 7;

/// GitHub #687: the most lanes a round at the extended shape runs, so that
/// `lanes x (window + 1)` stays within the 64 columns of 8 lanes of 8
/// (`IGNIS_EXTENDED_SHAPE_MAX_LANES`).
pub const EXTENDED_SHAPE_MAX_LANES: u32 = 4;

/// A verify round's shape (GitHub #687): its draft window, a round at it
/// `window + 1` columns a lane, and the most lanes a round at it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundShape {
    pub window: u32,
    pub lanes: u32,
}

/// The widest draft window of a model's MTP head, the 27B's and
/// Flash-Next's (`IGNIS_MTP_MAX_DRAFT_TOKENS`).
pub const MTP_MAX_DRAFT_TOKENS: u32 = 7;

/// The widest draft window of the verify substrate alone, a test's fake
/// drafter (`IGNIS_VERIFY_ONLY_MAX_DRAFT_TOKENS`).
pub const VERIFY_ONLY_MAX_DRAFT_TOKENS: u32 = 7;

const fn wider(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

/// The widest draft window any drafter accepts: the bound of the server's
/// one `spec.draft_tokens` field, inside which each drafter refuses past its
/// own cap, and the positions [`crate::types::SpecCounters`] can carry.
pub const WIDEST_DRAFT_TOKENS: u32 =
    wider(DFLASH2_MAX_DRAFT_TOKENS, wider(MTP_MAX_DRAFT_TOKENS, VERIFY_ONLY_MAX_DRAFT_TOKENS));

/// The fewest draft positions a load reports: the seven every load
/// reported while one cap of 7 bounded every drafter, so that a load at a
/// window of 7 or fewer reports what it did then (a Flash-Next load at 2
/// drafts still prints positions 1 to 7 on the server). Past it a load
/// reports its own window ([`draft_positions`]).
pub const MIN_DRAFT_POSITIONS: u32 = 7;

/// The draft positions a load at a window of `draft_tokens` reports (GitHub
/// #685): its rounds' per-position counters ([`crate::types::SpecCounters`])
/// and the `ignis_speculative_position_*` series the server prints, the
/// window but never fewer than [`MIN_DRAFT_POSITIONS`].
pub const fn draft_positions(draft_tokens: u32) -> u32 {
    wider(draft_tokens, MIN_DRAFT_POSITIONS)
}

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

/// The 27B's hidden width, which both its drafters share: DFlash2, and the
/// MTP head's stem and continuation hidden ([`crate::mtp_head`]).
pub(crate) const HIDDEN: u64 = 5120;

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
    /// The checkpoint's own MTP layer. On Qwen3.8-Flash-Next its draft head
    /// (spec flash-next/07, GitHub #307), bound from its companion container;
    /// on the 27B its second drafter (GitHub #620, spec #618), the 12 `mtp/`
    /// objects of its own container ([`crate::mtp_head`]), which a load
    /// reaches through the leaf. The server offers it as `--spec-backend mtp`
    /// (GitHub #623), opt-in beside DFlash2, every profile's drafter.
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

    /// The widest draft window this drafter accepts (GitHub #685): each
    /// holds its own cap, refused past it naming the drafter.
    pub const fn max_draft_tokens(&self) -> u32 {
        match self {
            Self::Dflash2 => DFLASH2_MAX_DRAFT_TOKENS,
            Self::VerifyOnly => VERIFY_ONLY_MAX_DRAFT_TOKENS,
            Self::Mtp => MTP_MAX_DRAFT_TOKENS,
        }
    }

    /// The refusal of a window outside `1..=max_draft_tokens`, naming the
    /// drafter, its range and the value.
    fn refuse_window(&self, draft_tokens: u32) -> Option<String> {
        let widest = self.max_draft_tokens();
        (!(1..=widest).contains(&draft_tokens))
            .then(|| format!("draft tokens must be in 1..{widest} under {}, got {draft_tokens}", self.as_str()))
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

/// The draft window a 27B MTP load verifies when the operator names none
/// (`--spec-backend mtp` alone, GitHub #623): the window that committed the
/// most tokens per second on the 16 GB artifact in a 13 GiB budget over
/// recorded agent turns (GitHub #628, finding
/// 2026-11-15-the-27bs-mtp-head-measured-on-the-rtx-5090), where windows 5 to
/// 7 came within 0.4% of each other; on the NVFP4 artifact 5 is estimated
/// to commit about 3.5% more.
pub const QWEN38_MTP_DEFAULT_DRAFT_TOKENS: u32 = 7;

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
    /// `draft_tokens` in `1..=` the backend's
    /// [`SpeculativeBackend::max_draft_tokens`], a row budget of 0 (the
    /// default's [`FLASH_NEXT_DEFAULT_VERIFY_ROWS`]) or
    /// 2..=[`FLASH_NEXT_MAX_VERIFY_ROWS`]; DFlash2 is the 27B's.
    pub fn new(backend: SpeculativeBackend, draft_tokens: u32, row_budget: u32) -> Result<Self, String> {
        if backend == SpeculativeBackend::Dflash2 {
            return Err("DFlash2 is the 27B's drafter; Qwen3.8-Flash-Next drafts with mtp".to_string());
        }
        if let Some(refusal) = backend.refuse_window(draft_tokens) {
            return Err(refusal);
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

    /// The draft positions this load reports ([`draft_positions`]).
    pub fn draft_positions(&self) -> u32 {
        draft_positions(self.draft_tokens)
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

/// GitHub #689 (spec #684): the **shape rule**'s thresholds on a DFlash2
/// load with an extended round shape (`kernel/src/round_shape.h`
/// `asks_extended`): a lane asks the extended shape after a base round that
/// drafted the base shape's whole window and had at least `enter_accepted` of
/// it taken, and keeps asking while its extended rounds take at least
/// `stay_accepted`. The server's `spec.shape_enter_accepted` and
/// `spec.shape_stay_accepted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeRule {
    enter_accepted: u32,
    stay_accepted: u32,
}

impl ShapeRule {
    /// Every draft of a full base window, to enter.
    pub const DEFAULT_ENTER_ACCEPTED: u32 = BASE_SHAPE_DRAFT_TOKENS;
    /// Six drafts of an extended round, to stay.
    pub const DEFAULT_STAY_ACCEPTED: u32 = 6;

    /// `enter_accepted` in `1..=`[`BASE_SHAPE_DRAFT_TOKENS`] and
    /// `stay_accepted` in `1..=`[`DFLASH2_MAX_DRAFT_TOKENS`]; the refusal
    /// names the threshold and its range. Whether a load's own window can
    /// meet the stay threshold is [`Speculation::with_shape_rule`]'s.
    pub fn new(enter_accepted: u32, stay_accepted: u32) -> Result<Self, String> {
        if !(1..=BASE_SHAPE_DRAFT_TOKENS).contains(&enter_accepted) {
            return Err(format!(
                "the shape rule's entry threshold {enter_accepted} is not in 1..={BASE_SHAPE_DRAFT_TOKENS}, the drafts a base round takes"
            ));
        }
        if !(1..=DFLASH2_MAX_DRAFT_TOKENS).contains(&stay_accepted) {
            return Err(format!(
                "the shape rule's stay threshold {stay_accepted} is not in 1..={DFLASH2_MAX_DRAFT_TOKENS}, the drafts an extended round takes"
            ));
        }
        Ok(Self { enter_accepted, stay_accepted })
    }

    pub fn enter_accepted(&self) -> u32 {
        self.enter_accepted
    }

    pub fn stay_accepted(&self) -> u32 {
        self.stay_accepted
    }
}

impl Default for ShapeRule {
    fn default() -> Self {
        Self { enter_accepted: Self::DEFAULT_ENTER_ACCEPTED, stay_accepted: Self::DEFAULT_STAY_ACCEPTED }
    }
}

/// GitHub #690 (spec #684): the **shape policy** on a DFlash2 load with an
/// extended round shape (`kernel/src/round_shape.h` `policy_extends`): how
/// the lanes' asks of the [`ShapeRule`] become a verify round's shape. A
/// round of 5 to 8 lanes runs at the base shape under either. The server's
/// `spec.shape_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShapePolicy {
    /// At one lane, the lane's ask. At 2 to 4 lanes, extended when the drafts
    /// the asking lanes are expected to take past the base window exceed
    /// what the wider round costs every lane in it ([`ShapeEstimate`]). The
    /// default.
    #[default]
    Aggregate,
    /// Extended only when every lane in the round asks: no lane pays for
    /// another.
    All,
}

impl ShapePolicy {
    /// The operator's spellings (`--spec-shape-policy`).
    pub const NAMES: [&'static str; 2] = ["aggregate", "all"];

    /// Parse the operator's spelling, naming both on a refusal.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim() {
            "aggregate" => Ok(Self::Aggregate),
            "all" => Ok(Self::All),
            other => Err(format!("unknown shape policy `{other}` (expected aggregate or all)")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Aggregate => "aggregate",
            Self::All => "all",
        }
    }

    /// `enum ignis_shape_policy` (`kernel/include/ignis_model.h`).
    pub fn abi_code(&self) -> u32 {
        match self {
            Self::Aggregate => 0,
            Self::All => 1,
        }
    }
}

/// GitHub #690: the most verify rounds of its own a lane's expectations read
/// over (`IGNIS_SHAPE_ESTIMATE_MAX_ROUNDS`).
pub const SHAPE_ESTIMATE_MAX_ROUNDS: u32 = 64;

/// GitHub #690: what [`ShapePolicy::Aggregate`] reads beside the lanes' asks:
/// the recent verify rounds of its own each lane's expectations are read over
/// (the server's `spec.shape_estimate_rounds`), and the extended round's
/// period over the base round's at 1 to [`EXTENDED_SHAPE_MAX_LANES`] lanes, in
/// thousandths (`spec.shape_period_ratios`; the one-lane entry is not read, a
/// lane alone running at the shape it asks). The window's default is
/// provisional; the ratios' are measured on the RTX 5090
/// ([`Self::DEFAULT_PERIOD_RATIOS_MILLI`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeEstimate {
    rounds: u32,
    period_ratios_milli: [u32; EXTENDED_SHAPE_MAX_LANES as usize],
}

impl ShapeEstimate {
    /// 16 rounds: a lane whose extended rounds do not pay is granted one
    /// again once its window holds none, which a lane beside it pays for
    /// once in 17 rounds -- under 2% of its rate at two lanes -- and a lane
    /// whose text turns predictable waits at most 16 rounds.
    pub const DEFAULT_ROUNDS: u32 = 16;
    /// 1.128, 1.29, 1.363 and 1.344 at 1 to 4 lanes: measured on the RTX
    /// 5090 on the GPU clock (round timing), FP8 exception weights, the
    /// `hq-e8-2b` cache, the default cost levers and route, the extended
    /// round's period over the base round's at the same lane count and a
    /// matched context, pooled over two orders. At 3 and 4 lanes the card's
    /// 575 W power limit lowers the extended round's clock, so 3 lanes
    /// measured above 4 in either order (not a slip); and the ratio grows with
    /// context, these being its value at about 3.5-5K tokens.
    pub const DEFAULT_PERIOD_RATIOS_MILLI: [u32; EXTENDED_SHAPE_MAX_LANES as usize] = [1128, 1290, 1363, 1344];
    /// The narrowest and widest ratio taken: an extended round costs at
    /// least a base round and at most twice one, for at most twice its
    /// columns.
    pub const PERIOD_RATIO_MILLI_RANGE: std::ops::RangeInclusive<u32> = 1000..=2000;

    /// `rounds` in `1..=`[`SHAPE_ESTIMATE_MAX_ROUNDS`] and every ratio in
    /// [`Self::PERIOD_RATIO_MILLI_RANGE`]; the refusal names which and its
    /// range.
    pub fn new(rounds: u32, period_ratios_milli: [u32; EXTENDED_SHAPE_MAX_LANES as usize]) -> Result<Self, String> {
        if !(1..=SHAPE_ESTIMATE_MAX_ROUNDS).contains(&rounds) {
            return Err(format!(
                "the shape policy's estimate window of {rounds} rounds is not in 1..={SHAPE_ESTIMATE_MAX_ROUNDS}"
            ));
        }
        if let Some(lanes) = period_ratios_milli.iter().position(|ratio| !Self::PERIOD_RATIO_MILLI_RANGE.contains(ratio)) {
            return Err(format!(
                "the shape policy's period ratio at {} lane{} ({}) is not in 1..=2",
                lanes + 1,
                if lanes == 0 { "" } else { "s" },
                f64::from(period_ratios_milli[lanes]) / 1000.0
            ));
        }
        Ok(Self { rounds, period_ratios_milli })
    }

    pub fn rounds(&self) -> u32 {
        self.rounds
    }

    pub fn period_ratios_milli(&self) -> [u32; EXTENDED_SHAPE_MAX_LANES as usize] {
        self.period_ratios_milli
    }
}

impl Default for ShapeEstimate {
    fn default() -> Self {
        Self { rounds: Self::DEFAULT_ROUNDS, period_ratios_milli: Self::DEFAULT_PERIOD_RATIOS_MILLI }
    }
}

/// A validated speculation option: a backend, its draft window, the
/// drafter's proposal head, the rounds' draft rule and (GitHub #689, #690)
/// the shape rule's thresholds and the shape policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Speculation {
    backend: SpeculativeBackend,
    draft_tokens: u32,
    proposal_head: ProposalHead,
    draft_rule: DraftRule,
    drafter_sampling: DrafterSampling,
    shape_rule: ShapeRule,
    shape_policy: ShapePolicy,
    shape_estimate: ShapeEstimate,
}

impl Speculation {
    /// `draft_tokens` must be in `1..=` the backend's
    /// [`SpeculativeBackend::max_draft_tokens`]; the refusal names the
    /// drafter, its range and the value.
    pub fn new(backend: SpeculativeBackend, draft_tokens: u32) -> Result<Self, String> {
        if let Some(refusal) = backend.refuse_window(draft_tokens) {
            return Err(refusal);
        }
        Ok(Self {
            backend,
            draft_tokens,
            proposal_head: ProposalHead::Full,
            draft_rule: DraftRule::Fixed,
            drafter_sampling: DrafterSampling::default(),
            shape_rule: ShapeRule::default(),
            shape_policy: ShapePolicy::default(),
            shape_estimate: ShapeEstimate::default(),
        })
    }

    /// The same option with its rounds' shapes chosen by `shape_rule`
    /// (GitHub #689). On a load with an extended shape a stay threshold past
    /// the load's window could never keep a lane there, and is refused naming
    /// the threshold and the window; every other load never reads the rule.
    pub fn with_shape_rule(self, shape_rule: ShapeRule) -> Result<Self, String> {
        if self.has_extended_shape() && shape_rule.stay_accepted > self.draft_tokens {
            return Err(format!(
                "the shape rule's stay threshold {} is past the window of {} drafts, which no extended round can take",
                shape_rule.stay_accepted, self.draft_tokens
            ));
        }
        Ok(Self { shape_rule, ..self })
    }

    pub fn shape_rule(&self) -> ShapeRule {
        self.shape_rule
    }

    /// The same option with its rounds' shapes chosen from the lanes' asks
    /// by `shape_policy`, its expectations read as `shape_estimate` says
    /// (GitHub #690). Every load without an extended shape never reads them.
    pub fn with_shape_policy(self, shape_policy: ShapePolicy, shape_estimate: ShapeEstimate) -> Self {
        Self { shape_policy, shape_estimate, ..self }
    }

    pub fn shape_policy(&self) -> ShapePolicy {
        self.shape_policy
    }

    pub fn shape_estimate(&self) -> ShapeEstimate {
        self.shape_estimate
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

    /// The draft positions this load reports ([`draft_positions`]).
    pub fn draft_positions(&self) -> u32 {
        draft_positions(self.draft_tokens)
    }

    /// Whether a load of this option holds an extended round shape beside its
    /// base shape ([`Speculation::round_shapes`]): DFlash2 past
    /// [`BASE_SHAPE_DRAFT_TOKENS`] drafts, at every lane count.
    pub fn has_extended_shape(&self) -> bool {
        self.backend == SpeculativeBackend::Dflash2 && self.draft_tokens > BASE_SHAPE_DRAFT_TOKENS
    }

    /// The round shapes a load of `decode_lanes` lanes holds (GitHub #687,
    /// `plan_load_sizes`, `kernel/src/model.cu`): its base shape, and its
    /// extended shape where it has one. Up to [`BASE_SHAPE_DRAFT_TOKENS`], and
    /// under every backend but DFlash2, one shape: the load's window at its
    /// lanes. Past it a DFlash2 load's base shape is
    /// [`BASE_SHAPE_DRAFT_TOKENS`] at its lanes, and its extended shape the
    /// load's window at no more than [`EXTENDED_SHAPE_MAX_LANES`]. The two
    /// hold one set of device buffers, each the larger of the two shapes', so
    /// the plan's lines take each buffer once.
    pub fn round_shapes(&self, decode_lanes: u32) -> (RoundShape, Option<RoundShape>) {
        if self.has_extended_shape() {
            let extended = RoundShape { window: self.draft_tokens, lanes: decode_lanes.min(EXTENDED_SHAPE_MAX_LANES) };
            (RoundShape { window: BASE_SHAPE_DRAFT_TOKENS, lanes: decode_lanes }, Some(extended))
        } else {
            (RoundShape { window: self.draft_tokens, lanes: decode_lanes }, None)
        }
    }

    /// The device bytes the load's drafter keeps per sequence in a pool of
    /// `slot_count` slots (GitHub #619: a lane's share of the plan's
    /// `lane_state` line). The state lives in the sequence pool (P5-03,
    /// GitHub #152), one lane per slot, because snapshot, restore and prefix
    /// clone carry it with the rest of a sequence.
    ///
    /// DFlash2 keeps its drafter window: BF16, layers × window × KV heads ×
    /// head width × (K + V) per slot — 40 MiB. The window only: the reference
    /// also keeps a rewrite checkpoint of it, which ignis carried for every
    /// slot and never read, so it is not reserved. Independent of the draft
    /// window.
    ///
    /// The 27B's MTP head keeps the continuation hidden (GitHub #620): one
    /// BF16 hidden per slot. Its attention state is the KV pool's
    /// ([`crate::seq::kv_geometry`]), not a slot's.
    pub fn drafter_state_bytes(&self, slot_count: u32) -> u64 {
        match self.backend {
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
            SpeculativeBackend::Mtp => u64::from(slot_count) * crate::mtp_head::continuation_bytes(),
            // The verify substrate binds no drafter (P5-04, GitHub #153).
            SpeculativeBackend::VerifyOnly => 0,
        }
    }

    /// What the load's drafter adds to one slot's hq-e8-2b residual window
    /// under `kv_format` (the plan's `hq_residual_window` line): the 27B's MTP
    /// head's layer, its planes and its own ring words (GitHub #620). Nothing
    /// under BF16 KV, and nothing for a drafter that keeps no attention layer
    /// in the pool.
    pub fn hq_residual_window_bytes(&self, kv_format: crate::KvFormat) -> u64 {
        match (self.backend, kv_format) {
            (SpeculativeBackend::Mtp, crate::KvFormat::HqE8_2b) => crate::mtp_head::hq_residual_window_bytes(),
            _ => 0,
        }
    }

    /// What a `prefill_chunk_tokens`-wide chunk adds to the leaf's prefill
    /// scratch for the load's drafter.
    ///
    /// DFlash2's feature taps (`dflash2_prefill_scratch_bytes`,
    /// `kernel/src/model.cu`): the taps, their projection and normalization,
    /// and one drafter layer's query/key/value parent with the key and value
    /// rows kept from it — each at most one window of columns wide, each
    /// rounded to the arena's alignment.
    ///
    /// The 27B's MTP head writes its rows over the chunk in calls of at most
    /// [`crate::mtp_head::PREFILL_ROWS`] rows ([`crate::mtp_head::prefill_scratch_bytes`]),
    /// at three rotation axes a row on a load with `vision` (GitHub #624);
    /// DFlash2's taps are the same with or without it.
    pub fn prefill_scratch_bytes(&self, prefill_chunk_tokens: u32, vision: bool) -> u64 {
        match self.backend {
            SpeculativeBackend::Dflash2 => {
                use crate::arena::{bf16, words as i32};
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
            SpeculativeBackend::Mtp => crate::mtp_head::prefill_scratch_bytes(prefill_chunk_tokens, vision),
            // No drafter bound, nothing tapped (P5-04, GitHub #153).
            SpeculativeBackend::VerifyOnly => 0,
        }
    }

    /// What the load's drafter adds to the decode round's scratch, the plan's
    /// `decode_graph` line, at `decode_lanes` lanes: the 27B's MTP head's
    /// buffers over the widest round's `lanes * (k + 1)` columns (GitHub #620,
    /// [`crate::mtp_head::round_scratch_bytes`]), which every round runs the
    /// head in, with the rows' rotation on a load with `vision` (GitHub #624).
    /// DFlash2's round has its own line ([`Self::round_scratch_bytes`]).
    pub fn decode_round_bytes(&self, decode_lanes: u32, vision: bool) -> u64 {
        match self.backend {
            SpeculativeBackend::Mtp => crate::mtp_head::round_scratch_bytes(
                u64::from(decode_lanes) * u64::from(self.draft_tokens + 1),
                vision,
            ),
            SpeculativeBackend::Dflash2 | SpeculativeBackend::VerifyOnly => 0,
        }
    }

    /// What the load's drafter's verify round adds to a load of
    /// `decode_lanes` lanes: the plan's `drafter_round` line.
    ///
    /// DFlash2's (P5-05, GitHub #155; `kernel/src/model.cu`): the feature
    /// taps of every verify column of the widest round -- one of
    /// `decode_lanes` lanes since GitHub #336, where it was always 8 -- one
    /// append count per lane, and the round scratch -- the larger of the
    /// drafter's forward activations and the round's context append, each
    /// rounded to the arena's alignment (the top-k's row-split partials
    /// excepted, added as the leaf adds them), plus the fixed allowance for
    /// the vendored workspaces the forward calls into. Under
    /// [`DraftRule::Sampled`], also the drafter's distribution kept from the
    /// draw to the accept: each draft column's candidate ids and their
    /// probabilities (GitHub #576, `dflash2_draw_bytes`).
    ///
    /// GitHub #687: with two round shapes ([`Self::round_shapes`]) each
    /// buffer is the larger of the two shapes', and the extended shape asks
    /// the scratch by its forward's phases
    /// (`dflash2_extended_round_scratch_bytes`): while a layer runs, the
    /// query block, the layer's blocks and the allowance; after the last one,
    /// the query block and the head's buffers where the layers' were. That is
    /// under the base shape's sum at every lane count, so the scratch stays
    /// the base shape's.
    pub fn round_scratch_bytes(&self, decode_lanes: u32) -> u64 {
        match self.backend {
            // The fake drafter proposes from the host: nothing to reserve. The
            // 27B's MTP head runs in the round's own scratch
            // ([`Self::decode_round_bytes`]), and Flash-Next's MTP sizes its
            // own (spec flash-next/07).
            SpeculativeBackend::VerifyOnly | SpeculativeBackend::Mtp => 0,
            SpeculativeBackend::Dflash2 => {
                let (base, extended) = self.round_shapes(decode_lanes);
                let at = Dflash2RoundBuffers::at(base, self.draft_rule);
                let scratch = (at.query + at.layers + at.head).max(at.append) + DFLASH2_ROUND_WORKSPACE_BYTES;
                let mut lines = [at.features, at.append_counts, at.draws, scratch];
                if let Some(extended) = extended {
                    let at = Dflash2RoundBuffers::at(extended, self.draft_rule);
                    let scratch = (at.query + (at.layers + DFLASH2_ROUND_WORKSPACE_BYTES).max(at.head))
                        .max(at.append + DFLASH2_ROUND_WORKSPACE_BYTES);
                    for (line, extended) in lines.iter_mut().zip([at.features, at.append_counts, at.draws, scratch]) {
                        *line = (*line).max(extended);
                    }
                }
                lines.iter().sum()
            }
        }
    }
}

/// The DFlash2 drafter's round buffers at one round shape, in bytes
/// (`plan_load_sizes`, `dflash2_round_activations`, `kernel/src/model.cu`):
/// the feature taps, the append counts and the sampled draft's draws, and
/// what the drafter's round allocates from its scratch -- the forward's query
/// block (ids, positions, residual), one layer's attention and MLP blocks
/// counted together, the draft columns' head, top-k and selector, and the
/// round's context append.
struct Dflash2RoundBuffers {
    features: u64,
    append_counts: u64,
    draws: u64,
    query: u64,
    layers: u64,
    head: u64,
    append: u64,
}

impl Dflash2RoundBuffers {
    fn at(shape: RoundShape, draft_rule: DraftRule) -> Self {
        use crate::arena::{bf16, words as wide};
        let lanes = u64::from(shape.lanes);
        let columns = u64::from(shape.window + 1) * lanes;
        let drafts = u64::from(shape.window) * lanes;
        let kv_width = DFLASH2_KV_HEADS * DFLASH2_HEAD_DIM;
        let hidden_columns = bf16(HIDDEN * columns);
        // Added unrounded, as the leaf adds it (GitHub #259: this
        // term was missing since the top-k was replaced, 5dcfade).
        let topk_splits = VOCAB.div_ceil(DFLASH2_TOPK_ROWS_PER_SPLIT).min(DFLASH2_TOPK_MAX_SPLITS);
        let topk_partials = topk_splits * drafts * DFLASH2_SELECTOR_TOP_K * DFLASH2_TOPK_ENTRY_BYTES;

        let query = 2 * wide(columns) + hidden_columns;
        let layers = 2 * hidden_columns
            + bf16(DFLASH2_CONV_PROJ_ROWS * columns)
            + bf16((DFLASH2_QUERY_SIZE + 2 * kv_width) * columns)
            + 3 * bf16(DFLASH2_QUERY_SIZE * columns)
            + 3 * bf16(kv_width * columns)
            + 2 * hidden_columns
            + 2 * hidden_columns
            + bf16(DFLASH2_CONV_PROJ_ROWS * columns)
            + bf16(FFN_INTERMEDIATE * columns)
            + 2 * hidden_columns;
        let head = 2 * bf16(HIDDEN * drafts)
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

        Self {
            features: DFLASH2_FEATURE_TAPS * HIDDEN * columns * 2,
            append_counts: lanes * 4,
            draws: match draft_rule {
                DraftRule::Fixed => 0,
                DraftRule::Sampled => 2 * DFLASH2_SELECTOR_TOP_K * drafts * 4,
            },
            query,
            layers,
            head,
            append,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GitHub #689, #690 (ADR 0046): the shape rule's and the shape policy's
    /// defaults are declared once, here (and the server's schema takes
    /// them); the leaf repeats them only as what a load's 0 stands for, and
    /// the range of a period ratio, in named constants of
    /// `kernel/src/round_shape.h`, which this reads as text and holds to
    /// these. No toolchain: a value changed on one side alone goes red here.
    #[test]
    fn the_leafs_shape_defaults_are_the_cores() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernel/src/round_shape.h");
        let header = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let constant = |name: &str| -> String {
            let line = header
                .lines()
                .find(|line| line.starts_with("inline constexpr") && line.contains(&format!(" {name} = ")))
                .unwrap_or_else(|| panic!("round_shape.h declares no `{name} = ...;` line"));
            let value = line.split_once(" = ").expect("an initializer").1;
            value.trim_end().strip_suffix(';').unwrap_or_else(|| panic!("`{name}` ends its line")).trim().to_owned()
        };
        let number = |name: &str| constant(name).parse::<u32>().unwrap_or_else(|e| panic!("`{name}`: {e}"));
        assert_eq!(number("kDefaultEnterAccepted"), ShapeRule::DEFAULT_ENTER_ACCEPTED);
        assert_eq!(number("kDefaultStayAccepted"), ShapeRule::DEFAULT_STAY_ACCEPTED);
        assert_eq!(number("kDefaultEstimateRounds"), ShapeEstimate::DEFAULT_ROUNDS);
        let ratios: Vec<u32> = constant("kDefaultPeriodRatios")
            .trim_start_matches('{')
            .trim_end_matches('}')
            .split(',')
            .map(|ratio| ratio.trim().parse().unwrap_or_else(|e| panic!("kDefaultPeriodRatios: {e}")))
            .collect();
        assert_eq!(ratios, ShapeEstimate::DEFAULT_PERIOD_RATIOS_MILLI);
        assert_eq!(
            number("kPeriodRatioMinMilli")..=number("kPeriodRatioMaxMilli"),
            ShapeEstimate::PERIOD_RATIO_MILLI_RANGE
        );
    }

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

    /// GitHub #685: the widest window is the drafter's own. Each of DFlash2,
    /// the MTP head and the verify substrate alone holds its cap: GitHub #687,
    /// DFlash2 drafts up to 15, the other two keep 7.
    const WINDOW_RANGES: [(SpeculativeBackend, u32, &str); 3] = [
        (SpeculativeBackend::Dflash2, 15, "1..15"),
        (SpeculativeBackend::Mtp, 7, "1..7"),
        (SpeculativeBackend::VerifyOnly, 7, "1..7"),
    ];

    #[test]
    fn every_window_in_range_is_accepted() {
        for (backend, widest, _) in WINDOW_RANGES {
            assert_eq!(backend.max_draft_tokens(), widest, "{}", backend.as_str());
            for n in 1..=widest {
                let spec = Speculation::new(backend, n).unwrap_or_else(|e| panic!("{} at {n}: {e}", backend.as_str()));
                assert_eq!((spec.backend(), spec.draft_tokens()), (backend, n));
            }
        }
    }

    #[test]
    fn a_window_outside_the_range_is_refused_naming_the_range() {
        for (backend, widest, range) in WINDOW_RANGES {
            for n in [0, 8, 15, 16, 17, u32::MAX] {
                if (1..=widest).contains(&n) {
                    continue;
                }
                let err = Speculation::new(backend, n).expect_err("out of range");
                let at = format!("{} at {n}", backend.as_str());
                assert!(err.contains(&format!("must be in {range} ")), "{at}: {err}");
                assert!(err.contains(backend.as_str()), "{at}: the refusal names the drafter: {err}");
                assert!(err.contains(&format!("got {n}")), "{at}: {err}");
            }
            // The last window in the range, and the first one past it.
            assert!(Speculation::new(backend, widest).is_ok(), "{} at {widest}", backend.as_str());
            assert!(Speculation::new(backend, widest + 1).is_err(), "{} at {}", backend.as_str(), widest + 1);
        }
    }

    /// GitHub #685: the draft positions a load reports (its rounds'
    /// counters, the request log's profile, what the server prints) are sized by the
    /// load's own window, never fewer than the seven every load reported
    /// while one cap bounded every drafter: so every load a drafter at 7
    /// takes, DFlash2's, the 27B's MTP head's, the verify substrate's and
    /// Flash-Next's, reports the seven it reported then, whatever its window;
    /// GitHub #687: a DFlash2 load past 7 drafts reports its own window.
    #[test]
    fn every_load_reports_the_seven_draft_positions_it_reported_before_the_cap_split() {
        for (backend, widest, _) in WINDOW_RANGES {
            for n in 1..=widest {
                let positions = Speculation::new(backend, n).unwrap().draft_positions();
                assert_eq!(positions, n.max(7), "{} at {n}", backend.as_str());
                assert!(positions as usize <= crate::types::DRAFT_POSITIONS, "the counters hold every position a load reports");
            }
        }
        for backend in [SpeculativeBackend::Mtp, SpeculativeBackend::VerifyOnly] {
            for n in 1..=7 {
                let flash_next = FlashNextSpeculation::new(backend, n, 0).unwrap();
                assert_eq!(flash_next.draft_positions(), 7, "Flash-Next's {} at {n}", backend.as_str());
            }
        }
    }

    #[test]
    fn each_drafter_reserves_its_own_state_and_the_verify_substrate_none() {
        let spec = Speculation::new(SpeculativeBackend::VerifyOnly, 3).expect("in range");
        assert_eq!((spec.drafter_state_bytes(4), spec.prefill_scratch_bytes(1024, false), spec.round_scratch_bytes(4)), (0, 0, 0));
        assert_eq!((spec.decode_round_bytes(4, false), spec.hq_residual_window_bytes(crate::KvFormat::HqE8_2b)), (0, 0));
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 3).expect("in range");
        assert!(spec.drafter_state_bytes(4) > 0 && spec.prefill_scratch_bytes(1024, false) > 0 && spec.round_scratch_bytes(4) > 0);
        assert_eq!((spec.decode_round_bytes(4, false), spec.hq_residual_window_bytes(crate::KvFormat::HqE8_2b)), (0, 0));
        // GitHub #620: the 27B's MTP head keeps a continuation hidden per
        // slot, its layer in the hq window, and its buffers in a prefill chunk
        // and a round; it has no drafter round of its own.
        let spec = Speculation::new(SpeculativeBackend::Mtp, 3).expect("in range");
        assert_eq!(spec.drafter_state_bytes(4), 4 * 10_240);
        assert_eq!(spec.prefill_scratch_bytes(1024, false), crate::mtp_head::prefill_scratch_bytes(1024, false));
        assert_eq!(spec.round_scratch_bytes(4), 0);
        assert_eq!(spec.decode_round_bytes(4, false), crate::mtp_head::round_scratch_bytes(16, false));
        assert_eq!(spec.hq_residual_window_bytes(crate::KvFormat::HqE8_2b), 2_228_288);
        assert_eq!(spec.hq_residual_window_bytes(crate::KvFormat::Bf16), 0);
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
        for draft in 1..=MTP_MAX_DRAFT_TOKENS {
            assert_eq!(widths(draft, 0), widths(draft, FLASH_NEXT_DEFAULT_VERIFY_ROWS), "0 is the default budget");
        }
        // Every window fits its budget, and none goes past the maximum.
        for draft in 1..=MTP_MAX_DRAFT_TOKENS {
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
        for draft in 1..=MTP_MAX_DRAFT_TOKENS {
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
            assert!(err.contains("1..7") && err.contains("mtp"), "{err}");
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
        assert_eq!(spec.drafter_state_bytes(8), 0, "no drafter, no window pool");
        assert_eq!(spec.prefill_scratch_bytes(128, false), 0, "no drafter, no prefill taps");
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
        assert_eq!(spec.drafter_state_bytes(1), 40 * 1024 * 1024);
        assert_eq!(spec.drafter_state_bytes(crate::N_DECODE_LANES as u32), 8 * 40 * 1024 * 1024);
        let narrow = Speculation::new(SpeculativeBackend::Dflash2, 1).unwrap();
        assert_eq!(narrow.drafter_state_bytes(8), spec.drafter_state_bytes(8), "the window does not size the pool");
    }

    #[test]
    fn the_dflash2_prefill_scratch_is_bounded_by_the_window() {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, 7).unwrap();
        // 128 columns, worked by hand: taps 6,553,600 + positions 512 + two
        // scalars 256 each + projected and context 1,310,720 each + qkv
        // 1,572,864 + key_raw, key, value 262,144 each.
        assert_eq!(spec.prefill_scratch_bytes(128, false), 11_535_360);
        // A chunk wider than the window taps at most the window.
        assert_eq!(spec.prefill_scratch_bytes(4096, false), spec.prefill_scratch_bytes(2048, false));
        assert!(spec.prefill_scratch_bytes(1024, false) < spec.prefill_scratch_bytes(2048, false));
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

    /// GitHub #687: the round shapes a load holds. One, the option's own window
    /// at the load's lanes, up to a window of 7 and under every backend but
    /// DFlash2; past 7 DFlash2's two: the base shape at 7 and the load's lanes,
    /// the extended one at the window and at most four lanes, so that neither
    /// passes 64 columns. The two draft out of one set of buffers, each the
    /// larger of the two shapes': at eight lanes both top out at 64 columns, so
    /// the drafter's line is the 7-draft load's -- the scratch to the byte (the
    /// extended forward's larger head, 60 draft columns against 56, lands where
    /// its layers' blocks were), and under the sampled draft 512 bytes more of
    /// draws (16 candidates' ids and probabilities of 4 more draft columns).
    /// Below eight lanes the extended shape has the more columns, and its taps
    /// size the line.
    #[test]
    fn a_dflash2_load_past_seven_drafts_holds_a_base_and_an_extended_shape_on_one_set_of_buffers() {
        assert_eq!((BASE_SHAPE_DRAFT_TOKENS, EXTENDED_SHAPE_MAX_LANES), (7, 4));
        for lanes in 1..=8u32 {
            for window in 1..=7 {
                for (backend, _, _) in WINDOW_RANGES {
                    let spec = Speculation::new(backend, window).unwrap();
                    assert_eq!(spec.round_shapes(lanes), (RoundShape { window, lanes }, None), "{} at {window}", backend.as_str());
                }
            }
            for window in 8..=DFLASH2_MAX_DRAFT_TOKENS {
                let (base, extended) = Speculation::new(SpeculativeBackend::Dflash2, window).unwrap().round_shapes(lanes);
                assert_eq!(base, RoundShape { window: 7, lanes }, "window {window}");
                let extended = extended.unwrap_or_else(|| panic!("window {window}: an extended shape"));
                assert_eq!(extended, RoundShape { window, lanes: lanes.min(4) }, "window {window}, {lanes} lanes");
                assert!(base.lanes * 8 <= 64 && extended.lanes * (window + 1) <= 64, "window {window}, {lanes} lanes");
            }
        }

        let at = |window| Speculation::new(SpeculativeBackend::Dflash2, window).unwrap();
        let sampled = |window| at(window).with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
        assert_eq!(at(15).round_scratch_bytes(8), at(7).round_scratch_bytes(8));
        assert_eq!(at(15).round_scratch_bytes(8), 78_027_552);
        assert_eq!(sampled(15).round_scratch_bytes(8), sampled(7).round_scratch_bytes(8) + 512);
        // A window between the two: 4 x 10 columns and 36 draft columns, under
        // the base shape's 64 and 56 on every buffer.
        assert_eq!(sampled(9).round_scratch_bytes(8), sampled(7).round_scratch_bytes(8));
        // Four lanes: the extended shape's 64 columns against the base shape's
        // 32 double the taps (1,638,400 more); the scratch is still the base
        // shape's sum.
        assert_eq!(at(15).round_scratch_bytes(4), at(7).round_scratch_bytes(4) + 1_638_400);
        // The extended forward by its phases at 4 lanes of 16 columns: the query
        // block 655,872 with the layers' blocks 10,551,296 and the allowance
        // 33,554,432, against the query block with the head's 32,131,328; the
        // base forward's head at 8 lanes of 8 is 29,989,120.
        let extended = Dflash2RoundBuffers::at(RoundShape { window: 15, lanes: 4 }, DraftRule::Fixed);
        assert_eq!((extended.query, extended.layers, extended.head, extended.append), (655_872, 10_551_296, 32_131_328, 2_490_368));
        let base = Dflash2RoundBuffers::at(RoundShape { window: 7, lanes: 8 }, DraftRule::Fixed);
        assert_eq!((base.query, base.layers, base.head), (extended.query, extended.layers, 29_989_120));
        // At every lane count and window the extended forward by its phases fits
        // the base shape's sum, so the shared scratch is the base shape's.
        for lanes in 1..=8 {
            for window in 8..=DFLASH2_MAX_DRAFT_TOKENS {
                let (base, extended) = at(window).round_shapes(lanes);
                let base = Dflash2RoundBuffers::at(base, DraftRule::Fixed);
                let extended = Dflash2RoundBuffers::at(extended.unwrap(), DraftRule::Fixed);
                let base_scratch = base.query + base.layers + base.head + DFLASH2_ROUND_WORKSPACE_BYTES;
                let extended_scratch = extended.query + (extended.layers + DFLASH2_ROUND_WORKSPACE_BYTES).max(extended.head);
                assert!(extended_scratch < base_scratch, "window {window} at {lanes} lanes: {extended_scratch} against {base_scratch}");
            }
        }
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
