//! The 27B's extended verify round is in its base round's numerical class,
//! row by row (GitHub #692, spec #684; ADR 0104, ADR 0056's method on the
//! 27B).
//!
//! A DFlash2 load past seven drafts runs a verify round at its **extended
//! shape**, the window + 1 columns a lane at 1 to 4 lanes, beside its **base
//! shape**, 8 columns a lane. Past 8 columns a round's attention runs in two
//! chunks, its FP8 exception linears quantize their activations on another
//! tile and the GDN gating projection leaves its small-T route, so its logits
//! are the base round's up to summation order and rounding, not bit for bit,
//! and a free-running greedy text can leave at a near tie (as main's own text
//! already does between one lane and several). So the extended round is held
//! teacher-forced, as ADR 0056 holds a Flash-Next budget past 8 rows: the same
//! tokens through both shapes, logits row by row.
//!
//! # The rows
//!
//! On one load (the NVFP4 body, DFlash2 at 15 under the sampled draft,
//! hq-e8-2b, FP8 exception weights, the served lanes and context; #695: and,
//! in a test of its own, the trellis body the 16 GB and 24 GB profiles load,
//! under its own exception weights, `bf16`), per prompt:
//!
//! - **spec-off**: plain greedy rounds of one lane, one token each, to the
//!   turn's end or [`MAX_TOKENS`]: speculation off's own text `T`, and the row
//!   each round's draw read (`dflash2_readout::plain_round_logits`), row `i`
//!   the one after `T[i]`.
//! - Then each shape, at the same lane count, in two kinds of round, every
//!   round teacher-forced to `T` (`step::decode_program_verify_forced_with`,
//!   its shape named, never chosen) and its rows read at its own stride
//!   (`dflash2_readout::verify_logits` at the shape's window):
//!   - **forced**: the round is offered `T` as its drafts, so column `j` of a
//!     round at text index `t` holds `T[t + j]`, the round commits its whole
//!     shape, and its row `j` stands after the same tokens as spec-off's row
//!     `t + j`. Such a round runs eagerly (its drafts are not in a graph).
//!   - **graph**: the round verifies its drafter's own drafts and replays the
//!     shape's captured verify graph, the served round; the recording's
//!     verdict takes the drafts' leading agreement with `T`, so rows `0` to
//!     the accepted count stand after `T` and the next round starts where they
//!     end: every text index has one row.
//!
//!   Each kind runs at 1, 2, 3 and 4 lanes, a lane a prompt; the forced rounds
//!   of one group of lanes stand at one text index, the graph rounds of a lane
//!   whose text runs out leave the round, so a graph cell of `n` lanes ends
//!   on narrower rounds, and its base and extended rows at one index can come
//!   from rounds of different widths. Every round asserts the shape it ran at
//!   (its drafts, the shape's round count) and whether it ran eagerly.
//!
//! # What is held, and to what
//!
//! Not the extended row to speculation off's row by a bound of its own. On
//! this model the base round, main's own, already sits outside ADR 0056's
//! Flash-Next bounds against speculation off (#692: over 19,720 rows of three
//! long answers its rows' 99th percentile is 97.9 ulps and its widest 161.0
//! where that ADR allows 64, with 239 picks moved, 16 of them past its 16-ulp
//! near tie), and the extended round is the same class there (97.8, 200.5,
//! 213 picks moved). So the extended round is held to the base round:
//!
//! - **the row itself**: the extended row against the base row after the same
//!   tokens, at the same lane count ([`logit_drift::row_drift`]: the largest
//!   |difference| over the vocabulary in BF16 ulps at the base row's largest
//!   magnitude), every row within [`EXTENDED_VS_BASE_TOLERANCE_ULPS`], the
//!   median row of each lane within [`EXTENDED_VS_BASE_TYPICAL_ULPS`], and a
//!   pick that differs only where the base row rates the two within the
//!   class's near tie (below);
//! - **the class**: how far the extended rows sit from speculation off's
//!   against how far the base rows sit from them, at each lane count of a
//!   kind and over all of them together: the median, the 90th percentile and
//!   the widest row no further
//!   than the base rows' by [`CLASS_MEDIAN_MARGIN`], [`CLASS_P90_MARGIN`] and
//!   [`CLASS_WIDEST_MARGIN`]; no more picks moved from spec-off's than
//!   [`CLASS_MOVED_PICKS_MARGIN`] times those the base rows moved plus
//!   [`CLASS_MOVED_PICKS_ALLOWANCE`]; and, over every lane count of a kind
//!   together, none at a gap past the class's near tie: ADR 0056's
//!   [`PICK_GAP_ULPS`], or [`CLASS_PICK_GAP_MARGIN`] times the widest gap at
//!   which the base rows move a pick from spec-off's, whichever is more;
//! - **the text's likelihood**: each lane's mean log-probability of `T` under
//!   the extended rows within [`EXTENDED_VS_BASE_LOG_PROB_NATS`] of the base
//!   rows';
//! - **a control**: the extended row at index `i` against spec-off's row at
//!   `i + 1`, another context, what a row read from the wrong column looks
//!   like. Its median must be past [`CONTROL_MARGIN`] times both the right
//!   rows' median and [`EXTENDED_VS_BASE_TYPICAL_ULPS`], or the bounds could
//!   not tell a right row from a wrong one.
//!
//! Each shape's distance from speculation off prints per lane, per lane count
//! and per kind: a reading, not a bound. The margins are proposed, the
//! owner's to confirm or replace (ADR 0104).
//!
//! # Mixed rounds (#717, spec #711; ADR 0106)
//!
//! A **mixed round** carries lanes of both shapes, `w` at the extended shape
//! beside `n` at the base shape, one of the twelve mixes
//! ([`ignis_core::lane_lines::MIXES`]), and is held to the base round's
//! class as the extended round is, with the same method, margins and
//! windows. On the same load (no new load: each cell's, [`Cell::mixed`]; on
//! the trellis body bit for bit, below), speculation off writes the text of seven
//! prompts of its own ([`MIXED_PROMPTS`]), one a lane, its rows over a uniform
//! cell's window ([`MAX_TOKENS`]) and its text on past it
//! ([`MIXED_TEXT_TOKENS`]), and the decode graphs are captured again under
//! `IGNIS_MIXED_ROUNDS=1` ([`MIXED_ROUNDS`]), a verify graph for every mix.
//! Then, in each kind, at each lane count of 2 to 7, a leg of base rounds at
//! that lane count over the window, the lanes on the prompts in order, as a
//! uniform cell's, is the reference of every mix of that lane count, each a
//! leg of rounds at that mix (`TeacherForcing::lane_shapes`), its last `w`
//! lanes extended so that the round's rows, extended lanes first, are not the
//! caller's order: forced rounds offered the text as their drafts, placed by
//! group (eager), and graph rounds replaying the mix's graph on the drafter's
//! own drafts. A mixed leg rides its mix until every lane holds the rows its
//! base leg gave it, a lane that commits more a round riding on past them on
//! its text, so every one of its rounds is at the mix and at the cell's lane
//! count and each lane holds as many rows as a uniform cell's (a lane short of
//! them, its text run out first, is an excess); each round asserts every
//! lane's shape and extent, that it ran eagerly or from its graph, at neither
//! uniform shape and with no fallback. Its rows are read by group
//! ([`dflash2_readout::mixed_verify_logits`]), and every mixed row is held
//! to the base row at the same text index and lane count by the bounds above:
//! the row, a lane's median row and the text's likelihood per lane; the
//! class at each lane count (over its mixes) and over the kind's; the control,
//! one index off, over the kind's; and the near tie over the kind's, read, as
//! the extended rows' is, over the uniform cells' base rows of the kind.
//!
//! On the trellis body (#720) a lane's rows follow its round's columns, not its
//! lane count: the trellis projection routes by the call's token count, and on
//! the RTX 5090 a lane's rows are the same bits in every round of 24 to 56
//! columns, of either kind, where a graph round of 64 gives its own (its
//! drafter's drafts, and so the lane's rounds, move on two prompts). So the
//! base leg at the lane count is no reference there (a mix of 2 lanes is a
//! round of 24 columns, that leg's 16), and the owner decided on 2026-11-23
//! that each mixed row is held to the row of a uniform round of its own shape
//! at its round's columns, bit for bit, as it is ([`MixedRows::Uniform`], ADR
//! 0106): in each kind, uniform legs at one width on the mixed prompts' texts,
//! every lane riding until each holds a uniform cell's rows, so that no round
//! narrows, as no mixed round does ([`fixed_leg`]): for the mixes of 24 to 56
//! columns 7 base lanes (56 columns) and 3 extended (48: the extended shape has
//! no round of 24, 40 or 56), for the three of 64 8 base lanes and 4 extended;
//! then every mix as above, each lane riding at its mix to [`MAX_TOKENS`] rows,
//! every one of them its prompt's row in the uniform leg of its shape. The
//! uniform rounds are held to the class by the uniform cells above; no mixed
//! row is read against a base row there.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`, or
//! alone:
//!
//! ```text
//! cargo test -p ignis-core --features cuda --test extended_round_class_gpu -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout;
use ignis_core::gpu_profile;
use ignis_core::lane_lines::MIXES;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, model_scope, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::speculation::{BASE_SHAPE_DRAFT_TOKENS, EXTENDED_SHAPE_MAX_LANES};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch, decode_program_verify_forced_with, prefill_program_permitted,
    prefill_program_sampled, program_stats, SamplingParams, TeacherForcing, VerifyLane,
};
use ignis_core::{
    DraftRule, DrafterSampling, ExceptionWeights, KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation,
    SpeculativeBackend,
};

#[path = "support/logit_drift.rs"]
mod logit_drift;

use logit_drift::{row_drift, within, RowDrift};

// Measured on the RTX 5090. "Prototype": on 2026-11-19 by spec #684's
// prototype, one lane, eager rounds, in five cells -- a 240-row text of the
// first prompt below, the first 224 rows of each of three long answers (672
// rows), and those answers whole: a JSON structure (7,224 rows), a long JSON
// array (7,312) and a code module (5,184) -- and over this test's first three
// prompts (672 rows). "This test": its own run of 2026-11-20 (NVFP4 body,
// FP8 exception weights, hq-e8-2b), forced rounds over 3,360 rows and graph
// rounds over 3,205, at 1 to 4 lanes. "The trellis body" (#695): its own
// test's run of 2026-11-20 (the `sc30` artifact, BF16, hq-e8-2b), forced
// rounds over 3,360 rows and graph rounds over 3,255, at 1 to 4 lanes; its
// rows sit a median 1 ulp from speculation off's where the NVFP4 body's sit
// about 12 in this test, so it is far inside every bound below. "The mixed
// rounds" (#717): this test's run of 2026-11-22 on the NVFP4 cell, the twelve
// mixes at 2 to 7 lanes on the seven prompts of their own, each lane held over
// the uniform cells' window, forced rounds over 12,096 mixed rows (224 a lane)
// and graph rounds over 11,887 (217 to 223 a lane), each against the base rows
// at the same lane count. #720: on the trellis cell, its run of 2026-11-23,
// 12,096 forced and 12,096 graph mixed rows (224 a lane), each of them its
// uniform leg's bit for bit, so none is read by these bounds.

/// The extended row against the base row after the same tokens: no logit
/// further than this, in BF16 ulps at the base row's largest magnitude.
/// Measured widest: prototype 91.1, 113.8, 127.0, 138.9, 168.0, 96.9 by cell;
/// this test, forced rounds 96.9 at one lane, 0.0 at two and 1.0 at three
/// and four, graph rounds 114.1 at one lane and 115.2 at two to four; the
/// trellis body, forced rounds 66.6, 43.9 and 0.0 at one, two and three or
/// four lanes, graph rounds 64.0, 88.7, 41.1 and 39.8 at one to four; the
/// mixed rounds, forced 1.0, graph 110.8; 256 is the next power of two above,
/// ADR 0056's rule for its own tolerance.
const EXTENDED_VS_BASE_TOLERANCE_ULPS: f32 = 256.0;

/// What a lane's median row of that pair stays within. Measured medians:
/// prototype 14.0, 14.2, 19.5, 22.3, 20.1, 14.7 (the furthest of the three
/// prompts); this test, forced rounds 14.0 to 16.1 at one lane and 0.0 to 0.5
/// at two to four, graph rounds 13.6 to 16.0 at one lane and 11.8 to 14.2 at
/// two to four but for the code prompt's 0.0 there (neither split is
/// explained); the trellis body, 1.0 but for forced rounds' 0.0 at three and
/// four lanes; the mixed rounds, forced 0.0 to 0.5, graph 0.0 to 14.9 (the
/// extended lanes 0.0 to 14.9, the base lanes 0.0 to 13.0); 32 is the next
/// power of two above.
const EXTENDED_VS_BASE_TYPICAL_ULPS: f32 = 32.0;

/// ADR 0056's near tie, in the reference row's ulps: the floor of the class's
/// near tie ([`CLASS_PICK_GAP_MARGIN`]), inside which a pick that differs
/// between the shapes, or between the extended round and speculation off,
/// must sit.
const PICK_GAP_ULPS: f32 = logit_drift::PICK_GAP_ULPS;

/// The class, median: the extended rows' median distance from speculation
/// off's rows over the base rows'. Measured ratios: prototype 0.96, 1.01,
/// 1.01, 1.00, 1.01, 1.01; this test, forced rounds 1.00 (0.99 to 1.00 by lane
/// count), graph rounds 1.02 (0.99 to 1.03); the trellis body, 1.00 for both
/// kinds at every lane count; the mixed rounds, forced 1.00 at every lane
/// count, graph 1.00 (0.99 to 1.01). Held at each lane count too.
const CLASS_MEDIAN_MARGIN: f32 = 1.25;

/// The class, 90th percentile. Measured ratios: prototype 0.86, 1.03, 1.02,
/// 1.00, 1.02, 0.87; this test, forced rounds 1.00 (0.98 to 1.01), graph
/// rounds 0.99 (0.94 to 1.02); the trellis body, forced rounds 0.95 (0.75 to
/// 1.08), graph rounds 1.01 (1.00 to 1.04); the mixed rounds, forced 1.00,
/// graph 0.99 (0.95 to 1.02).
const CLASS_P90_MARGIN: f32 = 1.25;

/// The class, widest row: one tail logit each, so the loosest of the three.
/// Measured ratios: prototype 1.02, 1.20, 0.97, 1.07, 1.25, 1.02; this test,
/// forced rounds 1.00 (1.00 to 1.02), graph rounds 1.03 (1.03 to 1.06); the
/// trellis body, forced rounds 0.69 (0.59 to 1.00), graph rounds 0.49 (0.42 to
/// 0.53); the mixed rounds, forced 1.00, graph 1.00 (0.89 to 1.00).
const CLASS_WIDEST_MARGIN: f32 = 1.5;

/// The class, moved picks: the extended rows may move this many times the
/// picks the base rows move from speculation off's, and
/// [`CLASS_MOVED_PICKS_ALLOWANCE`] more (a count of a few picks has no ratio
/// worth a margin). Measured, extended against base: prototype 0 and 2, 3 and
/// 5, 0 and 0, 155 and 176, 58 and 63, 15 and 14; this test, forced rounds 90
/// and 89 (25 and 24, 25 and 25, 15 and 15, 25 and 25 at 1 to 4 lanes), graph
/// rounds 76 and 73 (19 and 20, 19 and 21, 12 and 11, 26 and 21: 26 against a
/// bound of 28 at 4 lanes); the trellis body, forced rounds 8 and 8 (2 and 2 at
/// each lane count), graph rounds 11 and 12 (3 and 2, 3 and 3, 2 and 3, 3 and
/// 4); the mixed rounds, forced 385 and 385 (the same at each lane count),
/// graph 311 and 286 (70 against a bound of 73 at 4 lanes the closest).
const CLASS_MOVED_PICKS_MARGIN: f32 = 1.25;
const CLASS_MOVED_PICKS_ALLOWANCE: usize = 2;

/// The class's near tie: a pick an extended row moves -- from speculation
/// off's row, or from the base row's -- only where the reference row rates the
/// two within [`PICK_GAP_ULPS`], or within this many times the widest gap at
/// which the base rows move a pick from speculation off's where that is more.
/// The base round itself moves picks from a plain round past ADR 0056's 16
/// ulps (to 42 on the long answers), so the near tie alone would hold the
/// extended round tighter than its base round is. Measured, the widest gap
/// from spec-off, extended over base: prototype none over 5.0, 4.0 over 4.0,
/// none over none, 26 over 42, 37 over 22 (1.68), 6.0 over 11.0; this test,
/// over each kind's lane counts together (the bound's reading), forced rounds
/// 12.0 over 12.0 and graph rounds 22.0 over 12.0 (1.83); at one lane count
/// the graph rounds' 4 lanes read 22.0 over 9.0 (2.44), past this margin there.
/// The widest gap between the shapes: prototype 0 to 8 on texts of this
/// test's length, 82.5 on the long answers; this test, forced rounds 8.0,
/// graph rounds 17.0 (one lane, over a base widest of 12.0 there), where ADR
/// 0056's 16 alone, the bound between the shapes before this one, failed the
/// test's first run. The trellis body: from spec-off 1.0 over 1.0 for both
/// kinds, so its near tie is ADR 0056's 16; between the shapes none moved on
/// forced rounds, 1.0 on graph rounds. The mixed rounds (#717, ADR 0106): their
/// near tie is the extended rows' of the same kind, read over the uniform
/// cells' base rows (12.0 for both kinds, so 24): a mixed lane's extended rows
/// are the extended round's, whose own 22.0 that bound passes, where a longer
/// leg's mixed base rows alone read 9.0, a bound of 18 that the same move
/// failed (#717's diagnostic). From spec-off, forced 9.0 over 9.0, graph 10.0
/// over 10.0 (the mixed cells' own base rows); against the base rows, none
/// moved on forced rounds, 12.0 on graph rounds.
const CLASS_PICK_GAP_MARGIN: f32 = 2.0;

/// A lane's mean log-probability of the text under the extended rows against
/// the base rows', in nats a token, either way. Measured apart by: prototype
/// 0.0074, 0.0031, 0.0001, 0.0040, 0.0001, and 0.0089 at the furthest of the
/// three prompts; this test, forced rounds at most 0.0089, graph rounds at
/// most 0.0124; the trellis body, forced rounds at most 0.0025, graph rounds
/// at most 0.0016; 0.02 is ADR 0056's bound for its own pair. It is a mean
/// over a uniform cell's window of about 210 rows, which a few near-tie rows
/// of prose move on a short one: the prose prompt's uniform extended graph
/// lane reads 0.0348 over its first 32 rows at one lane and 0.0303 at four,
/// 0.0242 and 0.0177 over 64, inside 0.02 for good only from row 87 and 180
/// (#717). The owner decided on 2026-11-22 that the mixed rows are held as the
/// extended round is, on legs that reach a uniform cell's rows; this test holds
/// a mixed lane over the same window (ADR 0106, proposed): the mixed rounds,
/// forced at most 0.0005, graph at most 0.0158, the essay prompt's extended
/// lane at the mix 3+2 over 218 rows, which reads 0.0229 over its first 64.
const EXTENDED_VS_BASE_LOG_PROB_NATS: f64 = logit_drift::MEAN_LOG_PROB_NATS;

/// The control's median, one index off, over the right rows' median and over
/// [`EXTENDED_VS_BASE_TYPICAL_ULPS`]. Measured: the control's median row sits
/// 166.5 ulps from speculation off's over the prototype's three prompts
/// (145.5 to 182.2 by prompt), the right rows' 11.6, 14 times apart; this
/// test, forced rounds 166.0 (14.0 times the right rows', 5.2 times the
/// typical bound), graph rounds 165.0 (13.8, 5.2); the trellis body, forced
/// rounds 167.5 (167.5 times the right rows' 1.0, 5.2 times the typical
/// bound), graph rounds 168.1 (168.1, 5.3); the mixed rounds, forced 160.5
/// (14.0, 5.0), graph 160.8 (14.0, 5.0); 2 is far inside both and far
/// outside a right pair's margins above.
const CONTROL_MARGIN: f32 = 2.0;

/// One body the class is held on: its artifact and the exception weights it
/// loads under.
struct Cell {
    name: &'static str,
    artifact: &'static str,
    exceptions: ExceptionWeights,
    /// #717: how the mixed rounds' class is held on it (#720: the trellis
    /// body's too).
    mixed: MixedRows,
}

/// #720: how a cell's mixed rows are held (ADR 0106).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MixedRows {
    /// #717: each mixed row against the base row at the same text index and
    /// lane count, by the class's bounds, as the extended rows are.
    Class,
    /// #720: each mixed row bit for bit the row of a uniform round of its
    /// lane's shape at its round's columns
    /// ([`hold_the_mixed_rows_bit_for_bit`]).
    Uniform,
}

/// The NVFP4 body the RTX 5090 profile serves, under FP8 exception weights.
const NVFP4: Cell = Cell { name: "nvfp4", artifact: "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer", exceptions: ExceptionWeights::Fp8, mixed: MixedRows::Class };
/// #695: the trellis body the 16 GB and 24 GB profiles load (the `sc30`
/// artifact), which holds no BF16 exception tensor: `bf16` is its own.
const TRELLIS: Cell = Cell { name: "trellis", artifact: "F:/ai/q38/ninfer-models/qwen3_8_27b_trellis_sc30-v2.ninfer", exceptions: ExceptionWeights::Bf16, mixed: MixedRows::Uniform };

/// The load: the widest DFlash2 window, at the served lanes, prefill chunk and
/// context. The context is the envelope the rounds' attention splits by, so it
/// is the served cell's, not the session's need; the pool below is small.
const WINDOW: u32 = 15;
const LANES: u32 = 8;
const CHUNK: u32 = 1024;
const MAX_CONTEXT: u32 = 262_144;
/// Each sequence's context: its prompt, its text and a round past it.
const SEQ_CONTEXT: u32 = 512;
/// The text speculation off writes per prompt, at most, and so the rows a lane
/// holds; and the fewest rows a lane must compare for its figures to mean
/// anything.
const MAX_TOKENS: usize = 224;
const MIN_ROWS: usize = 64;
/// #717: the text speculation off writes for each of [`MIXED_PROMPTS`], at
/// most: a mixed leg rides every lane at its mix until each holds the rows of
/// a uniform cell's lane ([`MAX_TOKENS`]), so a lane that commits more a round
/// rides past them on its text. Sized on #717's diagnostic: a prose lane
/// commits about 3 tokens a graph round, so a leg beside one rides about 78
/// rounds, and an extended lane on structured text about 13.5 a round, 1,050
/// tokens; this holds a third more. Measured on 2026-11-22: legs of 28 forced
/// rounds and of 45 to 88 graph rounds (1+6 the most), no lane short of its
/// rows.
const MIXED_TEXT_TOKENS: usize = 1408;
/// #717: a mixed leg's sequence's context: its prompt, its text and a round
/// past it.
const MIXED_SEQ_CONTEXT: u32 = 1536;
/// `<|im_end|>`, where the server stops a turn.
const IM_END: i32 = 248_046;

/// One prompt a lane: the first is the measured cell's prompt; the others are
/// text the extended shape is taken on (structured output, code) and text it
/// is not (prose).
const PROMPTS: [&str; 4] = [
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "Write a JSON array of 12 objects describing fictional employees, each with the fields id, name, email, department and city. Output only the JSON.",
    "Explain in three paragraphs why the sky is blue, why sunsets are red and why clouds are white.",
    "Write a Rust function that splits a CSV line into its fields, honouring double-quoted fields, with three unit tests.",
];

/// #717: the prompts of the mixed cells' lanes, one a lane, as [`PROMPTS`]'s
/// kinds of text in their order and then a table, prose and code, each asking
/// for more text than [`MIXED_TEXT_TOKENS`], which every one of them must reach
/// ([`PROMPTS`]' prose prompt ends at 337 tokens, and the shorter table prompt
/// this held before at 223, on 2026-11-22).
const MIXED_PROMPTS: [&str; 7] = [
    "Write a complete Python module for a library management system, with classes for books, authors, members, loans and reservations, a docstring on every method, and then a unittest test suite of at least forty test cases.",
    "Write a JSON array of 60 objects describing fictional employees, each with the fields id, name, email, department, city, start_date and salary. Output only the JSON.",
    "Write a long essay of thirty paragraphs on the history of the printing press, from its invention to the present day.",
    "Write a Rust module implementing a CSV parser that honours double-quoted fields and escaped quotes, with an error type, an iterator over records, a documentation comment on every item and thirty unit tests.",
    "Write a Markdown table of all 118 chemical elements, with columns for the atomic number, the symbol, the name, the atomic mass, the group and the period.",
    "Write a long story of thirty paragraphs about a lighthouse keeper who finds a message in a bottle.",
    "Write a SQL script for a library database: CREATE TABLE statements for books, authors, members, loans and reservations with their foreign keys and an index on each foreign key, then INSERT statements for thirty rows of sample data in each table.",
];

/// #717: the measurement switch that captures the mixed rounds' graphs, read
/// at capture: `1` captures a verify graph for every mix the load holds.
const MIXED_ROUNDS: &str = "IGNIS_MIXED_ROUNDS";

/// A verify round's shape, named to the round by its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Base,
    Extended,
}

impl Shape {
    fn window(self) -> u32 {
        match self {
            Shape::Base => BASE_SHAPE_DRAFT_TOKENS,
            Shape::Extended => WINDOW,
        }
    }

    fn columns(self) -> usize {
        self.window() as usize + 1
    }

    /// Its entry in the program's stats, the base shape first.
    fn stats_index(self) -> usize {
        match self {
            Shape::Base => 0,
            Shape::Extended => 1,
        }
    }
}

/// What a leg's rounds are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Offered the text as their drafts: every column holds it; eager.
    Forced,
    /// The drafter's own drafts, from the shape's verify graph: rows 0 to the
    /// accepted count hold the text.
    Graph,
}

/// The lanes of each group a kind runs at `width` lanes: the prompts in
/// order, `width` at a time, a short last group left out.
fn groups(width: usize) -> Vec<Vec<usize>> {
    (0..PROMPTS.len()).collect::<Vec<_>>().chunks(width).filter(|group| group.len() == width).map(<[usize]>::to_vec).collect()
}

/// One plain round of `seq`: the token it emitted, the row its draw read
/// (`[vocab]`, BF16) and the token that stands pending after it.
fn plain_round(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, what: &str) -> (i32, Vec<u16>, i32) {
    let emitted = decode_program_batch(model, pool, &mut [&mut *seq]).unwrap_or_else(|e| panic!("{what}: plain round: {e}"));
    let row = dflash2_readout::plain_round_logits(model, 1).unwrap_or_else(|e| panic!("{what}: the plain round's row: {e}"));
    (emitted[0], row, seq.pending_token().expect("a plain round draws"))
}

/// The load's stride from text index `at` on: `len` tokens of `text` after
/// `text[at]`, the -1 pad past its end (never read: a round reads only its
/// shape's).
fn after(text: &[i32], at: usize, len: usize) -> Vec<i32> {
    (0..len).map(|j| text.get(at + 1 + j).copied().unwrap_or(-1)).collect()
}

/// One teacher-forced verify round of `seqs` at `shape`, of `kind`, lane `i`
/// at index `at[i]` of `texts[i]`: each lane's rows that stand after its text,
/// row `j` after `texts[i][at[i] + j]`.
#[allow(clippy::too_many_arguments)]
fn forced_round(
    model: &Model,
    pool: &SeqPool,
    seqs: &mut [Seq<'_>],
    texts: &[&[i32]],
    at: &[usize],
    shape: Shape,
    kind: Kind,
    what: &str,
) -> Vec<Vec<Vec<u16>>> {
    let columns = shape.columns();
    let lanes = seqs.len();
    for (i, text) in texts.iter().enumerate() {
        assert!(at[i] + columns < text.len(), "{what}: lane {i}'s correction at {} is past its text", at[i] + columns);
    }
    // The load's stride whatever the round's shape: the recording, the text
    // after each anchor; the drafts, the same text.
    let recordings: Vec<Vec<i32>> = (0..lanes).map(|i| after(texts[i], at[i], WINDOW as usize + 1)).collect();
    let drafts: Vec<Vec<i32>> = (0..lanes).map(|i| after(texts[i], at[i], WINDOW as usize)).collect();
    let recording_refs: Vec<&[i32]> = recordings.iter().map(Vec::as_slice).collect();
    let draft_refs: Vec<&[i32]> = drafts.iter().map(Vec::as_slice).collect();
    let forcing = TeacherForcing {
        shape_window: Some(shape.window()),
        drafts: (kind == Kind::Forced).then_some(draft_refs.as_slice()),
        lane_shapes: None,
    };
    let before = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}")).shapes()[shape.stats_index()];
    let greedy: Vec<VerifyLane<'_>> = (0..lanes).map(|_| VerifyLane::greedy(&[])).collect();
    let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
    let runs = decode_program_verify_forced_with(model, pool, &mut refs, &greedy, WINDOW, &recording_refs, forcing)
        .unwrap_or_else(|e| panic!("{what}: a {kind:?} round at the {shape:?} shape: {e}"));
    let after_round = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}")).shapes()[shape.stats_index()];
    assert_eq!(after_round.verify_rounds, before.verify_rounds + 1, "{what}: the round ran at the {shape:?} shape");
    let eager = u64::from(kind == Kind::Forced);
    assert_eq!(
        after_round.eager_verify_rounds,
        before.eager_verify_rounds + eager,
        "{what}: a {kind:?} round at the {shape:?} shape {}",
        if eager == 1 { "runs eagerly" } else { "replays its graph" }
    );
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let logits = dflash2_readout::verify_logits(model, lanes as u32, shape.window())
        .unwrap_or_else(|e| panic!("{what}: the {shape:?} round's rows: {e}"));
    runs.iter()
        .enumerate()
        .map(|(i, run)| {
            assert_eq!(run.extent, shape.window(), "{what}: lane {i} drafts the {shape:?} shape's window");
            let committed = run.tokens.len();
            assert!(committed >= 1 && committed <= columns, "{what}: lane {i} committed {committed} of {columns} columns");
            if kind == Kind::Forced {
                assert_eq!(committed, columns, "{what}: lane {i} commits its whole {shape:?} round of its text's drafts");
            }
            assert_eq!(run.tokens[..], texts[i][at[i]..at[i] + committed], "{what}: lane {i} commits its text");
            (0..committed).map(|j| logits[(i * columns + j) * vocab..(i * columns + j + 1) * vocab].to_vec()).collect()
        })
        .collect()
}

/// The rows a leg gives each lane of `group`, by text index: every round of
/// `kind` at `shape`, from the prompt on, while the lane's text holds a whole
/// round.
#[allow(clippy::too_many_arguments)]
fn leg(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    texts: &[Vec<i32>],
    group: &[usize],
    shape: Shape,
    kind: Kind,
    what: &str,
) -> Vec<Vec<Vec<u16>>> {
    let greedy = SamplingParams::greedy();
    let mut seqs: Vec<Seq<'_>> = Vec::new();
    for &p in group {
        let mut seq = pool.alloc(SEQ_CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
        prefill_program_permitted(model, pool, &mut seq, &prompts[p], 0, greedy, &[texts[p][0]], None)
            .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
        seqs.push(seq);
    }
    let columns = shape.columns();
    let mut rows: Vec<Vec<Vec<u16>>> = vec![Vec::new(); group.len()];
    // The lanes still riding, by their place in `group`.
    let mut riding: Vec<usize> = (0..group.len()).collect();
    loop {
        // Forced rounds of one group stand at one index, so the group ends
        // with its shortest text; a graph lane whose text runs out leaves.
        let fits = |l: usize| rows[l].len() + columns < texts[group[l]].len();
        if kind == Kind::Forced && !riding.iter().all(|&l| fits(l)) {
            break;
        }
        riding.retain(|&l| fits(l));
        if riding.is_empty() {
            break;
        }
        let at: Vec<usize> = riding.iter().map(|&l| rows[l].len()).collect();
        let lane_texts: Vec<&[i32]> = riding.iter().map(|&l| texts[group[l]].as_slice()).collect();
        // The riding sequences, in order, out of `seqs` and back.
        let mut taken: Vec<Seq<'_>> = Vec::new();
        let mut rest: Vec<(usize, Seq<'_>)> = Vec::new();
        for (l, seq) in seqs.drain(..).enumerate() {
            if riding.contains(&l) {
                taken.push(seq);
            } else {
                rest.push((l, seq));
            }
        }
        let round_rows = forced_round(model, pool, &mut taken, &lane_texts, &at, shape, kind, what);
        for (&l, lane_rows) in riding.iter().zip(round_rows) {
            rows[l].extend(lane_rows);
        }
        let mut back: Vec<(usize, Seq<'_>)> = riding.iter().copied().zip(taken).chain(rest).collect();
        back.sort_by_key(|(l, _)| *l);
        seqs = back.into_iter().map(|(_, seq)| seq).collect();
    }
    rows
}

/// #717: a lane's shape in a mix, extended where its entry is `true`.
fn shape_of(extended: bool) -> Shape {
    if extended { Shape::Extended } else { Shape::Base }
}

/// #717: one teacher-forced mixed round of `seqs`, of `kind`, lane `i` at the
/// extended shape where `extended[i]` and at the base shape otherwise, at
/// index `at[i]` of `texts[i]`: each lane's rows that stand after its text, as
/// [`forced_round`]'s, read at its own group's stride.
#[allow(clippy::too_many_arguments)]
fn mixed_round(
    model: &Model,
    pool: &SeqPool,
    seqs: &mut [Seq<'_>],
    texts: &[&[i32]],
    at: &[usize],
    extended: &[bool],
    kind: Kind,
    what: &str,
) -> Vec<Vec<Vec<u16>>> {
    let lanes = seqs.len();
    for (i, text) in texts.iter().enumerate() {
        let columns = shape_of(extended[i]).columns();
        assert!(at[i] + columns < text.len(), "{what}: lane {i}'s correction at {} is past its text", at[i] + columns);
    }
    // The load's stride whatever each lane's shape, as a uniform round's; the
    // round reads each lane's first columns of its shape, at its group's place.
    let recordings: Vec<Vec<i32>> = (0..lanes).map(|i| after(texts[i], at[i], WINDOW as usize + 1)).collect();
    let drafts: Vec<Vec<i32>> = (0..lanes).map(|i| after(texts[i], at[i], WINDOW as usize)).collect();
    let recording_refs: Vec<&[i32]> = recordings.iter().map(Vec::as_slice).collect();
    let draft_refs: Vec<&[i32]> = drafts.iter().map(Vec::as_slice).collect();
    let forcing = TeacherForcing {
        shape_window: None,
        drafts: (kind == Kind::Forced).then_some(draft_refs.as_slice()),
        lane_shapes: Some(extended),
    };
    let before = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}"));
    let greedy: Vec<VerifyLane<'_>> = (0..lanes).map(|_| VerifyLane::greedy(&[])).collect();
    let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
    let runs = decode_program_verify_forced_with(model, pool, &mut refs, &greedy, WINDOW, &recording_refs, forcing)
        .unwrap_or_else(|e| panic!("{what}: a {kind:?} mixed round: {e}"));
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("{what}: stats: {e}"));
    // A mixed round is neither shape's round, and ran at its mix: from its
    // graph, or eagerly when offered drafts, never a uniform round in its place.
    for shape in [Shape::Base, Shape::Extended] {
        let (was, is) = (before.shapes()[shape.stats_index()].verify_rounds, stats.shapes()[shape.stats_index()].verify_rounds);
        assert_eq!(is, was, "{what}: a mixed round is no {shape:?} round");
    }
    assert_eq!(stats.mixed_fallback_rounds, before.mixed_fallback_rounds, "{what}: the round ran at its mix, not at a uniform shape");
    let graph = u64::from(kind == Kind::Graph);
    assert_eq!(stats.graph_launches, graph, "{what}: a {kind:?} mixed round {}", if graph == 1 { "replays its mix's graph" } else { "runs eagerly" });
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let logits = dflash2_readout::mixed_verify_logits(model, WINDOW, extended)
        .unwrap_or_else(|e| panic!("{what}: the mixed round's rows: {e}"));
    runs.iter()
        .enumerate()
        .map(|(i, run)| {
            let shape = shape_of(extended[i]);
            let columns = shape.columns();
            assert_eq!(run.extended, extended[i], "{what}: lane {i} ran at the {shape:?} shape");
            assert_eq!(run.extent, shape.window(), "{what}: lane {i} drafts the {shape:?} shape's window");
            let committed = run.tokens.len();
            assert!(committed >= 1 && committed <= columns, "{what}: lane {i} committed {committed} of {columns} columns");
            if kind == Kind::Forced {
                assert_eq!(committed, columns, "{what}: lane {i} commits its whole {shape:?} columns of its text's drafts");
            }
            assert_eq!(run.tokens[..], texts[i][at[i]..at[i] + committed], "{what}: lane {i} commits its text");
            assert_eq!(logits[i].len(), columns * vocab, "{what}: lane {i}'s rows");
            (0..committed).map(|j| logits[i][j * vocab..(j + 1) * vocab].to_vec()).collect()
        })
        .collect()
}

/// #717: the rows a mixed leg gives each lane of `group`, by text index, at
/// most `held[l]` to lane `l`: every round of `kind` at the mix `extended`
/// names, from the prompt on, until every lane holds its `held` rows or a
/// lane's text holds no whole round of its shape, so that every round is at
/// the mix and at the cell's lane count. A lane that holds its rows rides on,
/// its rows past them dropped; with the leg's rounds.
#[allow(clippy::too_many_arguments)]
fn mixed_leg(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    texts: &[Vec<i32>],
    group: &[usize],
    extended: &[bool],
    held: &[usize],
    kind: Kind,
    what: &str,
) -> (Vec<Vec<Vec<u16>>>, usize) {
    let greedy = SamplingParams::greedy();
    let mut seqs: Vec<Seq<'_>> = Vec::new();
    for &p in group {
        let mut seq = pool.alloc(MIXED_SEQ_CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
        prefill_program_permitted(model, pool, &mut seq, &prompts[p], 0, greedy, &[texts[p][0]], None)
            .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
        seqs.push(seq);
    }
    let mut rows: Vec<Vec<Vec<u16>>> = vec![Vec::new(); group.len()];
    // Each lane's text index, which runs past the rows it holds.
    let mut at: Vec<usize> = vec![0; group.len()];
    let lane_texts: Vec<&[i32]> = group.iter().map(|&p| texts[p].as_slice()).collect();
    let mut rounds = 0;
    while (0..group.len()).any(|l| rows[l].len() < held[l])
        && (0..group.len()).all(|l| at[l] + shape_of(extended[l]).columns() < lane_texts[l].len())
    {
        let round_rows = mixed_round(model, pool, &mut seqs, &lane_texts, &at, extended, kind, what);
        for (l, lane_rows) in round_rows.into_iter().enumerate() {
            at[l] += lane_rows.len();
            let room = held[l].saturating_sub(rows[l].len());
            rows[l].extend(lane_rows.into_iter().take(room));
        }
        rounds += 1;
    }
    assert!(rounds > 0, "{what}: no round at the mix");
    (rows, rounds)
}

/// #720: the rows a uniform leg at one width gives each lane of `group`, by
/// text index, at most `held[l]` to lane `l`: every round of `kind` at
/// `shape`, from the prompt on, every lane riding until each holds its rows or
/// a lane's text holds no whole round, so that no round narrows, as no mixed
/// leg's does. A lane held to no row rides only to fill out the width.
#[allow(clippy::too_many_arguments)]
fn fixed_leg(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    texts: &[Vec<i32>],
    group: &[usize],
    shape: Shape,
    kind: Kind,
    held: &[usize],
    what: &str,
) -> Vec<Vec<Vec<u16>>> {
    let greedy = SamplingParams::greedy();
    let mut seqs: Vec<Seq<'_>> = Vec::new();
    for &p in group {
        let mut seq = pool.alloc(MIXED_SEQ_CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
        prefill_program_permitted(model, pool, &mut seq, &prompts[p], 0, greedy, &[texts[p][0]], None)
            .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
        seqs.push(seq);
    }
    let mut rows: Vec<Vec<Vec<u16>>> = vec![Vec::new(); group.len()];
    let mut at: Vec<usize> = vec![0; group.len()];
    let lane_texts: Vec<&[i32]> = group.iter().map(|&p| texts[p].as_slice()).collect();
    while (0..group.len()).any(|l| rows[l].len() < held[l]) && (0..group.len()).all(|l| at[l] + shape.columns() < lane_texts[l].len()) {
        let round_rows = forced_round(model, pool, &mut seqs, &lane_texts, &at, shape, kind, what);
        for (l, lane_rows) in round_rows.into_iter().enumerate() {
            at[l] += lane_rows.len();
            let room = held[l].saturating_sub(rows[l].len());
            rows[l].extend(lane_rows.into_iter().take(room));
        }
    }
    rows
}

/// The `p`-th quantile of `values` (nearest rank), 0 where there is none.
fn quantile(values: &[f32], p: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

/// One pair of row sources over rows: every row's drift, in order.
#[derive(Default)]
struct Pair {
    drifts: Vec<RowDrift>,
}

impl Pair {
    fn ulps(&self) -> Vec<f32> {
        self.drifts.iter().map(|d| d.ulps).collect()
    }

    fn at(&self, p: f64) -> f32 {
        quantile(&self.ulps(), p)
    }

    /// The reference row's gap at every pick the other row moved.
    fn moved(&self) -> Vec<f32> {
        self.drifts.iter().filter(|d| d.pick_off != d.pick_on).map(|d| d.gap_ulps).collect()
    }

    fn widest_gap(&self) -> f32 {
        self.moved().into_iter().fold(0.0, f32::max)
    }

    fn extend(&mut self, other: &Pair) {
        self.drifts.extend_from_slice(&other.drifts);
    }

    fn line(&self) -> String {
        let moved = self.moved();
        // A few gaps print whole; the control moves every pick.
        let gaps = if moved.len() <= 16 { format!("at gaps {moved:?}") } else { format!("the widest gap {:.1}", self.widest_gap()) };
        format!(
            "{} rows, ulps median {:.1}, p90 {:.1}, p99 {:.1}, widest {:.1}; {} picks moved, {gaps}",
            self.drifts.len(),
            self.at(0.5),
            self.at(0.9),
            self.at(0.99),
            self.at(1.0),
            moved.len()
        )
    }
}

/// Every pair a kind compares, over the lanes it ran: the extended rows', or
/// #717 the mixed rows' in their place, named to the prints and the failures
/// as `tested`.
#[derive(Default)]
struct Class {
    extended_off: Pair,
    base_off: Pair,
    extended_base: Pair,
    control: Pair,
    /// Each pick the extended rows moved from the base rows', where: its lane
    /// and row, and its gap in the base row.
    moved_between: Vec<(String, f32)>,
}

impl Class {
    fn extend(&mut self, other: &Class) {
        self.extended_off.extend(&other.extended_off);
        self.base_off.extend(&other.base_off);
        self.extended_base.extend(&other.extended_base);
        self.control.extend(&other.control);
        self.moved_between.extend_from_slice(&other.moved_between);
    }

    fn print(&self, label: &str, tested: &str) {
        println!("extended_round_class_gpu {label}:");
        println!("  {tested} against base:     {}", self.extended_base.line());
        println!("  {tested} against spec-off: {}", self.extended_off.line());
        println!("  base against spec-off:     {}", self.base_off.line());
        println!("  control, one index off:    {}", self.control.line());
        for (name, p) in [("median", 0.5), ("90th percentile", 0.9), ("widest", 1.0)] {
            let (extended, base) = (self.extended_off.at(p), self.base_off.at(p));
            println!("  class, {name}: {tested} {extended:.1} ulps from spec-off, base {base:.1}, ratio {:.2}", extended / base);
        }
    }

    /// The class's distances and moved picks over these rows, held at each
    /// lane count and over a kind's rows together; each excess a line in
    /// `failures`.
    fn hold_distances(&self, label: &str, tested: &str, failures: &mut Vec<String>) {
        for (name, p, margin) in [("median", 0.5, CLASS_MEDIAN_MARGIN), ("90th percentile", 0.9, CLASS_P90_MARGIN), ("widest", 1.0, CLASS_WIDEST_MARGIN)] {
            let (extended, base) = (self.extended_off.at(p), self.base_off.at(p));
            if extended > margin * base {
                failures.push(format!(
                    "{label}, the class: the {name} {tested} row sits {extended:.1} ulps from spec-off's, the base one {base:.1}: past {margin} times it"
                ));
            }
        }
        let (moved_extended, moved_base) = (self.extended_off.moved().len(), self.base_off.moved().len());
        let moved_bound = (CLASS_MOVED_PICKS_MARGIN * moved_base as f32) as usize + CLASS_MOVED_PICKS_ALLOWANCE;
        println!("  class, picks moved from spec-off's: {tested} {moved_extended}, base {moved_base} (at most {moved_bound})");
        if moved_extended > moved_bound {
            failures.push(format!(
                "{label}, the class: the {tested} rows moved {moved_extended} picks from spec-off's, the base rows {moved_base}: past {CLASS_MOVED_PICKS_MARGIN} times them and {CLASS_MOVED_PICKS_ALLOWANCE}"
            ));
        }
    }

    /// The class's near tie and the control, over a kind's rows together. The
    /// near tie rests on the widest gap at which the base rows move a pick, one
    /// pick's of 11 to 25 a lane count moves, so it is read over every lane
    /// count of the kind, as the control's median is: `base_widest`, these
    /// rows' own base rows' for the extended rows, and #717 for the mixed rows
    /// the uniform cells' of the same kind (ADR 0106). Each excess a line in
    /// `failures`.
    fn hold_near_tie(&self, label: &str, tested: &str, base_widest: f32, failures: &mut Vec<String>) {
        let gap_bound = PICK_GAP_ULPS.max(CLASS_PICK_GAP_MARGIN * base_widest);
        println!(
            "  class, the near tie {gap_bound:.1} ulps, over base rows whose widest moved gap is {base_widest:.1}: the widest moved gap from spec-off's, {tested} {:.1}, base {:.1}; between the shapes {:.1}",
            self.extended_off.widest_gap(),
            self.base_off.widest_gap(),
            self.extended_base.widest_gap()
        );
        if self.extended_off.widest_gap() > gap_bound {
            failures.push(format!(
                "{label}, the class: one {tested} row moved a pick spec-off's row rates {:.1} ulps apart, past {gap_bound:.1} (the near tie, or {CLASS_PICK_GAP_MARGIN} times the widest gap its base rows move a pick at, {base_widest:.1})",
                self.extended_off.widest_gap()
            ));
        }
        for (place, gap) in &self.moved_between {
            if *gap > gap_bound {
                failures.push(format!(
                    "{place}: the {tested} row moved the base row's pick, {gap:.1} ulps apart in the base row, past the class's near tie {gap_bound:.1}"
                ));
            }
        }
        let (right, wrong) = (self.extended_off.at(0.5), self.control.at(0.5));
        println!(
            "  control: its median {wrong:.1} ulps, {:.1} times the right rows' {right:.1} and {:.1} times the typical bound",
            wrong / right,
            wrong / EXTENDED_VS_BASE_TYPICAL_ULPS
        );
        if wrong <= CONTROL_MARGIN * right || wrong <= CONTROL_MARGIN * EXTENDED_VS_BASE_TYPICAL_ULPS {
            failures.push(format!(
                "{label}: the control's median row, one index off, sits {wrong:.1} ulps from spec-off's, the right rows' median {right:.1}: within {CONTROL_MARGIN} times it or the typical bound {EXTENDED_VS_BASE_TYPICAL_ULPS}, so the bounds would not tell a right row from a wrong one"
            ));
        }
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn an_extended_rounds_rows_are_the_base_rounds_class() {
    hold_the_class(&NVFP4);
}

/// #695: the same on the trellis body, at the same lanes, chunk and context
/// (the 16 GB profile's context and chunk; 8 lanes, the 24 GB profile's, so
/// that the extended shape runs its 4; #720: and its mixed rounds, every mix at
/// those 8 lanes, bit for bit).
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn on_the_trellis_body_an_extended_rounds_rows_are_the_base_rounds_class() {
    hold_the_class(&TRELLIS);
}

fn hold_the_class(cell: &Cell) {
    let file = cell.artifact;
    let path = Path::new(file);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("{}: artifact absent: {file}", cell.name)) {
        return;
    }
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    // The figures the bounds rest on are the RTX 5090's, and the NVFP4 body's
    // FP8 exception weights need it: another generation has no such cell to
    // hold (#695: the trellis body's included, measured there alone).
    let (_, (major, minor)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("device identity: {e}"));
    if major != 12 && gpu_profile::skip_or_fail(&format!("compute capability {major}.{minor}: the extended round's cell is measured on 12.0 alone")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {file}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW)
        .unwrap_or_else(|e| panic!("speculation: {e}"))
        .with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, model_scope(Some(spec), None, cell.exceptions)).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut failures: Vec<String> = Vec::new();
    {
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            CHUNK,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            LANES,
        )
        .unwrap_or_else(|e| panic!("model load: {e}"));
        // #717: a mixed leg's sequences hold its longer texts.
        let context = MIXED_SEQ_CONTEXT;
        let pool = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::HqE8_2b,
                kv_page_group_count: context.div_ceil(64) * LANES,
                max_context_tokens: MAX_CONTEXT,
                slot_count: LANES,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
            Some(SpeculativeBackend::Dflash2),
        )
        .unwrap_or_else(|e| panic!("pool: {e}"));
        // Both shapes' verify graphs at every width they serve: the graph
        // rounds replay them, as served.
        capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        let shapes = stats.shapes();
        assert_eq!(shapes.len(), 2, "a load at {WINDOW} drafts holds two round shapes");
        for (shape, stats) in [(Shape::Base, &shapes[0]), (Shape::Extended, &shapes[1])] {
            assert_eq!(stats.window, shape.window(), "the {shape:?} shape's window");
            assert!(stats.widths_without_a_graph().is_empty(), "the {shape:?} shape's widths without a graph: {:?}", stats.widths_without_a_graph());
        }
        assert_eq!(shapes[1].lanes, EXTENDED_SHAPE_MAX_LANES, "the extended shape's lanes");
        // Speculation off: each prompt's own greedy text, and the row after
        // each of its tokens.
        let mut prompts: Vec<Vec<i32>> = Vec::new();
        let mut texts: Vec<Vec<i32>> = Vec::new();
        let mut rows_off: Vec<Vec<Vec<u16>>> = Vec::new();
        for (number, prompt) in PROMPTS.iter().enumerate() {
            let (ids, text, rows, _) = spec_off(&model, &pool, &frontend, &format!("prompt {number}"), prompt, SEQ_CONTEXT);
            prompts.push(ids);
            texts.push(text);
            rows_off.push(rows);
        }

        // #717: the widest gap at which each kind's base rows move a pick, the
        // mixed rows' near tie.
        let mut base_widest = [0f32; 2];
        for (k, kind) in [Kind::Forced, Kind::Graph].into_iter().enumerate() {
            let mut kind_class = Class::default();
            for width in 1..=EXTENDED_SHAPE_MAX_LANES as usize {
                let mut width_class = Class::default();
                for group in groups(width) {
                    let what = format!("{kind:?} rounds at {width} lanes, prompts {group:?}");
                    let base = leg(&model, &pool, &prompts, &texts, &group, Shape::Base, kind, &what);
                    let extended = leg(&model, &pool, &prompts, &texts, &group, Shape::Extended, kind, &what);
                    for (l, &p) in group.iter().enumerate() {
                        let lane = format!("{what}, lane {l} (prompt {p})");
                        let lane_class = hold_lane(&lane, "extended", &texts[p], &rows_off[p], &base[l], &extended[l], &mut failures);
                        width_class.extend(&lane_class);
                    }
                }
                let width_label = format!("{kind:?} rounds at {width} lanes");
                width_class.print(&width_label, "extended");
                width_class.hold_distances(&width_label, "extended", &mut failures);
                kind_class.extend(&width_class);
            }
            // The class: the extended rows' distance from speculation off's
            // against the base rows', over every lane of the kind.
            let label = format!("{kind:?} rounds, 1 to {EXTENDED_SHAPE_MAX_LANES} lanes");
            kind_class.print(&label, "extended");
            kind_class.hold_distances(&label, "extended", &mut failures);
            base_widest[k] = kind_class.base_off.widest_gap();
            kind_class.hold_near_tie(&label, "extended", base_widest[k], &mut failures);
        }

        // #717: the mixed rounds' class, on the same load.
        hold_the_mixed_class(&model, &pool, &frontend, base_widest, cell.mixed, &mut failures);
    }
    artifact.release_arena(&mut device).unwrap_or_else(|e| panic!("release: {e}"));
    assert!(failures.is_empty(), "{}: {} excesses, the first of them:\n{}", cell.name, failures.len(),failures.iter().take(24).cloned().collect::<Vec<_>>().join("\n"));
}

/// Speculation off on `prompt`, named `what`: its ids, its own greedy text
/// and the row after each of the text's tokens, and its sequence, of
/// `context`, its text's last token pending.
fn spec_off<'p>(
    model: &Model,
    pool: &'p SeqPool,
    frontend: &FrontendSet,
    what: &str,
    prompt: &str,
    context: u32,
) -> (Vec<i32>, Vec<i32>, Vec<Vec<u16>>, Seq<'p>) {
    let greedy = SamplingParams::greedy();
    let ids: Vec<i32> = frontend
        .tokenizer()
        .encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert!(ids.len() + MAX_TOKENS + 2 * (WINDOW as usize + 1) <= context as usize, "{what}: {} prompt tokens", ids.len());
    let mut off = pool.alloc(context).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
    prefill_program_sampled(model, pool, &mut off, &ids, 0, greedy, None).unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
    let mut text = vec![off.pending_token().expect("the prefill drew")];
    let mut rows: Vec<Vec<u16>> = Vec::new();
    while text.len() <= MAX_TOKENS && *text.last().expect("a token") != IM_END {
        let (emitted, row, next) = plain_round(model, pool, &mut off, what);
        assert_eq!(emitted, text[text.len() - 1], "{what}: the plain round emitted its pending token");
        assert_eq!(logit_drift::argmax(&row).0 as i32, next, "{what}: the plain round's row is the one its greedy draw read");
        rows.push(row);
        text.push(next);
    }
    println!(
        "extended_round_class_gpu {what}: {} prompt tokens, {} of text (ended: {})",
        ids.len(),
        text.len(),
        text.last() == Some(&IM_END)
    );
    (ids, text, rows, off)
}

/// #717: speculation off's texts continued past the rows a lane holds to
/// [`MIXED_TEXT_TOKENS`], `offs[i]` writing `texts[i]`: plain greedy rounds of
/// every lane whose text is short of them and has not ended, the lanes
/// together, no row read. A mixed leg's lane only rides on this text, so it
/// need not be one lane's.
fn continue_texts(model: &Model, pool: &SeqPool, offs: &mut [Seq<'_>], texts: &mut [Vec<i32>]) {
    loop {
        let riding: Vec<usize> =
            (0..offs.len()).filter(|&i| texts[i].len() <= MIXED_TEXT_TOKENS && texts[i].last() != Some(&IM_END)).collect();
        if riding.is_empty() {
            break;
        }
        let mut refs: Vec<&mut Seq<'_>> = offs.iter_mut().enumerate().filter(|(i, _)| riding.contains(i)).map(|(_, seq)| seq).collect();
        let emitted = decode_program_batch(model, pool, &mut refs).unwrap_or_else(|e| panic!("the mixed prompts' texts: plain round: {e}"));
        for (k, &i) in riding.iter().enumerate() {
            assert_eq!(emitted[k], *texts[i].last().expect("a token"), "mixed prompt {i}: the plain round emitted its pending token");
            texts[i].push(refs[k].pending_token().expect("a plain round draws"));
        }
    }
}

/// One lane's rows held: the `tested` rows (the extended ones, or #717 the
/// mixed ones) against the base rows after the same tokens at the same lane
/// count, and both against speculation off's, at every text index all three
/// reached, at least [`MIN_ROWS`]; each excess a line in `failures`. Its pairs,
/// for the class.
#[allow(clippy::too_many_arguments)]
fn hold_lane(
    lane: &str,
    tested: &str,
    text: &[i32],
    off: &[Vec<u16>],
    base: &[Vec<u16>],
    under_test: &[Vec<u16>],
    failures: &mut Vec<String>,
) -> Class {
    // Every text index all three reached.
    let rows = base.len().min(under_test.len()).min(off.len());
    if rows < MIN_ROWS {
        failures.push(format!("{lane}: {rows} rows to compare, under {MIN_ROWS}"));
    }
    let mut lane_class = Class::default();
    let (mut lp_off, mut lp_base, mut lp_tested) = (0f64, 0f64, 0f64);
    // #717: and over the lane's first MIN_ROWS rows, a reading beside the
    // bound: every lane of either cell reaches them, so a short mixed lane's
    // mean reads beside a uniform lane's over one window of its text.
    let (mut window_base, mut window_tested) = (0f64, 0f64);
    for i in 0..rows {
        lane_class.extended_off.drifts.push(row_drift(&off[i], &under_test[i]));
        lane_class.base_off.drifts.push(row_drift(&off[i], &base[i]));
        lane_class.extended_base.drifts.push(row_drift(&base[i], &under_test[i]));
        if i + 1 < off.len() {
            lane_class.control.drifts.push(row_drift(&off[i + 1], &under_test[i]));
        }
        let next = text[i + 1] as u32;
        lp_off += logit_drift::log_prob(&off[i], next);
        lp_base += logit_drift::log_prob(&base[i], next);
        lp_tested += logit_drift::log_prob(&under_test[i], next);
        if i + 1 == MIN_ROWS {
            (window_base, window_tested) = (lp_base, lp_tested);
        }
    }
    let mean = |sum: f64| sum / rows.max(1) as f64;
    println!("extended_round_class_gpu {lane}:");
    println!("  {tested} against base:     {}", lane_class.extended_base.line());
    println!("  {tested} against spec-off: {}", lane_class.extended_off.line());
    println!("  base against spec-off:     {}", lane_class.base_off.line());
    println!("  control, one index off:    {}", lane_class.control.line());
    let apart = mean(lp_tested) - mean(lp_base);
    println!(
        "  mean log-probability of the text, nats a token: spec-off {:.5}, base {:+.5}, {tested} {:+.5} from it; {tested} against base {apart:+.5}",
        mean(lp_off),
        mean(lp_base) - mean(lp_off),
        mean(lp_tested) - mean(lp_off)
    );
    if rows >= MIN_ROWS {
        println!(
            "  the same over its first {MIN_ROWS} rows: {tested} against base {:+.5} ({:+.3} nats over the {rows} rows)",
            (window_tested - window_base) / MIN_ROWS as f64,
            lp_tested - lp_base
        );
    }

    // The row itself: against base, every logit; its moved picks are held to
    // the class's near tie once the kind's rows are in.
    for (index, drift) in lane_class.extended_base.drifts.iter().enumerate() {
        if let Err(excess) = within(drift, EXTENDED_VS_BASE_TOLERANCE_ULPS, f32::INFINITY) {
            failures.push(format!("{lane}: {tested} against base, row {index}: {excess} (the logit that moved most: id {})", drift.worst));
        }
        if drift.pick_off != drift.pick_on {
            lane_class.moved_between.push((format!("{lane}, row {index}"), drift.gap_ulps));
        }
    }
    let median = lane_class.extended_base.at(0.5);
    if median > EXTENDED_VS_BASE_TYPICAL_ULPS {
        failures.push(format!(
            "{lane}: the median {tested} row sits {median:.1} ulps from the base row, past {EXTENDED_VS_BASE_TYPICAL_ULPS}"
        ));
    }
    if apart.abs() > EXTENDED_VS_BASE_LOG_PROB_NATS {
        failures.push(format!(
            "{lane}: the text's mean log-probability under the {tested} rows is {apart:+.5} nats a token from the base rows', past {EXTENDED_VS_BASE_LOG_PROB_NATS}"
        ));
    }
    lane_class
}

/// #717: the mixed rounds' class on the loaded cell. Speculation off writes
/// the text of each of [`MIXED_PROMPTS`], its rows the uniform cells' window
/// ([`MAX_TOKENS`]) and its text on to [`MIXED_TEXT_TOKENS`]
/// ([`continue_texts`]); the decode graphs are captured again under
/// [`MIXED_ROUNDS`] (the switch removed after, so that a later test of the
/// binary captures as it would). Then at each lane count of 2 to 7, a leg of
/// base rounds on the first prompts over the window, as a uniform cell's, and
/// every mix of that count on the same prompts, its last `w` lanes extended,
/// riding at its mix until each lane holds the rows its base leg gave it; each
/// lane's mixed rows held to its base rows, the class at each lane count and
/// over the kind's, in both kinds of round, the near tie read over the uniform
/// cells' base rows of the kind, `base_widest` (ADR 0106). #720: on a cell
/// whose mixed rows are held as [`MixedRows::Uniform`], after the capture,
/// [`hold_the_mixed_rows_bit_for_bit`] in place of the legs at each lane count.
fn hold_the_mixed_class(model: &Model, pool: &SeqPool, frontend: &FrontendSet, base_widest: [f32; 2], hold: MixedRows, failures: &mut Vec<String>) {
    let began = std::time::Instant::now();
    let mut prompts: Vec<Vec<i32>> = Vec::new();
    let mut texts: Vec<Vec<i32>> = Vec::new();
    let mut rows_off: Vec<Vec<Vec<u16>>> = Vec::new();
    let mut offs: Vec<Seq<'_>> = Vec::new();
    for (number, prompt) in MIXED_PROMPTS.iter().enumerate() {
        let what = format!("mixed prompt {number}");
        let (ids, text, rows, off) = spec_off(model, pool, frontend, &what, prompt, MIXED_SEQ_CONTEXT);
        assert!(ids.len() + MIXED_TEXT_TOKENS + 2 * (WINDOW as usize + 1) <= MIXED_SEQ_CONTEXT as usize, "{what}: {} prompt tokens", ids.len());
        prompts.push(ids);
        texts.push(text);
        rows_off.push(rows);
        offs.push(off);
    }
    continue_texts(model, pool, &mut offs, &mut texts);
    drop(offs);
    for (number, text) in texts.iter().enumerate() {
        println!("extended_round_class_gpu mixed prompt {number}: {} of text (ended: {})", text.len(), text.last() == Some(&IM_END));
        assert_eq!(text.len(), MIXED_TEXT_TOKENS + 1, "mixed prompt {number}: its text ended before the {MIXED_TEXT_TOKENS} tokens a mixed leg may ride on");
    }
    // The uniform cells' window of each text: the base legs' text, and the
    // rows a mixed lane holds.
    let window: Vec<Vec<i32>> = texts.iter().map(|text| text[..=MAX_TOKENS].to_vec()).collect();
    let texts_took = began.elapsed().as_secs_f64();

    // SAFETY (test-only): this binary's tests run one at a time on the card
    // (the profile's --test-threads=1, or one test by name), and the leaf
    // reads the environment on this thread.
    unsafe { std::env::set_var(MIXED_ROUNDS, "1") };
    let captured = capture_decode_graphs(model, pool);
    // SAFETY: as above.
    unsafe { std::env::remove_var(MIXED_ROUNDS) };
    captured.unwrap_or_else(|e| panic!("capture under {MIXED_ROUNDS}=1: {e}"));
    let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert!(stats.mixed_graphs_asked, "the capture asked for the mixed graphs");
    for (at, &(w, n)) in MIXES.iter().enumerate() {
        assert!(stats.mixed_verify_graph_nodes[at] > 0, "the {w}+{n} mixed round's verify graph, on {LANES} lanes and slots");
    }
    for shape in stats.shapes() {
        assert!(shape.widths_without_a_graph().is_empty(), "the {} drafts shape's widths without a graph: {:?}", shape.window, shape.widths_without_a_graph());
    }
    let most = MIXES.iter().map(|&(w, n)| (w + n) as usize).max().expect("twelve mixes");
    assert!(prompts.len() >= most, "{} prompts for a mix of {most} lanes", prompts.len());
    if hold == MixedRows::Uniform {
        drop(rows_off);
        hold_the_mixed_rows_bit_for_bit(model, pool, &prompts, &texts, failures);
        println!(
            "extended_round_class_gpu: the mixed rounds' rows took {:.0} s, {texts_took:.0} s of it speculation off's texts",
            began.elapsed().as_secs_f64()
        );
        return;
    }
    for (k, kind) in [Kind::Forced, Kind::Graph].into_iter().enumerate() {
        let mut kind_class = Class::default();
        for lanes in 2..=most {
            let group: Vec<usize> = (0..lanes).collect();
            let what = format!("{kind:?} rounds at {lanes} lanes, mixed prompts {group:?}");
            let base = leg(model, pool, &prompts, &window, &group, Shape::Base, kind, &what);
            let held: Vec<usize> = base.iter().map(Vec::len).collect();
            let mut width_class = Class::default();
            for &(w, n) in MIXES.iter().filter(|&&(w, n)| (w + n) as usize == lanes) {
                let extended: Vec<bool> = (0..lanes).map(|l| l >= n as usize).collect();
                let what = format!("{kind:?} rounds at the mix {w}+{n}, mixed prompts {group:?}");
                let (mixed, rounds) = mixed_leg(model, pool, &prompts, &texts, &group, &extended, &held, kind, &what);
                println!("extended_round_class_gpu {what}: {rounds} rounds at the mix");
                for (l, &p) in group.iter().enumerate() {
                    let lane = format!("{what}, lane {l} (prompt {p}, {:?})", shape_of(extended[l]));
                    // Every round at the mix: a lane short of its base leg's
                    // rows means a text ran out first.
                    if mixed[l].len() < held[l] {
                        failures.push(format!("{lane}: its leg's texts held {rounds} rounds at the mix, {} of the {} rows its base leg gave it", mixed[l].len(), held[l]));
                    }
                    let lane_class = hold_lane(&lane, "mixed", &texts[p], &rows_off[p], &base[l], &mixed[l], failures);
                    width_class.extend(&lane_class);
                }
            }
            let width_label = format!("{kind:?} mixed rounds at {lanes} lanes");
            width_class.print(&width_label, "mixed");
            width_class.hold_distances(&width_label, "mixed", failures);
            kind_class.extend(&width_class);
        }
        // The class: the mixed rows' distance from speculation off's against
        // the base rows', over every lane of every mix of the kind; its near
        // tie over the uniform cells' base rows of the kind, as the extended
        // rows' is.
        let label = format!("{kind:?} mixed rounds, 2 to {most} lanes");
        kind_class.print(&label, "mixed");
        kind_class.hold_distances(&label, "mixed", failures);
        kind_class.hold_near_tie(&label, "mixed", base_widest[k], failures);
    }
    println!(
        "extended_round_class_gpu: the mixed rounds' class took {:.0} s, {texts_took:.0} s of it speculation off's texts",
        began.elapsed().as_secs_f64()
    );
}

/// #720: the trellis cell's mixed rows, each bit for bit the row of a uniform
/// round of its lane's shape at its round's columns (ADR 0106). A mix's round
/// is `16w + 8n` columns, 24 to 64; on the RTX 5090 a lane's rows are the same
/// bits in every round of 24 to 56 columns and its own in one of 64, so one
/// uniform round of each shape stands for each: below the widest mixes'
/// columns the widest round of the shape under them (7 base lanes, 56 columns;
/// 3 extended, 48), at them the uniform round of as many (8 base lanes, 4
/// extended). In each kind and for each, the uniform legs of both shapes on
/// every prompt its mixes' lanes of that shape take ([`fixed_leg`]: a leg from
/// each such prompt not yet held, `width` prompts on, ending by the last prompt
/// where it can and past it on the first again), then each of its mixes, its
/// lanes and its extended group as [`hold_the_mixed_class`]'s, every lane
/// riding at the mix to [`MAX_TOKENS`] rows and each of them its prompt's row
/// in the uniform leg of its shape; each excess a line in `failures`.
fn hold_the_mixed_rows_bit_for_bit(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], texts: &[Vec<i32>], failures: &mut Vec<String>) {
    let columns = |w: u32, n: u32| w as usize * Shape::Extended.columns() + n as usize * Shape::Base.columns();
    let widest = MIXES.iter().map(|&(w, n)| columns(w, n)).max().expect("twelve mixes");
    // The lanes of the uniform round of `shape` that stands for a mix's round,
    // at the widest mixes' columns or below them.
    let standing = |shape: Shape, at_widest: bool| (if at_widest { widest } else { widest - 1 }) / shape.columns();
    for kind in [Kind::Forced, Kind::Graph] {
        let (mut lanes_held, mut rows_held, mut identical) = (0usize, 0usize, 0usize);
        for at_widest in [false, true] {
            let mixes: Vec<(u32, u32)> = MIXES.iter().copied().filter(|&(w, n)| (columns(w, n) == widest) == at_widest).collect();
            // Each shape's uniform rows of every prompt a lane of that shape
            // takes in these mixes, by its stats index and prompt.
            let mut uniform: [Vec<Vec<Vec<u16>>>; 2] = [vec![Vec::new(); prompts.len()], vec![Vec::new(); prompts.len()]];
            for shape in [Shape::Base, Shape::Extended] {
                let width = standing(shape, at_widest);
                let mut taken: Vec<usize> =
                    mixes.iter().flat_map(|&(w, n)| (0..(w + n) as usize).filter(move |&l| shape_of(l >= n as usize) == shape)).collect();
                taken.sort_unstable();
                taken.dedup();
                let mut legs: Vec<Vec<usize>> = Vec::new();
                for &p in &taken {
                    if !legs.iter().any(|leg| leg.contains(&p)) {
                        let start = p.min(prompts.len().saturating_sub(width));
                        legs.push((start..start + width).map(|q| q % prompts.len()).collect());
                    }
                }
                for group in legs {
                    // Each prompt's rows once, where a mix takes them.
                    let held: Vec<usize> = group
                        .iter()
                        .enumerate()
                        .map(|(l, &p)| if taken.contains(&p) && !group[..l].contains(&p) && uniform[shape.stats_index()][p].is_empty() { MAX_TOKENS } else { 0 })
                        .collect();
                    let what = format!("{kind:?} uniform rounds at the {shape:?} shape, {width} lanes ({} columns), mixed prompts {group:?}", width * shape.columns());
                    let rows = fixed_leg(model, pool, prompts, texts, &group, shape, kind, &held, &what);
                    for (l, lane_rows) in rows.into_iter().enumerate() {
                        if held[l] > 0 {
                            uniform[shape.stats_index()][group[l]] = lane_rows;
                        }
                    }
                }
            }
            for &(w, n) in &mixes {
                let lanes = (w + n) as usize;
                let group: Vec<usize> = (0..lanes).collect();
                let extended: Vec<bool> = (0..lanes).map(|l| l >= n as usize).collect();
                let what = format!("{kind:?} rounds at the mix {w}+{n}, {} columns, mixed prompts {group:?}", columns(w, n));
                let (mixed, rounds) = mixed_leg(model, pool, prompts, texts, &group, &extended, &vec![MAX_TOKENS; lanes], kind, &what);
                let mut mix_identical = 0;
                for (l, &p) in group.iter().enumerate() {
                    let shape = shape_of(extended[l]);
                    let width = standing(shape, at_widest);
                    let lane = format!("{what}, lane {l} (prompt {p}, {shape:?})");
                    let reference = &uniform[shape.stats_index()][p];
                    // Every lane at a uniform cell's rows, against as many:
                    // a short one means a text ran out first.
                    if mixed[l].len() != MAX_TOKENS || reference.len() != MAX_TOKENS {
                        failures.push(format!(
                            "{lane}: {} rows at the mix and {} in the uniform leg at {width} lanes, where each holds {MAX_TOKENS}",
                            mixed[l].len(),
                            reference.len()
                        ));
                        continue;
                    }
                    let same = mixed[l].iter().zip(reference).filter(|(row, uniform_row)| row == uniform_row).count();
                    if let Some(i) = (0..MAX_TOKENS).find(|&i| mixed[l][i] != reference[i]) {
                        failures.push(format!(
                            "{lane}: {same} of its {MAX_TOKENS} rows are the uniform round's at the {shape:?} shape and {} columns ({width} lanes) bit for bit; the first that is not, row {i}, sits {:.1} ulps from it",
                            width * shape.columns(),
                            row_drift(&reference[i], &mixed[l][i]).ulps
                        ));
                    }
                    mix_identical += same;
                    rows_held += MAX_TOKENS;
                    lanes_held += 1;
                }
                identical += mix_identical;
                println!(
                    "extended_round_class_gpu {what}: {rounds} rounds at the mix; {mix_identical} of its {} rows bit for bit the uniform rounds' of their lanes' shapes, the base at {} columns and the extended at {}",
                    lanes * MAX_TOKENS,
                    standing(Shape::Base, at_widest) * Shape::Base.columns(),
                    standing(Shape::Extended, at_widest) * Shape::Extended.columns()
                );
            }
        }
        println!(
            "extended_round_class_gpu {kind:?} mixed rounds, 2 to {} lanes: {identical} of {rows_held} rows, over {lanes_held} lanes, bit for bit the uniform rounds' of their lanes' shapes at their rounds' columns",
            MIXES.iter().map(|&(w, n)| w + n).max().expect("twelve mixes")
        );
    }
}
