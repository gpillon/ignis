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
//! hq-e8-2b, FP8 exception weights, the served lanes and context), per prompt:
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
//!   whose text runs out leave the round. Every round asserts the shape it ran
//!   at (its drafts, the shape's round count) and whether it ran eagerly.
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
//!   against how far the base rows sit from them, over every lane of a kind
//!   together: the median, the 90th percentile and the widest row no further
//!   than the base rows' by [`CLASS_MEDIAN_MARGIN`], [`CLASS_P90_MARGIN`] and
//!   [`CLASS_WIDEST_MARGIN`]; no more picks moved from spec-off's than
//!   [`CLASS_MOVED_PICKS_MARGIN`] times those the base rows moved plus
//!   [`CLASS_MOVED_PICKS_ALLOWANCE`], and none at a gap past the class's
//!   near tie: ADR 0056's [`PICK_GAP_ULPS`], or [`CLASS_PICK_GAP_MARGIN`]
//!   times the widest gap at which the base rows move a pick from spec-off's,
//!   whichever is more;
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
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`, or
//! alone:
//!
//! ```text
//! cargo test -p ignis-core --features cuda --test extended_round_class_gpu -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout;
use ignis_core::gpu_profile;
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
// rounds over 3,205, at 1 to 4 lanes.

/// The extended row against the base row after the same tokens: no logit
/// further than this, in BF16 ulps at the base row's largest magnitude.
/// Measured widest: prototype 91.1, 113.8, 127.0, 138.9, 168.0, 96.9 by cell;
/// this test, forced rounds 96.9 at one lane, 0.0 at two and 1.0 at three
/// and four, graph rounds 114.1 at one lane and 115.2 at two to four; 256 is
/// the next power of two above, ADR 0056's rule for its own tolerance.
const EXTENDED_VS_BASE_TOLERANCE_ULPS: f32 = 256.0;

/// What a lane's median row of that pair stays within. Measured medians:
/// prototype 14.0, 14.2, 19.5, 22.3, 20.1, 14.7 (the furthest of the three
/// prompts); this test, forced rounds 14.0 to 16.1 at one lane and 0.0 to 0.5
/// at two to four (both shapes then take the routes of more than 8 columns,
/// and commit every column), graph rounds 0.0 to 16.0; 32 is the next power
/// of two above.
const EXTENDED_VS_BASE_TYPICAL_ULPS: f32 = 32.0;

/// ADR 0056's near tie, in the reference row's ulps: the floor of the class's
/// near tie ([`CLASS_PICK_GAP_MARGIN`]), inside which a pick that differs
/// between the shapes, or between the extended round and speculation off,
/// must sit.
const PICK_GAP_ULPS: f32 = logit_drift::PICK_GAP_ULPS;

/// The class, median: the extended rows' median distance from speculation
/// off's rows over the base rows'. Measured ratios: prototype 0.96, 1.01,
/// 1.01, 1.00, 1.01, 1.01; this test, forced rounds 1.00 (0.99 to 1.00 by lane
/// count), graph rounds 1.02 (0.99 to 1.03).
const CLASS_MEDIAN_MARGIN: f32 = 1.25;

/// The class, 90th percentile. Measured ratios: prototype 0.86, 1.03, 1.02,
/// 1.00, 1.02, 0.87; this test, forced rounds 1.00 (0.98 to 1.01), graph
/// rounds 0.99 (0.94 to 1.02).
const CLASS_P90_MARGIN: f32 = 1.25;

/// The class, widest row: one tail logit each, so the loosest of the three.
/// Measured ratios: prototype 1.02, 1.20, 0.97, 1.07, 1.25, 1.02; this test,
/// forced rounds 1.00 (1.00 to 1.02), graph rounds 1.03 (1.03 to 1.06).
const CLASS_WIDEST_MARGIN: f32 = 1.5;

/// The class, moved picks: the extended rows may move this many times the
/// picks the base rows move from speculation off's, and
/// [`CLASS_MOVED_PICKS_ALLOWANCE`] more (a count of a few picks has no ratio
/// worth a margin). Measured, extended against base: prototype 0 and 2, 3 and
/// 5, 0 and 0, 155 and 176, 58 and 63, 15 and 14; this test, forced rounds 90
/// and 89, graph rounds 76 and 73.
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
/// forced rounds 12.0 over 12.0, graph rounds 22.0 over 12.0 (1.83). The
/// widest gap between the shapes: prototype 0 to 8 on texts of this test's
/// length, 82.5 on the long answers; this test, forced rounds 8.0, graph
/// rounds 17.0, where ADR 0056's 16 alone failed the test's first run.
const CLASS_PICK_GAP_MARGIN: f32 = 2.0;

/// A lane's mean log-probability of the text under the extended rows against
/// the base rows', in nats a token, either way. Measured apart by: prototype
/// 0.0074, 0.0031, 0.0001, 0.0040, 0.0001, and 0.0089 at the furthest of the
/// three prompts; this test, forced rounds at most 0.0089, graph rounds at
/// most 0.0124; 0.02 is ADR 0056's bound for its own pair.
const EXTENDED_VS_BASE_LOG_PROB_NATS: f64 = logit_drift::MEAN_LOG_PROB_NATS;

/// The control's median, one index off, over the right rows' median and over
/// [`EXTENDED_VS_BASE_TYPICAL_ULPS`]. Measured: the control's median row sits
/// 166.5 ulps from speculation off's over the prototype's three prompts
/// (145.5 to 182.2 by prompt), the right rows' 11.6, 14 times apart; this
/// test, forced rounds 166.0 (14.0 times the right rows', 5.2 times the
/// typical bound), graph rounds 165.0 (13.8, 5.2); 2 is far inside both and
/// far outside a right pair's margins above.
const CONTROL_MARGIN: f32 = 2.0;

const NVFP4: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer";
/// The load: the widest DFlash2 window, at the served lanes, prefill chunk and
/// context. The context is the envelope the rounds' attention splits by, so it
/// is the served cell's, not the session's need; the pool below is small.
const WINDOW: u32 = 15;
const LANES: u32 = 8;
const CHUNK: u32 = 1024;
const MAX_CONTEXT: u32 = 262_144;
/// Each sequence's context: its prompt, its text and a round past it.
const SEQ_CONTEXT: u32 = 512;
/// The text speculation off writes per prompt, at most; and the fewest rows a
/// lane must compare for its figures to mean anything.
const MAX_TOKENS: usize = 224;
const MIN_ROWS: usize = 64;
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

/// Every pair a kind compares, over the lanes it ran.
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

    fn print(&self, label: &str) {
        println!("extended_round_class_gpu {label}:");
        println!("  extended against base:     {}", self.extended_base.line());
        println!("  extended against spec-off: {}", self.extended_off.line());
        println!("  base against spec-off:     {}", self.base_off.line());
        println!("  control, one index off:    {}", self.control.line());
        for (name, p) in [("median", 0.5), ("90th percentile", 0.9), ("widest", 1.0)] {
            let (extended, base) = (self.extended_off.at(p), self.base_off.at(p));
            println!("  class, {name}: extended {extended:.1} ulps from spec-off, base {base:.1}, ratio {:.2}", extended / base);
        }
    }

    /// The class bounds over these rows; each excess a line in `failures`.
    fn hold(&self, label: &str, failures: &mut Vec<String>) {
        for (name, p, margin) in [("median", 0.5, CLASS_MEDIAN_MARGIN), ("90th percentile", 0.9, CLASS_P90_MARGIN), ("widest", 1.0, CLASS_WIDEST_MARGIN)] {
            let (extended, base) = (self.extended_off.at(p), self.base_off.at(p));
            if extended > margin * base {
                failures.push(format!(
                    "{label}, the class: the {name} extended row sits {extended:.1} ulps from spec-off's, the base one {base:.1}: past {margin} times it"
                ));
            }
        }
        let (moved_extended, moved_base) = (self.extended_off.moved().len(), self.base_off.moved().len());
        let moved_bound = (CLASS_MOVED_PICKS_MARGIN * moved_base as f32) as usize + CLASS_MOVED_PICKS_ALLOWANCE;
        println!("  class, picks moved from spec-off's: extended {moved_extended}, base {moved_base} (at most {moved_bound})");
        if moved_extended > moved_bound {
            failures.push(format!(
                "{label}, the class: the extended rows moved {moved_extended} picks from spec-off's, the base rows {moved_base}: past {CLASS_MOVED_PICKS_MARGIN} times them and {CLASS_MOVED_PICKS_ALLOWANCE}"
            ));
        }
        let gap_bound = PICK_GAP_ULPS.max(CLASS_PICK_GAP_MARGIN * self.base_off.widest_gap());
        println!(
            "  class, the near tie {gap_bound:.1} ulps: the widest moved gap from spec-off's, extended {:.1}, base {:.1}; between the shapes {:.1}",
            self.extended_off.widest_gap(),
            self.base_off.widest_gap(),
            self.extended_base.widest_gap()
        );
        if self.extended_off.widest_gap() > gap_bound {
            failures.push(format!(
                "{label}, the class: an extended row moved a pick spec-off's row rates {:.1} ulps apart, past {gap_bound:.1} (the near tie, or {CLASS_PICK_GAP_MARGIN} times the base rows' widest, {:.1})",
                self.extended_off.widest_gap(),
                self.base_off.widest_gap()
            ));
        }
        for (place, gap) in &self.moved_between {
            if *gap > gap_bound {
                failures.push(format!(
                    "{place}: the extended row moved the base row's pick, {gap:.1} ulps apart in the base row, past the class's near tie {gap_bound:.1}"
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
    let path = Path::new(NVFP4);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {NVFP4}")) {
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
    // The extended round, its FP8 exception weights and its figures are the
    // RTX 5090's: another generation has no such cell to hold.
    let (_, (major, minor)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("device identity: {e}"));
    if major != 12 && gpu_profile::skip_or_fail(&format!("compute capability {major}.{minor}: the extended round's cell is measured on 12.0 alone")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {NVFP4}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW)
        .unwrap_or_else(|e| panic!("speculation: {e}"))
        .with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, model_scope(Some(spec), None, ExceptionWeights::Fp8)).unwrap_or_else(|e| panic!("bind: {e}"));
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
        let pool = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::HqE8_2b,
                kv_page_group_count: SEQ_CONTEXT.div_ceil(64) * LANES,
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
        let greedy = SamplingParams::greedy();

        // Speculation off: each prompt's own greedy text, and the row after
        // each of its tokens.
        let mut prompts: Vec<Vec<i32>> = Vec::new();
        let mut texts: Vec<Vec<i32>> = Vec::new();
        let mut rows_off: Vec<Vec<Vec<u16>>> = Vec::new();
        for (number, prompt) in PROMPTS.iter().enumerate() {
            let what = format!("prompt {number}");
            let ids: Vec<i32> = frontend
                .tokenizer()
                .encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                .unwrap_or_else(|e| panic!("tokenize: {e}"))
                .into_iter()
                .map(|id| i32::try_from(id).expect("token id fits i32"))
                .collect();
            assert!(ids.len() + MAX_TOKENS + 2 * (WINDOW as usize + 1) <= SEQ_CONTEXT as usize, "{what}: {} prompt tokens", ids.len());
            let mut off = pool.alloc(SEQ_CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
            prefill_program_sampled(&model, &pool, &mut off, &ids, 0, greedy, None).unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
            let mut text = vec![off.pending_token().expect("the prefill drew")];
            let mut rows: Vec<Vec<u16>> = Vec::new();
            while text.len() <= MAX_TOKENS && *text.last().expect("a token") != IM_END {
                let (emitted, row, next) = plain_round(&model, &pool, &mut off, &what);
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
            prompts.push(ids);
            texts.push(text);
            rows_off.push(rows);
        }

        for kind in [Kind::Forced, Kind::Graph] {
            let mut kind_class = Class::default();
            for width in 1..=EXTENDED_SHAPE_MAX_LANES as usize {
                let mut width_class = Class::default();
                for group in groups(width) {
                    let what = format!("{kind:?} rounds at {width} lanes, prompts {group:?}");
                    let base = leg(&model, &pool, &prompts, &texts, &group, Shape::Base, kind, &what);
                    let extended = leg(&model, &pool, &prompts, &texts, &group, Shape::Extended, kind, &what);
                    for (l, &p) in group.iter().enumerate() {
                        let lane = format!("{what}, lane {l} (prompt {p})");
                        let text = &texts[p];
                        let off = &rows_off[p];
                        // Every text index all three reached.
                        let rows = base[l].len().min(extended[l].len()).min(off.len());
                        if rows < MIN_ROWS {
                            failures.push(format!("{lane}: {rows} rows to compare, under {MIN_ROWS}"));
                        }
                        let mut lane_class = Class::default();
                        let (mut lp_off, mut lp_base, mut lp_extended) = (0f64, 0f64, 0f64);
                        for i in 0..rows {
                            lane_class.extended_off.drifts.push(row_drift(&off[i], &extended[l][i]));
                            lane_class.base_off.drifts.push(row_drift(&off[i], &base[l][i]));
                            lane_class.extended_base.drifts.push(row_drift(&base[l][i], &extended[l][i]));
                            if i + 1 < off.len() {
                                lane_class.control.drifts.push(row_drift(&off[i + 1], &extended[l][i]));
                            }
                            let next = text[i + 1] as u32;
                            lp_off += logit_drift::log_prob(&off[i], next);
                            lp_base += logit_drift::log_prob(&base[l][i], next);
                            lp_extended += logit_drift::log_prob(&extended[l][i], next);
                        }
                        let mean = |sum: f64| sum / rows.max(1) as f64;
                        println!("extended_round_class_gpu {lane}:");
                        println!("  extended against base:     {}", lane_class.extended_base.line());
                        println!("  extended against spec-off: {}", lane_class.extended_off.line());
                        println!("  base against spec-off:     {}", lane_class.base_off.line());
                        println!("  control, one index off:    {}", lane_class.control.line());
                        let apart = mean(lp_extended) - mean(lp_base);
                        println!(
                            "  mean log-probability of the text, nats a token: spec-off {:.5}, base {:+.5}, extended {:+.5} from it; extended against base {apart:+.5}",
                            mean(lp_off),
                            mean(lp_base) - mean(lp_off),
                            mean(lp_extended) - mean(lp_off)
                        );

                        // The row itself: extended against base, every logit;
                        // its moved picks are held to the class's near tie
                        // once the kind's rows are in.
                        for (index, drift) in lane_class.extended_base.drifts.iter().enumerate() {
                            if let Err(excess) = within(drift, EXTENDED_VS_BASE_TOLERANCE_ULPS, f32::INFINITY) {
                                failures.push(format!("{lane}: extended against base, row {index}: {excess} (the logit that moved most: id {})", drift.worst));
                            }
                            if drift.pick_off != drift.pick_on {
                                lane_class.moved_between.push((format!("{lane}, row {index}"), drift.gap_ulps));
                            }
                        }
                        let median = lane_class.extended_base.at(0.5);
                        if median > EXTENDED_VS_BASE_TYPICAL_ULPS {
                            failures.push(format!(
                                "{lane}: the median extended row sits {median:.1} ulps from the base row, past {EXTENDED_VS_BASE_TYPICAL_ULPS}"
                            ));
                        }
                        if apart.abs() > EXTENDED_VS_BASE_LOG_PROB_NATS {
                            failures.push(format!(
                                "{lane}: the text's mean log-probability under the extended rows is {apart:+.5} nats a token from the base rows', past {EXTENDED_VS_BASE_LOG_PROB_NATS}"
                            ));
                        }
                        width_class.extend(&lane_class);
                    }
                }
                width_class.print(&format!("{kind:?} rounds at {width} lanes"));
                kind_class.extend(&width_class);
            }
            // The class: the extended rows' distance from speculation off's
            // against the base rows', over every lane of the kind.
            let label = format!("{kind:?} rounds, 1 to {EXTENDED_SHAPE_MAX_LANES} lanes");
            kind_class.print(&label);
            kind_class.hold(&label, &mut failures);
        }
    }
    artifact.release_arena(&mut device).unwrap_or_else(|e| panic!("release: {e}"));
    assert!(failures.is_empty(), "{} excesses, the first of them:\n{}", failures.len(), failures.iter().take(24).cloned().collect::<Vec<_>>().join("\n"));
}
