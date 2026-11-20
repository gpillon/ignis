//! The verify commit whose cut the host decides, on the 27B with DFlash2
//! (GitHub #590, spec #287, ADR 0089): one lane of a speculative round is
//! held to a grammar, the others are not.
//!
//! Three lanes run the same verify rounds twice, from the same prompts: once
//! all free, once with lane 2 held to a strict `json_schema` over the
//! artifact's own vocabulary, whose keys are names the model does not guess
//! and which its prompt does not describe. In the held run the host sees lane
//! 2's run after each pass, keeps it up to its first draft the grammar
//! refuses, and repairs that draft -- now the lane's pending token -- from the
//! verify pass's own scores (`ignis_program_repair`).
//!
//! The schema is what makes the cut happen: under `json_object` the model's
//! own object passes draft after draft (a run of #590's first GPU leg was
//! never cut inside in sixteen rounds), while a key the model would not have
//! written is refused where the target and the drafter agree on the key it
//! would have -- a draft the target accepted, inside the run.
//!
//! - **The neighbours keep their drafts.** Lanes 0 and 1 verify the same
//!   extents and commit the same tokens, round for round, as beside a free
//!   lane 2: a held lane costs them nothing.
//! - **The round runs in its graph.** Every round of both runs replays the
//!   width's verify graph.
//! - **The repair reads the column the cut ended at.** Each repair of a
//!   draft the host cut at is the argmax, over the grammar's permitted set,
//!   of the verify logits' column that drew it (`dflash2_readout::
//!   verify_logits`), and the next round consumes it as its anchor.
//! - **A cut happened.** At least one of lane 2's runs was cut inside, not
//!   only at its end, so the cut and the repair after it were exercised.
//!
//! # At the extended shape (#691, spec #684)
//!
//! The second test runs the same two runs on a load of two round shapes,
//! DFlash2 at 15 drafts (the RTX 5090 profile's load: hq-e8-2b, FP8
//! exception weights), every round under `IGNIS_ROUND_SHAPE=extended`, so the
//! three lanes' rounds are 16 columns a lane, the held lane's among them: a
//! round with a lane the host cuts takes its shape as any other does. Every
//! round asserts its shape, the cut reads the held lane's run at that
//! stride, and each repair is the masked argmax of the column that drew it,
//! read at that stride.
//!
//! Then a repair reads only its own round's scores. The two shapes write one
//! logits buffer, so a record a round of one shape kept is current only while
//! no later round of either shape has written it: lane 2 rides a cut round at
//! one shape and is repaired from it, lanes 0 and 1 ride a round of the
//! other, and a second repair of lane 2 is refused, not read; both ways.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile it is a failure.
//! The two tests load the 27B each: one at a time, as the profile runs them.
//!
//! ```text
//! cargo test -p ignis-core --features cuda --test verify_cut_gpu -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use ignis_artifact::{
    bind_model_scope_27b, bind_model_scope_27b_with, materialize, CudaDevice, DraftModule, FrontendSet, Reader, TokenBytes,
};
use ignis_core::compute::ModelConfig;
use ignis_core::constrained::PermittedMask;
use ignis_core::dflash2_readout::verify_logits;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, load_qwen38_27b_with_speculation, model_scope, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_cut, decode_program_verify_runs, prefill_program_draw,
    program_stats, repair_program, LaneDraw, LaneVerifyRun, SamplingParams, VerifyLane,
};
use ignis_core::{ExceptionWeights, KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation, SpeculativeBackend};
use ignis_grammar::{Grammar, Matcher, TokenSet, Vocab, VocabEntry};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 512;
/// DFlash2-7, every built-in profile's window.
const WINDOW: u32 = 7;
/// #691: DFlash2 at 15 drafts, a load of two round shapes: the base shape's 7
/// drafts a lane at every width and the extended shape's 15 at up to 4
/// lanes, which the three lanes are under.
const EXTENDED_WINDOW: u32 = 15;
const BASE_WINDOW: u32 = 7;
/// The measurement switch that names a two-shape load's round shape.
const ROUND_SHAPE: &str = "IGNIS_ROUND_SHAPE";
/// Rounds compared: enough runs of up to eight tokens for the held lane to
/// miss inside one.
const ROUNDS: usize = 16;
const LANES: usize = 3;
/// The held lane.
const HELD: usize = 2;

/// Two free prompts the drafter does well on, and one the model answers in
/// prose, which the schema turns into an object of its own: misses on keys
/// and values.
const PROMPTS: [&str; LANES] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "Name three rivers of Europe and say how long each one is.",
];

/// The held lane's schema, strict: an object the prompt does not describe,
/// with key names the model has no reason to write.
const SCHEMA: &str = r#"{"type":"object","properties":{"waterways":{"type":"array","items":{"type":"object","properties":{"designation":{"type":"string"},"span_km":{"type":"integer"},"basin_states":{"type":"array","items":{"type":"string"}}},"required":["designation","span_km","basin_states"],"additionalProperties":false}}},"required":["waterways"],"additionalProperties":false}"#;

fn vocab() -> usize {
    ModelConfig::qwen38_27b().vocab as usize
}

/// The prompts in the model's chat format with thinking closed, the stop
/// ids, and the schema's grammar over the artifact's vocabulary (the end of
/// turn its stop, as the server builds it).
fn inputs(reader: &Reader) -> (Vec<Vec<i32>>, Vec<i32>, Arc<Grammar>) {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let tokenizer = frontend.tokenizer();
    let encode = |text: &str| -> Vec<i32> {
        tokenizer.encode(text).unwrap_or_else(|e| panic!("tokenize: {e}")).into_iter().map(|id| id as i32).collect()
    };
    let stops: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect();
    assert_eq!(stops.len(), 2, "the stop markers must tokenize as special tokens");
    let prompts = PROMPTS
        .iter()
        .map(|prompt| encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")))
        .collect();
    let entries = tokenizer
        .token_bytes_table(vocab())
        .unwrap_or_else(|e| panic!("byte table: {e}"))
        .into_iter()
        .map(|entry| match entry {
            TokenBytes::Text(bytes) => VocabEntry::Text(bytes),
            TokenBytes::Added { .. } => VocabEntry::Control,
            TokenBytes::Absent => VocabEntry::Absent,
        })
        .collect();
    let grammar = Grammar::json_schema(Vocab::new(entries, &[stops[0] as u32]), SCHEMA, true)
        .unwrap_or_else(|e| panic!("the held lane's schema: {e:?}"));
    (prompts, stops, grammar)
}

fn pool(kv_format: KvFormat) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * LANES as u32,
            max_context_tokens: MAX_CONTEXT,
            slot_count: LANES as u32,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        Some(SpeculativeBackend::Dflash2),
    )
    .unwrap_or_else(|e| panic!("seq pool: {e}"))
}

/// The argmax of `logits` over `set`, the lower id on a tie: the vendored
/// sampler's greedy rule.
fn masked_argmax(logits: &[f32], set: &TokenSet) -> i32 {
    let mut best: Option<u32> = None;
    for id in set.iter() {
        if best.is_none_or(|b| logits[id as usize] > logits[b as usize]) {
            best = Some(id);
        }
    }
    best.expect("a grammar never leaves an empty permitted set") as i32
}

/// The latest verify round's logits row of `lane`'s column `column`, widened
/// to f32, read at the stride of the round's shape: `window` drafts a lane.
fn verify_column(model: &Model, window: u32, lane: usize, column: usize) -> Vec<f32> {
    let logits = verify_logits(model, LANES as u32, window).unwrap_or_else(|e| panic!("verify logits: {e}"));
    let row = (lane * (window as usize + 1) + column) * vocab();
    logits[row..row + vocab()].iter().map(|&bits| f32::from_bits(u32::from(bits) << 16)).collect()
}

/// #691: a repair's mask permitting every token of the vocabulary but
/// `excluded`, and the argmax of `logits` under it, the lower id on a tie.
fn all_but(excluded: i32) -> PermittedMask {
    let vocab = vocab();
    let mut words = vec![u64::MAX; vocab.div_ceil(64)];
    if vocab % 64 != 0 {
        words[vocab / 64] = (1u64 << (vocab % 64)) - 1;
    }
    words[excluded as usize / 64] &= !(1u64 << (excluded as usize % 64));
    PermittedMask::from_words(vocab as u32, words).unwrap_or_else(|e| panic!("mask: {e}"))
}

fn argmax_but(logits: &[f32], excluded: i32) -> i32 {
    let mut best: Option<usize> = None;
    for (id, &logit) in logits.iter().enumerate() {
        if id as i32 != excluded && best.is_none_or(|b| logit > logits[b]) {
            best = Some(id);
        }
    }
    best.expect("a vocabulary of more than one token") as i32
}

/// #691: [`ROUND_SHAPE`] set to `named` for the rounds after, or removed.
fn set_round_shape(named: Option<&str>) {
    // SAFETY (test-only): this binary's tests run one at a time
    // (`--test-threads=1`), each reading and writing the environment on its
    // own thread, where the leaf reads it too.
    unsafe {
        match named {
            Some(value) => std::env::set_var(ROUND_SHAPE, value),
            None => std::env::remove_var(ROUND_SHAPE),
        }
    }
}

/// The held lane's grammar and what it saw.
struct Held {
    matcher: Matcher,
    set: TokenSet,
    stops: Vec<i32>,
    /// The anchor the next round must consume: the last draw held.
    anchor: Option<i32>,
    /// Runs the host cut inside, and repairs made after a verify round.
    cuts: usize,
    repairs: usize,
}

impl Held {
    /// The host's cut: the run up to its first draft the grammar refuses or
    /// a stop, which then waits as the lane's pending token.
    fn cut(&mut self, run: &[i32]) -> usize {
        if let Some(anchor) = self.anchor {
            assert_eq!(run[0], anchor, "the round consumed the held draw as its anchor");
        }
        let kept = 1 + run[1..]
            .iter()
            .take_while(|&&token| !self.stops.contains(&token) && self.matcher.accept(token as u32))
            .count();
        if kept < run.len() {
            self.cuts += 1;
        }
        kept
    }

    /// Hold `seq`'s pending token: taken when the grammar passes it, else
    /// repaired from the scores its call kept. `column` is the verify logits'
    /// row that drew it, when a verify round did. Returns whether the answer
    /// is complete.
    fn hold(&mut self, model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, column: Option<&[f32]>) -> bool {
        let pending = seq.pending_token().expect("a pending token");
        if self.stops.contains(&pending) && self.matcher.is_accepting() {
            return true;
        }
        if !self.stops.contains(&pending) && self.matcher.accept(pending as u32) {
            self.anchor = Some(pending);
            return false;
        }
        self.matcher.mask(&mut self.set);
        let mask = PermittedMask::from_words(vocab() as u32, self.set.words().to_vec())
            .unwrap_or_else(|e| panic!("mask: {e}"));
        let repaired = repair_program(model, pool, seq, SamplingParams::greedy(), &mask)
            .unwrap_or_else(|e| panic!("repair of {pending}: {e}"));
        if let Some(column) = column {
            assert_eq!(
                repaired,
                masked_argmax(column, &self.set),
                "the repair of {pending} is the masked argmax of the verify column that drew it"
            );
            self.repairs += 1;
        }
        assert!(self.set.contains(repaired as u32), "the repair stayed in its mask");
        let done = self.stops.contains(&repaired);
        if !done {
            assert!(self.matcher.accept(repaired as u32));
        }
        self.anchor = Some(repaired);
        done
    }
}

/// Each round's runs of the free lanes 0 and 1: `(extent, tokens)`.
type FreeRounds = Vec<Vec<(u32, Vec<i32>)>>;

/// `ROUNDS` greedy verify rounds over the three prompts at the load's draft
/// `window`, on a pool of `kv_format`, lane 2 held to `grammar` when `held`
/// is given. Every round must replay its graph, and (#691) run at the
/// extended shape on a load past [`BASE_WINDOW`] drafts, whose rounds the
/// caller names `extended`.
fn run(model: &Model, prompts: &[Vec<i32>], kv_format: KvFormat, window: u32, mut held: Option<&mut Held>) -> FreeRounds {
    let pool = pool(kv_format);
    let capture = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    assert!(capture.is_ready(LANES as u32), "width {LANES}: no decode graph");
    let mut seqs: Vec<Seq<'_>> =
        (0..LANES).map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    for (lane, (seq, prompt)) in seqs.iter_mut().zip(prompts).enumerate() {
        // #287: the held lane's prefill keeps its draw's scores, as a checked
        // job's does; the free lanes' keep none.
        let keeps_scores = held.is_some() && lane == HELD;
        let draw = LaneDraw::free(SamplingParams::greedy());
        prefill_program_draw(model, &pool, seq, prompt, 0, 0, None, false, keeps_scores, draw, None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let mut done = false;
    if let Some(held) = held.as_deref_mut() {
        // The prefill's own draw, repaired from the row it kept.
        done = held.hold(model, &pool, &mut seqs[HELD], None);
    }
    let mut free = Vec::new();
    for round in 0..ROUNDS {
        let stops = held.as_ref().map_or(Vec::new(), |held| held.stops.clone());
        let lanes: Vec<VerifyLane<'_>> = (0..LANES)
            .map(|lane| VerifyLane {
                stop_ids: if lane == HELD { &stops } else { &[] },
                ..VerifyLane::greedy(&[])
            })
            .collect();
        let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
        let runs: Vec<LaneVerifyRun> = match held.as_deref_mut().filter(|_| !done) {
            None => decode_program_verify_runs(model, &pool, &mut refs, &lanes, window),
            Some(held) => {
                let flags: Vec<bool> = (0..LANES).map(|lane| lane == HELD).collect();
                decode_program_verify_cut(model, &pool, &mut refs, &lanes, window, &flags, &mut |lane, run| {
                    assert_eq!(lane, HELD, "only the held lane is shown to the host");
                    held.cut(run)
                })
            }
        }
        .unwrap_or_else(|e| panic!("round {round}: {e}"));
        let stats = program_stats(model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, 1, "round {round}: the round replayed its verify graph");
        let extended = window > BASE_WINDOW;
        assert!(
            runs.iter().all(|run| run.extended == extended) && runs[..HELD].iter().all(|run| run.extent == window),
            "round {round}: the round ran at the {} shape, its free lanes drafting {window}",
            if extended { "extended" } else { "base" }
        );
        free.push(runs[..HELD].iter().map(|run| (run.extent, run.tokens.clone())).collect());
        if let Some(held) = held.as_deref_mut().filter(|_| !done) {
            let committed = runs[HELD].tokens.len();
            let column = verify_column(model, window, HELD, committed - 1);
            done = held.hold(model, &pool, &mut seqs[HELD], Some(&column));
        }
    }
    free
}

/// The free run and the held run on `model` at its draft `window`, held to
/// each other: the free lanes verify and commit the same, the held lane's
/// runs are cut inside and repaired from the verify pass's scores.
fn held_beside_free(model: &Model, prompts: &[Vec<i32>], stops: Vec<i32>, grammar: &Arc<Grammar>, kv_format: KvFormat, window: u32) {
    let alone = run(model, prompts, kv_format, window, None);
    let mut held = Held {
        matcher: Matcher::new(grammar.clone()),
        set: TokenSet::new(vocab()),
        stops,
        anchor: None,
        cuts: 0,
        repairs: 0,
    };
    let beside = run(model, prompts, kv_format, window, Some(&mut held));
    for (round, (alone, beside)) in alone.iter().zip(&beside).enumerate() {
        assert_eq!(beside, alone, "round {round}: the free lanes verified and committed what they did beside a free lane");
    }
    let drafted: usize = alone.iter().flatten().map(|(extent, _)| *extent as usize).sum();
    let accepted: usize = alone.iter().flatten().map(|(_, tokens)| tokens.len() - 1).sum();
    eprintln!(
        "verify cut gpu, window {window}: free lanes drafted {drafted}, accepted {accepted}; held lane cut {} runs inside, {} repairs from the verify scores",
        held.cuts, held.repairs
    );
    assert!(accepted > 0, "the free lanes kept drafts");
    assert!(held.cuts > 0, "the held lane's runs were never cut inside: the prompt exercised nothing");
    assert!(held.repairs > 0, "no repair read the verify pass's scores");
}

/// #691: on the two-shape `model`, lane 2 rides one verify round at the
/// shape `cut_at` names, the host cutting its run (keeping it whole, so its
/// pending token is the draw of the column the run ended at), and is
/// repaired from it with every token but that one permitted: the second best
/// of the column, read back at the round's stride. Then lanes 0 and 1 alone
/// ride a round of the other shape, which writes the logits both shapes
/// share, and a second repair of lane 2 is refused, its pending token left
/// as the first repair made it.
fn refused_after_the_other_shape(model: &Model, prompts: &[Vec<i32>], cut_at: &str) {
    let extended = cut_at == "extended";
    let other = if extended { "base" } else { "extended" };
    let what = format!("a cut at the {cut_at} shape, then a round at the {other} shape");
    let pool = pool(KvFormat::HqE8_2b);
    capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("{what}: capture: {e}"));
    let mut seqs: Vec<Seq<'_>> =
        (0..LANES).map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"))).collect();
    for (seq, prompt) in seqs.iter_mut().zip(prompts) {
        let draw = LaneDraw::free(SamplingParams::greedy());
        prefill_program_draw(model, &pool, seq, prompt, 0, 0, None, false, false, draw, None)
            .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
    }
    let lanes = vec![VerifyLane::greedy(&[]); LANES];
    let flags: Vec<bool> = (0..LANES).map(|lane| lane == HELD).collect();
    set_round_shape(Some(cut_at));
    let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
    let runs = decode_program_verify_cut(model, &pool, &mut refs, &lanes, EXTENDED_WINDOW, &flags, &mut |_, run| run.len())
        .unwrap_or_else(|e| panic!("{what}: the cut round: {e}"));
    assert!(runs.iter().all(|run| run.extended == extended), "{what}: the cut round ran at the {cut_at} shape");
    let window = if extended { EXTENDED_WINDOW } else { BASE_WINDOW };
    let column = verify_column(model, window, HELD, runs[HELD].tokens.len() - 1);
    let pending = seqs[HELD].pending_token().expect("a pending token");
    let mask = all_but(pending);
    let repaired = repair_program(model, &pool, &mut seqs[HELD], SamplingParams::greedy(), &mask)
        .unwrap_or_else(|e| panic!("{what}: the repair after the cut: {e}"));
    assert_eq!(
        repaired,
        argmax_but(&column, pending),
        "{what}: the repair is the best but one of the column the cut round drew {pending} from"
    );
    set_round_shape(Some(other));
    let mut others: Vec<&mut Seq<'_>> = seqs[..HELD].iter_mut().collect();
    let runs = decode_program_verify_runs(model, &pool, &mut others, &lanes[..HELD], EXTENDED_WINDOW)
        .unwrap_or_else(|e| panic!("{what}: the round of the other shape: {e}"));
    set_round_shape(None);
    assert!(runs.iter().all(|run| run.extended != extended), "{what}: lanes 0 and 1 rode the {other} shape");
    let refused = repair_program(model, &pool, &mut seqs[HELD], SamplingParams::greedy(), &mask);
    let error = refused.expect_err(&format!("{what}: a repair read scores a later round wrote over"));
    assert!(error.contains("written over by a later round"), "{what}: the refusal names the later round: {error}");
    assert_eq!(seqs[HELD].pending_token(), Some(repaired), "{what}: the refused repair left the pending token");
    println!("verify cut gpu: {what}: the repair after the cut drew {repaired}, the one after the round was refused");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_held_lane_keeps_speculation_and_its_neighbours_keep_their_drafts_and_the_graph() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (prompts, stops, grammar) = inputs(&reader);
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = materialize(&reader, &plan, &mut device, None).unwrap_or_else(|e| panic!("materialize: {e}"));
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
    let model =
        load_qwen38_27b_with_speculation(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT, KvFormat::Bf16, Some(spec))
            .unwrap_or_else(|e| panic!("model load (dflash2, window {WINDOW}): {e}"));
    held_beside_free(&model, &prompts, stops, &grammar, KvFormat::Bf16, WINDOW);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn at_the_extended_shape_a_held_lane_is_cut_and_repaired_from_its_rounds_scores_and_a_repair_after_a_round_of_the_other_shape_is_refused() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
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
    let (_, (major, minor)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("device identity: {e}"));
    if major != 12 && gpu_profile::skip_or_fail(&format!("compute capability {major}.{minor}: FP8 exception weights need 12.0")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (prompts, stops, grammar) = inputs(&reader);
    let spec = Speculation::new(SpeculativeBackend::Dflash2, EXTENDED_WINDOW).unwrap();
    let (plan, handles) = bind_model_scope_27b_with(&reader, model_scope(Some(spec), None, ExceptionWeights::Fp8))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let mut artifact = materialize(&reader, &plan, &mut device, None).unwrap_or_else(|e| panic!("materialize: {e}"));
    {
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            MAX_CONTEXT,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            ignis_core::N_DECODE_LANES as u32,
        )
        .unwrap_or_else(|e| panic!("model load (dflash2, window {EXTENDED_WINDOW}): {e}"));
        set_round_shape(Some("extended"));
        held_beside_free(&model, &prompts, stops, &grammar, KvFormat::HqE8_2b, EXTENDED_WINDOW);
        refused_after_the_other_shape(&model, &prompts, "extended");
        refused_after_the_other_shape(&model, &prompts, "base");
        set_round_shape(None);
    }
    artifact.release_arena(&mut device).unwrap_or_else(|e| panic!("release: {e}"));
}
