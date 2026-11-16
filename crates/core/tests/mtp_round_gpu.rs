//! GPU coverage for the 27B's MTP head in the verify round (GitHub #621,
//! spec #618). On a load built for MTP the head fills the round's proposal
//! seam: each lane's chain runs inside the round's pass (and its captured
//! graph), the first step from the sequence's continuation hidden and its
//! pending token, each later one from the head's own hidden and the draft it
//! chose, up to the load's window; the drafts are the head's best tokens (the
//! fixed draft). After the cut the head's rows the committed run licenses are
//! written again from the run itself, and the continuation hidden moves to
//! the run's last column (kernel/src/mtp_head.h, "the head in a verify
//! round"). Only the runs and each lane's extent cross the seam.
//!
//! - **The drafts the target accepts.** Every round commits its anchor and
//!   the drafts the target accepts: the verify substrate alone (no head),
//!   given the same drafts -- each round's committed ones, then one the
//!   target rejects -- commits the same runs at the same extents, bit for bit,
//!   at windows 1 to 7 and one to eight lanes, and on the 16 GB artifact
//!   (whose output head, which the chain scores with, is trellis) at one to
//!   eight lanes. So the head leaves the target's computation as it was.
//! - Greedy spec-on equals the same load's spec-off text (plain rounds) on
//!   the canary prompts at one, four and eight lanes, each stream cut at the
//!   turn's first stop id, up to the near-tie rule (`support/near_tie.rs`,
//!   probed on a text-scope load: an MTP load takes no per-token prefill) --
//!   or, where the rule refuses, up to a pick the verify substrate makes with
//!   the same drafts and the two prefill routes rate within
//!   `VERIFY_TILE_PICK_ULPS` BF16 ulps of their top logit (ADR 0056's moved
//!   pick; ADR 0095): the verify tile's drift from a one-column decode, not
//!   the head's, printed as a finding. Every round's acceptance is printed,
//!   with the mean committed tokens per full-window round beside the
//!   reference engine's figure of about 3 (a mean below it is a finding).
//! - Graph replay and eager rounds are one computation at every lane count,
//!   and a prefill chunk between two replays changes nothing.
//! - A lane at extent 0 by its budget or its context, and one whose context
//!   cuts its extent, ride a round beside a drafting lane as under DFlash2.
//! - **The head's state after a round is the committed run's.** In every
//!   round, drafts accepted and rejected, under BF16 KV: the continuation
//!   hidden is the round's last committed column's, bit for bit, and the
//!   head's rows at the run's positions are the head's in double precision
//!   (`f64_reference::evaluate_mtp_head`) over what the run consumed -- the
//!   continuation hidden before the round and the round's own committed
//!   columns, with the run's tokens -- within #620's BF16 bounds, while a row
//!   built as the chain builds its own (the head's hidden in place of the
//!   target's) lands far outside them. Not bit for bit against a sequence
//!   that committed the same tokens otherwise: the target's hidden at a
//!   position is the round's that computed it (a plain round's, a verify
//!   round's at another extent), which already parts at a near-tie. Against
//!   plain rounds held to the same tokens the drafted rows stand within twice
//!   the distance a prefill of the same text stands from them (two correct
//!   routes), and a chain-built row stands past both.
//! - A sampled lane with a fixed seed reproduces its stream, which does not
//!   depend on its row in the round.
//! - Under hq-e8-2b each round leaves the head's own ring words as the host
//!   rule says: the chain's rows past the commit cleared, the committed ones
//!   set, the target's words as for DFlash2 (GitHub #257); at eight lanes,
//!   every width's verify graph captured.
//! - **The head is the right head** (GitHub #625): on the same sixteen greedy
//!   prompts, at one lane, a fixed window of 7 and the full proposal head,
//!   under BF16 and hq-e8-2b KV, ignis commits no fewer tokens a round than
//!   the reference engine's MTP less 5%, both counted as the reference counts
//!   them; every prompt's and the pooled acceptance at each position are
//!   printed beside the reference's (`fixtures/mtp_reference_acceptance.json`).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/near_tie.rs"]
#[allow(dead_code)]
mod near_tie;

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, f64_reference, materialize, model_scope_27b_of, CudaDevice, DraftModule, FrontendSet,
    MaterializationPlan, MaterializedArtifact, ModelScope, ObjectHandle, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::hq_ring::{ring_after_prefill, HqRing};
use ignis_core::model_load::{load_qwen38_27b, load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_round, decode_program_verify_runs,
    prefill_program_sampled, prefill_program_with_route, program_stats, LaneDraw, LaneVerifyRun, PrefillRoute, Redirect,
    SamplingParams, VerifyLane,
};
use ignis_core::{mtp_readout, KvFormat, SpecCounters, Speculation, SpeculativeBackend};

use near_tie::first_divergence;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The 16 GB artifact: a trellis body and output head, the same W8 head.
const TRELLIS: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_trellis_sc30-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// The 27B MTP load's chunk: a multiple of 128 (`mtp_head::PREFILL_ROWS`).
const CHUNK: u32 = 128;
/// The window the reference engine's figure is at, and the widest.
const WINDOW: u32 = 7;
/// Tokens each lane emits in a comparison: several full windows.
const TOTAL: usize = 48;
/// The reference engine's committed tokens per full-window round with its
/// MTP head at draft 7, the full proposal head and the fixed window, as this
/// test drafts: 4.207 under BF16 KV and 4.164 under hq-e8-2b on #625's
/// sixteen short prompts, 3.97 on its four long ones (2K to 15K tokens).
/// The spec's "about 3" was the shortlist head under the adaptive window.
const REFERENCE_PER_ROUND: f64 = 4.2;
const HIDDEN: usize = 5120;
const VOCAB: usize = 248_320;
const KV_WIDTH: usize = mtp_readout::KV_WIDTH;
/// A pick the verify tile moves from a one-column decode's, where the near-tie
/// rule's two prefill routes agree, is held within this many BF16 ulps of
/// their top logit: ADR 0056's moved-pick bound (measured moved picks 0-8
/// ulps on Flash-Next's verify rows), recorded for the 27B in ADR 0095.
const VERIFY_TILE_PICK_ULPS: f64 = 16.0;
/// BF16's unit roundoff and the roundings on the device's path to a head
/// row's key and value (crates/core/tests/mtp_head_gpu.rs, #620's bounds).
const BF16_UNIT: f64 = 1.0 / 256.0;
const KEY_ROUNDINGS: f64 = 6.0;
const VALUE_ROUNDINGS: f64 = 5.0;

fn tolerance(roundings: f64) -> f64 {
    BF16_UNIT * roundings.sqrt()
}

/// The canary suite's prompts (`crates/bench/src/canary.rs::CANARIES`), each
/// in a user turn of the model's chat format with thinking closed.
const CANARY_PROMPTS: [&str; 4] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "Explain in one sentence what `x.reverse()` does to a `Vec<i32>` named `x`.",
];

fn mtp(window: u32) -> Speculation {
    Speculation::new(SpeculativeBackend::Mtp, window).unwrap()
}

fn budget(kv_format: KvFormat, slot_count: u32, context: u32) -> SeqPoolBudget {
    SeqPoolBudget {
        kv_format,
        kv_page_group_count: context.div_ceil(64) * slot_count,
        max_context_tokens: context,
        slot_count,
        retained_slot_count: 0,
        retained_host_slot_count: 0,
    }
}

/// A pool built for the head: its layer in every page, a continuation
/// hidden in every slot.
fn head_pool(kv_format: KvFormat, slot_count: u32, context: u32) -> SeqPool {
    SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(), &budget(kv_format, slot_count, context), Some(SpeculativeBackend::Mtp))
        .unwrap_or_else(|e| panic!("MTP pool create: {e}"))
}

fn plain_pool(slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(), &budget(KvFormat::Bf16, slot_count, MAX_CONTEXT), None)
        .unwrap_or_else(|e| panic!("plain pool create: {e}"))
}

/// The verify substrate alone at `window`: no drafter, each round's drafts
/// from its caller (the text scope's handles).
fn load_substrate(reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle], window: u32) -> Model {
    let spec = Speculation::new(SpeculativeBackend::VerifyOnly, window).unwrap();
    load_qwen38_27b_with_speculation(reader, artifact, handles, CHUNK, MAX_CONTEXT, KvFormat::Bf16, Some(spec))
        .unwrap_or_else(|e| panic!("model load (verify substrate, window {window}): {e}"))
}

fn substrate_pool(slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &budget(KvFormat::Bf16, slot_count, MAX_CONTEXT),
        Some(SpeculativeBackend::VerifyOnly),
    )
    .unwrap_or_else(|e| panic!("verify-only pool create: {e}"))
}

fn load_head(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    window: u32,
    kv_format: KvFormat,
    context: u32,
) -> Model {
    load_qwen38_27b_with_speculation(reader, artifact, handles, CHUNK, context, kv_format, Some(mtp(window)))
        .unwrap_or_else(|e| panic!("model load (MTP, window {window}, {kv_format:?}): {e}"))
}

/// The card and an artifact materialized on it; dropping it gives the arena
/// back, on a panic too.
struct OnDevice {
    artifact: MaterializedArtifact,
    device: CudaDevice,
}

impl Drop for OnDevice {
    fn drop(&mut self) {
        let _ = self.artifact.release_arena(&mut self.device);
    }
}

fn on_device(reader: &Reader, plan: &MaterializationPlan) -> Option<OnDevice> {
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    match materialize(reader, plan, &mut device, None) {
        Ok(artifact) => Some(OnDevice { artifact, device }),
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp: {e}")) {
                return None;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    }
}

/// The artifact at `artifact`'s reader, its text + MTP scope and the scope on
/// the card, or `None` when the test skips.
fn materialized(artifact: &str) -> Option<(Reader, Vec<ObjectHandle>, OnDevice)> {
    let path = Path::new(artifact);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {artifact}")) {
        return None;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Mtp)).unwrap_or_else(|e| panic!("bind with the MTP head: {e}"));
    let on_device = on_device(&reader, &plan)?;
    Some((reader, handles, on_device))
}

/// How many of the scope's handles are the text scope's: its first.
fn text_len(reader: &Reader) -> usize {
    model_scope_27b_of(reader, ModelScope::default()).unwrap_or_else(|e| panic!("text scope: {e}")).len()
}

fn encoder(reader: &Reader) -> impl Fn(&str) -> Vec<i32> + '_ {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    move |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    }
}

fn turn(encode: &dyn Fn(&str) -> Vec<i32>, text: &str) -> Vec<i32> {
    encode(&format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))
}

/// The canary prompts in the model's chat format, and the two stop markers'
/// ids (the turn ends at either).
fn chat_canaries(reader: &Reader) -> (Vec<Vec<i32>>, Vec<i32>) {
    let encode = encoder(reader);
    assert_eq!(encode("<|im_start|>").len(), 1, "the chat markers must tokenize as special tokens");
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    (CANARY_PROMPTS.iter().map(|p| turn(&encode, p)).collect(), stop_ids)
}

fn prefill_all(model: &Model, pool: &SeqPool, sequences: &mut [Seq<'_>], prompts: &[Vec<i32>]) {
    for (seq, prompt) in sequences.iter_mut().zip(prompts) {
        prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
}

fn alloc_prefilled<'p>(model: &Model, pool: &'p SeqPool, prompts: &[Vec<i32>], context: u32) -> Vec<Seq<'p>> {
    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(context).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    prefill_all(model, pool, &mut sequences, prompts);
    sequences
}

fn cut_at_stop(stream: &mut Vec<i32>, stop_ids: &[i32]) {
    if let Some(stop) = stream.iter().position(|t| stop_ids.contains(t)) {
        stream.truncate(stop + 1);
    }
}

/// The same load's spec-off text: `total` greedy tokens per lane, one a
/// plain round, every lane in one batch, each stream cut at its first stop.
fn spec_off(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], stop_ids: &[i32], total: usize) -> Vec<Vec<i32>> {
    let mut sequences = alloc_prefilled(model, pool, prompts, MAX_CONTEXT);
    let params = vec![SamplingParams::greedy(); prompts.len()];
    let mut streams = vec![Vec::with_capacity(total); prompts.len()];
    for _ in 0..total {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let round = decode_program_batch_sampled(model, pool, &mut refs, &params)
            .unwrap_or_else(|e| panic!("spec-off decode: {e}"));
        for (stream, token) in streams.iter_mut().zip(round) {
            stream.push(token);
        }
    }
    for stream in &mut streams {
        cut_at_stop(stream, stop_ids);
    }
    streams
}

/// One verify round, the head proposing, each lane under its sampling,
/// budgeted to `budgets[i]` and cut at `stop_ids`.
fn round(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
    samplings: &[SamplingParams],
    budgets: &[u32],
    stop_ids: &[i32],
    window: u32,
) -> Vec<LaneVerifyRun> {
    let lanes: Vec<VerifyLane<'_>> = budgets
        .iter()
        .zip(samplings)
        .map(|(&remaining_tokens, &sampling)| VerifyLane { sampling, remaining_tokens, stop_ids, ..VerifyLane::greedy(&[]) })
        .collect();
    decode_program_verify_runs(model, pool, sequences, &lanes, window).unwrap_or_else(|e| panic!("verify round: {e}"))
}

/// What one run recorded: each lane's text, and per round the `(lane,
/// extent, committed)` of every lane that rode it.
#[derive(Debug, Clone, PartialEq)]
struct SpecRun {
    emitted: Vec<Vec<i32>>,
    rounds: Vec<Vec<(usize, u32, usize)>>,
}

/// Runs `sequences` through verify rounds until every lane has emitted
/// `total` tokens or a stop id, each lane budgeted to what it has left; lanes
/// ride every round until done, so widths shrink as lanes finish. Checks each
/// round's run opens with the lane's anchor, its reported extent and, when
/// `expect_graph` is given, whether the round replayed a verify graph.
#[allow(clippy::too_many_arguments)]
fn run_lanes(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [Seq<'_>],
    samplings: &[SamplingParams],
    stop_ids: &[i32],
    total: usize,
    window: u32,
    expect_graph: Option<bool>,
) -> SpecRun {
    let lanes = sequences.len();
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); lanes];
    let mut done = vec![false; lanes];
    let mut rounds = Vec::new();
    while done.iter().any(|d| !d) {
        let active: Vec<usize> = (0..lanes).filter(|&i| !done[i]).collect();
        let budgets: Vec<u32> = active.iter().map(|&i| (total - emitted[i].len()) as u32).collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> =
            active.iter().map(|&i| remaining[i].take().expect("each active lane once")).collect();
        let lane_samplings: Vec<SamplingParams> = active.iter().map(|&i| samplings[i]).collect();
        let anchors: Vec<i32> = refs.iter().map(|seq| seq.pending_token().expect("a pending token")).collect();
        let results = round(model, pool, &mut refs, &lane_samplings, &budgets, stop_ids, window);
        for (result, anchor) in results.iter().zip(&anchors) {
            assert_eq!(result.tokens.first(), Some(anchor), "round {}: a run that does not open with its anchor", rounds.len());
        }
        if let Some(expect_graph) = expect_graph {
            let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
            assert_eq!(stats.graph_launches, u64::from(expect_graph), "round {} at width {}: graph replay", rounds.len(), active.len());
        }
        let mut this_round = Vec::new();
        for ((&i, result), &budget) in active.iter().zip(results).zip(&budgets) {
            assert_eq!(result.extent, window.min(budget - 1), "lane {i}: the leaf's reported extent");
            assert!(
                !result.tokens.is_empty() && result.tokens.len() <= result.extent as usize + 1,
                "lane {i}: a run of {} at extent {}",
                result.tokens.len(),
                result.extent
            );
            this_round.push((i, result.extent, result.tokens.len()));
            done[i] = result.tokens.iter().any(|t| stop_ids.contains(t));
            emitted[i].extend(result.tokens);
            done[i] |= emitted[i].len() >= total;
        }
        rounds.push(this_round);
        assert!(rounds.len() <= total, "the run does not converge");
    }
    SpecRun { emitted, rounds }
}

#[allow(clippy::too_many_arguments)]
fn spec_on(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    samplings: &[SamplingParams],
    stop_ids: &[i32],
    total: usize,
    window: u32,
    expect_graph: Option<bool>,
) -> SpecRun {
    let mut sequences = alloc_prefilled(model, pool, prompts, MAX_CONTEXT);
    run_lanes(model, pool, &mut sequences, samplings, stop_ids, total, window, expect_graph)
}

fn greedy(lanes: usize) -> Vec<SamplingParams> {
    vec![SamplingParams::greedy(); lanes]
}

/// Prints every round's acceptance and the mean committed tokens per
/// full-window round beside the reference engine's figure; a mean below it
/// is printed as a finding. Returns (drafts accepted, mean).
fn report_acceptance(label: &str, run: &SpecRun, window: u32) -> (usize, f64) {
    let mut committed_full = 0usize;
    let mut full_rounds = 0usize;
    let mut accepted = 0usize;
    for (index, lanes) in run.rounds.iter().enumerate() {
        let cells: Vec<String> =
            lanes.iter().map(|&(lane, extent, committed)| format!("lane {lane}: {}/{extent}", committed - 1)).collect();
        println!("{label} round {index}: accepted/drafted {}", cells.join(", "));
        for &(_, extent, committed) in lanes {
            accepted += committed - 1;
            if extent == window {
                committed_full += committed;
                full_rounds += 1;
            }
        }
    }
    let mean = committed_full as f64 / full_rounds.max(1) as f64;
    println!(
        "{label}: {mean:.2} committed tokens per full-window round over {full_rounds} rounds (the reference engine: \
         about {REFERENCE_PER_ROUND:.1} at draft {WINDOW})"
    );
    if window == WINDOW && mean < REFERENCE_PER_ROUND {
        println!("{label}: FINDING -- fewer committed tokens per round than the reference engine's");
    }
    (accepted, mean)
}

/// The verify substrate alone, given each round of `run` the drafts that
/// round committed and then one the target rejects (one past the run's next
/// token), at the same lanes, budgets and stop ids: each round's extent and
/// run, which a head that leaves the target alone makes the same.
fn replay(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    stop_ids: &[i32],
    total: usize,
    window: u32,
    run: &SpecRun,
) -> Vec<Vec<(usize, u32, Vec<i32>)>> {
    let mut sequences = alloc_prefilled(model, pool, prompts, MAX_CONTEXT);
    let mut at = vec![0usize; prompts.len()];
    let mut rounds = Vec::new();
    for mtp_round in &run.rounds {
        let mut drafts: Vec<Vec<i32>> = Vec::new();
        let mut budgets = Vec::new();
        for &(lane, extent, committed) in mtp_round {
            let text = &run.emitted[lane];
            let o = at[lane];
            let mut lane_drafts: Vec<i32> = text[o + 1..o + committed].to_vec();
            let filler = text.get(o + committed).map_or(0, |&next| (next + 1) % VOCAB as i32);
            lane_drafts.resize(extent as usize, filler);
            drafts.push(lane_drafts);
            budgets.push((total - o) as u32);
        }
        let lanes: Vec<VerifyLane<'_>> = drafts
            .iter()
            .zip(&budgets)
            .map(|(d, &remaining_tokens)| VerifyLane { remaining_tokens, stop_ids, ..VerifyLane::greedy(d) })
            .collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> =
            mtp_round.iter().map(|&(lane, _, _)| remaining[lane].take().expect("each lane once")).collect();
        let results = decode_program_verify_runs(model, pool, &mut refs, &lanes, window)
            .unwrap_or_else(|e| panic!("the substrate's round: {e}"));
        let mut this_round = Vec::new();
        for (&(lane, _, _), result) in mtp_round.iter().zip(results) {
            at[lane] += result.tokens.len();
            this_round.push((lane, result.extent, result.tokens));
        }
        rounds.push(this_round);
    }
    rounds
}

/// `run`'s rounds as [`replay`] returns them.
fn rounds_of(run: &SpecRun) -> Vec<Vec<(usize, u32, Vec<i32>)>> {
    let mut at = vec![0usize; run.emitted.len()];
    run.rounds
        .iter()
        .map(|round| {
            round
                .iter()
                .map(|&(lane, extent, committed)| {
                    let tokens = run.emitted[lane][at[lane]..at[lane] + committed].to_vec();
                    at[lane] += committed;
                    (lane, extent, tokens)
                })
                .collect()
        })
        .collect()
}

/// The BF16 ulp at `x`'s magnitude.
fn bf16_ulp(x: f32) -> f64 {
    let magnitude = f64::from(x.abs()).max(f64::MIN_POSITIVE);
    2f64.powi(magnitude.log2().floor() as i32 - 7)
}

/// Spec-on `on` against spec-off `off` (lane by lane, `support/near_tie.rs`'s
/// rule, probed on `plain`), with one exception the rule does not know: a
/// divergence the verify substrate reproduced with the same drafts (the
/// caller asserted [`replay`]) where both prefill routes rate spec-off's pick
/// ahead by at most [`VERIFY_TILE_PICK_ULPS`] of their top logit's ulp -- the
/// verify tile's drift, ADR 0056. Returns a line per such divergence; panics
/// on any other.
fn judge(plain: &Model, prompts: &[Vec<i32>], off: &[Vec<i32>], on: &[Vec<i32>], what: &str) -> Vec<String> {
    let mut findings = Vec::new();
    for (lane, (prompt, (off, on))) in prompts.iter().zip(off.iter().zip(on)).enumerate() {
        let Some(d) = first_divergence(off, on) else { continue };
        assert!(d < off.len() && d < on.len(), "{what} lane {lane}: one stream ended early at {d}");
        let (a, b) = (off[d], on[d]);
        let mut tokens = prompt.clone();
        tokens.extend_from_slice(&off[..d]);
        let probe = |route: PrefillRoute| -> (f32, f32) {
            let pool = plain_pool(1);
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
            let mut logits = vec![0f32; VOCAB];
            prefill_program_with_route(plain, &pool, &mut seq, &tokens, 0, route, Some(&mut logits))
                .unwrap_or_else(|e| panic!("near-tie probe prefill: {e}"));
            let top = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
            (logits[a as usize] - logits[b as usize], top)
        };
        let (chunked, top) = probe(PrefillRoute::Chunked);
        let (per_token, _) = probe(PrefillRoute::PerToken);
        let routes_disagree = (chunked > 0.0) != (per_token > 0.0) || chunked == 0.0 || per_token == 0.0;
        let within_spread = chunked.abs().min(per_token.abs()) <= (chunked - per_token).abs();
        let routes_side_with_spec_on = chunked < 0.0 && per_token < 0.0;
        if routes_disagree || within_spread || routes_side_with_spec_on {
            continue;
        }
        let ulps = f64::from(chunked.max(per_token)) / bf16_ulp(top);
        let line = format!(
            "{what} lane {lane}: after {d} tokens spec-on picked {b} where spec-off picked {a}; both prefill routes put \
             {a} ahead by {chunked} / {per_token} ({ulps:.1} BF16 ulps at their top logit {top}), a pick the verify \
             substrate makes with the same drafts"
        );
        assert!(
            ulps <= VERIFY_TILE_PICK_ULPS,
            "{line}: past the verify tile's moved-pick bound of {VERIFY_TILE_PICK_ULPS} ulps"
        );
        println!("FINDING: {line}");
        findings.push(line);
    }
    findings
}

fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// `ours` against `reference`, both BF16 bits, as a relative L2.
fn relative_l2_bits(ours: &[u16], reference: &[u16]) -> f64 {
    let (mut error, mut norm) = (0f64, 0f64);
    for (&a, &b) in ours.iter().zip(reference) {
        let (a, b) = (bf16_to_f64(a), bf16_to_f64(b));
        error += (a - b) * (a - b);
        norm += b * b;
    }
    error.sqrt() / norm.sqrt().max(1e-30)
}

fn relative_l2(ours: &[f64], reference: &[f64]) -> f64 {
    let error: f64 = ours.iter().zip(reference).map(|(a, b)| (a - b) * (a - b)).sum();
    let norm: f64 = reference.iter().map(|b| b * b).sum();
    error.sqrt() / norm.sqrt().max(1e-30)
}

/// The device's embedding of `token`, as the embedding op rounds it.
fn embedding(reader: &Reader, token: i32) -> Vec<f64> {
    f64_reference::matrix_row(reader, "text/token_embedding", VOCAB, HIDDEN, token as usize)
        .unwrap_or_else(|e| panic!("embedding row {token}: {e}"))
        .into_iter()
        .map(|value| bf16_to_f64(f32_to_bf16(value as f32)))
        .collect()
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_head_drafts_in_the_verify_round_and_the_text_stays_the_spec_off_text() {
    let Some((reader, handles, on_device)) = materialized(ARTIFACT) else {
        return;
    };
    let artifact = &on_device.artifact;
    let text_len = text_len(&reader);
    let (canaries, stop_ids) = chat_canaries(&reader);
    // Eight lanes carry the four canaries twice.
    let prompts8: Vec<Vec<i32>> = canaries.iter().chain(&canaries).cloned().collect();
    assert!(prompts8.iter().all(|p| p.len() + TOTAL <= MAX_CONTEXT as usize), "a canary outgrows the context");

    // --- the text at one, four and eight lanes, and its acceptance ---------
    let widths = [1usize, 4, 8];
    let mut off = Vec::new();
    let mut on = Vec::new();
    let mut accepted = 0usize;
    {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        for &width in &widths {
            let pool = head_pool(KvFormat::Bf16, width as u32, MAX_CONTEXT);
            let prompts = &prompts8[..width];
            off.push(spec_off(&model, &pool, prompts, &stop_ids, TOTAL));
            let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            assert!(capture.is_ready(width as u32), "width {width}: no decode graph");
            let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
            assert!(
                stats.verify_graph_ready_mask & (1 << (width - 1)) != 0,
                "width {width}: no verify graph with the head's chain in it (mask {:#010b})",
                stats.verify_graph_ready_mask
            );
            let run = spec_on(&model, &pool, prompts, &greedy(width), &stop_ids, TOTAL, WINDOW, Some(true));
            accepted += report_acceptance(&format!("width {width}"), &run, WINDOW).0;
            on.push(run);
        }
        // The head proposes every lane's drafts: a caller's are refused.
        let pool = head_pool(KvFormat::Bf16, 1, MAX_CONTEXT);
        let mut seq = alloc_prefilled(&model, &pool, &canaries[..1], MAX_CONTEXT).remove(0);
        let drafts = [canaries[0][0]];
        let refused = decode_program_verify_runs(&model, &pool, &mut [&mut seq], &[VerifyLane::greedy(&drafts)], WINDOW);
        let refusal = refused.err().expect("an MTP load took caller drafts");
        assert!(refusal.contains("must be NULL") && refusal.contains("MTP head"), "the wrong refusal: {refusal}");
    }
    assert!(accepted > 0, "the head never landed a draft: its chain is not reading the sequence");
    {
        // Every round committed its anchor and the drafts the target accepts:
        // the verify substrate alone, given the same drafts, commits the same.
        let substrate = load_substrate(&reader, artifact, &handles[..text_len], WINDOW);
        for (&width, on) in widths.iter().zip(&on) {
            let pool = substrate_pool(width as u32);
            let _ = capture_decode_graphs(&substrate, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            let replayed = replay(&substrate, &pool, &prompts8[..width], &stop_ids, TOTAL, WINDOW, on);
            assert!(replayed == rounds_of(on), "width {width}: the substrate given the head's drafts committed other runs");
        }
    }
    {
        // The near-tie judge reads the target's two prefill routes, on a load
        // with no head (an MTP load takes no per-token prefill).
        let plain = load_qwen38_27b(&reader, artifact, &handles[..text_len], CHUNK, MAX_CONTEXT, KvFormat::Bf16)
            .unwrap_or_else(|e| panic!("plain load: {e}"));
        let mut findings = 0;
        for ((&width, off), on) in widths.iter().zip(&off).zip(&on) {
            findings += judge(&plain, &prompts8[..width], off, &on.emitted, &format!("width {width} MTP")).len();
        }
        println!("spec-on against spec-off: {findings} divergence(s) past the near-tie rule, each the verify tile's");
    }

    // --- graph replay and eager rounds are one computation ---------------
    let eager: Vec<SpecRun> = {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        (1..=8usize)
            .map(|width| {
                let pool = head_pool(KvFormat::Bf16, width as u32, MAX_CONTEXT);
                spec_on(&model, &pool, &prompts8[..width], &greedy(width), &stop_ids, TOTAL, WINDOW, Some(false))
            })
            .collect()
    };
    {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        for (width, eager) in (1..=8usize).zip(eager) {
            let pool = head_pool(KvFormat::Bf16, width as u32, MAX_CONTEXT);
            let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
            assert!(
                stats.verify_graph_ready_mask & (1 << (width - 1)) != 0,
                "width {width}: no verify graph with the head's chain in it (mask {:#010b})",
                stats.verify_graph_ready_mask
            );
            let graph =
                spec_on(&model, &pool, &prompts8[..width], &greedy(width), &stop_ids, TOTAL, WINDOW, Some(true));
            assert_eq!(eager, graph, "width {width}: the head's graph replay diverged from the eager rounds");
        }

        // A prefill chunk between two replays changes nothing: the chain and
        // the commit take the round's scratch, never the prefill's, and the
        // interloper's head state is its own slot's.
        const WIDTH: usize = 2;
        let pool = head_pool(KvFormat::Bf16, WIDTH as u32 + 1, MAX_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let two_rounds = |interleave: bool| -> Vec<Vec<LaneVerifyRun>> {
            let mut lanes = alloc_prefilled(&model, &pool, &canaries[..WIDTH], MAX_CONTEXT);
            let mut out = Vec::new();
            for index in 0..2 {
                let _interloper = (interleave && index == 1).then(|| alloc_prefilled(&model, &pool, &canaries[3..4], MAX_CONTEXT));
                let mut refs: Vec<&mut Seq<'_>> = lanes.iter_mut().collect();
                out.push(round(&model, &pool, &mut refs, &greedy(WIDTH), &[TOTAL as u32; WIDTH], &[], WINDOW));
                let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
                assert_eq!(stats.graph_launches, 1, "the head's round did not replay a verify graph");
            }
            out
        };
        assert_eq!(two_rounds(false), two_rounds(true), "a prefill chunk between two replays changed the second");
    }

    // --- extent 0 by budget and by context, an extent cut by the context ---
    {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        let pool = head_pool(KvFormat::Bf16, 4, MAX_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let mut budgeted = alloc_prefilled(&model, &pool, &canaries[..1], MAX_CONTEXT).remove(0);
        // One KV page, prefilled to one short of it: extent 0 by its context.
        let mut cramped = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
        // The same page prefilled to four short: its context cuts it to 3.
        let mut cut = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut drafting = alloc_prefilled(&model, &pool, &canaries[2..3], MAX_CONTEXT).remove(0);
        let long: Vec<i32> = canaries[1].iter().cycle().take(63).copied().collect();
        prefill_all(&model, &pool, std::slice::from_mut(&mut cramped), &[long.clone()]);
        prefill_all(&model, &pool, std::slice::from_mut(&mut cut), &[long[..60].to_vec()]);
        let before: Vec<u64> = [&budgeted, &cramped, &cut, &drafting].iter().map(|s| s.stats().position).collect();
        let results = round(
            &model,
            &pool,
            &mut [&mut budgeted, &mut cramped, &mut cut, &mut drafting],
            &greedy(4),
            &[1, 0, TOTAL as u32, TOTAL as u32],
            &[],
            WINDOW,
        );
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, 1, "the round with lanes at extent 0 replays the graph");
        for (label, result) in [("out of budget", &results[0]), ("out of context", &results[1])] {
            assert_eq!(result.extent, 0, "{label}: the lane must run at extent 0");
            assert_eq!(result.tokens.len(), 1, "{label}: the lane commits its anchor alone");
        }
        assert_eq!(results[2].extent, 3, "the context cuts the lane's extent to what it leaves after the anchor");
        assert!(results[2].tokens.len() <= 4, "a context-cut lane committed past its context");
        assert_eq!(results[3].extent, WINDOW, "the drafting lane proposes a full window beside them");
        for (i, (seq, result)) in [&budgeted, &cramped, &cut, &drafting].iter().zip(&results).enumerate() {
            let position = seq.stats().position;
            assert_eq!(position, before[i] + result.tokens.len() as u64, "lane {i}: the round committed its run");
            // The head holds every committed position but the last, whose
            // entry is the next step's.
            let frontier = position as u32 - 1;
            mtp_readout::head_state(&pool, seq, 0, frontier).unwrap_or_else(|e| panic!("lane {i}: the head's rows: {e}"));
            let past = mtp_readout::head_state(&pool, seq, frontier, 1).map(|_| ()).expect_err("a row past the head's frontier");
            assert!(past.contains("written"), "lane {i}: {past}");
        }
        // Its budget restored, the lane at extent 0 drafts a full window.
        let continued = round(&model, &pool, &mut [&mut budgeted], &greedy(1), &[TOTAL as u32], &[], WINDOW).remove(0);
        assert_eq!(continued.extent, WINDOW, "the continued lane drafts a full window");
    }

    // --- the head's state after a round is the committed run's -------------
    {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        let prompts = [canaries[0].clone(), canaries[2].clone()];
        let width = prompts.len();
        let pool = head_pool(KvFormat::Bf16, 3 * width as u32, MAX_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let mut lanes = alloc_prefilled(&model, &pool, &prompts, MAX_CONTEXT);
        let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); width];
        let mut done = vec![false; width];
        // Each lane's expected continuation hidden: the last committed
        // column's, read back at the next round's start.
        let mut expected: Vec<Option<Vec<u16>>> = vec![None; width];
        let (mut worst_key, mut worst_value, mut nearest_chain) = (0f64, 0f64, f64::INFINITY);
        let (mut rounds, mut accepted, mut rejected, mut rows_checked) = (0usize, 0usize, 0usize, 0usize);
        while done.iter().any(|d| !d) {
            let active: Vec<usize> = (0..width).filter(|&i| !done[i]).collect();
            // Before the round: what the chain's first step reads, and the
            // head's own hidden there (the chain's next step's input).
            let mut before = Vec::new();
            for &i in &active {
                let position = lanes[i].stats().position;
                let proposal = mtp_readout::propose(&model, &pool, &mut lanes[i]).unwrap_or_else(|e| panic!("readout: {e}"));
                if let Some(expected) = expected[i].take() {
                    assert!(
                        proposal.continuation == expected,
                        "lane {i} round {rounds}: the continuation hidden is not the last round's last committed column"
                    );
                }
                before.push((position, proposal));
            }
            let budgets: Vec<u32> = active.iter().map(|&i| (TOTAL - emitted[i].len()) as u32).collect();
            let mut remaining: Vec<Option<&mut Seq<'_>>> = lanes.iter_mut().map(Some).collect();
            let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
            let results = round(&model, &pool, &mut refs, &greedy(active.len()), &budgets, &stop_ids, WINDOW);
            let hiddens = mtp_readout::round_hidden(&model, active.len() as u32, WINDOW)
                .unwrap_or_else(|e| panic!("the round's hiddens: {e}"));
            let column = |row: usize, j: usize| -> Vec<u16> {
                let at = (row * (WINDOW as usize + 1) + j) * HIDDEN;
                hiddens[at..at + HIDDEN].to_vec()
            };
            for (row, ((&i, result), (position, proposal))) in active.iter().zip(&results).zip(&before).enumerate() {
                let c = result.tokens.len();
                accepted += c - 1;
                rejected += result.extent as usize + 1 - c;
                // The run consumed positions P .. P + c - 1, so it licenses the
                // head's rows P - 1 .. P + c - 2: from the continuation hidden
                // and the anchor, then each committed column and the token
                // after it.
                let first = *position as u32 - 1;
                let (keys, values) =
                    mtp_readout::head_state(&pool, &lanes[i], first, c as u32).unwrap_or_else(|e| panic!("rows: {e}"));
                let inputs: Vec<Vec<f64>> = std::iter::once(proposal.continuation.clone())
                    .chain((0..c - 1).map(|j| column(row, j)))
                    .map(|bits| bits.into_iter().map(bf16_to_f64).collect())
                    .collect();
                let embeddings: Vec<Vec<f64>> = result.tokens.iter().map(|&t| embedding(&reader, t)).collect();
                let positions: Vec<i32> = (0..c as i32).map(|j| first as i32 + j).collect();
                let reference = f64_reference::evaluate_mtp_head(&reader, &inputs, &embeddings, &positions)
                    .unwrap_or_else(|e| panic!("f64 head: {e}"));
                for r in 0..c {
                    let device_key: Vec<f64> = keys[r * KV_WIDTH..(r + 1) * KV_WIDTH].iter().copied().map(bf16_to_f64).collect();
                    let device_value: Vec<f64> =
                        values[r * KV_WIDTH..(r + 1) * KV_WIDTH].iter().copied().map(bf16_to_f64).collect();
                    let key = relative_l2(&device_key, &reference.keys[r]);
                    let value = relative_l2(&device_value, &reference.values[r]);
                    assert!(
                        key <= tolerance(KEY_ROUNDINGS) && value <= tolerance(VALUE_ROUNDINGS),
                        "lane {i} round {rounds}: the head's row at {} is not the committed run's (keys {key}, values {value})",
                        first as usize + r
                    );
                    worst_key = worst_key.max(key);
                    worst_value = worst_value.max(value);
                    rows_checked += 1;
                }
                // What a chain write left below the frontier would be: the row
                // at P built from the head's hidden at P - 1, in place of the
                // target's, with the same draft.
                if c >= 2 {
                    let chain = f64_reference::evaluate_mtp_head(
                        &reader,
                        &[proposal.head_hidden.iter().copied().map(bf16_to_f64).collect()],
                        &[embedding(&reader, result.tokens[1])],
                        &[first as i32 + 1],
                    )
                    .unwrap_or_else(|e| panic!("f64 head: {e}"));
                    let device_key: Vec<f64> = keys[KV_WIDTH..2 * KV_WIDTH].iter().copied().map(bf16_to_f64).collect();
                    let device_value: Vec<f64> = values[KV_WIDTH..2 * KV_WIDTH].iter().copied().map(bf16_to_f64).collect();
                    nearest_chain = nearest_chain
                        .min(relative_l2(&device_key, &chain.keys[0]).min(relative_l2(&device_value, &chain.values[0])));
                }
                expected[i] = Some(column(row, c - 1));
                done[i] = result.tokens.iter().any(|t| stop_ids.contains(t));
                emitted[i].extend(result.tokens.iter().copied());
                done[i] |= emitted[i].len() >= TOTAL;
            }
            rounds += 1;
            assert!(rounds <= TOTAL, "the run does not converge");
        }
        for i in 0..width {
            let proposal = mtp_readout::propose(&model, &pool, &mut lanes[i]).unwrap_or_else(|e| panic!("readout: {e}"));
            assert!(Some(proposal.continuation) == expected[i], "lane {i}: the last continuation hidden");
        }
        println!(
            "state: {rounds} rounds, {accepted} drafts accepted, {rejected} rejected; {rows_checked} committed rows against \
             the f64 head over the run: keys {worst_key:.6} (bound {:.6}), values {worst_value:.6} (bound {:.6}); a \
             chain-built row {nearest_chain:.6} at the nearest",
            tolerance(KEY_ROUNDINGS),
            tolerance(VALUE_ROUNDINGS)
        );
        assert!(accepted > 0 && rejected > 0, "the run must both accept and reject drafts");
        assert!(
            nearest_chain > 4.0 * tolerance(KEY_ROUNDINGS),
            "a chain-built row {nearest_chain} is within four bounds of the committed run's"
        );

        // The same tokens by plain rounds, each draw held to the run's next
        // token: the drift from the target's other hiddens, which a
        // chain-built row stands far past.
        let mut plain = alloc_prefilled(&model, &pool, &prompts, MAX_CONTEXT);
        let longest = emitted.iter().map(Vec::len).max().unwrap();
        let forced: Vec<Vec<i32>> = lanes
            .iter()
            .zip(&emitted)
            .map(|(seq, text)| text[1..].iter().copied().chain(std::iter::once(seq.pending_token().unwrap())).collect())
            .collect();
        for r in 0..longest {
            let active: Vec<usize> = (0..width).filter(|&i| r < emitted[i].len()).collect();
            let mut remaining: Vec<Option<&mut Seq<'_>>> = plain.iter_mut().map(Some).collect();
            let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
            let draws: Vec<LaneDraw<'_>> = active
                .iter()
                .map(|&i| LaneDraw { sampling: SamplingParams::greedy(), permitted: &forced[i][r..=r], redirect: Redirect::NONE })
                .collect();
            let drawn = decode_program_round(&model, &pool, &mut refs, &draws, None).unwrap_or_else(|e| panic!("plain round: {e}"));
            for (&i, draw) in active.iter().zip(drawn) {
                assert_eq!(draw.token, emitted[i][r], "lane {i}: the plain round emitted another token");
            }
        }
        // A third route to the same rows: one prefill of the prompt and the
        // text, whose hiddens are the prefill chunk's.
        let mut prefilled: Vec<Seq<'_>> = (0..width).map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
        for (i, seq) in prefilled.iter_mut().enumerate() {
            let mut text = prompts[i].clone();
            text.extend_from_slice(&emitted[i]);
            prefill_program_sampled(&model, &pool, seq, &text, 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("prefill of the text: {e}"));
        }
        let (mut worst_plain, mut worst_route) = (0f64, 0f64);
        for i in 0..width {
            let frontier = plain[i].stats().position as u32 - 1;
            assert_eq!(frontier, lanes[i].stats().position as u32 - 1, "lane {i}: the plain rounds' frontier");
            let first = prompts[i].len() as u32 - 1;
            let rows = |seq: &Seq<'_>| mtp_readout::head_state(&pool, seq, first, frontier - first).unwrap_or_else(|e| panic!("rows: {e}"));
            let (ours, theirs, route) = (rows(&lanes[i]), rows(&plain[i]), rows(&prefilled[i]));
            let drift = relative_l2_bits(&ours.0, &theirs.0).max(relative_l2_bits(&ours.1, &theirs.1));
            let floor = relative_l2_bits(&route.0, &theirs.0).max(relative_l2_bits(&route.1, &theirs.1));
            println!(
                "lane {i}: {} generated rows against plain rounds of the same tokens: drafted {drift:.6}, the prefill \
                 route {floor:.6}",
                frontier - first
            );
            worst_plain = worst_plain.max(drift);
            worst_route = worst_route.max(floor);
        }
        println!("plain rounds: drafted {worst_plain:.6}, the prefill route {worst_route:.6}, a chain-built row {nearest_chain:.6}");
        // Two correct routes to the same rows set the tolerance, as for
        // DFlash2's window (dflash2_round_gpu.rs): the drafted rows stand
        // within twice the prefill's distance from the plain rounds' ...
        assert!(
            worst_plain <= 2.0 * worst_route,
            "the drift to plain rounds {worst_plain} is past twice the prefill route's {worst_route}"
        );
        // ... and a chain-built row far past them.
        assert!(
            worst_plain < nearest_chain / 2.0,
            "the drift to plain rounds {worst_plain} is not far below a chain-built row's {nearest_chain}"
        );
    }

    // --- a sampled lane: a fixed seed reproduces it, its row does not matter
    {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        let pool = head_pool(KvFormat::Bf16, 4, MAX_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let wide = SamplingParams {
            temperature: 1.2,
            top_k: 0,
            top_p: 1.0,
            presence_penalty: 0.1,
            frequency_penalty: 0.1,
            seed: 4242,
        };
        let mixed: Vec<SamplingParams> = vec![wide, SamplingParams::greedy(), wide, SamplingParams { seed: 7, ..wide }];
        let together = spec_on(&model, &pool, &canaries, &mixed, &stop_ids, TOTAL, WINDOW, Some(true));
        let again = spec_on(&model, &pool, &canaries, &mixed, &stop_ids, TOTAL, WINDOW, Some(true));
        assert_eq!(together.emitted, again.emitted, "a fixed seed must reproduce the sampled lanes' streams");
        let greedy_run = spec_on(&model, &pool, &canaries, &greedy(4), &stop_ids, TOTAL, WINDOW, Some(true));
        assert_ne!(together.emitted[0], greedy_run.emitted[0], "the wide support never reached the sampler");
        let reversed_prompts: Vec<Vec<i32>> = canaries.iter().rev().cloned().collect();
        let reversed_sampling: Vec<SamplingParams> = mixed.iter().rev().copied().collect();
        let reversed =
            spec_on(&model, &pool, &reversed_prompts, &reversed_sampling, &stop_ids, TOTAL, WINDOW, Some(true));
        let unreversed: Vec<Vec<i32>> = reversed.emitted.into_iter().rev().collect();
        assert_eq!(unreversed, together.emitted, "a lane's sampled stream changed with the row it occupied");
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn every_window_and_lane_count_commits_the_anchor_and_the_drafts_the_target_accepts() {
    const WINDOW_TOTAL: usize = 24;
    let Some((reader, handles, on_device)) = materialized(ARTIFACT) else {
        return;
    };
    let artifact = &on_device.artifact;
    let (canaries, stop_ids) = chat_canaries(&reader);
    let prompts8: Vec<Vec<i32>> = canaries.iter().chain(&canaries).cloned().collect();
    let text_len = text_len(&reader);
    for window in 1..=WINDOW {
        let runs: Vec<SpecRun> = {
            let model = load_head(&reader, artifact, &handles, window, KvFormat::Bf16, MAX_CONTEXT);
            let pool = head_pool(KvFormat::Bf16, 8, MAX_CONTEXT);
            let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
            assert_eq!(stats.verify_graph_ready_mask, 0xFF, "window {window}: every width's verify graph");
            (1..=8usize)
                .map(|width| {
                    spec_on(&model, &pool, &prompts8[..width], &greedy(width), &stop_ids, WINDOW_TOTAL, window, Some(true))
                })
                .collect()
        };
        // The verify substrate alone, given the same drafts, commits the same
        // runs at the same extents: each round committed its anchor and the
        // drafts the target accepts.
        let substrate = load_substrate(&reader, artifact, &handles[..text_len], window);
        let pool = substrate_pool(8);
        let _ = capture_decode_graphs(&substrate, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let mut accepted = 0usize;
        let mut full = (0usize, 0usize);
        for (width, run) in (1..=8usize).zip(&runs) {
            let replayed = replay(&substrate, &pool, &prompts8[..width], &stop_ids, WINDOW_TOTAL, window, run);
            assert!(
                replayed == rounds_of(run),
                "window {window}, {width} lanes: the substrate given the head's drafts committed other runs"
            );
            for &(_, extent, committed) in run.rounds.iter().flatten() {
                accepted += committed - 1;
                if extent == window {
                    full.0 += committed;
                    full.1 += 1;
                }
            }
        }
        println!(
            "window {window}: {accepted} drafts accepted, {:.2} committed tokens per full-window round over {} rounds",
            full.0 as f64 / full.1.max(1) as f64,
            full.1
        );
        assert!(accepted > 0, "window {window}: the head never landed a draft");
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn hq_rounds_clear_the_heads_ring_bits_of_every_chain_row_past_the_commit() {
    const CONTEXT: u32 = 1024;
    let Some((reader, handles, on_device)) = materialized(ARTIFACT) else {
        return;
    };
    let artifact = &on_device.artifact;
    let encode = encoder(&reader);
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect();
    let long = |topic: &str| {
        format!(
            "{}Now answer in one sentence: {topic}",
            "A paged KV cache keeps each sequence's keys and values in fixed-size pages, and a block \
             table maps every logical page to a physical one, so many sequences share one pool. "
                .repeat(16)
        )
    };
    // Eight lanes, the widest round: two canaries and two prompts longer than
    // the ring, twice.
    let prompts: Vec<Vec<i32>> = [
        turn(&encode, CANARY_PROMPTS[0]),
        turn(&encode, CANARY_PROMPTS[2]),
        turn(&encode, &long("what does the block table map?")),
        turn(&encode, &long("why do pages make many sequences fit?")),
    ]
    .iter()
    .cycle()
    .take(8)
    .cloned()
    .collect();
    assert!(prompts[2].len() > 544 && prompts[3].len() > 544, "the long prompts must wrap the ring past the sinks");
    let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::HqE8_2b, CONTEXT);
    let width = prompts.len();
    let pool = head_pool(KvFormat::HqE8_2b, width as u32, CONTEXT);
    let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
    assert_eq!(stats.verify_graph_ready_mask, 0xFF, "hq-e8-2b: every width's verify graph, the chain in it");
    let mut lanes = alloc_prefilled(&model, &pool, &prompts, CONTEXT);
    // The target's ring: each chunk appended as the fill kernel appends it.
    // The head's: every row the prefill wrote, the prompt's rows but its last.
    let mut target_rings: Vec<HqRing> = prompts
        .iter()
        .map(|p| {
            let chunks: Vec<(u64, u64)> = (0..p.len() as u64)
                .step_by(CHUNK as usize)
                .map(|start| (start, u64::from(CHUNK).min(p.len() as u64 - start)))
                .collect();
            ring_after_prefill(&chunks)
        })
        .collect();
    let mut head_rings: Vec<HqRing> = prompts
        .iter()
        .map(|p| {
            let mut ring = HqRing::new();
            ring.append_decode(0, p.len() as u64 - 1);
            ring
        })
        .collect();
    let check = |lanes: &[Seq<'_>], target: &[HqRing], head: &[HqRing], what: &str| {
        for (i, lane) in lanes.iter().enumerate() {
            let (target_words, head_words) = mtp_readout::ring_words(&pool, lane).unwrap_or_else(|e| panic!("ring words: {e}"));
            assert_eq!(target_words, target[i].words(), "lane {i}, {what}: the target's ring words are the rule's");
            assert_eq!(head_words, head[i].words(), "lane {i}, {what}: the head's ring words are the rule's");
        }
    };
    check(&lanes, &target_rings, &head_rings, "after the prefill");

    let mut frontier: Vec<u64> = prompts.iter().map(|p| p.len() as u64).collect();
    let mut emitted = vec![0usize; width];
    let mut done = vec![false; width];
    let (mut rounds, mut cleared, mut cleared_wrapped) = (0usize, 0u64, 0u64);
    while done.iter().any(|d| !d) {
        let active: Vec<usize> = (0..width).filter(|&i| !done[i]).collect();
        let budgets: Vec<u32> = active.iter().map(|&i| (TOTAL - emitted[i]) as u32).collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = lanes.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
        let results = round(&model, &pool, &mut refs, &greedy(active.len()), &budgets, &stop_ids, WINDOW);
        for (&i, result) in active.iter().zip(results) {
            let (p, extent, kept) = (frontier[i], u64::from(result.extent), result.tokens.len() as u64);
            target_rings[i].append_decode(p, extent + 1);
            target_rings[i].invalidate(p + kept, p + extent + 1);
            // The head: its chain's rows, the commit's, the chain's past the
            // commit cleared.
            head_rings[i].append_decode(p - 1, extent);
            head_rings[i].append_decode(p - 1, kept);
            head_rings[i].invalidate(p - 1 + kept, p - 1 + extent);
            let past = (extent + 1).saturating_sub(kept + 1);
            cleared += past;
            if p >= 512 + 32 {
                cleared_wrapped += past;
            }
            frontier[i] += kept;
            done[i] = result.tokens.iter().any(|t| stop_ids.contains(t));
            emitted[i] += result.tokens.len();
            done[i] |= emitted[i] >= TOTAL;
        }
        check(&lanes, &target_rings, &head_rings, &format!("round {rounds}"));
        rounds += 1;
        assert!(rounds <= TOTAL, "the run does not converge");
    }
    println!("hq: {rounds} rounds over {width} lanes, {cleared} chain rows past the commit cleared, {cleared_wrapped} on a wrapped ring");
    assert!(cleared_wrapped > 0, "no chain row past a commit on a wrapped ring: the clear this test exists for never ran");
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn on_the_16gb_artifact_the_head_drafts_and_the_substrate_commits_the_same_runs() {
    const RUN_TOTAL: usize = 24;
    let Some((reader, handles, on_device)) = materialized(TRELLIS) else {
        return;
    };
    let artifact = &on_device.artifact;
    let text_len = text_len(&reader);
    let (canaries, stop_ids) = chat_canaries(&reader);
    let prompts8: Vec<Vec<i32>> = canaries.iter().chain(&canaries).cloned().collect();
    // Its chain scores with the trellis output head, inside every width's
    // verify graph.
    let runs: Vec<SpecRun> = {
        let model = load_head(&reader, artifact, &handles, WINDOW, KvFormat::Bf16, MAX_CONTEXT);
        let pool = head_pool(KvFormat::Bf16, 8, MAX_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.verify_graph_ready_mask, 0xFF, "16 GB artifact: every width's verify graph, the chain in it");
        (1..=8usize)
            .map(|width| spec_on(&model, &pool, &prompts8[..width], &greedy(width), &stop_ids, RUN_TOTAL, WINDOW, Some(true)))
            .collect()
    };
    let substrate = load_substrate(&reader, artifact, &handles[..text_len], WINDOW);
    let pool = substrate_pool(8);
    let _ = capture_decode_graphs(&substrate, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let mut accepted = 0usize;
    let mut full = (0usize, 0usize);
    for (width, run) in (1..=8usize).zip(&runs) {
        let replayed = replay(&substrate, &pool, &prompts8[..width], &stop_ids, RUN_TOTAL, WINDOW, run);
        assert!(
            replayed == rounds_of(run),
            "16 GB artifact, {width} lanes: the substrate given the head's drafts committed other runs"
        );
        for &(_, extent, committed) in run.rounds.iter().flatten() {
            accepted += committed - 1;
            if extent == WINDOW {
                full.0 += committed;
                full.1 += 1;
            }
        }
    }
    println!(
        "16 GB artifact: {accepted} drafts accepted, {:.2} committed tokens per full-window round over {} rounds",
        full.0 as f64 / full.1.max(1) as f64,
        full.1
    );
    assert!(accepted > 0, "16 GB artifact: the head never landed a draft");
}

// --- the head against the reference engine's (GitHub #625) ---------------

/// The prompts of the comparison with the reference engine, by id: the four
/// canaries and twelve more (code, prose, lists, arithmetic, a translation).
const ACCEPTANCE_PROMPTS: [(&str, &str); 16] = [
    ("rust-hello", CANARY_PROMPTS[0]),
    ("rust-sort", CANARY_PROMPTS[1]),
    ("math-greedy", CANARY_PROMPTS[2]),
    ("explain-reverse", CANARY_PROMPTS[3]),
    ("py-fib", "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and two doctest examples."),
    ("rust-struct", "Write a Rust struct `Stack<T>` backed by a `Vec<T>` with `push`, `pop` and `peek` methods, and a short test module."),
    (
        "sql-join",
        "Given tables `orders(id, customer_id, total)` and `customers(id, name)`, write a SQL query that lists each customer's \
         name and the sum of their order totals, highest first, and explain it briefly.",
    ),
    ("json-convert", "Convert this list into a JSON array of objects with fields name and age: Alice 31, Bob 27, Carol 45, Dave 19."),
    ("prose-story", "Write a short story of about 150 words about a lighthouse keeper who finds a message in a bottle."),
    ("prose-explain", "Explain to a high-school student how a refrigerator moves heat from inside to outside."),
    ("history", "Summarize the main causes of the French Revolution in five bullet points."),
    (
        "translate",
        "Translate into Italian: The meeting has been moved to Thursday afternoon because the room is not available on Wednesday.",
    ),
    ("math-word", "A train leaves at 9:40 and travels 210 km at 84 km/h. At what time does it arrive? Show your reasoning."),
    ("list-tips", "Give eight practical tips for writing clear commit messages."),
    ("regex", "Write a regular expression that matches ISO 8601 dates like 2024-03-17 and explain each part of it."),
    (
        "bash-script",
        "Write a bash script that finds every .log file older than 7 days under /var/log/app, compresses it with gzip, and \
         prints how many files it compressed.",
    ),
];

/// The reference engine's runs of [`ACCEPTANCE_PROMPTS`] under each KV format,
/// in their order; its settings are the fixture's `about`.
const REFERENCE_ACCEPTANCE: &str = include_str!("fixtures/mtp_reference_acceptance.json");
/// The reference's pooled rounds and drafts accepted as its runs printed them:
/// a guard on the fixture's copy.
const REFERENCE_POOLED: [(&str, u32, u32); 2] = [("bf16", 634, 2033), ("hq-e8-2b", 654, 2069)];
/// The reference engine's run: max context 4096, at most 256 new tokens.
const ACCEPTANCE_CONTEXT: u32 = 4096;
const ACCEPTANCE_NEW_TOKENS: usize = 256;
/// How far ignis's pooled tokens a round may stand under the reference's.
const ACCEPTANCE_MARGIN: f64 = 0.05;

/// One engine's greedy run of one prompt.
struct EngineRun {
    /// The rounds that drafted (extent > 0), with their leading-run counters.
    counters: SpecCounters,
    /// The generated text, the prefill's token first, through the first stop id.
    generated: Vec<i32>,
}

fn tokens_a_round(counters: &SpecCounters) -> f64 {
    1.0 + f64::from(counters.accepted) / f64::from(counters.rounds.max(1))
}

/// Accepted over drafted at position `j + 1`, the leading-run definition.
fn rate(counters: &SpecCounters, j: usize) -> f64 {
    f64::from(counters.accepted_at[j]) / f64::from(counters.drafted_at[j].max(1))
}

fn by_position(counters: &SpecCounters) -> String {
    (0..WINDOW as usize).map(|j| format!("{}/{}", counters.accepted_at[j], counters.drafted_at[j])).collect::<Vec<_>>().join(" ")
}

/// The reference engine's runs under `kv` (the fixture's key): each prompt's
/// id, its prompt tokens and its run.
fn reference_runs(kv: &str) -> Vec<(String, usize, EngineRun)> {
    let fixture: serde_json::Value = serde_json::from_str(REFERENCE_ACCEPTANCE).expect("the reference fixture parses");
    let rows = fixture["runs"][kv].as_array().unwrap_or_else(|| panic!("the fixture has no run under {kv}"));
    rows.iter()
        .map(|row| {
            let numbers = |key: &str| -> Vec<i64> {
                row[key].as_array().unwrap_or_else(|| panic!("{key}")).iter().map(|x| x.as_i64().expect(key)).collect()
            };
            let counts = |key: &str| -> Vec<u32> { numbers(key).into_iter().map(|x| x as u32).collect() };
            let (accepted_at, drafted_at) = (counts("accepted_per_position"), counts("drafted_per_position"));
            let mut counters = SpecCounters {
                rounds: drafted_at[0],
                drafted: drafted_at.iter().sum(),
                accepted: accepted_at.iter().sum(),
                ..SpecCounters::default()
            };
            counters.accepted_at.copy_from_slice(&accepted_at);
            counters.drafted_at.copy_from_slice(&drafted_at);
            let generated = numbers("generated_ids").into_iter().map(|x| x as i32).collect();
            let id = row["id"].as_str().expect("id").to_string();
            let prompt_tokens = row["prompt_tokens"].as_u64().expect("prompt_tokens") as usize;
            (id, prompt_tokens, EngineRun { counters, generated })
        })
        .collect()
}

/// ignis's greedy run of `prompt` at one lane, run and counted as the
/// reference engine runs and counts its own. Each round is given what is left
/// of the generation's budget, so its extent is the reference's, `min(7,
/// remaining - 1)`, and no stop ids, so it commits the whole run the target
/// accepts, past a stop too, as the reference's round licenses and counts it;
/// the run ends after the round whose generated tokens (its drafts committed
/// and the target's next token) hold a stop id, or at the cap. A round at
/// extent 0 is a plain step, not a round. Returns the run, the counters with
/// each round's run cut at its stop (as a round given the stop ids commits
/// and reports it), and the rounds at extent 0.
fn acceptance_run(model: &Model, pool: &SeqPool, prompt: &[i32], stop_ids: &[i32]) -> (EngineRun, SpecCounters, u32) {
    let mut seq = alloc_prefilled(model, pool, &[prompt.to_vec()], ACCEPTANCE_CONTEXT).remove(0);
    let mut generated = vec![seq.pending_token().expect("the prefill's token")];
    let (mut counters, mut cut_at, mut fallback_steps) = (SpecCounters::default(), SpecCounters::default(), 0u32);
    let stops = |t: &i32| stop_ids.contains(t);
    while !generated.iter().any(stops) && generated.len() < ACCEPTANCE_NEW_TOKENS {
        let remaining = (ACCEPTANCE_NEW_TOKENS - generated.len()) as u32;
        let run = round(model, pool, &mut [&mut seq], &greedy(1), &[remaining], &[], WINDOW).remove(0);
        assert_eq!(run.tokens.first(), generated.last(), "a run that does not open with its anchor");
        assert_eq!(run.extent, WINDOW.min(remaining - 1), "the leaf's extent is not the reference's");
        let accepted = run.tokens.len() as u32 - 1;
        if run.extent == 0 {
            fallback_steps += 1;
        } else {
            counters = counters + SpecCounters::round(run.extent, accepted);
            let served = run.tokens[1..].iter().position(stops).map_or(accepted, |k| k as u32 + 1);
            cut_at = cut_at + SpecCounters::round(run.extent, served);
        }
        generated.extend_from_slice(&run.tokens[1..]);
        generated.push(seq.pending_token().expect("the round's next token"));
        assert!(counters.rounds + fallback_steps <= ACCEPTANCE_NEW_TOKENS as u32, "the run does not converge");
    }
    cut_at_stop(&mut generated, stop_ids);
    (EngineRun { counters, generated }, cut_at, fallback_steps)
}

/// GitHub #625: a head read wrongly shows only as drafts the target refuses,
/// so the measure of the reading is the reference engine's acceptance with
/// the same head on the same prompts. Both engines: the NVFP4 artifact, one
/// lane, MTP at a fixed window of 7, the full proposal head, greedy, thinking
/// off and no system message, at most 256 new tokens, the same KV format;
/// ignis's loop guard is off, since the leaf has none. ignis's pooled tokens a
/// round may not stand under the reference's by more than 5%. A larger gap is
/// a defect in the reading of the head until shown otherwise: the test prints
/// the first position where ignis's rate falls under the reference's by more
/// than the margin, and for each prompt the first token where the texts part.
#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_heads_acceptance_is_within_five_percent_of_the_reference_engines_on_the_same_prompts() {
    for (kv, rounds, accepted) in REFERENCE_POOLED {
        let pooled = reference_runs(kv).into_iter().fold(SpecCounters::default(), |sum, (_, _, run)| sum + run.counters);
        assert_eq!((pooled.rounds, pooled.accepted), (rounds, accepted), "{kv}: the fixture is not the reference's runs");
    }
    let Some((reader, handles, on_device)) = materialized(ARTIFACT) else {
        return;
    };
    let artifact = &on_device.artifact;
    let encode = encoder(&reader);
    let (_, stop_ids) = chat_canaries(&reader);
    let prompts: Vec<Vec<i32>> = ACCEPTANCE_PROMPTS.iter().map(|(_, text)| turn(&encode, text)).collect();
    println!(
        "#625: one lane, MTP at a fixed window of {WINDOW}, the full proposal head, greedy, thinking off, at most \
         {ACCEPTANCE_NEW_TOKENS} new tokens, max context {ACCEPTANCE_CONTEXT}; ignis's loop guard off (the leaf has none)"
    );
    let mut verdicts = Vec::new();
    for (kv_format, kv) in [(KvFormat::Bf16, "bf16"), (KvFormat::HqE8_2b, "hq-e8-2b")] {
        let reference = reference_runs(kv);
        assert_eq!(reference.len(), ACCEPTANCE_PROMPTS.len(), "{kv}: a reference run per prompt");
        for ((id, _), (prompt, (reference_id, prompt_tokens, _))) in ACCEPTANCE_PROMPTS.iter().zip(prompts.iter().zip(&reference)) {
            assert_eq!(*id, reference_id.as_str(), "{kv}: the fixture's prompts are not in the test's order");
            assert_eq!(prompt.len(), *prompt_tokens, "{id}: the prompt is not the reference engine's");
        }
        let model = load_head(&reader, artifact, &handles, WINDOW, kv_format, ACCEPTANCE_CONTEXT);
        let pool = head_pool(kv_format, 1, ACCEPTANCE_CONTEXT);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let (mut ours, mut cut, mut theirs) = (SpecCounters::default(), SpecCounters::default(), SpecCounters::default());
        let (mut same_ours, mut same_theirs, mut same_prompts, mut fallback) =
            (SpecCounters::default(), SpecCounters::default(), 0usize, 0u32);
        for ((id, _), (prompt, (_, _, reference))) in ACCEPTANCE_PROMPTS.iter().zip(prompts.iter().zip(&reference)) {
            let (run, served, fallback_steps) = acceptance_run(&model, &pool, prompt, &stop_ids);
            let text = match first_divergence(&run.generated, &reference.generated) {
                None => {
                    same_ours = same_ours + run.counters;
                    same_theirs = same_theirs + reference.counters;
                    same_prompts += 1;
                    format!("the same {} generated tokens", run.generated.len())
                }
                Some(d) => format!(
                    "parts at generated token {d} (ignis {:?}, the reference {:?}; {} and {} generated)",
                    run.generated.get(d),
                    reference.generated.get(d),
                    run.generated.len(),
                    reference.generated.len()
                ),
            };
            println!(
                "{kv} {id:15} prompt {:2} | ignis {:3} rounds (+{fallback_steps} at extent 0), {:3} accepted, {:.3} tokens a \
                 round, accepted/drafted by position {} | the reference {:3} rounds, {:3} accepted, {:.3}, {} | {text}",
                prompt.len(),
                run.counters.rounds,
                run.counters.accepted,
                tokens_a_round(&run.counters),
                by_position(&run.counters),
                reference.counters.rounds,
                reference.counters.accepted,
                tokens_a_round(&reference.counters),
                by_position(&reference.counters),
            );
            ours = ours + run.counters;
            cut = cut + served;
            theirs = theirs + reference.counters;
            fallback += fallback_steps;
        }
        let (mine, reference_figure) = (tokens_a_round(&ours), tokens_a_round(&theirs));
        println!(
            "{kv} pooled over {} prompts: ignis {} rounds (+{fallback} at extent 0), {} accepted, {mine:.3} tokens a round \
             ({:.3} with each round's run cut at its stop, as the served counters count it) | the reference engine {} \
             rounds, {} accepted, {reference_figure:.3} tokens a round | ignis / reference {:.4}",
            ACCEPTANCE_PROMPTS.len(),
            ours.rounds,
            ours.accepted,
            tokens_a_round(&cut),
            theirs.rounds,
            theirs.accepted,
            mine / reference_figure
        );
        for j in 0..WINDOW as usize {
            println!(
                "{kv} position {}: ignis {}/{} = {:.3}, the reference {}/{} = {:.3}, ignis / reference {:.3}",
                j + 1,
                ours.accepted_at[j],
                ours.drafted_at[j],
                rate(&ours, j),
                theirs.accepted_at[j],
                theirs.drafted_at[j],
                rate(&theirs, j),
                rate(&ours, j) / rate(&theirs, j).max(f64::MIN_POSITIVE)
            );
        }
        match (0..WINDOW as usize).find(|&j| rate(&ours, j) < rate(&theirs, j) * (1.0 - ACCEPTANCE_MARGIN)) {
            Some(j) => println!("{kv}: the first position where ignis's rate is under the reference's by more than 5%: {}", j + 1),
            None => println!("{kv}: at no position is ignis's rate under the reference's by more than 5%"),
        }
        println!(
            "{kv}: the {same_prompts} prompt(s) whose text is the reference's end to end: ignis {:.3}, the reference {:.3} \
             tokens a round",
            tokens_a_round(&same_ours),
            tokens_a_round(&same_theirs)
        );
        verdicts.push((kv, mine, reference_figure));
    }
    for (kv, mine, reference_figure) in verdicts {
        assert!(
            mine >= reference_figure * (1.0 - ACCEPTANCE_MARGIN),
            "{kv}: ignis commits {mine:.3} tokens a round, under the reference engine's {reference_figure:.3} by more than \
             {:.0}%",
            100.0 * ACCEPTANCE_MARGIN
        );
    }
}
