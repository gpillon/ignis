//! GPU coverage for the instruments the replay of recorded agent turns stands
//! on under MTP (GitHub #626, spec #618), on both 27B bodies: the served NVFP4
//! one and the 16 GB artifact's trellis one, whose output head the chain
//! scores with is trellis. Neither seam is a serving path; the replay
//! (`crates/runtime/examples/mtp_replay.rs`) stands on both, as the DFlash2
//! replay stands on its own (`dflash2_replay_seams_gpu.rs`, unchanged).
//!
//! - **The chain readout** (`mtp_readout::chain`): the drafts it reads at a
//!   sequence's state are the drafts the next verify round from that state
//!   proposes -- a teacher-forced round whose recording is the read drafts
//!   takes every one of them, and a live round accepts exactly a prefix of
//!   them -- each with the head's probability of it and its 16 leading
//!   candidates, consistent with the step's scores. It writes nothing a later
//!   step reads: a sequence read before every round generates bit for bit what
//!   one never read generates, a second read gives back the first's scores bit
//!   for bit, and the round's logits are the round's still. Under hq-e8-2b,
//!   past the residual window's first lap (where a chain's appends land on
//!   ring rows older keys inside the window still name), the head's ring
//!   words after a read are the words before it. It refuses a load without
//!   the head, a pool of another drafter, a window not the load's, an extent
//!   of 0 or past the window and a sequence with no frontier.
//! - **The teacher-forced round on an MTP load**
//!   (`step::decode_program_verify_forced`): forced with the text the engine's
//!   own rounds commit, it is those rounds bit for bit -- the same runs, the
//!   same extents, the same pending token, and the same chain read after them,
//!   drafts and scores -- at widths 1 and 2, under BF16 KV and hq-e8-2b;
//!   forced with any other text it commits that text, each round accepting
//!   the read drafts' leading agreement with it. Only greedy lanes are forced:
//!   an MTP load drafts under the fixed draft only, so a sampling lane is
//!   refused, as is a forced run of the wrong length and a successor outside
//!   the vocabulary, the sequence unmoved.
//!
//! The state "bit for bit" is held to: the readout leaves the head's rows past
//! its frontier, which a snapshot carries, so the comparison is the runs, the pending token and
//! the chain read at the end, every draft and every score -- the
//! continuation hidden, the head's rows and the target's state are all in
//! it.
//!
//! BF16 KV is the correctness oracle (ADR 0022); hq-e8-2b is what the replay
//! loads. Explicit GPU profile (ADR 0006, GitHub #38): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact, GPU, or kernel error is a skip;
//! under the profile it is a hard failure. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, body_27b, materialize, model_scope_27b_of, Body27b, CudaDevice, DraftModule, FrontendSet, ModelScope,
    Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::mtp_readout::{self, MtpChain, TOP_K};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_forced, decode_program_verify_runs, prefill_program_sampled,
    LaneVerifyRun, SamplingParams, VerifyLane,
};
use ignis_core::{KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation, SpeculativeBackend};

const NVFP4: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const TRELLIS: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_trellis_sc30-v2.ninfer";
/// Room for the long prompts below and a run after them.
const MAX_CONTEXT: u32 = 1024;
/// The 27B MTP load's chunk: a multiple of 128 (`mtp_head::PREFILL_ROWS`).
const CHUNK: u32 = 256;
const WINDOW: u32 = 7;
const LANES: u32 = 2;
/// Tokens each lane emits: several full windows.
const TOTAL: usize = 48;
/// hq-e8-2b's sink and recent ring (`hq_ring`): keys past this many are
/// appended over ring rows older keys held.
const HQ_FIRST_LAP: usize = 32 + 512;

const PROMPTS: [&str; 3] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "List three prime numbers larger than ten, separated by commas, and say why each is prime.",
];

fn head_pool(kv_format: KvFormat, slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        Some(SpeculativeBackend::Mtp),
    )
    .unwrap_or_else(|e| panic!("MTP pool create: {e}"))
}

fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, prompt: &[i32]) {
    prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
}

/// `text[at..at + WINDOW + 1]`, padded with -1 past its end.
fn forced_slice(text: &[i32], at: usize) -> Vec<i32> {
    (0..=WINDOW as usize).map(|j| text.get(at + j).copied().unwrap_or(-1)).collect()
}

fn leading_agreement(drafts: &[i32], forced: &[i32]) -> usize {
    drafts.iter().zip(forced).take_while(|(d, f)| d == f).count()
}

/// `run` cut at its first stop id, inclusive.
fn cut_at_stop(run: &[i32], stop_ids: &[i32]) -> Vec<i32> {
    let end = run.iter().position(|t| stop_ids.contains(t)).map_or(run.len(), |p| p + 1);
    run[..end].to_vec()
}

/// A step's own consistency: its candidates are its scores' best 16,
/// descending, the draft first, each with its share of the softmax over the
/// whole row (recomputed here in f64 from the scores the readout returned).
fn check_chain(chain: &MtpChain, extent: u32, label: &str) {
    assert_eq!(chain.steps.len(), extent as usize, "{label}: one step a draft position");
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let scores = chain.scores.as_ref().expect("asked for the scores");
    for (s, step) in chain.steps.iter().enumerate() {
        let row = &scores[s * vocab..(s + 1) * vocab];
        let value = |id: i32| f64::from(f32::from_bits(u32::from(row[id as usize]) << 16));
        let max = (0..vocab as i32).map(value).fold(f64::NEG_INFINITY, f64::max);
        let partition: f64 = (0..vocab as i32).map(|id| (value(id) - max).exp()).sum();
        assert_eq!(step.candidate_ids.len(), TOP_K, "{label}: step {s}'s candidates");
        assert_eq!(step.candidate_ids[0], step.draft, "{label}: step {s}'s draft is not its first candidate");
        assert_eq!(value(step.draft), max, "{label}: step {s}'s draft is not its best score");
        for (i, (&id, &p)) in step.candidate_ids.iter().zip(&step.candidate_probabilities).enumerate() {
            let expected = (value(id) - max).exp() / partition;
            assert!((f64::from(p) - expected).abs() <= 1e-6 * expected.max(1e-3), "{label}: step {s} candidate {i}: {p} against {expected}");
            if i > 0 {
                assert!(value(id) <= value(step.candidate_ids[i - 1]), "{label}: step {s}'s candidates are not descending");
            }
        }
        let floor = value(step.candidate_ids[TOP_K - 1]);
        let above = (0..vocab as i32).filter(|&id| value(id) > floor && !step.candidate_ids.contains(&id)).count();
        assert_eq!(above, 0, "{label}: step {s} has {above} tokens above its 16th candidate");
        assert_eq!(step.probability, step.candidate_probabilities[0], "{label}: step {s}'s probability");
    }
}

/// Each lane's verify runs, the token its last round left pending, and the
/// chain read at its end (every draft and every score).
#[derive(Debug, PartialEq)]
struct Lanes {
    runs: Vec<Vec<LaneVerifyRun>>,
    pending: Vec<i32>,
    end: Vec<MtpChain>,
}

fn read(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, extent: u32) -> MtpChain {
    mtp_readout::chain(model, pool, seq, extent, WINDOW, true).unwrap_or_else(|e| panic!("chain readout: {e}"))
}

fn finish(model: &Model, pool: &SeqPool, seqs: &mut [Seq<'_>], runs: Vec<Vec<LaneVerifyRun>>) -> Lanes {
    let pending = seqs.iter().map(|seq| seq.pending_token().expect("a lane that ran a round has a pending token")).collect();
    let end = seqs.iter_mut().map(|seq| read(model, pool, seq, WINDOW)).collect();
    Lanes { runs, pending, end }
}

/// One body's claims, at one KV format, over `prompts`.
struct Bench<'a> {
    model: &'a Model,
    kv_format: KvFormat,
    prompts: &'a [Vec<i32>],
    stop_ids: &'a [i32],
}

impl Bench<'_> {
    fn alloc_prefilled<'p>(&self, pool: &'p SeqPool, lanes: &[usize]) -> Vec<Seq<'p>> {
        lanes
            .iter()
            .map(|&lane| {
                let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
                prefill(self.model, pool, &mut seq, &self.prompts[lane]);
                seq
            })
            .collect()
    }

    /// The engine's own rounds over `lanes`, `TOTAL` tokens a lane, each lane
    /// budgeted to what it has left. With `reads`, the head's chain is read
    /// before every round (twice, the second read the first's bit for bit)
    /// and each round's accepted drafts are the read ones.
    fn generate(&self, pool: &SeqPool, lanes: &[usize], reads: bool) -> Lanes {
        let model = self.model;
        let mut seqs = self.alloc_prefilled(pool, lanes);
        let mut emitted = vec![0usize; lanes.len()];
        let mut done = vec![false; lanes.len()];
        let mut per_lane: Vec<Vec<LaneVerifyRun>> = vec![Vec::new(); lanes.len()];
        while done.iter().any(|d| !d) {
            let active: Vec<usize> = (0..lanes.len()).filter(|&i| !done[i]).collect();
            let mut chains: Vec<Option<MtpChain>> = Vec::new();
            for &i in &active {
                // The round's extent: its budget leaves room for this many drafts.
                let extent = WINDOW.min((TOTAL - emitted[i]) as u32 - 1);
                chains.push((reads && extent > 0).then(|| {
                    let rings = (self.kv_format == KvFormat::HqE8_2b)
                        .then(|| mtp_readout::ring_words(pool, &seqs[i]).unwrap_or_else(|e| panic!("ring words: {e}")));
                    let first = read(model, pool, &mut seqs[i], extent);
                    let second = read(model, pool, &mut seqs[i], extent);
                    assert!(first == second, "lane {i}: a second read is not the first's");
                    if let Some(before) = rings {
                        let after = mtp_readout::ring_words(pool, &seqs[i]).unwrap_or_else(|e| panic!("ring words: {e}"));
                        assert_eq!(before, after, "lane {i}: the read moved the ring words");
                    }
                    check_chain(&first, extent, &format!("lane {i}"));
                    first
                }));
            }
            let mut remaining: Vec<Option<&mut Seq<'_>>> = seqs.iter_mut().map(Some).collect();
            let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
            let lanes_now: Vec<VerifyLane<'_>> = active
                .iter()
                .map(|&i| VerifyLane { remaining_tokens: (TOTAL - emitted[i]) as u32, stop_ids: self.stop_ids, ..VerifyLane::greedy(&[]) })
                .collect();
            let runs = decode_program_verify_runs(model, pool, &mut refs, &lanes_now, WINDOW).unwrap_or_else(|e| panic!("round: {e}"));
            for ((&i, run), chain) in active.iter().zip(runs).zip(&chains) {
                if let Some(chain) = chain {
                    // A greedy round accepts a draft only as the token it
                    // commits there, so the accepted drafts are read ones.
                    let accepted = run.tokens.len() - 1;
                    assert_eq!(&run.tokens[1..], &chain.drafts()[..accepted], "lane {i}: the round's accepted drafts");
                }
                emitted[i] += run.tokens.len();
                done[i] = emitted[i] >= TOTAL || run.tokens.iter().any(|t| self.stop_ids.contains(t));
                per_lane[i].push(run);
            }
        }
        finish(model, pool, &mut seqs, per_lane)
    }

    /// The same lanes through forced rounds over `recorded` -- each lane's
    /// text, then the successor its live rounds left pending -- prefilled as
    /// `generate` prefilled them. A text that ends at its budget has no stop
    /// to end it, so its last forced round takes the successor in place of
    /// the pad, as the live round drew one.
    fn replay(&self, pool: &SeqPool, lanes: &[usize], recorded: &[Vec<i32>]) -> Lanes {
        let model = self.model;
        let mut seqs = self.alloc_prefilled(pool, lanes);
        let mut at = vec![0usize; lanes.len()];
        let mut forced_runs: Vec<Vec<LaneVerifyRun>> = vec![Vec::new(); lanes.len()];
        let unwalked = |i: usize, at: &[usize]| at[i] + 1 < recorded[i].len();
        while (0..lanes.len()).any(|i| unwalked(i, &at)) {
            let active: Vec<usize> = (0..lanes.len()).filter(|&i| unwalked(i, &at)).collect();
            let forced: Vec<Vec<i32>> = active.iter().map(|&i| forced_slice(&recorded[i], at[i] + 1)).collect();
            let forced_refs: Vec<&[i32]> = forced.iter().map(Vec::as_slice).collect();
            let mut remaining: Vec<Option<&mut Seq<'_>>> = seqs.iter_mut().map(Some).collect();
            let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
            let lanes_now: Vec<VerifyLane<'_>> = active
                .iter()
                .map(|&i| VerifyLane { remaining_tokens: (TOTAL - at[i]) as u32, stop_ids: self.stop_ids, ..VerifyLane::greedy(&[]) })
                .collect();
            let runs = decode_program_verify_forced(model, pool, &mut refs, &lanes_now, WINDOW, &forced_refs)
                .unwrap_or_else(|e| panic!("forced round: {e}"));
            for (&i, run) in active.iter().zip(runs) {
                at[i] += run.tokens.len();
                forced_runs[i].push(run);
            }
        }
        finish(model, pool, &mut seqs, forced_runs)
    }

    /// From each lane's state after two live rounds: a forced round whose
    /// recording is the lane's read drafts takes every one of them -- the
    /// read drafts are the drafts that round proposes, not only the first.
    fn forced_with_the_read_drafts(&self, pool: &SeqPool, lanes: &[usize]) {
        let model = self.model;
        let mut seqs = self.alloc_prefilled(pool, lanes);
        for _ in 0..2 {
            let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
            let lanes_now = vec![VerifyLane { stop_ids: self.stop_ids, ..VerifyLane::greedy(&[]) }; lanes.len()];
            decode_program_verify_runs(model, pool, &mut refs, &lanes_now, WINDOW).unwrap_or_else(|e| panic!("round: {e}"));
        }
        let anchors: Vec<i32> = seqs.iter().map(|seq| seq.pending_token().unwrap()).collect();
        let drafts: Vec<Vec<i32>> = seqs.iter_mut().map(|seq| read(model, pool, seq, WINDOW).drafts()).collect();
        // The recorded token after the drafts: any vocabulary id.
        let forced: Vec<Vec<i32>> = drafts.iter().map(|d| d.iter().copied().chain([d[0]]).collect()).collect();
        let forced_refs: Vec<&[i32]> = forced.iter().map(Vec::as_slice).collect();
        let mut refs: Vec<&mut Seq<'_>> = seqs.iter_mut().collect();
        let lanes_now = vec![VerifyLane { stop_ids: self.stop_ids, ..VerifyLane::greedy(&[]) }; lanes.len()];
        let runs = decode_program_verify_forced(model, pool, &mut refs, &lanes_now, WINDOW, &forced_refs)
            .unwrap_or_else(|e| panic!("forced round: {e}"));
        for (i, run) in runs.iter().enumerate() {
            let expected = cut_at_stop(&std::iter::once(anchors[i]).chain(drafts[i].iter().copied()).collect::<Vec<_>>(), self.stop_ids);
            assert_eq!(run.extent, WINDOW, "lane {i}: the round's extent");
            assert_eq!(run.tokens, expected, "lane {i}: forced with its read drafts, the round did not take them all");
        }
    }

    /// Widths 1 and 2: reading changes nothing, forced rounds fed a run's own
    /// text are that run's rounds bit for bit, and a recording of the read
    /// drafts is taken whole.
    fn rounds(&self) {
        for lanes in [&[0usize][..], &[1, 0][..]] {
            let label = format!("{:?}, width {}", self.kv_format, lanes.len());
            let pool = head_pool(self.kv_format, lanes.len() as u32);
            let capture = capture_decode_graphs(self.model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            assert!(capture.is_ready(lanes.len() as u32), "{label}: no verify graph at this width");
            let plain = self.generate(&pool, lanes, false);
            let read = self.generate(&pool, lanes, true);
            assert!(plain == read, "{label}: reading the head's chain changed the rounds");
            let recorded: Vec<Vec<i32>> = plain
                .runs
                .iter()
                .zip(&plain.pending)
                .map(|(runs, &after)| runs.iter().flat_map(|r| r.tokens.clone()).chain([after]).collect())
                .collect();
            let forced = self.replay(&pool, lanes, &recorded);
            assert!(forced == plain, "{label}: forced with its own text, the rounds are not the live rounds");
            self.forced_with_the_read_drafts(&pool, lanes);
            let committed: usize = recorded.iter().map(|text| text.len() - 1).sum();
            let rounds: usize = plain.runs.iter().map(Vec::len).sum();
            let first_probability: Vec<String> =
                plain.end.iter().map(|chain| format!("{:.3}", chain.steps[0].probability)).collect();
            println!(
                "{label}: {committed} tokens in {rounds} rounds, {:.2} per round; the end chains' first-draft probabilities {}",
                committed as f64 / rounds as f64,
                first_probability.join(", ")
            );
        }
    }

    /// The forced round fed another text, the readout's refusals and the
    /// forced round's, at width 1 under BF16.
    fn other_text_and_refusals(&self) {
        let model = self.model;
        // Two slots: a fresh sequence beside the one under test, below.
        let pool = head_pool(self.kv_format, 2);
        let _ = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        // Prompt 0's anchor, then prompt 1's answer: text the target never
        // chose. The runs commit it exactly, each forced round accepting the
        // read drafts' leading agreement with it.
        let own = self.generate(&pool, &[0], false);
        let other = self.generate(&pool, &[1], false);
        let mut text = vec![own.runs[0][0].tokens[0]];
        text.extend(other.runs[0].iter().flat_map(|r| r.tokens.clone()).skip(1));
        let recorded: Vec<i32> = text.iter().copied().chain([other.pending[0]]).collect();
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
        prefill(model, &pool, &mut seq, &self.prompts[0]);
        let (mut at, mut committed, mut accepted_any) = (0usize, Vec::new(), false);
        while at < text.len() {
            let remaining = (text.len() - at) as u32;
            let extent = WINDOW.min(remaining - 1);
            let drafts = if extent > 0 { read(model, &pool, &mut seq, extent).drafts() } else { Vec::new() };
            let forced = forced_slice(&recorded, at + 1);
            let run = decode_program_verify_forced(
                model,
                &pool,
                &mut [&mut seq],
                &[VerifyLane { remaining_tokens: remaining, stop_ids: self.stop_ids, ..VerifyLane::greedy(&[]) }],
                WINDOW,
                &[&forced],
            )
            .unwrap_or_else(|e| panic!("forced round: {e}"))
            .remove(0);
            assert_eq!(run.tokens[0], text[at], "the round's anchor is the recorded token after the last run");
            assert_eq!(run.extent, extent, "the readout's extent is not the round's");
            let expected = leading_agreement(&drafts, &forced).min(extent as usize);
            let cut = run.tokens.iter().position(|t| self.stop_ids.contains(t)).map_or(usize::MAX, |p| p + 1);
            assert_eq!(run.tokens.len(), (expected + 1).min(cut), "the forced round's acceptance is the read drafts' agreement");
            accepted_any |= expected > 0;
            at += run.tokens.len();
            committed.extend(run.tokens);
        }
        assert_eq!(committed, text, "the forced rounds did not commit the forced text");
        assert!(accepted_any, "the head never agreed with the forced text: not a test of the agreement rule");
        drop(seq);

        // The chain leaves the round's logits as the round left them.
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
        prefill(model, &pool, &mut seq, &self.prompts[2]);
        decode_program_verify_runs(model, &pool, &mut [&mut seq], &[VerifyLane::greedy(&[])], WINDOW).unwrap();
        let logits = dflash2_readout::verify_logits(model, 1, WINDOW).unwrap_or_else(|e| panic!("verify logits: {e}"));
        let _ = read(model, &pool, &mut seq, WINDOW);
        let again = dflash2_readout::verify_logits(model, 1, WINDOW).unwrap_or_else(|e| panic!("verify logits: {e}"));
        assert!(logits == again, "the chain readout wrote the round's logits");

        // The readout's refusals: a window not the load's, an extent of 0 or
        // past the window, a sequence with no frontier, a pool of another
        // drafter.
        for (extent, window, what) in [(WINDOW - 1, WINDOW - 1, "not the load's draft window"), (0, WINDOW, "is not in 1.."), (WINDOW + 1, WINDOW, "is not in 1..")] {
            let err = mtp_readout::chain(model, &pool, &mut seq, extent, window, false).expect_err(what);
            assert!(err.contains(what), "extent {extent} at window {window}: the wrong refusal: {err}");
        }
        let mut fresh = pool.alloc(MAX_CONTEXT).unwrap();
        let err = mtp_readout::chain(model, &pool, &mut fresh, WINDOW, WINDOW, false).expect_err("no frontier");
        assert!(err.contains("no frontier"), "the wrong refusal: {err}");
        drop(fresh);
        let verify_only = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: self.kv_format,
                kv_page_group_count: MAX_CONTEXT.div_ceil(64),
                max_context_tokens: MAX_CONTEXT,
                slot_count: 1,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
            Some(SpeculativeBackend::VerifyOnly),
        )
        .unwrap_or_else(|e| panic!("verify-only pool create: {e}"));
        let mut foreign = verify_only.alloc(MAX_CONTEXT).unwrap();
        let err = mtp_readout::chain(model, &verify_only, &mut foreign, WINDOW, WINDOW, false).expect_err("another pool");
        assert!(err.contains("not built for the MTP head"), "the wrong refusal: {err}");
        drop(foreign);

        // The forced round's refusals, the sequence unmoved: a successor
        // outside the vocabulary (an id past it, the pad with no stop), a
        // sampling lane (an MTP load drafts under the fixed draft only), and
        // a forced run of the wrong length.
        let anchor = seq.pending_token().expect("a pending anchor");
        assert!(!self.stop_ids.contains(&anchor), "the refusals below need an anchor that is no stop id");
        let position = seq.stats().position;
        let vocab_id = ModelConfig::qwen38_27b().vocab as i32;
        let greedy = [VerifyLane::greedy(&[])];
        for (forced, what) in [(vec![vocab_id; WINDOW as usize + 1], "outside the vocabulary"), (vec![-1; WINDOW as usize + 1], "outside the vocabulary")] {
            let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &greedy, WINDOW, &[&forced]).expect_err(what);
            assert!(err.contains(what), "the wrong refusal: {err}");
            assert_eq!((seq.stats().position, seq.pending_token()), (position, Some(anchor)), "a refused round moved the sequence");
        }
        let sampled = VerifyLane {
            sampling: SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 626, ..SamplingParams::greedy() },
            ..VerifyLane::greedy(&[])
        };
        let forced = forced_slice(&text, 0);
        let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &[sampled], WINDOW, &[&forced])
            .expect_err("a sampling lane was forced");
        assert!(err.contains("greedy"), "the wrong refusal: {err}");
        let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &greedy, WINDOW, &[&forced[..3]])
            .expect_err("a short forced run was taken");
        assert!(err.contains("window + 1"), "the wrong refusal: {err}");
        assert_eq!((seq.stats().position, seq.pending_token()), (position, Some(anchor)), "a refused round moved the sequence");
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_replay_seams_read_the_heads_chain_and_force_the_recorded_text_on_the_nvfp4_body() {
    the_replay_seams_hold(NVFP4, Body27b::Nvfp4);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_replay_seams_read_the_heads_chain_and_force_the_recorded_text_on_the_trellis_body() {
    the_replay_seams_hold(TRELLIS, Body27b::Trellis);
}

fn the_replay_seams_hold(artifact_path: &str, body: Body27b) {
    let path = Path::new(artifact_path);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {artifact_path}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    assert_eq!(body_27b(&reader), body, "{artifact_path}: the body under test");
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    let turn = |text: &str| encode(&format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|m| encode(m)).collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    let short: Vec<Vec<i32>> = PROMPTS.iter().map(|p| turn(p)).collect();
    // Past the hq ring's first lap: a log the answer reads.
    let long: Vec<Vec<i32>> = ["the sum of each line's two values", "which lines repeat a value"]
        .iter()
        .enumerate()
        .map(|(n, question)| {
            let log: String = (0..70).map(|i| format!("line {i}: {} and {}\n", (i * 7 + n) % 13, (i * 5 + 3 * n) % 11)).collect();
            turn(&format!("Here is a log.\n{log}In two sentences, describe {question}."))
        })
        .collect();
    for prompt in &long {
        assert!(prompt.len() > HQ_FIRST_LAP + 16, "a long prompt of {} tokens is inside the ring's first lap", prompt.len());
    }

    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Mtp)).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let load = |kv_format: KvFormat| -> Model {
        load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            CHUNK,
            MAX_CONTEXT,
            kv_format,
            Some(Speculation::new(SpeculativeBackend::Mtp, WINDOW).unwrap()),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            LANES,
        )
        .unwrap_or_else(|e| panic!("model load ({kv_format:?}): {e}"))
    };
    {
        let model = load(KvFormat::Bf16);
        let bench = Bench { model: &model, kv_format: KvFormat::Bf16, prompts: &short, stop_ids: &stop_ids };
        bench.rounds();
        bench.other_text_and_refusals();
    }
    {
        let model = load(KvFormat::HqE8_2b);
        Bench { model: &model, kv_format: KvFormat::HqE8_2b, prompts: &long, stop_ids: &stop_ids }.rounds();
    }
    {
        // A load without the head refuses the readout: the verify substrate
        // alone, on the text scope's handles (the scope's first).
        let text = model_scope_27b_of(&reader, ModelScope::default()).unwrap_or_else(|e| panic!("text scope: {e}")).len();
        let spec = Speculation::new(SpeculativeBackend::VerifyOnly, WINDOW).unwrap();
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles[..text],
            CHUNK,
            MAX_CONTEXT,
            KvFormat::Bf16,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            1,
        )
        .unwrap_or_else(|e| panic!("model load (verify-only): {e}"));
        let pool = head_pool(KvFormat::Bf16, 1);
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
        let err = mtp_readout::chain(&model, &pool, &mut seq, WINDOW, WINDOW, false).expect_err("a load without the head");
        assert!(err.contains("not loaded with the 27B's MTP head"), "the wrong refusal: {err}");
    }
    // Two bodies in one process: give this one's weights back before the
    // next materializes (the arena has no Drop of its own).
    let _ = artifact.release_arena(&mut device);
}
