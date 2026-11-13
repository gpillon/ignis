//! GPU coverage for the acceptance replay's two measurement seams under the
//! **sampled draft** (GitHub #577, spec #574, ADR 0090), on both 27B bodies.
//! `dflash2_replay_seams_gpu.rs` holds them under the fixed draft, unedited;
//! this file holds what the sampled draft adds. Neither seam is a serving
//! path.
//!
//! - **The proposal readout** (`dflash2_readout::propose_sampled`): for a lane
//!   that samples, the drafts it reads are the drafts the lane's next round
//!   draws -- the round accepts exactly a prefix of them -- each with the
//!   distribution q it was drawn from, which is the host reference's
//!   (`sampled_draft::drafter_distribution`) over the read lattice after the
//!   draft before it, at the lane's temperature times the load's factor and
//!   the load's top-p; a greedy lane's are the walk's, q one-hot on them. And
//!   it writes nothing: a sequence read before every round generates bit for
//!   bit what one never read generates, and its snapshot does not move.
//! - **The teacher-forced round** (`step::decode_program_verify_forced`):
//!   forced with the text the engine's own rounds commit, greedy or sampled
//!   (a sampling lane forced as the same sampling lane, its seed the live
//!   one's), it is those rounds bit for bit -- the same runs, the same state
//!   after them -- at widths 1 and 2. Forced with another text, under a seed
//!   of its own, it commits that text: each run is the recorded tokens the
//!   forced verdict took, never one that is not among its column's
//!   candidates, then the recorded token after them -- and the verdict is the
//!   coin's, not the agreement with the drafts the lane would have drawn: some
//!   runs go past it. A recording whose pad follows no stop is refused before
//!   the pass, the sequence unmoved.
//!
//! Forced and live are compared at the same width only (`support/near_tie.rs`).
//! Each body loads at its profile's lanes, BF16 KV (ADR 0022), under the
//! sampled draft at its default drafter settings. Explicit GPU profile (ADR
//! 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1` a missing artifact, GPU,
//! or kernel error is a skip; under the profile it is a hard failure. Run via
//! `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, body_27b, materialize, Body27b, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout::{self, Dflash2Proposal, TOP_K};
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::sampled_draft::drafter_distribution;
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_forced, decode_program_verify_runs, prefill_program_sampled,
    LaneVerifyRun, SamplingParams, VerifyLane,
};
use ignis_core::{
    DraftRule, DrafterSampling, KvFormat, PrefillAttention, PrefillLinear, RopeScaling, Speculation, SpeculativeBackend,
};

const NVFP4: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MAX_CONTEXT: u32 = 256;
const PREFILL_CHUNK: u32 = 128;
const WINDOW: u32 = 7;
/// Tokens each lane emits: several full windows.
const TOTAL: usize = 64;

const PROMPTS: [&str; 3] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "List three prime numbers larger than ten, separated by commas, and say why each is prime.",
];

fn pool(slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
        Some(SpeculativeBackend::Dflash2),
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

/// The agents' thinking sampling at `seed`, or greedy.
fn sampling(sampled: bool, seed: u64) -> SamplingParams {
    match sampled {
        false => SamplingParams::greedy(),
        true => SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed, ..SamplingParams::greedy() },
    }
}

/// `text[at..at + WINDOW + 1]`, padded with -1 past its end.
fn forced_slice(text: &[i32], at: usize) -> Vec<i32> {
    (0..=WINDOW as usize).map(|j| text.get(at + j).copied().unwrap_or(-1)).collect()
}

/// How many leading forced tokens are among their column's candidates: the
/// most a forced round under the sampled draft can take.
fn leading_candidates(proposal: &Dflash2Proposal, forced: &[i32]) -> usize {
    (0..proposal.window()).take_while(|&s| proposal.candidates(s).contains(&forced[s])).count()
}

/// Each lane's verify runs, and the token its last round left pending.
type Lanes = (Vec<Vec<LaneVerifyRun>>, Vec<i32>);

fn pending(seqs: &[Seq<'_>]) -> Vec<i32> {
    seqs.iter().map(|seq| seq.pending_token().expect("a lane that ran a round has a pending token")).collect()
}

/// The readout's q against the host reference: for a sampling lane, each
/// column's row is the drafter's distribution over the read scores after the
/// draft read before it, and the draft holds some of it; for a greedy lane
/// the drafts are the walk's and q is one-hot on them.
fn check_q(proposal: &Dflash2Proposal, lane_sampling: SamplingParams, label: &str) {
    if !(lane_sampling.temperature > 0.0) {
        let path = proposal.walk();
        for (column, &index) in path.iter().enumerate() {
            assert_eq!(proposal.candidates(column)[index], proposal.drafts[column], "{label}: greedy draft {column} is not the walk's");
            let one_hot: Vec<f32> = (0..TOP_K).map(|i| if i == index { 1.0 } else { 0.0 }).collect();
            assert_eq!(proposal.q(column), &one_hot[..], "{label}: a greedy draft's q is not one-hot on it");
        }
        return;
    }
    let drafter = DrafterSampling::default();
    let temperature = lane_sampling.temperature * drafter.temperature_factor();
    let mut pred = 0usize;
    for column in 0..proposal.window() {
        let row = &proposal.scores[(column * TOP_K + pred) * TOP_K..][..TOP_K];
        let want = drafter_distribution(row, temperature, drafter.top_p());
        for (i, (&got, &want)) in proposal.q(column).iter().zip(&want).enumerate() {
            assert!((got - want).abs() <= 1e-5, "{label}: column {column} candidate {i}: q {got} against the host's {want}");
        }
        let index = proposal
            .candidates(column)
            .iter()
            .position(|&id| id == proposal.drafts[column])
            .unwrap_or_else(|| panic!("{label}: draft {column} is none of its column's candidates"));
        assert!(proposal.q(column)[index] > 0.0, "{label}: draft {column} was drawn where q holds nothing");
        pred = index;
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn under_the_sampled_draft_the_replay_seams_read_the_draws_and_force_the_recorded_text_on_the_nvfp4_body() {
    the_replay_seams_hold(NVFP4, Body27b::Nvfp4, 8);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn under_the_sampled_draft_the_replay_seams_read_the_draws_and_force_the_recorded_text_on_the_trellis_body() {
    the_replay_seams_hold(TRELLIS, Body27b::Trellis, 2);
}

fn the_replay_seams_hold(artifact_path: &str, body: Body27b, decode_lanes: u32) {
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
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|m| encode(m)).collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    let prompts: Vec<Vec<i32>> = PROMPTS
        .iter()
        .map(|p| encode(&format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")))
        .collect();

    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    {
        let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW)
            .unwrap()
            .with_draft_rule(DraftRule::Sampled, DrafterSampling::default());
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            PREFILL_CHUNK,
            MAX_CONTEXT,
            KvFormat::Bf16,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            decode_lanes,
        )
        .unwrap_or_else(|e| panic!("model load: {e}"));
        seams(&model, &prompts, &stop_ids);
    }
    // Two bodies in one process: give this one's weights back before the
    // next materializes (the arena has no Drop of its own).
    let _ = artifact.release_arena(&mut device);
}

fn seams(model: &Model, prompts: &[Vec<i32>], stop_ids: &[i32]) {
    // Lane `lane`'s sampling: greedy, or the agents' thinking sampling with
    // the lane's own seed.
    let sampling_of = |sampled: bool, lane: usize| sampling(sampled, 577 + lane as u64);

    // --- the engine's own rounds: the text and the runs ------------------
    // `readout`: read the drafter before every round, as the lane, and check
    // the reads.
    let generate = |pool: &SeqPool, lanes: &[usize], sampled: bool, readout: bool| -> Lanes {
        let mut seqs: Vec<Seq<'_>> = lanes.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap()).collect();
        for (seq, &lane) in seqs.iter_mut().zip(lanes) {
            prefill_program_sampled(model, pool, seq, &prompts[lane], 0, sampling_of(sampled, lane), None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
        }
        let mut emitted = vec![0usize; lanes.len()];
        let mut done = vec![false; lanes.len()];
        let mut per_lane: Vec<Vec<LaneVerifyRun>> = vec![Vec::new(); lanes.len()];
        while done.iter().any(|d| !d) {
            let active: Vec<usize> = (0..lanes.len()).filter(|&i| !done[i]).collect();
            let reads: Vec<Option<Dflash2Proposal>> = active
                .iter()
                .map(|&i| {
                    readout.then(|| {
                        let extent = WINDOW.min((TOTAL - emitted[i]) as u32 - 1);
                        let before = seqs[i].snapshot().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
                        let lane_sampling = sampling_of(sampled, lanes[i]);
                        let proposal =
                            dflash2_readout::propose_sampled(model, pool, &seqs[i], extent, WINDOW, false, lane_sampling)
                                .unwrap_or_else(|e| panic!("readout: {e}"));
                        let after = seqs[i].snapshot().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
                        assert!(before == after, "the readout moved the sequence's state");
                        check_q(&proposal, lane_sampling, &format!("lane {i}"));
                        proposal
                    })
                })
                .collect();
            let mut remaining: Vec<Option<&mut Seq<'_>>> = seqs.iter_mut().map(Some).collect();
            let mut refs: Vec<&mut Seq<'_>> = active.iter().map(|&i| remaining[i].take().unwrap()).collect();
            let lanes_now: Vec<VerifyLane<'_>> = active
                .iter()
                .map(|&i| VerifyLane {
                    sampling: sampling_of(sampled, lanes[i]),
                    remaining_tokens: (TOTAL - emitted[i]) as u32,
                    stop_ids,
                    ..VerifyLane::greedy(&[])
                })
                .collect();
            let runs = decode_program_verify_runs(model, pool, &mut refs, &lanes_now, WINDOW)
                .unwrap_or_else(|e| panic!("round: {e}"));
            for ((&i, run), read) in active.iter().zip(runs).zip(&reads) {
                if let Some(proposal) = read {
                    // The round's accepted drafts are the read ones: a round
                    // takes a draft only as the token it commits there.
                    let accepted = run.tokens.len() - 1;
                    assert_eq!(&run.tokens[1..], &proposal.drafts[..accepted], "lane {i}: the round's accepted drafts");
                }
                emitted[i] += run.tokens.len();
                done[i] = emitted[i] >= TOTAL || run.tokens.iter().any(|t| stop_ids.contains(t));
                per_lane[i].push(run);
            }
        }
        (per_lane, pending(&seqs))
    };

    // The same lanes replayed through forced rounds over `recorded`, each
    // lane forced as the lane that recorded it -- its sampling and its seed
    // -- and prefilled as `generate` prefilled it.
    let replay = |pool: &SeqPool, lanes: &[usize], sampled: bool, recorded: &[Vec<i32>]| -> Lanes {
        let mut seqs: Vec<Seq<'_>> = lanes.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap()).collect();
        for (seq, &lane) in seqs.iter_mut().zip(lanes) {
            prefill_program_sampled(model, pool, seq, &prompts[lane], 0, sampling_of(sampled, lane), None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
        }
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
                .map(|&i| VerifyLane {
                    sampling: sampling_of(sampled, lanes[i]),
                    remaining_tokens: (TOTAL - at[i]) as u32,
                    stop_ids,
                    ..VerifyLane::greedy(&[])
                })
                .collect();
            let runs = decode_program_verify_forced(model, pool, &mut refs, &lanes_now, WINDOW, &forced_refs)
                .unwrap_or_else(|e| panic!("forced round: {e}"));
            for (&i, run) in active.iter().zip(runs) {
                at[i] += run.tokens.len();
                forced_runs[i].push(run);
            }
        }
        (forced_runs, pending(&seqs))
    };

    // Widths 1 and 2, greedy and sampled: the readout reads what the rounds
    // draw and writes nothing, and forced rounds fed a run's own text, as the
    // lane that made it, are that run's rounds bit for bit -- the forced
    // verdict is the live one wherever the lane's own draws emit the recorded
    // token, which on its own text they always do.
    for sampled in [false, true] {
        for lanes in [&[0usize][..], &[1, 2][..]] {
            let label = format!("sampled draft, width {}, {}", lanes.len(), if sampled { "sampled" } else { "greedy" });
            let pool = pool(lanes.len() as u32);
            let _ = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
            let plain = generate(&pool, lanes, sampled, false);
            let read = generate(&pool, lanes, sampled, true);
            assert_eq!(plain, read, "{label}: reading the drafter changed the rounds");
            let texts: Vec<Vec<i32>> =
                plain.0.iter().map(|runs| runs.iter().flat_map(|r| r.tokens.clone()).collect()).collect();
            let recorded: Vec<Vec<i32>> =
                texts.iter().zip(&plain.1).map(|(text, &after)| text.iter().copied().chain([after]).collect()).collect();
            let forced = replay(&pool, lanes, sampled, &recorded);
            assert_eq!(forced, plain, "{label}: forced with its own text, the rounds are not the live rounds");
            let committed: usize = texts.iter().map(Vec::len).sum();
            let rounds: usize = plain.0.iter().map(Vec::len).sum();
            println!("{label}: {committed} tokens in {rounds} rounds, {:.2} per round", committed as f64 / rounds as f64);
        }
    }

    // --- the forced round, fed another text, under a seed of its own ------
    // Prompt 0's anchor, then prompt 1's sampled answer: text the lane never
    // drew. The runs commit it exactly; each round takes at most the leading
    // recorded tokens that are candidates; and some round takes more than the
    // lane's own draws agree with -- the coin, not the agreement.
    let pool = pool(1);
    let _ = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let (own, _) = generate(&pool, &[0], true, false);
    let (other, other_after) = generate(&pool, &[1], true, false);
    let mut text = vec![own[0][0].tokens[0]];
    text.extend(other[0].iter().flat_map(|r| r.tokens.clone()).skip(1));
    let recorded: Vec<i32> = text.iter().copied().chain([other_after[0]]).collect();
    let replaying = sampling(true, 4242);
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
    prefill_program_sampled(model, &pool, &mut seq, &prompts[0], 0, sampling_of(true, 0), None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    assert_eq!(seq.pending_token(), Some(text[0]), "the forced text starts at the prefill's own anchor");
    let mut at = 0usize;
    let mut committed = Vec::new();
    let (mut rounds, mut past_agreement, mut short_of_candidates) = (0usize, 0usize, 0usize);
    while at < text.len() {
        let remaining = (text.len() - at) as u32;
        let extent = WINDOW.min(remaining - 1);
        let proposal = dflash2_readout::propose_sampled(model, &pool, &seq, extent, WINDOW, false, replaying)
            .unwrap_or_else(|e| panic!("readout: {e}"));
        let forced = forced_slice(&recorded, at + 1);
        let run = decode_program_verify_forced(
            model,
            &pool,
            &mut [&mut seq],
            &[VerifyLane { sampling: replaying, remaining_tokens: remaining, stop_ids, ..VerifyLane::greedy(&[]) }],
            WINDOW,
            &[&forced],
        )
        .unwrap_or_else(|e| panic!("forced round: {e}"))
        .remove(0);
        assert_eq!(run.tokens[0], text[at], "the round's anchor is the recorded token after the last run");
        assert_eq!(run.extent, extent, "the readout's extent is not the round's");
        let cut = run.tokens.iter().position(|t| stop_ids.contains(t)).map_or(usize::MAX, |p| p + 1);
        let accepted = run.tokens.len() - 1;
        let reachable = leading_candidates(&proposal, &forced).min(extent as usize);
        assert!(
            accepted <= reachable,
            "a forced round took {accepted} recorded tokens, of which only {reachable} lead among their candidates"
        );
        let agreement = proposal.drafts.iter().zip(&forced).take_while(|(d, f)| d == f).count().min(extent as usize);
        if run.tokens.len() < cut {
            past_agreement += usize::from(accepted > agreement);
            short_of_candidates += usize::from(accepted < reachable);
        }
        rounds += 1;
        at += run.tokens.len();
        committed.extend(run.tokens);
    }
    assert_eq!(committed, text, "the forced rounds did not commit the forced text");
    assert!(past_agreement > 0, "no forced round went past the lane's own draws: the coin went unexercised");
    assert!(short_of_candidates > 0, "every forced round took every candidate it could: the coin never refused");
    println!(
        "another text: {} tokens in {rounds} rounds; {past_agreement} past the lane's own draws, {short_of_candidates} \
         short of the candidates",
        text.len()
    );

    // A recording whose pad follows no stop could stop at that pad, under the
    // forced verdict: refused before the pass, the sequence unmoved.
    let anchor = seq.pending_token().expect("a pending anchor");
    if !stop_ids.contains(&anchor) {
        let position = seq.stats().position;
        let mut bad = vec![-1; WINDOW as usize + 1];
        bad[0] = 13;
        let err = decode_program_verify_forced(
            model,
            &pool,
            &mut [&mut seq],
            &[VerifyLane { sampling: replaying, stop_ids, ..VerifyLane::greedy(&[]) }],
            WINDOW,
            &[&bad],
        )
        .expect_err("a pad with no stop before it");
        assert!(err.contains("outside the vocabulary"), "the wrong refusal: {err}");
        assert_eq!((seq.stats().position, seq.pending_token()), (position, Some(anchor)), "the refused round moved the sequence");
    }
}
