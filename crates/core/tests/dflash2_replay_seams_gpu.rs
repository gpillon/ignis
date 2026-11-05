//! GPU coverage for the acceptance replay's two measurement seams (GitHub
//! #267, docs/specs/runtime/08-dflash2-sampled-acceptance.md), on both 27B
//! bodies: the served NVFP4 one and the 16 GB artifact's trellis one (GitHub
//! #339, #340). Neither seam is a serving path; the replay of recorded agent
//! turns (`crates/runtime/examples/dflash2_replay.rs`) stands on both.
//!
//! - **The proposal readout** (`dflash2_readout::propose`): the drafts it
//!   reads at a sequence's state are the drafts the next verify round from
//!   that state verifies -- a forced round accepts exactly their leading
//!   agreement with the forced text -- its lattice walks to those drafts on
//!   the host, its candidates are the top 16 of the head logits it returns,
//!   and it writes nothing: a sequence read before every round generates
//!   bit for bit what one never read generates, and its snapshot does not
//!   move. A sequence whose window is one column behind (an extent-0 round's
//!   carried anchor, GitHub #157) is refused, not read.
//! - **The teacher-forced verify round** (`step::decode_program_verify_forced`):
//!   forced with the text the engine's own rounds commit, greedy or sampled,
//!   it is those rounds bit for bit -- the same runs, the same state after
//!   them -- at widths 1 and 2; forced with any other text it commits that
//!   text, the recorded token after each run pending as the next anchor. It
//!   refuses a sampling lane and a forced run of the wrong length.
//! - **The verify logits readout** (`dflash2_readout::verify_logits`): after a
//!   greedy round each row's argmax is the round's own licensed token -- the
//!   round's real logits, through the body's own head -- and it refuses more
//!   lanes than the load decodes (GitHub #336: the round's logits are sized
//!   for them).
//!
//! Forced and live are compared at the same width only: between widths a
//! near-tie may resolve differently (`support/near_tie.rs`), which is no
//! property of either seam.
//!
//! Each body loads at its profile's lanes: eight for the NVFP4 artifact, two
//! for the trellis one (`rtx50-16gb`). BF16 KV: the correctness oracle (ADR
//! 0022). Explicit GPU profile (ADR 0006, GitHub #38): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact, GPU, or kernel error is a skip;
//! under the profile it is a hard failure. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, body_27b, materialize, Body27b, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout::{self, Dflash2Proposal, TOP_K};
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_forced, decode_program_verify_runs, prefill_program_sampled,
    LaneVerifyRun, SamplingParams, VerifyLane,
};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend};

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

fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, prompt: &[i32]) {
    prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
}

fn greedy_lanes(n: usize) -> Vec<VerifyLane<'static>> {
    vec![VerifyLane::greedy(&[]); n]
}

/// `text[at..at + WINDOW + 1]`, padded with -1 past its end.
fn forced_slice(text: &[i32], at: usize) -> Vec<i32> {
    (0..=WINDOW as usize).map(|j| text.get(at + j).copied().unwrap_or(-1)).collect()
}

fn leading_agreement(drafts: &[i32], forced: &[i32]) -> usize {
    drafts.iter().zip(forced).take_while(|(d, f)| d == f).count()
}

/// Each lane's verify runs, and the token its last round left pending.
type Lanes = (Vec<Vec<LaneVerifyRun>>, Vec<i32>);

fn pending(seqs: &[Seq<'_>]) -> Vec<i32> {
    seqs.iter().map(|seq| seq.pending_token().expect("a lane that ran a round has a pending token")).collect()
}

/// The lattice's own consistency: the host walk over the scores lands on the
/// drafts, every draft is one of its column's candidates, the candidates are
/// the head logits' top 16 in descending order and carry those logits
/// exactly.
fn check_lattice(proposal: &Dflash2Proposal, label: &str) {
    let path = proposal.walk();
    for (column, &index) in path.iter().enumerate() {
        assert_eq!(
            proposal.candidates(column)[index],
            proposal.drafts[column],
            "{label}: the host walk over the read scores does not land on draft {column}"
        );
    }
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let logits = proposal.drafter_logits.as_ref().expect("asked for the head logits");
    for column in 0..proposal.window() {
        let row = &logits[column * vocab..(column + 1) * vocab];
        let value = |id: i32| f32::from_bits(u32::from(row[id as usize]) << 16);
        let candidates = proposal.candidates(column);
        let unary = &proposal.candidate_logits[column * TOP_K..(column + 1) * TOP_K];
        for i in 0..TOP_K {
            assert_eq!(unary[i], value(candidates[i]), "{label}: column {column} candidate {i}'s logit");
            if i > 0 {
                assert!(unary[i] <= unary[i - 1], "{label}: column {column}'s candidates are not descending");
            }
        }
        let floor = unary[TOP_K - 1];
        let above = (0..vocab as i32).filter(|&id| value(id) > floor && !candidates.contains(&id)).count();
        assert_eq!(above, 0, "{label}: column {column} has {above} tokens above its 16th candidate");
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_replay_seams_read_the_rounds_drafts_and_force_the_recorded_text_on_the_nvfp4_body() {
    the_replay_seams_hold(NVFP4, Body27b::Nvfp4, 8);
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_replay_seams_read_the_rounds_drafts_and_force_the_recorded_text_on_the_trellis_body() {
    the_replay_seams_hold(TRELLIS, Body27b::Trellis, 2);
}

/// Every seam's claim on `artifact_path`, whose body must be `body`, loaded
/// at `decode_lanes` lanes.
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
        let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
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
            ignis_core::PrefillLinear::default(),
            decode_lanes,
        )
        .unwrap_or_else(|e| panic!("model load: {e}"));
        assert_eq!(
            model.stats().reserved.trellis_workspace_bytes > 0,
            body == Body27b::Trellis,
            "only the trellis body plans the trellis workspace"
        );
        seams(&model, &prompts, &stop_ids, decode_lanes);
    }
    // Two bodies in one process: give this one's weights back before the
    // next materializes (the arena has no Drop of its own).
    let _ = artifact.release_arena(&mut device);
}

fn seams(model: &Model, prompts: &[Vec<i32>], stop_ids: &[i32], decode_lanes: u32) {
    // Lane `lane`'s sampling: greedy, or the agents' thinking sampling with
    // the lane's own seed.
    let sampling_of = |sampled: bool, lane: usize| -> SamplingParams {
        match sampled {
            false => SamplingParams::greedy(),
            true => SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 267 + lane as u64, ..SamplingParams::greedy() },
        }
    };

    // --- the engine's own rounds: the text and the runs ------------------
    // `readout`: read the drafter before every round (and check the reads).
    // Beside each lane's runs, the token its last round left pending: the
    // successor a recording of the text carries after it.
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
                        // The round's extent: its budget leaves room for this many drafts.
                        let extent = WINDOW.min((TOTAL - emitted[i]) as u32 - 1);
                        let before = seqs[i].snapshot().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
                        let proposal = dflash2_readout::propose(model, pool, &seqs[i], extent, WINDOW, true)
                            .unwrap_or_else(|e| panic!("readout: {e}"));
                        let after = seqs[i].snapshot().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
                        assert!(before == after, "the readout moved the sequence's state");
                        check_lattice(&proposal, &format!("lane {i}"));
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
                    // A round -- greedy or sampled -- accepts a one-hot draft
                    // only as the token it commits there, so the accepted
                    // drafts are the read ones.
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

    // The same lanes replayed through forced rounds over `recorded` -- each
    // lane's text, then the successor its live rounds left pending -- prefilled
    // as `generate` prefilled them (a sampled prefill draws the same first
    // anchor: the draw is keyed by seed and position). A text that ends at its
    // budget has no stop to end it, so its last forced round takes the
    // successor in place of the pad, as the live round drew one.
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
                .map(|&i| VerifyLane { remaining_tokens: (TOTAL - at[i]) as u32, stop_ids, ..VerifyLane::greedy(&[]) })
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
    // verify and writes nothing -- the text with reads is the text without --
    // and forced rounds fed a run's own text are that run's rounds, bit for
    // bit. Sampled, that is the replay's whole claim: a live round accepts a
    // one-hot draft exactly when the text it samples agrees, so forcing the
    // text gives back the live rounds -- and the token they leave pending.
    for sampled in [false, true] {
        for lanes in [&[0usize][..], &[1, 2][..]] {
            let label = format!("width {}, {}", lanes.len(), if sampled { "sampled" } else { "greedy" });
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

    // --- the forced round, fed another text ------------------------------
    // Prompt 0's anchor, then prompt 1's answer: text the target never chose.
    // The runs commit it exactly; each forced round accepts precisely the
    // read drafts' leading agreement with it; the token after each run is the
    // next round's anchor. The lane's budget ends with the text, as the
    // rounds that made it ended, and the token prompt 1's rounds left pending
    // follows it: a text that ends at its budget has no stop to end it.
    let pool = pool(1);
    let _ = capture_decode_graphs(model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let (own, _) = generate(&pool, &[0], false, false);
    let (other, other_after) = generate(&pool, &[1], false, false);
    let mut text = vec![own[0][0].tokens[0]];
    text.extend(other[0].iter().flat_map(|r| r.tokens.clone()).skip(1));
    let recorded: Vec<i32> = text.iter().copied().chain([other_after[0]]).collect();
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
    prefill(model, &pool, &mut seq, &prompts[0]);
    let mut at = 0usize;
    let mut committed = Vec::new();
    let mut accepted_any = false;
    while at < text.len() {
        let remaining = (text.len() - at) as u32;
        let extent = WINDOW.min(remaining - 1);
        let proposal = dflash2_readout::propose(model, &pool, &seq, extent, WINDOW, false)
            .unwrap_or_else(|e| panic!("readout: {e}"));
        let forced = forced_slice(&recorded, at + 1);
        let run = decode_program_verify_forced(
            model,
            &pool,
            &mut [&mut seq],
            &[VerifyLane { remaining_tokens: remaining, stop_ids, ..VerifyLane::greedy(&[]) }],
            WINDOW,
            &[&forced],
        )
        .unwrap_or_else(|e| panic!("forced round: {e}"))
        .remove(0);
        assert_eq!(run.tokens[0], text[at], "the round's anchor is the recorded token after the last run");
        assert_eq!(run.extent, extent, "the readout's extent is not the round's");
        let expected = leading_agreement(&proposal.drafts, &forced).min(extent as usize);
        let cut = run.tokens.iter().position(|t| stop_ids.contains(t)).map_or(usize::MAX, |p| p + 1);
        assert_eq!(run.tokens.len(), (expected + 1).min(cut), "the forced round's acceptance is the read drafts' agreement");
        accepted_any |= expected > 0;
        at += run.tokens.len();
        committed.extend(run.tokens);
    }
    assert_eq!(committed, text, "the forced rounds did not commit the forced text");
    assert!(accepted_any, "the drafter never agreed with the forced text: not a test of the agreement rule");
    drop(seq);

    // The verify logits readout, after a greedy round: row j's argmax is
    // what the round licensed at column j.
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap();
    prefill(model, &pool, &mut seq, &prompts[2]);
    let first = decode_program_verify_runs(model, &pool, &mut [&mut seq], &greedy_lanes(1), WINDOW).unwrap().remove(0);
    let logits = dflash2_readout::verify_logits(model, 1, WINDOW).unwrap_or_else(|e| panic!("verify logits: {e}"));
    let next = decode_program_verify_runs(model, &pool, &mut [&mut seq], &greedy_lanes(1), WINDOW).unwrap().remove(0);
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let argmax = |row: usize| -> i32 {
        let values = &logits[row * vocab..(row + 1) * vocab];
        let value = |v: u16| f32::from_bits(u32::from(v) << 16);
        let mut best = 0usize;
        for (id, &v) in values.iter().enumerate() {
            if value(v) > value(values[best]) {
                best = id;
            }
        }
        best as i32
    };
    let accepted = first.tokens.len() - 1;
    for j in 0..accepted {
        assert_eq!(argmax(j), first.tokens[j + 1], "verify logits row {j}: not the round's licensed draft");
    }
    assert_eq!(argmax(accepted), next.tokens[0], "verify logits row {accepted}: not the round's correction");
    // GitHub #336: the round's logits hold the load's lanes, no more.
    let err = dflash2_readout::verify_logits(model, decode_lanes + 1, WINDOW).expect_err("more lanes than the load decodes");
    assert!(err.contains("decode lanes"), "the wrong refusal: {err}");
    // Both readouts size their buffers by the caller's window, the leaf
    // writes the load's: another window is refused, not over- or under-run.
    let err = dflash2_readout::verify_logits(model, 1, WINDOW - 1).expect_err("a window not the load's");
    assert!(err.contains("not the load's draft window"), "the wrong refusal: {err}");
    let err = dflash2_readout::propose(model, &pool, &seq, WINDOW - 1, WINDOW - 1, false).expect_err("a window not the load's");
    assert!(err.contains("not the load's draft window"), "the wrong refusal: {err}");

    // A forced successor outside the vocabulary would be the next round's
    // anchor: refused, the sequence unmoved -- an id past the vocabulary, and
    // the pad after a recording that ends without a stop (the anchor here is
    // mid-answer).
    let anchor = seq.pending_token().expect("a pending anchor");
    assert!(!stop_ids.contains(&anchor), "the refusal below needs an anchor that is no stop id");
    let position = seq.stats().position;
    let vocab_id = ModelConfig::qwen38_27b().vocab as i32;
    for (forced, what) in [(vec![vocab_id; WINDOW as usize + 1], "an id past the vocabulary"), (vec![-1; WINDOW as usize + 1], "the pad with no stop")] {
        let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &greedy_lanes(1), WINDOW, &[&forced])
            .expect_err(what);
        assert!(err.contains("outside the vocabulary"), "{what}: the wrong refusal: {err}");
        assert_eq!((seq.stats().position, seq.pending_token()), (position, Some(anchor)), "{what}: the refused round moved the sequence");
    }

    // Refusals: a sampling lane, a forced run of the wrong length, and a
    // sequence whose window carries an extent-0 round's anchor.
    let sampled = VerifyLane {
        sampling: SamplingParams { temperature: 1.0, top_k: 20, top_p: 0.95, seed: 7, ..SamplingParams::greedy() },
        ..VerifyLane::greedy(&[])
    };
    let forced = forced_slice(&text, 0);
    let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &[sampled], WINDOW, &[&forced])
        .expect_err("a sampling lane was forced");
    assert!(err.contains("greedy"), "the wrong refusal: {err}");
    let err = decode_program_verify_forced(model, &pool, &mut [&mut seq], &greedy_lanes(1), WINDOW, &[&forced[..3]])
        .expect_err("a short forced run was taken");
    assert!(err.contains("window + 1"), "the wrong refusal: {err}");
    let ended = decode_program_verify_runs(
        model,
        &pool,
        &mut [&mut seq],
        &[VerifyLane { remaining_tokens: 1, ..VerifyLane::greedy(&[]) }],
        WINDOW,
    )
    .unwrap()
    .remove(0);
    assert_eq!(ended.extent, 0, "a one-token budget runs at extent 0");
    let err = dflash2_readout::propose(model, &pool, &seq, WINDOW, WINDOW, false)
        .expect_err("a window one column behind was read");
    assert!(err.contains("extent-0"), "the wrong refusal: {err}");
}
