//! GPU coverage for the DFlash2 drafter in the verify round (P5-05, GitHub
//! #155, spec 05). On a load with the drafter the leaf proposes every lane's
//! drafts itself: the round's pass opens with the drafter's forward over each
//! lane's window, the verify traversal taps the target's layer-5/19/33/47/61
//! outputs for every column, and after the cut the committed columns' taps
//! are appended to the lane's window. Only the runs and each lane's extent
//! cross the seam.
//!
//! - AC 1: greedy spec-on equals spec-off on the canary prompts at widths 1,
//!   4 and 8 -- identical, or identical up to a divergence the engine's own
//!   two prefill routes call a near-tie (`support/near_tie.rs`; why, measured,
//!   in `speculative_round_gpu.rs`'s header). The drafter runs inside the
//!   captured verify graph, so a width-4 run on an uncaptured load must
//!   replay it bit for bit.
//! - AC 2: acceptance is printed per round, with the mean committed tokens per
//!   full-window round against the reference's 3.4-5.75 band at draft 7. A
//!   mean below the band is printed as a finding, not failed; a drafter that
//!   never lands a single draft fails, because that is a drafter reading the
//!   wrong window.
//! - AC 3: a lane at extent 0 -- once by its budget, once by its context --
//!   completes its round beside a drafting lane and leaves its window and
//!   the window's frontier exactly as they were.
//! - AC 4: a sequence snapshotted mid-generation, released and restored into
//!   a fresh handle continues exactly as the one never evicted.
//!
//! BF16 KV throughout: it is the correctness oracle (ADR 0022), and the hq
//! verify tile's masked-column dependence (#153) is not this ticket's.
//!
//! One `#[test]`, several `Model`/`SeqPool` loads against one shared
//! `materialize()` of the text scope plus the drafter: the text scope is the
//! plan's first handles, so a plain load takes exactly those.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/near_tie.rs"]
mod near_tie;
#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, materialize, text_scope_27b, CudaDevice, DraftModule, FrontendSet, MaterializedArtifact,
    ObjectHandle, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b, load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{snapshot_format_version, Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_rounds, prefill_program_sampled,
    program_stats, SamplingParams, VerifyLane, VerifyRound,
};
use ignis_core::{KvFormat, Speculation, SpeculativeBackend};

use near_tie::assert_equivalent;
use snapshot_blob::{section, read_u64, PROGRESS_DRAFTER_FRONTIER, SECTION_DFLASH_WINDOW, SECTION_PROGRESS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// DFlash2-7, the reference lane G5 measures.
const WINDOW: u32 = 7;
/// Tokens each lane emits in a comparison: several full windows.
const TOTAL: usize = 48;
/// The reference's committed tokens per round at draft 7 (the issue's band).
const REFERENCE_BAND: (f64, f64) = (3.4, 5.75);

/// The canary suite's prompts (`crates/bench/src/canary.rs::CANARIES`),
/// each in a user turn of the model's chat format with thinking closed, so
/// the drafter continues the kind of text it was trained on.
const CANARY_PROMPTS: [&str; 4] = [
    "In one sentence, what does `fn main() { println!(\"hi\"); }` do?",
    "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "Explain in one sentence what `x.reverse()` does to a `Vec<i32>` named `x`.",
];

fn pool_for(backend: Option<SpeculativeBackend>, slot_count: u32) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
        backend,
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

fn drafter_pool(slot_count: u32) -> SeqPool {
    pool_for(Some(SpeculativeBackend::Dflash2), slot_count)
}

fn load_drafter(reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle]) -> Model {
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
    load_qwen38_27b_with_speculation(reader, artifact, handles, MAX_CONTEXT, MAX_CONTEXT, KvFormat::Bf16, Some(spec))
        .unwrap_or_else(|e| panic!("model load (dflash2, window {WINDOW}): {e}"))
}

fn prefill_all(model: &Model, pool: &SeqPool, sequences: &mut [Seq<'_>], prompts: &[Vec<i32>]) {
    for (seq, prompt) in sequences.iter_mut().zip(prompts) {
        prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
}

/// The spec-off text: `total` greedy tokens per lane, one per round, every
/// lane in one batch, each lane's stream cut after its first stop id -- the
/// turn's answer, which is what the server emits. Past the end of a turn the
/// model loops on chat markers, where the verify tile and the one-column
/// decode tile part by several logits (#153's recorded route difference);
/// nothing a request emits lives there.
fn spec_off(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], stop_ids: &[i32], total: usize) -> Vec<Vec<i32>> {
    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    prefill_all(model, pool, &mut sequences, prompts);
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
        if let Some(stop) = stream.iter().position(|t| stop_ids.contains(t)) {
            stream.truncate(stop + 1);
        }
    }
    streams
}

/// One greedy verify round over `sequences`, each lane budgeted to
/// `budgets[i]` tokens and cut at `stop_ids`, the drafter proposing.
fn round(
    model: &Model,
    pool: &SeqPool,
    sequences: &mut [&mut Seq<'_>],
    budgets: &[u32],
    stop_ids: &[i32],
) -> Vec<VerifyRound> {
    let lanes: Vec<VerifyLane<'_>> = budgets
        .iter()
        .map(|&remaining_tokens| VerifyLane { remaining_tokens, stop_ids, ..VerifyLane::greedy(&[]) })
        .collect();
    decode_program_verify_rounds(model, pool, sequences, &lanes, WINDOW).unwrap_or_else(|e| panic!("verify round: {e}"))
}

/// What one spec-on run recorded: each lane's text, and per round the
/// `(lane, drafted, committed)` of every lane that rode it.
#[derive(Debug, PartialEq)]
struct SpecRun {
    emitted: Vec<Vec<i32>>,
    rounds: Vec<Vec<(usize, u32, usize)>>,
}

/// Runs `prompts` through drafter rounds until every lane has emitted
/// `total` tokens or a stop id, each lane budgeted to exactly that. Lanes
/// ride every round until done, so widths shrink as lanes finish.
fn spec_on(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    stop_ids: &[i32],
    total: usize,
    expect_graph: bool,
) -> SpecRun {
    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    prefill_all(model, pool, &mut sequences, prompts);
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); prompts.len()];
    let mut done = vec![false; prompts.len()];
    let mut rounds = Vec::new();
    while done.iter().any(|d| !d) {
        let active: Vec<usize> = (0..prompts.len()).filter(|&i| !done[i]).collect();
        let budgets: Vec<u32> = active.iter().map(|&i| (total - emitted[i].len()) as u32).collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> =
            active.iter().map(|&i| remaining[i].take().expect("each active lane once")).collect();
        let results = round(model, pool, &mut refs, &budgets, stop_ids);
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, u64::from(expect_graph), "round {} at width {}: graph replay", rounds.len(), active.len());
        let mut this_round = Vec::new();
        for ((&i, result), &budget) in active.iter().zip(results).zip(&budgets) {
            let expected_extent = WINDOW.min(budget - 1);
            assert_eq!(result.drafted, expected_extent, "lane {i}: the leaf's reported extent");
            assert!(!result.tokens.is_empty() && result.tokens.len() <= result.drafted as usize + 1, "lane {i}: run of {}", result.tokens.len());
            this_round.push((i, result.drafted, result.tokens.len()));
            done[i] = result.tokens.iter().any(|t| stop_ids.contains(t));
            emitted[i].extend(result.tokens);
            done[i] |= emitted[i].len() >= total;
        }
        rounds.push(this_round);
        assert!(rounds.len() <= total, "the run does not converge");
    }
    SpecRun { emitted, rounds }
}

/// AC 2: prints every round's acceptance and the mean committed tokens per
/// full-window round. Returns the drafts accepted over the run.
fn report_acceptance(label: &str, run: &SpecRun) -> usize {
    let mut committed_full = 0usize;
    let mut full_rounds = 0usize;
    let mut accepted = 0usize;
    for (index, lanes) in run.rounds.iter().enumerate() {
        let cells: Vec<String> = lanes
            .iter()
            .map(|&(lane, drafted, committed)| format!("lane {lane}: {}/{drafted}", committed - 1))
            .collect();
        println!("{label} round {index}: accepted/drafted {}", cells.join(", "));
        for &(_, drafted, committed) in lanes {
            accepted += committed - 1;
            if drafted == WINDOW {
                committed_full += committed;
                full_rounds += 1;
            }
        }
    }
    let mean = committed_full as f64 / full_rounds.max(1) as f64;
    println!(
        "{label}: {mean:.2} committed tokens per full-window round over {full_rounds} rounds \
         (reference band {:.2}-{:.2} at draft {WINDOW})",
        REFERENCE_BAND.0, REFERENCE_BAND.1
    );
    if mean < REFERENCE_BAND.0 {
        println!("{label}: FINDING -- acceptance below the reference's band");
    }
    accepted
}

/// A sequence's drafter window and its frontier, and its program frontier,
/// as a snapshot carries them.
fn window_state(seq: &Seq<'_>) -> (Vec<u8>, u64, u64) {
    let blob = seq.snapshot().unwrap_or_else(|e| panic!("snapshot: {e:?}"));
    let (window_at, window_bytes) = section(&blob, SECTION_DFLASH_WINDOW);
    let (progress_at, _) = section(&blob, SECTION_PROGRESS);
    (
        blob[window_at..window_at + window_bytes].to_vec(),
        read_u64(&blob, progress_at + PROGRESS_DRAFTER_FRONTIER),
        read_u64(&blob, progress_at),
    )
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_drafter_proposes_from_its_window_and_the_text_stays_the_spec_off_text() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
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
    assert_eq!(encode("<|im_start|>").len(), 1, "the chat markers must tokenize as special tokens");
    // The turn ends at either marker; the server stops there.
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    let canaries: Vec<Vec<i32>> = CANARY_PROMPTS
        .iter()
        .map(|prompt| encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")))
        .collect();
    // Width 8 carries the four canaries twice: the suite has four prompts.
    let prompts8: Vec<Vec<i32>> = canaries.iter().chain(&canaries).cloned().collect();
    assert!(prompts8.iter().all(|p| p.len() + TOTAL <= MAX_CONTEXT as usize), "a canary outgrows the context");
    assert_eq!(snapshot_format_version(), 2, "this test reads the version-2 blob layout");

    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let text_len = text_scope_27b().len();
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

    // --- the spec-off text at each width AC 1 compares -------------------
    let widths = [1usize, 4, 8];
    let off: Vec<Vec<Vec<i32>>> = {
        let model = load_qwen38_27b(&reader, &artifact, &handles[..text_len], MAX_CONTEXT, MAX_CONTEXT, KvFormat::Bf16)
            .unwrap_or_else(|e| panic!("model load (spec-off): {e}"));
        widths
            .iter()
            .map(|&width| {
                let pool = pool_for(None, width as u32);
                spec_off(&model, &pool, &prompts8[..width], &stop_ids, TOTAL)
            })
            .collect()
    };

    // --- AC 1 + AC 2: the drafter's text, and its acceptance --------------
    let model = load_drafter(&reader, &artifact, &handles);
    let probe = || drafter_pool(1);
    let mut graph_width4 = None;
    let mut accepted = 0usize;
    for (&width, off) in widths.iter().zip(&off) {
        let pool = drafter_pool(width as u32);
        let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        assert!(capture.is_ready(width as u32), "width {width}: no decode graph");
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert!(
            stats.verify_graph_ready_mask & (1 << (width - 1)) != 0,
            "width {width}: no verify graph with the drafter in it (mask {:#010b})",
            stats.verify_graph_ready_mask
        );
        let prompts = &prompts8[..width];
        let run = spec_on(&model, &pool, prompts, &stop_ids, TOTAL, true);
        assert_equivalent(&model, &probe, MAX_CONTEXT, prompts, off, &run.emitted, &format!("width {width} drafter"));
        accepted += report_acceptance(&format!("width {width}"), &run);
        if width == 4 {
            graph_width4 = Some(run);
        }
    }
    assert!(accepted > 0, "the drafter never landed a draft: it is not proposing from its window");
    // The drafter proposes every lane's drafts: a caller's are refused.
    {
        let pool = drafter_pool(1);
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_all(&model, &pool, std::slice::from_mut(&mut seq), &canaries[..1]);
        let drafts = [canaries[0][0]];
        let refused = decode_program_verify_rounds(&model, &pool, &mut [&mut seq], &[VerifyLane::greedy(&drafts)], WINDOW);
        assert!(refused.is_err(), "a DFlash2 load took caller drafts");
    }
    drop(model);

    // The drafter's forward runs inside the captured verify graph: a load
    // that captured nothing runs the same round eagerly, bit for bit.
    {
        let model = load_drafter(&reader, &artifact, &handles);
        let pool = drafter_pool(4);
        let eager = spec_on(&model, &pool, &prompts8[..4], &stop_ids, TOTAL, false);
        assert_eq!(Some(eager), graph_width4, "width 4: the drafter's graph replay diverged from eager");
    }

    // --- AC 3: a lane at extent 0 leaves its window alone -----------------
    {
        let model = load_drafter(&reader, &artifact, &handles);
        let pool = drafter_pool(3);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        // Out of budget: its round may commit its anchor and nothing more.
        let mut budgeted = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        // Out of context: one KV page, prefilled to one short of it.
        let mut cramped = pool.alloc(64).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut drafting = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        let cramped_prompt: Vec<i32> = canaries[1].iter().cycle().take(63).copied().collect();
        prefill_all(&model, &pool, std::slice::from_mut(&mut budgeted), &canaries[..1]);
        prefill_all(&model, &pool, std::slice::from_mut(&mut cramped), &[cramped_prompt]);
        prefill_all(&model, &pool, std::slice::from_mut(&mut drafting), &canaries[2..3]);
        let budgeted_before = window_state(&budgeted);
        let cramped_before = window_state(&cramped);
        let drafting_before = window_state(&drafting);

        let results =
            round(&model, &pool, &mut [&mut budgeted, &mut cramped, &mut drafting], &[1, 0, TOTAL as u32], &[]);
        for (label, result) in [("out of budget", &results[0]), ("out of context", &results[1])] {
            assert_eq!(result.drafted, 0, "{label}: the lane must run at extent 0");
            assert_eq!(result.tokens.len(), 1, "{label}: the lane commits its anchor alone");
        }
        assert_eq!(results[2].drafted, WINDOW, "the drafting lane proposes a full window beside them");

        for (label, seq, before) in
            [("out of budget", &budgeted, &budgeted_before), ("out of context", &cramped, &cramped_before)]
        {
            let (window, frontier, position) = window_state(seq);
            assert!(window == before.0, "{label}: the round wrote the lane's drafter window");
            assert_eq!(frontier, before.1, "{label}: the round moved the window's frontier");
            assert_eq!(position, before.2 + 1, "{label}: the round committed its anchor");
        }
        let (window, frontier, position) = window_state(&drafting);
        assert!(window != drafting_before.0, "the drafting lane's committed columns never reached its window");
        assert_eq!(frontier, position, "the drafting lane's window stands at its frontier");
    }

    // --- AC 4: snapshot and restore mid-generation ------------------------
    {
        let model = load_drafter(&reader, &artifact, &handles);
        let pool = drafter_pool(2);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let prompt = &canaries[3];
        let generate = |seq: &mut Seq<'_>, emitted: &mut Vec<i32>, rounds: Option<usize>| {
            let mut ran = 0;
            let ended = |emitted: &[i32]| emitted.iter().any(|t| stop_ids.contains(t));
            while emitted.len() < TOTAL && !ended(emitted) && rounds.is_none_or(|r| ran < r) {
                let result =
                    round(&model, &pool, &mut [seq], &[(TOTAL - emitted.len()) as u32], &stop_ids).remove(0);
                emitted.extend(result.tokens);
                ran += 1;
            }
        };

        let mut never_evicted = Vec::new();
        {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
            prefill_all(&model, &pool, std::slice::from_mut(&mut seq), std::slice::from_ref(prompt));
            generate(&mut seq, &mut never_evicted, None);
        }

        let mut evicted = Vec::new();
        let blob = {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
            prefill_all(&model, &pool, std::slice::from_mut(&mut seq), std::slice::from_ref(prompt));
            generate(&mut seq, &mut evicted, Some(2));
            assert!(evicted.len() < TOTAL, "the snapshot must fall mid-generation");
            seq.snapshot().unwrap_or_else(|e| panic!("snapshot mid-generation: {e:?}"))
        };
        let mut restored = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        restored.restore(&blob).unwrap_or_else(|e| panic!("restore: {e:?}"));
        generate(&mut restored, &mut evicted, None);
        assert_eq!(evicted, never_evicted, "the restored sequence continued differently from the one never evicted");
    }

    let _ = artifact.release_arena(&mut device);
}
