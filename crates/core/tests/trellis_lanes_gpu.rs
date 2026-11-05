//! GitHub #339 (spec 27b-16gb/05a): the trellis body at two lanes -- the
//! profile `rtx50-16gb` serves two, and every other trellis test runs one.
//!
//! The round bodies a width of two reaches are leaf-internal: the decode
//! graph's GDN and GQA bodies (`run_gdn_layer_graph`, `run_gqa_layer_batch`
//! at one column a lane) and the verify round's (`run_gdn_layer_record`, the
//! trellis projection into the conv record of every lane's columns and the
//! vendored record kernel over it, and `run_gqa_layer_batch` at `k + 1`
//! columns a lane). The step ABI reaches them only through a whole round of
//! the 64 layers, so they are held the way the NVFP4 body's are
//! (`decode_lanes_gpu.rs`, `dflash2_round_gpu.rs`): by the tokens a round
//! commits, against each prompt served alone.
//!
//! - Plain rounds: two prompts in one width-2 round, from the captured decode
//!   graph, give each prompt's width-1 text; the same rounds run eagerly on a
//!   load that captured nothing give the graph's text bit for bit (the
//!   trellis op's one workspace, replayed).
//! - Verify rounds with the DFlash2 drafter at window 7: at width 1 and at
//!   width 2 (16 columns, the decode route's second token class) each lane's
//!   committed text is its plain width-1 text, identical or identical up to a
//!   near-tie the engine's own two prefill routes call one
//!   (`support/near_tie.rs`); the eager width-2 rounds are the graph's bit
//!   for bit.
//!
//! Every stream is cut at the turn's first stop id, where the server stops.
//! BF16 KV, the correctness oracle (ADR 0022).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

#[path = "support/near_tie.rs"]
mod near_tie;

use std::path::Path;

use ignis_artifact::{
    bind_model_scope_27b, body_27b, materialize, model_scope_27b_of, Body27b, CudaDevice, DraftModule, FrontendSet,
    MaterializedArtifact, ModelScope, ObjectHandle, Reader,
};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, Model};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_batch_sampled, decode_program_verify_runs, prefill_program_sampled,
    program_stats, SamplingParams, VerifyLane,
};
use ignis_core::{KvFormat, PrefillAttention, RopeScaling, Speculation, SpeculativeBackend};

use near_tie::{assert_equivalent, first_divergence};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// The window `rtx50-16gb` drafts at.
const WINDOW: u32 = 7;
/// The profile's lanes.
const LANES: u32 = 2;
/// Tokens each lane emits in a comparison: several full windows.
const TOTAL: usize = 64;

/// Two of the canary suite's prompts (`crates/bench/src/canary.rs::CANARIES`)
/// in a user turn of the model's chat format with thinking closed: two whose
/// answers run past `TOTAL`, so the pair shares its verify rounds until the
/// lane that accepts more reaches its budget.
const PROMPTS: [&str; 2] = [
    "Compute, step by step, 2 * 3 + 4 and give the final number on the last line.",
    "What does `let v = vec![3,1,2]; v.sort();` set `v` to, after the call?",
];

fn pool_for(backend: Option<SpeculativeBackend>, slot_count: u32) -> SeqPool {
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
        backend,
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

fn load(
    reader: &Reader,
    artifact: &MaterializedArtifact,
    handles: &[ObjectHandle],
    speculation: Option<Speculation>,
) -> Model {
    let model = load_qwen38_27b_with_decode_lanes(
        reader,
        artifact,
        handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        KvFormat::Bf16,
        speculation,
        None,
        RopeScaling::NONE,
        PrefillAttention::default(),
        ignis_core::PrefillLinear::default(),
        LANES,
    )
    .unwrap_or_else(|e| panic!("model load ({speculation:?}, {LANES} lanes): {e}"));
    assert!(model.stats().reserved.trellis_workspace_bytes > 0, "the trellis body plans its workspace");
    model
}

fn prefill_all(model: &Model, pool: &SeqPool, sequences: &mut [Seq<'_>], prompts: &[Vec<i32>]) {
    for (seq, prompt) in sequences.iter_mut().zip(prompts) {
        prefill_program_sampled(model, pool, seq, prompt, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
}

fn cut_at_stop(stream: &mut Vec<i32>, stop_ids: &[i32]) {
    if let Some(stop) = stream.iter().position(|t| stop_ids.contains(t)) {
        stream.truncate(stop + 1);
    }
}

/// `TOTAL` greedy tokens per lane, one per round, every lane in one batch,
/// each round from the captured graph (`expect_graph`) or eager.
fn plain(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], stop_ids: &[i32], expect_graph: bool) -> Vec<Vec<i32>> {
    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    prefill_all(model, pool, &mut sequences, prompts);
    let params = vec![SamplingParams::greedy(); prompts.len()];
    let mut streams = vec![Vec::with_capacity(TOTAL); prompts.len()];
    for round in 0..TOTAL {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let tokens = decode_program_batch_sampled(model, pool, &mut refs, &params)
            .unwrap_or_else(|e| panic!("width {} decode round {round}: {e}", prompts.len()));
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, u64::from(expect_graph), "width {} round {round}: graph replay", prompts.len());
        for (stream, token) in streams.iter_mut().zip(tokens) {
            stream.push(token);
        }
    }
    for stream in &mut streams {
        cut_at_stop(stream, stop_ids);
    }
    streams
}

/// A spec-on run: each lane's text, and per round the `(lane, extent,
/// committed)` of every lane that rode it.
#[derive(Debug, PartialEq)]
struct SpecRun {
    emitted: Vec<Vec<i32>>,
    rounds: Vec<Vec<(usize, u32, usize)>>,
}

/// Drafter rounds until every lane has emitted `TOTAL` tokens or a stop id,
/// each lane budgeted to exactly that; lanes ride every round until done, so
/// the width shrinks as lanes finish.
fn spec_on(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], stop_ids: &[i32], expect_graph: bool) -> SpecRun {
    let mut sequences: Vec<Seq<'_>> =
        prompts.iter().map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"))).collect();
    prefill_all(model, pool, &mut sequences, prompts);
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); prompts.len()];
    let mut done = vec![false; prompts.len()];
    let mut rounds = Vec::new();
    while done.iter().any(|d| !d) {
        let active: Vec<usize> = (0..prompts.len()).filter(|&i| !done[i]).collect();
        let budgets: Vec<u32> = active.iter().map(|&i| (TOTAL - emitted[i].len()) as u32).collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> =
            active.iter().map(|&i| remaining[i].take().expect("each active lane once")).collect();
        let lanes: Vec<VerifyLane<'_>> = budgets
            .iter()
            .map(|&remaining_tokens| VerifyLane { remaining_tokens, stop_ids, ..VerifyLane::greedy(&[]) })
            .collect();
        let results = decode_program_verify_runs(model, pool, &mut refs, &lanes, WINDOW)
            .unwrap_or_else(|e| panic!("width {} verify round {}: {e}", active.len(), rounds.len()));
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.graph_launches, u64::from(expect_graph), "verify round {} at width {}: graph replay", rounds.len(), active.len());
        let mut this_round = Vec::new();
        for ((&i, result), &budget) in active.iter().zip(results).zip(&budgets) {
            assert_eq!(result.extent, WINDOW.min(budget - 1), "lane {i}: the leaf's reported extent");
            assert!(!result.tokens.is_empty() && result.tokens.len() <= result.extent as usize + 1, "lane {i}: run of {}", result.tokens.len());
            this_round.push((i, result.extent, result.tokens.len()));
            done[i] = result.tokens.iter().any(|t| stop_ids.contains(t));
            emitted[i].extend(result.tokens);
            done[i] |= emitted[i].len() >= TOTAL;
        }
        rounds.push(this_round);
        assert!(rounds.len() <= TOTAL, "the run does not converge");
    }
    SpecRun { emitted, rounds }
}

/// The tokens per round of a run's full-window rounds, printed.
fn report(label: &str, run: &SpecRun) {
    let (mut committed, mut full) = (0usize, 0usize);
    for lanes in &run.rounds {
        for &(_, extent, tokens) in lanes {
            if extent == WINDOW {
                committed += tokens;
                full += 1;
            }
        }
    }
    let widths: Vec<usize> = run.rounds.iter().map(Vec::len).collect();
    println!(
        "{label}: {:.2} committed tokens per full-window round over {full} rounds; round widths {widths:?}",
        committed as f64 / full.max(1) as f64
    );
}

/// The prompts in the model's chat format, and the two stop markers' ids.
fn chat_prompts(reader: &Reader) -> (Vec<Vec<i32>>, Vec<i32>) {
    let frontend = FrontendSet::from_reader(reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    let stop_ids: Vec<i32> = ["<|im_end|>", "<|endoftext|>"].iter().flat_map(|marker| encode(marker)).collect();
    assert_eq!(stop_ids.len(), 2, "the stop markers must tokenize as special tokens");
    let prompts = PROMPTS
        .iter()
        .map(|prompt| encode(&format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")))
        .collect();
    (prompts, stop_ids)
}

fn print_agreement(label: &str, reference: &[Vec<i32>], other: &[Vec<i32>]) {
    for (lane, (a, b)) in reference.iter().zip(other).enumerate() {
        match first_divergence(a, b) {
            None => println!("{label} lane {lane}: identical, {} tokens", a.len()),
            Some(d) => println!("{label} lane {lane}: first divergence at token {d} of {}", a.len()),
        }
    }
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn two_trellis_lanes_in_one_round_give_each_lanes_own_text() {
    let path = Path::new(TRELLIS);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {TRELLIS}: {e}"));
    assert_eq!(body_27b(&reader), Body27b::Trellis);
    let (prompts, stop_ids) = chat_prompts(&reader);
    assert!(prompts.iter().all(|p| p.len() + TOTAL <= MAX_CONTEXT as usize), "a prompt outgrows the context");
    let (plan, handles) = bind_model_scope_27b(&reader, Some(DraftModule::Dflash2))
        .unwrap_or_else(|e| panic!("bind the trellis scope with dflash2: {e}"));
    // The text scope is the plan's first handles, so a plain load takes
    // exactly those.
    let text_len = model_scope_27b_of(&reader, ModelScope::default()).unwrap_or_else(|e| panic!("{e}")).len();
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize the trellis scope + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let both_lanes = (1u32 << LANES) - 1;

    // --- plain rounds: the decode graph's bodies at one and two lanes ------
    let (alone, together) = {
        let model = load(&reader, &artifact, &handles[..text_len], None);
        let pool = pool_for(None, LANES);
        let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        assert_eq!(capture.ready_mask, both_lanes, "decode graphs at widths 1 and 2");
        let alone: Vec<Vec<i32>> =
            prompts.iter().map(|p| plain(&model, &pool, std::slice::from_ref(p), &stop_ids, true).remove(0)).collect();
        let together = plain(&model, &pool, &prompts, &stop_ids, true);
        (alone, together)
    };
    for (lane, text) in alone.iter().enumerate() {
        println!("lane {lane} alone: {text:?}");
    }
    print_agreement("plain width 2 vs width 1", &alone, &together);
    let plain_probe = || pool_for(None, 1);
    {
        let model = load(&reader, &artifact, &handles[..text_len], None);
        assert_equivalent(&model, &plain_probe, MAX_CONTEXT, &prompts, &alone, &together, "plain width 2");
        // The same rounds with nothing captured: the eager traversal of the
        // same bodies, which the graph replays.
        let pool = pool_for(None, LANES);
        let eager = plain(&model, &pool, &prompts, &stop_ids, false);
        assert_eq!(eager, together, "the width-2 decode graph's replay diverged from the eager rounds");
    }

    // --- verify rounds: the drafter's window 7 at one and two lanes -------
    let spec = Speculation::new(SpeculativeBackend::Dflash2, WINDOW).unwrap();
    let drafter_probe = || pool_for(Some(SpeculativeBackend::Dflash2), 1);
    let graph_two = {
        let model = load(&reader, &artifact, &handles, Some(spec));
        let pool = pool_for(Some(SpeculativeBackend::Dflash2), LANES);
        let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        assert_eq!(capture.ready_mask, both_lanes, "decode graphs at widths 1 and 2");
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.verify_graph_ready_mask, both_lanes, "verify graphs at widths 1 and 2");
        let one: Vec<Vec<i32>> = prompts
            .iter()
            .map(|p| {
                let run = spec_on(&model, &pool, std::slice::from_ref(p), &stop_ids, true);
                report("verify width 1", &run);
                run.emitted.into_iter().next().unwrap()
            })
            .collect();
        print_agreement("verify width 1 vs plain width 1", &alone, &one);
        assert_equivalent(&model, &drafter_probe, MAX_CONTEXT, &prompts, &alone, &one, "verify width 1");
        let two = spec_on(&model, &pool, &prompts, &stop_ids, true);
        report("verify width 2", &two);
        assert!(two.rounds.iter().any(|lanes| lanes.len() == 2), "no verify round ran both lanes");
        print_agreement("verify width 2 vs plain width 1", &alone, &two.emitted);
        assert_equivalent(&model, &drafter_probe, MAX_CONTEXT, &prompts, &alone, &two.emitted, "verify width 2");
        two
    };
    {
        let model = load(&reader, &artifact, &handles, Some(spec));
        let pool = pool_for(Some(SpeculativeBackend::Dflash2), LANES);
        let eager = spec_on(&model, &pool, &prompts, &stop_ids, false);
        assert_eq!(eager, graph_two, "the width-2 verify graph's replay diverged from the eager rounds");
    }
}
