//! GPU integration coverage for P3-03's device-side sampling (GitHub #99,
//! ADR 0016's size-prefixed `ignis_sampling_params`): a sequence's stochastic
//! output depends only on its own seed and logical position, never on which
//! other lanes share its decode round, their count, or their arrival order;
//! greedy sampling stays exactly reproducible and independent of `seed`.
//!
//! One `#[test]`, several `Model`/`SeqPool` loads against one shared
//! `materialize()` (mirrors `chunked_prefill_gpu.rs`'s own note on the same
//! constraint: materialized device weights are never freed until the
//! artifact drops, so more than one `materialize()` call in this process
//! would exhaust VRAM).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    SamplingParams, decode_program_batch, decode_program_batch_sampled, prefill_program_sampled,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
const PROMPT: [i32; 4] = [5, 9, 20, 42];
const ROUNDS: usize = 4;

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
    )
}

/// Allocates a sequence, prefills the shared `PROMPT` under `sampling`, and
/// decodes `ROUNDS` tokens *alone* (batch size 1) -- the "no other lanes"
/// baseline every shared-round scenario below is compared against.
fn run_alone(model: &Model, pool: &SeqPool, sampling: SamplingParams) -> Vec<i32> {
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    prefill_program_sampled(model, pool, &mut seq, &PROMPT, 0, sampling, None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let mut tokens = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        tokens.extend(
            decode_program_batch_sampled(model, pool, &mut [&mut seq], &[sampling])
                .unwrap_or_else(|e| panic!("decode: {e}")),
        );
    }
    tokens
}

fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, sampling: SamplingParams) {
    prefill_program_sampled(model, pool, seq, &PROMPT, 0, sampling, None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn decode_sampling_is_independent_of_batch_composition_and_arrival_order() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    let seed_a = SamplingParams {
        temperature: 0.8,
        top_k: 5,
        top_p: 0.95,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 111,
    };
    let seed_b = SamplingParams {
        seed: 222,
        ..seed_a
    };
    let greedy = SamplingParams::greedy();

    // --- baseline: each sequence decoded alone, batch size 1 --------------
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let alone_a = run_alone(&model, &pool, seed_a);
    drop(pool);
    let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let alone_greedy = run_alone(&model, &pool, greedy);
    drop(pool);
    drop(model);

    // --- shared rounds: A, B and a greedy sequence together, in one order --
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(3).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let mut seq_a = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc a: {e}"));
    let mut seq_b = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc b: {e}"));
    let mut seq_g = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc g: {e}"));
    prefill(&model, &pool, &mut seq_a, seed_a);
    prefill(&model, &pool, &mut seq_b, seed_b);
    prefill(&model, &pool, &mut seq_g, greedy);
    let mut shared_a = Vec::with_capacity(ROUNDS);
    let mut shared_greedy = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        // Arrival order [g, a, b]: neither A's slot, A's row position in the
        // batch, nor the presence of B/G changes A's own draw.
        let tokens = decode_program_batch_sampled(
            &model,
            &pool,
            &mut [&mut seq_g, &mut seq_a, &mut seq_b],
            &[greedy, seed_a, seed_b],
        )
        .unwrap_or_else(|e| panic!("shared decode: {e}"));
        shared_greedy.push(tokens[0]);
        shared_a.push(tokens[1]);
    }
    drop(seq_a);
    drop(seq_b);
    drop(seq_g);
    drop(pool);
    drop(model);

    assert_eq!(
        alone_a, shared_a,
        "seed {}'s tokens must not depend on sharing its rounds with other lanes",
        seed_a.seed
    );
    assert_eq!(
        alone_greedy, shared_greedy,
        "greedy decoding must not depend on batch composition either"
    );
    assert_ne!(
        alone_a, alone_greedy,
        "sanity: the stochastic and greedy sequences actually diverge for this prompt"
    );
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn greedy_via_sampling_params_matches_the_plain_greedy_entry_point() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
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
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let via_sampling_params = run_alone(&model, &pool, SamplingParams::greedy());
    drop(pool);
    drop(model);

    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
    ignis_core::step::prefill_program(&model, &pool, &mut seq, &PROMPT, 0, None)
        .unwrap_or_else(|e| panic!("prefill: {e}"));
    let mut via_greedy_entry_point = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        via_greedy_entry_point
            .extend(decode_program_batch(&model, &pool, &mut [&mut seq]).unwrap_or_else(|e| {
                panic!("decode: {e}")
            }));
    }

    assert_eq!(
        via_sampling_params, via_greedy_entry_point,
        "SamplingParams::greedy() and the plain greedy entry point must be bit-identical"
    );
}
