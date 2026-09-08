//! P2-04 diagnostic (GitHub #86): the per-chunk wall time of an 8K chunked
//! prefill at the default 1,024-token chunk width. The leaf synchronizes
//! once per chunk (P2-02, GitHub #84), so the call's wall time divided by
//! the chunk count is the chunk time the ticket's acceptance asks to be
//! reported against the #84 baseline (a diagnostic, not a gate). The test
//! prints the reading under the explicit GPU profile; it is kept so a
//! future gate run (GitHub #88) can re-take it without a new harness.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::Instant;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program;

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The G2 gate's 8K cell (the spec's fixed prompt length) at the default
/// chunk width: eight 1,024-token chunks.
const SPAN_TOKENS: usize = 8192;
const PREFILL_CHUNK: u32 = 1024;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + 64) as u32;

fn pages_for(max_context_tokens: u32) -> u32 {
    max_context_tokens.div_ceil(64)
}

fn token_span(frontend: &FrontendSet, len: usize) -> Vec<i32> {
    let mut ids: Vec<i32> = Vec::with_capacity(len);
    let filler = "The quick brown fox jumps over the lazy dog, again and again. ";
    while ids.len() < len {
        let more = frontend
            .tokenizer()
            .encode(filler)
            .unwrap_or_else(|e| panic!("tokenize filler: {e}"));
        ids.extend(more.into_iter().map(|id| i32::try_from(id).expect("token id fits i32")));
    }
    ids.truncate(len);
    ids
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn chunked_prefill_reports_its_per_chunk_wall_time() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend =
        FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let (plan, handles) =
        bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
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
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let span = token_span(&frontend, SPAN_TOKENS);
    let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("ignis_model_load(chunk={PREFILL_CHUNK}): {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_page_group_count: pages_for(MAX_CONTEXT) * 2,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let mut logits = vec![0f32; vocab];

    // The first 8K prefill in the process warms the CUDA context's kernel
    // module caches; the reading below is the warm one.
    let mut cold = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
    prefill_program(&model, &pool, &mut cold, &span, 0, Some(&mut logits))
        .unwrap_or_else(|e| panic!("cold prefill: {e}"));
    let mut warm = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
    let start = Instant::now();
    prefill_program(&model, &pool, &mut warm, &span, 0, Some(&mut logits))
        .unwrap_or_else(|e| panic!("warm prefill: {e}"));
    let elapsed = start.elapsed();
    let chunks = SPAN_TOKENS / PREFILL_CHUNK as usize;
    println!(
        "P2-04 diagnostic: 8K chunked prefill ({chunks} x {PREFILL_CHUNK}-token chunks) \
         took {:.1} ms wall -> {:.1} ms/chunk (the leaf synchronizes once per chunk)",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e3 / chunks as f64,
    );
    assert!(elapsed > std::time::Duration::ZERO, "the warm prefill took no time at all");
}