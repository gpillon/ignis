//! P2-02 (GitHub #84): the chunked prefill route's self-oracle (ADR 0016).
//!
//! There is no reference engine or fixture here: the chunked route (the
//! production default) and the per-token route (retained test-only) are run
//! on the *same* long prompt, and their teacher-forced next-token
//! predictions must agree at the same >= 95% floor and with the same
//! scoring `crates/server/tests/oracle_teacher_forced_gpu.rs` uses for the
//! G1 canary (ADR 0014) -- only here the "oracle" is the per-token route
//! itself. Agreement proves the chunk loop, the state carry across chunk
//! boundaries, and (once P2-03/P2-04 turn on the multi-token kernel routes)
//! those routes did not change the function being computed.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b, Model};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{prefill_program_with_route, PrefillRoute};

use ignis_bench::oracle::{G1_AGREEMENT_FLOOR, meets_g1_floor, score_teacher_forced};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The spec's own example (`.scratch/runtime/specs/02-real-prefill.md`):
/// "An 8K prompt warms its sequence in eight [1,024-token] traversals."
const SPAN_TOKENS: usize = 8192;
const PREFILL_CHUNK: u32 = 1024;
/// Teacher-forced positions compared after the shared span, the same window
/// width the G1 canary gate uses (`FIRST_N`, `oracle_teacher_forced_gpu.rs`).
const FIRST_N: usize = 32;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + FIRST_N + 64) as u32;

/// The lowest-token-id argmax tie-break, matching the vendored
/// `ninfer::ops::argmax` kernel's own convention (mirrors
/// `oracle_teacher_forced_gpu.rs::argmax_lowest_id`).
fn argmax_lowest_id(logits: &[f32]) -> u32 {
    let mut best_id = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (id, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_id = id;
        }
    }
    best_id as u32
}

/// A deterministic token span of exactly `len` ids from the artifact's own
/// tokenizer -- content is irrelevant here (this is an agreement check
/// between two routes, not against any ground truth), only that both routes
/// see the identical span.
fn token_span(frontend: &ignis_artifact::FrontendSet, len: usize) -> Vec<i32> {
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

/// Prefills `prefill_span` through `route`, then teacher-forces
/// `forced_tail` one token at a time, returning the route's own
/// teacher-forced next-token prediction at each forced position.
fn run_route(
    model: &Model,
    pool: &SeqPool,
    prefill_span: &[i32],
    forced_tail: &[i32],
    route: PrefillRoute,
    vocab: usize,
) -> Result<Vec<u32>, String> {
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    let mut logits = vec![0f32; vocab];
    let mut position = 0u64;
    prefill_program_with_route(
        model,
        pool,
        &mut sequence,
        prefill_span,
        position,
        route,
        Some(&mut logits),
    )?;
    position += prefill_span.len() as u64;

    let mut predictions = Vec::with_capacity(forced_tail.len());
    for &token in forced_tail {
        predictions.push(argmax_lowest_id(&logits));
        let forced = [token];
        prefill_program_with_route(
            model,
            pool,
            &mut sequence,
            &forced,
            position,
            route,
            Some(&mut logits),
        )?;
        position += 1;
    }
    Ok(predictions)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn chunked_and_per_token_prefill_agree_on_a_long_prompt() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));

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
    let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));

    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let span = token_span(&frontend, SPAN_TOKENS + FIRST_N);
    let (prefill_span, forced_tail) = span.split_at(SPAN_TOKENS);

    // Every prefilled sequence within `SPAN_TOKENS + FIRST_N` tokens of
    // context, from a single-page-of-headroom-worth of pages upward; the
    // pool is sized generously enough for two concurrent slots (one per
    // route) rather than tuned tightly.
    let pages_per_slot = MAX_CONTEXT.div_ceil(64);
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: pages_per_slot * 2,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 2,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));

    let chunked = match run_route(&model, &pool, prefill_span, forced_tail, PrefillRoute::Chunked, vocab)
    {
        Ok(p) => p,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("chunked route: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let per_token = match run_route(
        &model,
        &pool,
        prefill_span,
        forced_tail,
        PrefillRoute::PerToken,
        vocab,
    ) {
        Ok(p) => p,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("per-token route: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    let result = score_teacher_forced("chunked-vs-per-token", &per_token, &chunked, FIRST_N);
    for m in &result.mismatches {
        eprintln!(
            "self-oracle position {}: chunked={:?} per_token={}",
            m.position, m.predicted, m.expected
        );
    }
    eprintln!(
        "self-oracle: chunked/per-token agreement {}/{} = {:.1}% (floor {:.0}%)",
        result.agree,
        result.compared,
        result.agreement * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    assert!(result.compared > 0, "the span must contribute scored positions");
    assert!(
        meets_g1_floor(result.agreement),
        "self-oracle: chunked vs per-token agreement {}/{} = {:.1}% < {:.0}%. A drop here means \
         the chunk loop, the chunk-boundary state carry, or a newly turned-on multi-token route \
         changed the function being computed.",
        result.agree,
        result.compared,
        result.agreement * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
}
