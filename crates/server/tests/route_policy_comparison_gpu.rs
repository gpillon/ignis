//! P2-03 (GitHub #85): comparing the engine-default (AllowA4) prefill route
//! against the forced A16-only policy on the *same* long prompt.
//!
//! The A16-only policy is the pre-#85 behaviour (the W8A16 path on every
//! projection), so it plays the role the reference engine plays for the G1
//! canary: a known-good route. The engine-default policy leaves the route
//! decision to the vendored dispatch's own per-projection token thresholds
//! (the A4 multi-token kernels where they exist). The two routes are
//! teacher-forced on the identical span and tail, and their agreement must
//! clear the same G1 floor (>= 95%, ADR 0014) the canary suite uses -- a
//! drop below the floor is filed as its own ticket, never waived here.
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
use ignis_core::step::{ComputePolicy, PrefillRoute, prefill_program_with_policy};

use ignis_bench::oracle::{G1_AGREEMENT_FLOOR, meets_g1_floor, score_teacher_forced};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// Same 8K span the self-oracle test warms over (`.scratch/runtime/specs/
/// 02-real-prefill.md`): eight [1,024-token] traversals of the chunked
/// route, which is where the A4 multi-token kernels actually engage.
const SPAN_TOKENS: usize = 8192;
const PREFILL_CHUNK: u32 = 1024;
/// Teacher-forced window scored after the shared span (same width the G1
/// canary floor uses).
const FIRST_N: usize = 32;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + FIRST_N + 64) as u32;

/// The lowest-token-id argmax tie-break, matching the vendored
/// `ninfer::ops::argmax` kernel's own convention (mirrors
/// `chunked_prefill_self_oracle_gpu.rs::argmax_lowest_id`).
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
/// between two routes on identical inputs, not against any ground truth).
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

/// Prefills `prefill_span` through the chunked route under `policy`, then
/// teacher-forces `forced_tail` one token at a time, returning the route's
/// own teacher-forced next-token prediction at each forced position.
fn run_policy(
    model: &Model,
    pool: &SeqPool,
    prefill_span: &[i32],
    forced_tail: &[i32],
    policy: ComputePolicy,
    vocab: usize,
) -> Result<Vec<u32>, String> {
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    let mut logits = vec![0f32; vocab];
    let mut position = 0u64;
    prefill_program_with_policy(
        model,
        pool,
        &mut sequence,
        prefill_span,
        position,
        PrefillRoute::Chunked,
        policy,
        Some(&mut logits),
    )?;
    position += prefill_span.len() as u64;

    let mut predictions = Vec::with_capacity(forced_tail.len());
    for &token in forced_tail {
        predictions.push(argmax_lowest_id(&logits));
        let forced = [token];
        prefill_program_with_policy(
            model,
            pool,
            &mut sequence,
            &forced,
            position,
            PrefillRoute::Chunked,
            policy,
            Some(&mut logits),
        )?;
        position += 1;
    }
    Ok(predictions)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a4_route_agrees_with_a16_route_on_the_same_prompt() {
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

    // One slot is enough: the two route runs are sequential, and each
    // sequence is released back to the pool when `run_policy` returns.
    let pages_per_slot = MAX_CONTEXT.div_ceil(64);
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: pages_per_slot,
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
        },
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"));

    // The A16-only policy is the known-good pre-#85 route: it plays the
    // oracle's role, the AllowA4 route is what gets scored against it.
    let a16 = match run_policy(
        &model,
        &pool,
        prefill_span,
        forced_tail,
        ComputePolicy::A16Only,
        vocab,
    ) {
        Ok(p) => p,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("A16-only route: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let a4 = match run_policy(
        &model,
        &pool,
        prefill_span,
        forced_tail,
        ComputePolicy::EngineDefault,
        vocab,
    ) {
        Ok(p) => p,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("engine-default route: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    let result =
        score_teacher_forced("a4-vs-a16-route", &a16, &a4, FIRST_N);
    for m in &result.mismatches {
        eprintln!(
            "route-comparison position {}: a16={} a4={:?}",
            m.position, m.expected, m.predicted
        );
    }
    eprintln!(
        "route-comparison: A4-route vs A16-route agreement {}/{} = {:.1}% (floor {:.0}%)",
        result.agree,
        result.compared,
        result.agreement * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
    assert!(result.compared > 0, "the span must contribute scored positions");
    assert!(
        meets_g1_floor(result.agreement),
        "A4 route vs A16 route agreement {}/{} = {:.1}% < {:.0}%. A drop below the G1 \
         floor means the W4A4 route changed the function being computed; file it as its \
         own ticket, never waive it.",
        result.agree,
        result.compared,
        result.agreement * 100.0,
        G1_AGREEMENT_FLOOR * 100.0
    );
}