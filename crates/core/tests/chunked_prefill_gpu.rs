//! GPU integration coverage for P2-02's chunked prefill route (GitHub #84,
//! ADR 0016): the span-split property and chunk-width invariance the
//! chunked path must hold so G4's prefix reuse and a chunk-size sweep can
//! rest on them later, plus determinism across fresh loads at chunk width.
//! The chunked-vs-per-token self-oracle lives in
//! `crates/server/tests/chunked_prefill_self_oracle_gpu.rs` (it needs the
//! G1 scoring helpers in `ignis-bench`, which this crate does not depend
//! on); the G1 canary floor re-measured on the (now default) chunked route
//! is `crates/server/tests/oracle_teacher_forced_gpu.rs`, unchanged.
//!
//! One `#[test]`, not three: each model load's device weights are never
//! freed until the artifact drops (`ignis_device_destroy` does not free
//! materialized VRAM, mirroring `openai_http_gpu.rs`'s own note on the same
//! constraint), so three independent `materialize()` calls in one test
//! *binary* -- which share a process regardless of how many `#[test]` fns
//! they are split across -- would exhaust a 32 GB card. One shared
//! materialize, several `Model` loads against it, in one test.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// Long enough to span several chunks at every chunk width this file uses,
/// and to let a split land on a boundary that is not a chunk multiple.
const SPAN_TOKENS: usize = 1536;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + 64) as u32;
const GENERATED: usize = 4;

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

fn pages_for(max_context_tokens: u32) -> u32 {
    max_context_tokens.div_ceil(64)
}

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: pages_for(MAX_CONTEXT) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
    )
}

/// Prefills `span` through `prefill_chunk_tokens`-wide chunks (the
/// production default route) then decodes `GENERATED` greedy tokens.
/// Returns the span's last-position logits and the decoded ids.
fn prefill_whole_then_decode(
    model: &Model,
    pool: &SeqPool,
    span: &[i32],
    vocab: usize,
) -> Result<(Vec<f32>, Vec<i32>), String> {
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    let mut logits = vec![0f32; vocab];
    prefill_program(model, pool, &mut sequence, span, 0, Some(&mut logits))?;
    let mut decoded = Vec::with_capacity(GENERATED);
    for _ in 0..GENERATED {
        decoded.extend(decode_program_batch(model, pool, &mut [&mut sequence])?);
    }
    Ok((logits, decoded))
}

/// Prefills `span` as two consecutive calls split at `split_at` (not
/// necessarily a chunk multiple), then decodes `GENERATED` greedy tokens.
fn prefill_split_then_decode(
    model: &Model,
    pool: &SeqPool,
    span: &[i32],
    split_at: usize,
    vocab: usize,
) -> Result<(Vec<f32>, Vec<i32>), String> {
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    let mut logits = vec![0f32; vocab];
    let (head, tail) = span.split_at(split_at);
    prefill_program(model, pool, &mut sequence, head, 0, None)?;
    prefill_program(
        model,
        pool,
        &mut sequence,
        tail,
        head.len() as u64,
        Some(&mut logits),
    )?;
    let mut decoded = Vec::with_capacity(GENERATED);
    for _ in 0..GENERATED {
        decoded.extend(decode_program_batch(model, pool, &mut [&mut sequence])?);
    }
    Ok((logits, decoded))
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_id = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (id, &v) in logits.iter().enumerate() {
        if v > best {
            best = v;
            best_id = id;
        }
    }
    best_id
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn chunked_prefill_holds_its_span_split_width_and_determinism_properties() {
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
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let span = token_span(&frontend, SPAN_TOKENS);

    // --- span split at a non-chunk boundary -------------------------------
    {
        const PREFILL_CHUNK: u32 = 512;
        const SPLIT_AT: usize = 700; // not a multiple of PREFILL_CHUNK
        assert_ne!(SPLIT_AT % PREFILL_CHUNK as usize, 0);

        let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
        let pool = new_pool(2).unwrap_or_else(|e| panic!("seq pool create: {e}"));

        let (one_span_logits, one_span_decoded) =
            match prefill_whole_then_decode(&model, &pool, &span, vocab) {
                Ok(r) => r,
                Err(e) => {
                    if gpu_profile::skip_or_fail(&format!("one-span prefill: {e}")) {
                        return;
                    }
                    unreachable!("skip_or_fail panics under the profile");
                }
            };
        let (split_logits, split_decoded) =
            match prefill_split_then_decode(&model, &pool, &span, SPLIT_AT, vocab) {
                Ok(r) => r,
                Err(e) => {
                    if gpu_profile::skip_or_fail(&format!("split prefill: {e}")) {
                        return;
                    }
                    unreachable!("skip_or_fail panics under the profile");
                }
            };

        assert_eq!(
            argmax(&one_span_logits),
            argmax(&split_logits),
            "one span and a two-call split at a non-chunk boundary must predict the same next token"
        );
        assert_eq!(
            one_span_decoded, split_decoded,
            "one span and a two-call split at a non-chunk boundary must decode the same first tokens"
        );
    }

    // --- chunk width is a performance knob, not a correctness parameter ---
    {
        const NARROW_CHUNK: u32 = 256;
        const WIDE_CHUNK: u32 = 512;
        let mut results = Vec::with_capacity(2);
        for chunk in [NARROW_CHUNK, WIDE_CHUNK] {
            let model = load_qwen38_27b(&reader, &artifact, &handles, chunk, MAX_CONTEXT)
                .unwrap_or_else(|e| panic!("ignis_model_load(chunk={chunk}): {e}"));
            let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create(chunk={chunk}): {e}"));
            match prefill_whole_then_decode(&model, &pool, &span, vocab) {
                Ok(r) => results.push(r),
                Err(e) => {
                    if gpu_profile::skip_or_fail(&format!("prefill(chunk={chunk}): {e}")) {
                        return;
                    }
                    unreachable!("skip_or_fail panics under the profile");
                }
            }
        }
        let (narrow_logits, narrow_decoded) = &results[0];
        let (wide_logits, wide_decoded) = &results[1];
        assert_eq!(
            argmax(narrow_logits),
            argmax(wide_logits),
            "a {NARROW_CHUNK}-token and a {WIDE_CHUNK}-token prefill chunk must predict the same \
             next token for the same prompt"
        );
        assert_eq!(
            narrow_decoded, wide_decoded,
            "a {NARROW_CHUNK}-token and a {WIDE_CHUNK}-token prefill chunk must decode the same \
             first tokens for the same prompt"
        );
    }

    // --- determinism across fresh loads, at a genuinely multi-chunk width -
    {
        const PREFILL_CHUNK: u32 = 512;
        assert!(
            (span.len() as u32) > PREFILL_CHUNK,
            "the prompt must actually span more than one chunk"
        );
        let mut runs = Vec::with_capacity(2);
        for _ in 0..2 {
            let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, MAX_CONTEXT)
                .unwrap_or_else(|e| panic!("ignis_model_load: {e}"));
            let pool = new_pool(1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
            match prefill_whole_then_decode(&model, &pool, &span, vocab) {
                Ok((_, decoded)) => runs.push(decoded),
                Err(e) => {
                    if gpu_profile::skip_or_fail(&format!("prefill: {e}")) {
                        return;
                    }
                    unreachable!("skip_or_fail panics under the profile");
                }
            }
        }
        assert_eq!(
            runs[0], runs[1],
            "two fresh model loads chunk-prefilling the same long prompt must decode identical tokens"
        );
    }
}
