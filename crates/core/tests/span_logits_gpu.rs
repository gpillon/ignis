//! The measurement readout (`ignis_prefill_options::out_span_logits`,
//! [`prefill_program_span_logits`]): the BF16 logits of every position of a
//! span, which the KLD against the BF16 checkpoint reads (2026-09-24).
//!
//! What it must hold: row `i` is the distribution a prompt ending at
//! `token_ids[i]` gets -- the one [`prefill_program`] reports for that
//! prompt's last position -- bit for bit where the two cut the same chunks,
//! and within a KL bound (the chunk width picks the GEMM routes) where not.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{prefill_program, prefill_program_span_logits};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The serving chunk width, and a span past one chunk of it, so the
/// readout's chunk offset is exercised.
const PREFILL_CHUNK: usize = 1024;
const SPAN_TOKENS: usize = 1100;
const MAX_CONTEXT: u32 = 1152;

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    let lse = max + sum.ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

/// KL(p || q) in nats, both given as log-probabilities.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&lp, &lq)| lp.exp() * (lp - lq)).sum()
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in logits.iter().enumerate() {
        if x > logits[best] {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn every_row_of_the_span_logits_is_that_prefix_prompts_last_position() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    // Real, varied text (this crate's own scheduler), not a repeated filler.
    let text = include_str!("../src/scheduler.rs");
    let span: Vec<i32> = frontend
        .tokenizer()
        .encode(text)
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .take(SPAN_TOKENS)
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    assert_eq!(span.len(), SPAN_TOKENS, "the source text is long enough");

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
            if gpu_profile::skip_or_fail(&format!("materialize the text scope: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK as u32, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64),
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
        },
    )
    .unwrap_or_else(|e| panic!("pool: {e}"));
    let vocab = ModelConfig::qwen38_27b().vocab as usize;

    let mut rows = vec![0u16; SPAN_TOKENS * vocab];
    {
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_span_logits(&model, &pool, &mut sequence, &span, 0, &mut rows)
            .unwrap_or_else(|e| panic!("span logits: {e}"));
    }
    let mut short = vec![0u16; 3];
    {
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        let err = prefill_program_span_logits(&model, &pool, &mut sequence, &span, 0, &mut short)
            .expect_err("a buffer that does not fit the span is refused");
        assert!(err.contains("logits for"), "{err}");
    }

    // Both sides of the 32-column head block and of the 1024-token chunk.
    // (position, bound): where the prefix prompt's chunks are the span's own
    // -- the chunk's last row, and the whole span's -- the row is the prompt's
    // last-position logits bit for bit, whatever the head's block width. Past
    // the chunk boundary the prefix prompt's second chunk is one token wide
    // and the span's 76, and a mid-chunk prefix prompt is one narrower chunk:
    // different GEMM routes, so a KL bound that a row off by one position
    // fails by nats.
    for (position, bound) in [(PREFILL_CHUNK - 1, 0.0), (SPAN_TOKENS - 1, 0.0), (PREFILL_CHUNK, 0.01), (517, 0.2), (31, 0.2)] {
        let row: Vec<f32> = rows[position * vocab..(position + 1) * vocab]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();
        let mut last = vec![0f32; vocab];
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program(&model, &pool, &mut sequence, &span[..=position], 0, Some(&mut last))
            .unwrap_or_else(|e| panic!("prefix prefill at {position}: {e}"));
        drop(sequence);
        let divergence = kl(&log_softmax(&last), &log_softmax(&row));
        println!(
            "position {position}: argmax {} vs {}, KL {divergence:.2e}",
            argmax(&last),
            argmax(&row)
        );
        if bound == 0.0 {
            assert_eq!(last, row, "position {position}: the same chunks, a different row");
        } else {
            assert!(divergence < bound, "position {position}: KL {divergence} against the prefix prompt");
        }
    }
}
