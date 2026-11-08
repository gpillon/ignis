//! GitHub #489 (ADR 0018 as amended): a load whose prefill scratch is
//! reserved for the solo width runs the prefill chunk on the same kernels as
//! one reserved for the chunk alone. The 27B's 16 GB trellis artifact on the
//! routes an RTX 30 or RTX 40 card runs (the dense prefill and the integer
//! band, named so the RTX 5090 runs them, and our GDN prefill, every card's
//! since GitHub #536), where `auto`
//! reserves a solo width: the same prompt, cut at the same 2,048-token jobs,
//! gives bit-identical last-position logits and the same greedy tokens at a
//! 2,048-token and at an 8,192-token reservation.
//!
//! One materialize, two loads, in one `#[test]`: device weights are freed
//! only when the artifact drops (`chunked_prefill_gpu.rs`). Explicit GPU
//! profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing artifact or
//! GPU is a skip; under it, a failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, FrontendSet, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b_with_decode_lanes};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{decode_program_batch, prefill_program};
use ignis_core::{KvFormat, PrefillAttention, PrefillLinear, RopeScaling};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
/// The profiles' prefill chunk, and the solo width `auto` tries first.
const CHUNK: u32 = 2048;
const SOLO: u32 = 8192;
/// Three jobs at the chunk: two whole, one short.
const SPAN_TOKENS: usize = 5000;
/// At least the solo width, which a load refuses past its context.
const MAX_CONTEXT: u32 = 16_384;
const GENERATED: usize = 4;

fn token_span(frontend: &FrontendSet, len: usize) -> Vec<i32> {
    let filler = "A lighthouse keeper logs the tide, the wind and every ship that passes, one line an hour. ";
    let mut ids: Vec<i32> = Vec::with_capacity(len);
    while ids.len() < len {
        let more = frontend.tokenizer().encode(filler).unwrap_or_else(|e| panic!("tokenize: {e}"));
        ids.extend(more.into_iter().map(|id| i32::try_from(id).expect("token id fits i32")));
    }
    ids.truncate(len);
    ids
}

fn new_pool() -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: KvFormat::HqE8_2b,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64),
            max_context_tokens: MAX_CONTEXT,
            slot_count: 1,
            retained_slot_count: 0,
            retained_host_slot_count: 0,
        },
    )
}

/// `span` prefilled one `CHUNK`-wide job at a time, as the scheduler cuts it
/// while a lane decodes, then `GENERATED` greedy tokens: the last
/// position's logits and the tokens.
fn prefill_at_the_chunk(model: &Model, pool: &SeqPool, span: &[i32], vocab: usize) -> Result<(Vec<f32>, Vec<i32>), String> {
    let mut sequence = pool.alloc(MAX_CONTEXT)?;
    let mut logits = vec![0f32; vocab];
    let jobs: Vec<&[i32]> = span.chunks(CHUNK as usize).collect();
    let mut start = 0u64;
    for (n, job) in jobs.iter().enumerate() {
        let last = n + 1 == jobs.len();
        prefill_program(model, pool, &mut sequence, job, start, last.then_some(&mut logits[..]))?;
        start += job.len() as u64;
    }
    let mut decoded = Vec::with_capacity(GENERATED);
    for _ in 0..GENERATED {
        decoded.extend(decode_program_batch(model, pool, &mut [&mut sequence])?);
    }
    Ok((logits, decoded))
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_prefill_chunk_runs_bit_identically_under_a_solo_width_reservation() {
    let path = Path::new(TRELLIS);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}"));
            return;
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("materialize: {e}"));
            return;
        }
    };
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let span = token_span(&frontend, SPAN_TOKENS);

    let mut runs = Vec::with_capacity(2);
    for reserved in [CHUNK, SOLO] {
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            reserved,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            None,
            None,
            RopeScaling::NONE,
            PrefillAttention::Int8,
            PrefillLinear::Int8,
            1,
        )
        .unwrap_or_else(|e| panic!("load with the scratch at {reserved}: {e}"));
        let pool = new_pool().unwrap_or_else(|e| panic!("seq pool: {e}"));
        match prefill_at_the_chunk(&model, &pool, &span, vocab) {
            Ok(run) => runs.push(run),
            Err(e) => {
                gpu_profile::skip_or_fail(&format!("prefill with the scratch at {reserved}: {e}"));
                return;
            }
        }
    }
    let (narrow_logits, narrow_decoded) = &runs[0];
    let (solo_logits, solo_decoded) = &runs[1];
    let differing = narrow_logits.iter().zip(solo_logits).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(differing, 0, "logits at {CHUNK}-token jobs: {differing} of {vocab} differ between a {CHUNK} and a {SOLO} reservation");
    assert_eq!(narrow_decoded, solo_decoded, "the greedy tokens after the prompt");
    println!("{CHUNK}-token jobs, scratch at {CHUNK} and at {SOLO}: logits bit-identical, tokens {solo_decoded:?}");
}
