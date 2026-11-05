//! GitHub #509 (spec #503): the load option `prefill_linear` runs the trellis
//! body's prefill chunks on the dense prefill (`kernel/include/ignis_dense_prefill.h`).
//! On the card, the base 16 GB artifact prefills one prompt in 512-token
//! chunks, once on the fused prefill (the default) and once on the dense one:
//!
//! - the dense load holds its one buffer, the body's largest matrix in INT8
//!   (`trellis_dense_bytes`), and the fused load none;
//! - every span logit of the dense load is finite;
//! - the logits moved (the option reached the body), and the two greedy
//!   readings agree almost everywhere: the dense prefill's numerics are the
//!   prototype's W8 and per-token A8, which the KLD gate measured within a
//!   percent or two of the fused prefill (spec #503);
//! - GitHub #511: the dense prefill with its A8 fused into the producers of
//!   its inputs (the default) gives every span logit of a third run with the
//!   standalone A8 passes (`trellis::set_dense_prefill_fused_producers(false)`)
//!   bit for bit: the fusion changes the passes, not the values.
//!
//! The FP16 band, so the test runs on every card the route serves.
//! Explicit GPU profile (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing
//! artifact or GPU is a skip, under it a failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_decode_lanes;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program_span_logits;
use ignis_core::trellis;
use ignis_core::{KvFormat, PrefillAttention, PrefillLinear, RopeScaling};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const CHUNK: u32 = 512;
const MAX_CONTEXT: u32 = 2048;
/// The 27B's largest trellis matrix, 5120 x 17408, in INT8.
const DENSE_BYTES: u64 = 5_120 * 17_408;
/// The least share of positions whose argmax the two routes agree on.
const ARGMAX_AGREEMENT_FLOOR: f64 = 0.9;

fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn argmax(row: &[u16]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if bf16(v) > bf16(row[best]) {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_dense_prefill_runs_the_trellis_body_and_agrees_with_the_fused_one() {
    let path = Path::new(TRELLIS);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {TRELLIS}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let text = "A paged KV cache hands out 64-token pages from a free list; a sequence that grows past its \
                last page takes another, and one that ends gives every page back. Prefix sharing lets two \
                requests point at the same pages until one of them writes. "
        .repeat(24);
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode(&text)
        .unwrap_or_else(|e| panic!("tokenize prompt: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();
    // Two whole chunks and a ragged third, every one wider than the decode route.
    let prompt = &prompt[..prompt.len().min(2 * CHUNK as usize + 200)];
    assert!(prompt.len() > 2 * CHUNK as usize + 64, "{} tokens", prompt.len());

    let (plan, handles) = bind_text_scope_27b(&reader).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = Vec::new();
    // The fused prefill, the dense one with its A8 fused into its producers
    // (the default), and the dense one with the standalone A8 passes.
    assert!(trellis::dense_prefill_fused_producers(), "the dense prefill fuses A8 into its producers by default");
    for (route, fused_producers) in [(PrefillLinear::Fp16, true), (PrefillLinear::Int8, true), (PrefillLinear::Int8, false)] {
        trellis::set_dense_prefill_fused_producers(fused_producers);
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            CHUNK,
            MAX_CONTEXT,
            KvFormat::HqE8_2b,
            None,
            None,
            RopeScaling::NONE,
            PrefillAttention::Fp16,
            route,
            1,
        )
        .unwrap_or_else(|e| panic!("{route}: model load: {e}"));
        let reserved = model.stats().reserved;
        let want = if route == PrefillLinear::Int8 { DENSE_BYTES } else { 0 };
        assert_eq!(reserved.trellis_dense_bytes, want, "{route}: the dense buffer's line");
        let pool = SeqPool::create(
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
        .unwrap_or_else(|e| panic!("{route}: seq pool: {e}"));
        let mut sequence = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{route}: alloc: {e}"));
        let mut span = vec![0u16; prompt.len() * vocab];
        let launches = trellis::dense_prefill_fused_norm_launches();
        prefill_program_span_logits(&model, &pool, &mut sequence, prompt, 0, &mut span)
            .unwrap_or_else(|e| panic!("{route}: prefill: {e}"));
        // The fused RMSnorm ran on the dense prefill under the switch alone.
        let fused_norms = trellis::dense_prefill_fused_norm_launches() - launches;
        if route == PrefillLinear::Int8 && fused_producers {
            assert!(fused_norms > 0, "{route}: the fused producers ran no fused RMSnorm");
        } else {
            assert_eq!(fused_norms, 0, "{route}, fused producers {fused_producers}: a fused RMSnorm ran");
        }
        logits.push(span);
    }
    trellis::set_dense_prefill_fused_producers(true);
    artifact.release_arena(&mut device).expect("release");

    let (fused, dense, standalone) = (&logits[0], &logits[1], &logits[2]);
    let unfused = dense.iter().zip(standalone).filter(|(a, b)| a != b).count();
    assert_eq!(
        unfused, 0,
        "{unfused} span logits of the dense prefill with its A8 fused into its producers differ from the standalone A8's"
    );
    let non_finite = dense.iter().filter(|&&v| !bf16(v).is_finite()).count();
    assert_eq!(non_finite, 0, "the dense prefill's span logits hold {non_finite} non-finite values");
    let differing = fused.iter().zip(dense).filter(|(a, b)| a != b).count();
    assert!(differing > 0, "the dense prefill computed the fused one's logits: the option did not reach the body");
    let agreeing = (0..prompt.len())
        .filter(|&p| argmax(&fused[p * vocab..(p + 1) * vocab]) == argmax(&dense[p * vocab..(p + 1) * vocab]))
        .count();
    let agreement = agreeing as f64 / prompt.len() as f64;
    assert!(
        agreement >= ARGMAX_AGREEMENT_FLOOR,
        "the two routes agree on the argmax at {agreeing} of {} positions",
        prompt.len()
    );
    println!(
        "prefill linear: {} prompt tokens in {} chunks of {CHUNK}; dense buffer {DENSE_BYTES} bytes; {differing} of {} \
         logits moved, none non-finite; argmax agreed at {agreeing} of {}; the fused producers' logits are the \
         standalone A8's bit for bit",
        prompt.len(),
        prompt.len().div_ceil(CHUNK as usize),
        prompt.len() * vocab,
        prompt.len()
    );
}
