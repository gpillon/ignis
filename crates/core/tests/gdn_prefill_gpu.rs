//! GitHub #514 (spec #503): the GDN prefill on the leaf's own full-rate kernels
//! (`kernel/src/gdn_prefill.h`), the route every card runs, the RTX 5090 since
//! it measured them faster than the vendored chunked kernels with its KLD
//! holding (GitHub #536). On the card, the base 16 GB artifact prefills one
//! prompt in 512-token chunks, once on the vendored kernels, named by the
//! measurement switch `IGNIS_GDN_PREFILL=vendored`, and once with the switch
//! unset:
//!
//! - the leaf reports the route each load took (`Model::gdn_prefill`), and
//!   unset it is ours;
//! - every span logit of our route is finite;
//! - the logits moved (the switch reached the 48 GDN layers);
//! - the two greedy readings agree almost everywhere: both routes compute the
//!   same recurrence, ours in f16 spans folded into fp32 where the vendored one
//!   stores bf16 intermediates, and the leaf test holds each within a few bf16
//!   roundings of an fp64 recurrence (`kernel/tests/test_gdn_prefill.cu`).
//!
//! The FP16 band and the fused prefill, so the test runs on every card the route
//! serves. This binary holds this one test: it sets the process environment
//! before each load. Explicit GPU profile (ADR 0006): outside
//! `IGNIS_GPU_PROFILE=1` a missing artifact or GPU is a skip, under it a failure.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{bind_text_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::load_qwen38_27b_with_decode_lanes;
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::prefill_program_span_logits;
use ignis_core::{KvFormat, PrefillAttention, PrefillLinear, RopeScaling};

const TRELLIS: &str = "F:/ai/models/Qwen3.8-27B-trellis-3bpw-ignis/qwen3_8_27b_trellis_sc30-v2.ninfer";
const CHUNK: u32 = 512;
const MAX_CONTEXT: u32 = 2048;
/// The measurement switch the leaf reads at each load (`kernel/src/model.cu`).
const SWITCH: &str = "IGNIS_GDN_PREFILL";
/// The least share of positions whose argmax the two routes agree on.
const ARGMAX_AGREEMENT_FLOOR: f64 = 0.95;

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
fn our_gdn_prefill_runs_the_48_layers_and_agrees_with_the_vendored_one() {
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
    // Two whole chunks and a ragged third, every one at least the route's 64 tokens and none a
    // multiple of its 32-token chunk at the end.
    let prompt = &prompt[..prompt.len().min(2 * CHUNK as usize + 201)];
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
    // The vendored kernels by name, then the card's own route, ours on every card.
    for (route, switch) in [("vendored", Some("vendored")), ("ours", None)] {
        // SAFETY: this binary holds this one test, which runs on one thread; nothing else reads
        // or writes the environment while it is set, and the leaf reads it inside the load below.
        match switch {
            Some(value) => unsafe { std::env::set_var(SWITCH, value) },
            None => unsafe { std::env::remove_var(SWITCH) },
        }
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
            PrefillLinear::Fp16,
            1,
        )
        .unwrap_or_else(|e| panic!("{route}: model load: {e}"));
        // The review of spec #503: the leaf reports the route the load took.
        assert_eq!(model.gdn_prefill().as_str(), route, "the route the load reports, the switch {switch:?}");
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
        prefill_program_span_logits(&model, &pool, &mut sequence, prompt, 0, &mut span)
            .unwrap_or_else(|e| panic!("{route}: prefill: {e}"));
        logits.push(span);
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var(SWITCH) };
    artifact.release_arena(&mut device).expect("release");

    let (vendored, ours) = (&logits[0], &logits[1]);
    let non_finite = ours.iter().filter(|&&v| !bf16(v).is_finite()).count();
    assert_eq!(non_finite, 0, "our GDN prefill's span logits hold {non_finite} non-finite values");
    let differing = vendored.iter().zip(ours).filter(|(a, b)| a != b).count();
    assert!(differing > 0, "our GDN prefill computed the vendored one's logits: the switch did not reach the layers");
    let agreeing = (0..prompt.len())
        .filter(|&p| argmax(&vendored[p * vocab..(p + 1) * vocab]) == argmax(&ours[p * vocab..(p + 1) * vocab]))
        .count();
    let agreement = agreeing as f64 / prompt.len() as f64;
    assert!(
        agreement >= ARGMAX_AGREEMENT_FLOOR,
        "the two GDN routes agree on the argmax at {agreeing} of {} positions",
        prompt.len()
    );
    println!(
        "gdn prefill: {} prompt tokens in {} chunks of {CHUNK}; {differing} of {} logits moved, none non-finite; \
         argmax agreed at {agreeing} of {}",
        prompt.len(),
        prompt.len().div_ceil(CHUNK as usize),
        prompt.len() * vocab,
        prompt.len()
    );
}
