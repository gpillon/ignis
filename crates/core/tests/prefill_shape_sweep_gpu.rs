//! The prefill wall time of `ignis_core::step::prefill_program` under the
//! shape `ignis-server` loads with, next to the shape the P2-04 chunk-timing
//! diagnostic loads with (GitHub #86). Both reach the identical FFI entry
//! (`ignis_program_prefill`, chunked route, NULL options), so what this
//! separates is the load-time shape -- `ignis_model_load`'s `max_context`
//! bound and the `SeqPool` budget -- and cold-vs-warm.
//!
//! Written to settle GitHub #93. The first G2 gate run measured ~18.7 s for
//! a 1,024-token prompt against the server and concluded the engine was
//! ~175x slower than the reference, while #86's diagnostic reported 94.9 ms
//! per 1,024-token chunk in-process. This sweep showed the kernel meets the
//! diagnostic's number under the server's own shape, cold, which placed the
//! fault above the FFI: a stale kernel archive (GitHub #94), from before the
//! chunk loop of #84, still walking the span one token at a time.
//!
//! Kept as a standing check because nothing else measures prefill under the
//! shape the server actually loads: a regression that only shows up at the
//! server's `max_context` or pool budget would otherwise reach a gate run
//! before anyone saw it. Prints; asserts only that the calls succeed.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

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
const PREFILL_CHUNK: u32 = 1024;

/// (label, max_context_tokens, kv_pool_tokens, slot_count)
///
/// `diagnostic` is `chunk_timing_diagnostic_gpu`'s shape verbatim;
/// `server` is what `ignis-server`'s startup line reported for the #93
/// run ("prefill chunk 1024 tokens, context 40960 tokens, KV pool 65536
/// tokens", 8 decode lanes); `server-ctx` isolates the context bound from
/// the pool/slot sizing.
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("diagnostic", 8_256, 8_256 * 2, 2),
    ("server-ctx", 40_960, 8_256 * 2, 2),
    ("server", 40_960, 65_536, 8),
];

/// The spans to prefill under each shape, in tokens.
const SPANS: &[usize] = &[1_024, 8_192];

fn pages_for(tokens: u32) -> u32 {
    tokens.div_ceil(64)
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
fn prefill_wall_time_across_the_load_shapes() {
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
    // Materialized once: the ~19 GB weight arena is the same for every
    // shape below (only `ignis_model_load`'s scratch reservation and the
    // pool differ), and a second upload would not fit beside the first.
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let cfg = ModelConfig::qwen38_27b();
    let longest = *SPANS.iter().max().expect("at least one span");
    let span = token_span(&frontend, longest);

    println!("--- #93 prefill shape sweep (chunk {PREFILL_CHUNK}) ---");
    for &(label, max_context, kv_pool_tokens, slot_count) in SHAPES {
        let model = load_qwen38_27b(&reader, &artifact, &handles, PREFILL_CHUNK, max_context)
            .unwrap_or_else(|e| panic!("{label}: model load: {e}"));
        let pool = SeqPool::create(
            &cfg,
            &SeqPoolBudget {
                kv_page_group_count: pages_for(kv_pool_tokens),
                max_context_tokens: max_context,
                slot_count,
            },
        )
        .unwrap_or_else(|e| panic!("{label}: seq pool create: {e}"));

        for &tokens in SPANS {
            let chunks = tokens.div_ceil(PREFILL_CHUNK as usize);
            // Two identical prefills on two fresh sequences: the first is
            // what the server's first request sees, the second is what
            // the #86 diagnostic reports.
            for (pass, name) in ["cold", "warm"].iter().enumerate() {
                let context = u32::try_from(tokens).expect("span fits u32") + 64;
                let mut seq = pool
                    .alloc(context)
                    .unwrap_or_else(|e| panic!("{label}/{tokens}/{name}: seq alloc: {e}"));
                let start = Instant::now();
                prefill_program(&model, &pool, &mut seq, &span[..tokens], 0, None)
                    .unwrap_or_else(|e| panic!("{label}/{tokens}/{name}: prefill: {e}"));
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                println!(
                    "shape={label:<11} ctx={max_context:<6} pool={kv_pool_tokens:<6} \
                     slots={slot_count} span={tokens:<5} pass={name:<4} \
                     total={elapsed:>10.1} ms  per_chunk={:>8.1} ms  per_token={:>7.3} ms",
                    elapsed / chunks as f64,
                    elapsed / tokens as f64,
                );
                let _ = pass;
            }
        }
    }
    println!("--- end sweep ---");
}
