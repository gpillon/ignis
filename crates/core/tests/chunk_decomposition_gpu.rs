//! GitHub #92, acceptance criterion 1: where the ~94.9 ms of per-chunk wall
//! time in the default chunked prefill route actually goes.
//!
//! Two independent measurements of the same question, so neither has to be
//! taken on faith:
//!
//! 1. **A chunk-width sweep over a fixed span.** The same 8,192-token span is
//!    prefilled at chunk widths 256..8192. A span costs
//!    `chunks * fixed + span_tokens * per_token` if there is a real fixed
//!    per-chunk-boundary cost, so regressing total wall on the chunk count
//!    reads that fixed cost off directly -- model-free, no instrumentation,
//!    nothing to believe about the leaf's internals. This is the one that
//!    answers the ticket's question: a route dominated by the '8 semaphores'
//!    has a large `fixed`, one that is compute-bound has a `fixed` near zero.
//!
//! 2. **The leaf's own event-level decomposition.** With `IGNIS_CHUNK_PROFILE`
//!    naming a file, `run_program_chunk` (kernel/src/step.cu) appends one
//!    JSONL record per chunk splitting that chunk into host enqueue time, the
//!    `cudaStreamSynchronize` stall, the device idle the forced sync opens at
//!    the chunk boundary, the 64 layer bodies' own device spans, and the
//!    device idle *between* layer bodies. That splits the chunk into the
//!    ticket's (a) dispatch / (b) sync / (c) compute buckets.
//!
//! This test drives both and prints; the JSONL is analyzed out of band. It
//! asserts only that every call succeeded -- it is a diagnostic, not a gate.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**.

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
/// The G2 gate's 8K cell, the span every existing reading on this ticket
/// (#86's diagnostic, #88's sweep) was taken over.
const SPAN_TOKENS: usize = 8192;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + 64) as u32;
/// Chunk widths the sweep visits. 1,024 is the production default; 8,192
/// is the degenerate single-chunk case (one synchronization for the whole
/// span) and 256 the 32-chunk case.
const WIDTHS: &[u32] = &[256, 512, 1024, 2048, 4096, 8192];
/// Timed repetitions per width, after one untimed warm-up at that width.
const REPS: usize = 3;

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
#[ignore = "GPU profile only: .scratch/issue-92/run.ps1"]
fn chunk_wall_time_decomposition() {
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
    let cfg = ModelConfig::qwen38_27b();
    let vocab = cfg.vocab as usize;
    let span = token_span(&frontend, SPAN_TOKENS);
    let mut logits = vec![0f32; vocab];

    // `IGNIS_DECOMP_WIDTHS=1024` narrows the sweep to one width, so an
    // external profiler (Nsight Systems) can be pointed at a run that
    // contains nothing but the width under study.
    let widths: Vec<u32> = match std::env::var("IGNIS_DECOMP_WIDTHS") {
        Ok(list) => list
            .split(',')
            .filter(|piece| !piece.trim().is_empty())
            .map(|piece| piece.trim().parse().unwrap_or_else(|e| panic!("IGNIS_DECOMP_WIDTHS: {e}")))
            .collect(),
        Err(_) => WIDTHS.to_vec(),
    };
    println!("#92 decomposition: span {SPAN_TOKENS} tokens, {REPS} timed reps per width");
    println!("width  chunks   wall_ms(mean)   wall_ms(min)   ms/chunk(mean)   ms/token(mean)");
    for &width in &widths {
        // One model per width: the chunk width is fixed at load (it sizes the
        // leaf's scratch arena), so the sweep has to reload. The materialized
        // weights are shared -- only the arena is re-reserved.
        let model = load_qwen38_27b(&reader, &artifact, &handles, width, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("ignis_model_load(chunk={width}): {e}"));
        let pool = SeqPool::create(
            &cfg,
            &SeqPoolBudget {
                kv_page_group_count: pages_for(MAX_CONTEXT) * 2,
                max_context_tokens: MAX_CONTEXT,
                slot_count: 2,
            },
        )
        .unwrap_or_else(|e| panic!("seq pool create(chunk={width}): {e}"));

        // Warm-up: the first span at a width faults in this arena's pages and
        // this width's kernel configurations. Never timed.
        {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            prefill_program(&model, &pool, &mut seq, &span, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("warm-up prefill(chunk={width}): {e}"));
        }

        let mut walls_ms: Vec<f64> = Vec::with_capacity(REPS);
        for rep in 0..REPS {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let start = Instant::now();
            prefill_program(&model, &pool, &mut seq, &span, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("prefill(chunk={width}, rep={rep}): {e}"));
            walls_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let chunks = SPAN_TOKENS.div_ceil(width as usize);
        let mean = walls_ms.iter().sum::<f64>() / walls_ms.len() as f64;
        let min = walls_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{width:5}  {chunks:6}   {mean:13.1}   {min:12.1}   {:14.2}   {:14.4}",
            mean / chunks as f64,
            mean / SPAN_TOKENS as f64,
        );
    }
    println!("#92 decomposition: done");
}
