//! GitHub #92, acceptance criterion 1: where the ~94.9 ms of per-chunk wall
//! time in the default chunked prefill route actually goes.
//!
//! Two sweeps, run back to back off one materialized artifact:
//!
//! 1. **A chunk-width sweep over a fixed span.** The same 8,192-token span is
//!    prefilled at a range of chunk widths. A span costs
//!    `chunks * fixed + span_tokens * per_token` if there is a real fixed
//!    per-chunk-boundary cost, so regressing total wall on the chunk count
//!    reads that fixed cost off directly -- model-free, no instrumentation,
//!    nothing to believe about the leaf's internals. This is the one that
//!    answers the ticket's question: a route dominated by the '8 semaphores'
//!    has a large `fixed`, one that is compute-bound has a `fixed` near zero.
//!    Kept as a standing check because it is the only thing in the suite that
//!    would notice a per-boundary cost appearing, which is exactly what
//!    changing the synchronization contract would introduce.
//!
//! 2. **A traversal-width sweep over isolated spans.** Each span is prefilled
//!    on its own, from position zero, in exactly one chunk of its own width,
//!    so `ms/token` is the cost of a traversal of that many tokens with no KV
//!    prefix behind it. Sweep 1 cannot answer that question: a 256-token
//!    chunk there still attends to up to 8K of prefix that a standalone
//!    256-token request would never carry. Kept as a standing check because
//!    it is the sharpest regression signal on the prefill path -- the engine
//!    switches kernel route with token count, and if it stopped crossing into
//!    the wide route at 1,024 tokens the per-token cost would jump ~25 % and
//!    this table would show it.
//!
//! Orthogonal to both, and driven independently: with `IGNIS_CHUNK_PROFILE`
//! naming a file, `run_program_chunk` (`kernel/src/step.cu`) appends one JSONL
//! record per chunk splitting it into host enqueue time, the
//! `cudaStreamSynchronize` stall, the device idle the forced sync opens at the
//! chunk boundary, the 64 layer bodies' own device spans, and the device idle
//! *between* layer bodies -- the ticket's (a) dispatch / (b) sync / (c)
//! compute buckets. That happens whatever this test is doing; the JSONL is
//! analyzed out of band.
//!
//! Both sweeps print the leaf's own VRAM reservation per row, so a run that
//! came close to the card's limit says so rather than being inferred later.
//!
//! **One test function on purpose.** Both sweeps need the materialized
//! artifact, and two `materialize` calls do not fit on a 32 GiB card at this
//! model's size: the second leaves `ignis_model_load` with zero free bytes.
//! Integration tests in separate files get separate processes and never meet
//! this; two tests in one file share one. So this is one test that sets up
//! once, not two that each set up.
//!
//! Asserts only that every call succeeded: a diagnostic, not a gate.
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
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{SeqPool, SeqPoolBudget};
use ignis_core::step::{prefill_program, program_stats};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
/// The G2 gate's 8K cell, the span every existing reading on this ticket
/// (#86's diagnostic, #88's sweep) was taken over. Sweep 1's span.
const SPAN_TOKENS: usize = 8192;
const MAX_CONTEXT: u32 = (SPAN_TOKENS + 64) as u32;
/// Chunk widths sweep 1 visits by default. 1,024 is the production width and
/// 256 the 32-chunk case.
///
/// It stops at 4,096 deliberately. The leaf reserves its prefill scratch
/// arena for the chunk width at load (P2-01, GitHub #83), so an 8,192-wide
/// chunk takes the run's footprint from 18.6 GiB to 19.9 GiB on a 32 GiB
/// card -- and `ignis_model_load`'s guard checks `cudaMemGetInfo`, which on
/// WDDM lets an over-large reservation be served from system memory instead
/// of refusing it (GitHub #75 is what that looks like). A standing check has
/// no business running that close to the edge. The single-chunk case is one
/// `IGNIS_DECOMP_WIDTHS=8192` away when someone wants it.
const WIDTHS: &[u32] = &[256, 512, 1024, 2048, 4096];
/// Span lengths sweep 2 visits by default, each prefilled on its own in a
/// single chunk of its own width. Stops at 4,096 for the reason `WIDTHS`
/// does; `IGNIS_DECOMP_SPANS=8192` restores the widest row.
const SPANS: &[usize] = &[256, 512, 1024, 2048, 4096];
/// Timed repetitions per row, after one untimed warm-up at that shape.
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

/// A comma-separated environment override, or the compiled-in default.
fn list_from_env<T>(name: &str, fallback: &[T]) -> Vec<T>
where
    T: Copy + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(list) => list
            .split(',')
            .filter(|piece| !piece.trim().is_empty())
            .map(|piece| piece.trim().parse().unwrap_or_else(|e| panic!("{name}: {e}")))
            .collect(),
        Err(_) => fallback.to_vec(),
    }
}

/// The leaf's own reservation for this shape, in GiB: weights, prefill
/// scratch arena, KV and GDN pools, sampling and decode-graph buffers.
///
/// Printed per row because the arena is sized for the traversal width at load
/// (P2-01, GitHub #83) and grows with it, while `ignis_model_load`'s guard
/// compares against `cudaMemGetInfo` -- which on WDDM lets an over-large
/// reservation be served from system memory rather than refusing it. A row
/// that paid for that over PCIe is not a row to trust, and this column is how
/// a later reader can tell.
fn leaf_vram_gib(model: &Model, pool: &SeqPool, shape: usize) -> f64 {
    program_stats(model, pool)
        .unwrap_or_else(|e| panic!("program_stats({shape}): {e}"))
        .vram_bytes as f64
        / (1024.0 * 1024.0 * 1024.0)
}

#[test]
#[ignore = "GPU profile only: .scratch/issue-92/run.ps1"]
fn prefill_chunk_and_traversal_sweeps() {
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
    // The one materialization both sweeps share -- see the module note.
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
    let mut logits = vec![0f32; vocab];

    // `IGNIS_DECOMP_REPS=1` shortens the run for an external profiler that
    // serializes every launch (Nsight Compute), where three timed reps cost
    // minutes for no extra signal.
    let reps: usize = std::env::var("IGNIS_DECOMP_REPS")
        .ok()
        .map(|v| v.trim().parse().unwrap_or_else(|e| panic!("IGNIS_DECOMP_REPS: {e}")))
        .unwrap_or(REPS);
    // `IGNIS_DECOMP_WIDTHS=1024` / `IGNIS_DECOMP_SPANS=1024` narrow a sweep to
    // one row, so an external profiler can be pointed at a run containing
    // nothing but the shape under study.
    let widths: Vec<u32> = list_from_env("IGNIS_DECOMP_WIDTHS", WIDTHS);
    let spans: Vec<usize> = list_from_env("IGNIS_DECOMP_SPANS", SPANS);

    let budget = SeqPoolBudget {
        kv_format: ignis_core::KvFormat::Bf16,
        kv_page_group_count: pages_for(MAX_CONTEXT) * 2,
        max_context_tokens: MAX_CONTEXT,
        slot_count: 2,
    };
    let longest = spans.iter().copied().max().unwrap_or(SPAN_TOKENS).max(SPAN_TOKENS);
    let corpus = token_span(&frontend, longest);

    // ---------------------------------------------------------------------
    // Sweep 1: chunk width over a fixed 8,192-token span.
    // ---------------------------------------------------------------------
    let span = &corpus[..SPAN_TOKENS];
    println!("#92 decomposition: span {SPAN_TOKENS} tokens, {reps} timed reps per width");
    println!("width  chunks   wall_ms(mean)   wall_ms(min)   ms/chunk   ms/token   leaf VRAM GiB");
    for &width in &widths {
        // One model per width: the chunk width is fixed at load (it sizes the
        // leaf's scratch arena), so the sweep has to reload. The materialized
        // weights are shared -- only the arena is re-reserved.
        let model = load_qwen38_27b(&reader, &artifact, &handles, width, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("ignis_model_load(chunk={width}): {e}"));
        let pool = SeqPool::create(&cfg, &budget)
            .unwrap_or_else(|e| panic!("seq pool create(chunk={width}): {e}"));

        // Warm-up: the first span at a width faults in this arena's pages and
        // this width's kernel configurations. Never timed.
        {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            prefill_program(&model, &pool, &mut seq, span, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("warm-up prefill(chunk={width}): {e}"));
        }
        let mut walls_ms: Vec<f64> = Vec::with_capacity(reps);
        for rep in 0..reps {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let start = Instant::now();
            prefill_program(&model, &pool, &mut seq, span, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("prefill(chunk={width}, rep={rep}): {e}"));
            walls_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let chunks = SPAN_TOKENS.div_ceil(width as usize);
        let mean = walls_ms.iter().sum::<f64>() / walls_ms.len() as f64;
        let min = walls_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "{width:5}  {chunks:6}   {mean:13.1}   {min:12.1}   {:10.2}   {:10.4}   {:11.2}",
            mean / chunks as f64,
            mean / SPAN_TOKENS as f64,
            leaf_vram_gib(&model, &pool, width as usize),
        );
    }

    // ---------------------------------------------------------------------
    // Sweep 2: tokens per traversal, isolated spans, no KV prefix.
    //
    // Read it as: packing N requests of L tokens gives every projection and
    // FFN GEMM the shapes of a single N*L-token traversal. The row at N*L
    // against the row at L bounds what that packing can buy. It is a
    // conservative bound -- one N*L-token span does *more* attention work
    // than N separate L-token spans -- so a real packed traversal can only
    // beat the figure here.
    // ---------------------------------------------------------------------
    println!("#92 packing proxy: one span, one chunk, no KV prefix");
    println!("tokens  wall_ms(min)   ms/token   vs first   leaf VRAM GiB");
    let mut baseline: Option<f64> = None;
    for &tokens in &spans {
        let width = u32::try_from(tokens).expect("span fits u32");
        let model = load_qwen38_27b(&reader, &artifact, &handles, width, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("ignis_model_load(chunk={width}): {e}"));
        let pool = SeqPool::create(&cfg, &budget)
            .unwrap_or_else(|e| panic!("seq pool create({tokens}): {e}"));
        let prompt = &corpus[..tokens];

        {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            prefill_program(&model, &pool, &mut seq, prompt, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("warm-up prefill({tokens}): {e}"));
        }
        let mut walls_ms: Vec<f64> = Vec::with_capacity(reps);
        for rep in 0..reps {
            let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("seq alloc: {e}"));
            let start = Instant::now();
            prefill_program(&model, &pool, &mut seq, prompt, 0, Some(&mut logits))
                .unwrap_or_else(|e| panic!("prefill({tokens}, rep={rep}): {e}"));
            walls_ms.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let min = walls_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let per_token = min / tokens as f64;
        let base = *baseline.get_or_insert(per_token);
        println!(
            "{tokens:6}  {min:12.1}   {per_token:8.4}   {:8.2}x   {:11.2}",
            base / per_token,
            leaf_vram_gib(&model, &pool, tokens),
        );
    }
    println!("#92 decomposition: done");
}
