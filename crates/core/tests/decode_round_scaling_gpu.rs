//! Decode-round scaling measurement (GitHub #111): what a decode round costs at batch
//! 4 relative to its own width-1 round.
//!
//! This is the issue's own no-profiler check that the round is a single
//! batch-wide traversal of the model rather than B traversals. #111 measured
//! ignis at **4.81x** its own B=1 round while the live reference paid 1.07x,
//! which is the signature of streaming the weights once per lane; requirement
//! 17 of `.scratch/runtime/specs/03-serving-loop.md` asks for the reference's
//! shape. A round that traverses the model once adds only the per-lane
//! attention, recurrence and sampling work as the width grows, so the ratio
//! sits near one and nowhere near the width.
//!
//! The threshold below is deliberately loose. It is a regression guard on the
//! *shape* of the round -- it must fail if the per-lane loop ever comes back,
//! and it must not fail on ordinary run-to-run variance or on a debug build's
//! host overhead. The printed ratio is the number worth reading; the gate's
//! own ITL cell (GitHub #110) is measured live/live against the reference and
//! is not this test's job.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;
use std::time::{Duration, Instant};

use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    SamplingParams, capture_decode_graphs, decode_program_batch_sampled, prefill_program_sampled,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 256;
const PROMPT_TOKENS: usize = 64;
/// Rounds timed per width after the warm-up. The median is reported, so a
/// single scheduling hiccup does not decide the reading.
const TIMED_ROUNDS: usize = 24;
const WARM_ROUNDS: usize = 8;
/// The measured width, and the width the G3 gate's C=4 cell runs.
const WIDE: usize = 4;
/// A width-4 round may cost no more than this multiple of the width-1 round.
/// The per-lane loop this ticket removed cost 4.81x; one traversal costs
/// near 1x plus the per-lane attention and sampling work.
const MAX_RATIO: f64 = 2.0;

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format: ignis_core::KvFormat::Bf16,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
    )
}

fn prompt_for(lane: usize) -> Vec<i32> {
    (0..PROMPT_TOKENS).map(|i| (11 + lane * 13 + i) as i32).collect()
}

/// Median wall time of one decode round at `width`, with every lane already
/// prefilled and the width's decode graph captured -- the production path.
///
/// The pool is the caller's, and it must be the one the graphs were captured
/// against: a captured graph holds the pool's KV planes, block-table matrix
/// and GDN state planes at their capture-time addresses (ADR 0019), so
/// replaying it against a pool allocated later reads freed memory.
fn median_round(model: &Model, pool: &SeqPool, width: usize) -> Duration {
    let mut sequences: Vec<Seq<'_>> = (0..width)
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for (lane, seq) in sequences.iter_mut().enumerate() {
        prefill_program_sampled(
            model,
            pool,
            seq,
            &prompt_for(lane),
            0,
            SamplingParams::greedy(),
            None,
        )
        .unwrap_or_else(|e| panic!("prefill lane {lane}: {e}"));
    }
    let sampling = vec![SamplingParams::greedy(); width];
    let mut timings = Vec::with_capacity(TIMED_ROUNDS);
    for round in 0..WARM_ROUNDS + TIMED_ROUNDS {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let start = Instant::now();
        decode_program_batch_sampled(model, pool, &mut refs, &sampling)
            .unwrap_or_else(|e| panic!("decode: {e}"));
        // The leaf synchronizes its stream before returning, so the call's
        // wall time is the round's device time plus the host's own copies.
        let elapsed = start.elapsed();
        if round >= WARM_ROUNDS {
            timings.push(elapsed);
        }
    }
    timings.sort_unstable();
    timings[timings.len() / 2]
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_wide_decode_round_costs_about_what_its_single_lane_round_costs() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
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

    // One model and one pool for both widths, so the two readings differ in
    // the round's width and nothing else -- and so both replay graphs
    // captured against the pool they actually run on.
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT, ignis_core::KvFormat::Bf16)
        .unwrap_or_else(|e| panic!("model load: {e}"));
    let pool = new_pool(WIDE as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let capture =
        capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("decode graph capture: {e}"));
    for width in [1, WIDE] {
        assert!(
            capture.is_ready(width as u32),
            "width {width} did not capture a decode graph (ready_mask {:#010b}) -- \
             the reading would compare a replay against an eager round",
            capture.ready_mask
        );
    }

    // Width 1 first: its lane is released before the wide run allocates,
    // so the wide round has the whole pool and neither reading is taken
    // against a pool holding another run's sequences.
    let single = median_round(&model, &pool, 1);
    let wide = median_round(&model, &pool, WIDE);
    drop(pool);
    drop(model);

    let ratio = wide.as_secs_f64() / single.as_secs_f64();
    println!(
        "#111 measurement: decode round B=1 {:.2} ms, B={WIDE} {:.2} ms -> {ratio:.2}x \
         (#111's pre-fix baseline was 4.81x at B=4; the live reference pays 1.07x)",
        single.as_secs_f64() * 1e3,
        wide.as_secs_f64() * 1e3,
    );
    assert!(
        ratio < MAX_RATIO,
        "a width-{WIDE} decode round cost {ratio:.2}x its width-1 round (limit {MAX_RATIO:.1}x) -- \
         the round is traversing the model once per lane again, not once per round"
    );
}
