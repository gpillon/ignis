//! GPU integration coverage for P3-05's decode CUDA graphs (GitHub #102,
//! ADR 0019) and P3-06's batch-wide decode round (GitHub #111): a decode
//! graph replayed at exact batch width 1..=8 must emit the identical token
//! stream the same traversal run eagerly would have, a prefill chunk run
//! between two replays must not disturb a later replay's output, the round
//! must traverse the model once whatever its width, and which row of the
//! round a sequence occupies must not change what it generates.
//!
//! Graph capture is triggered explicitly (`step::capture_decode_graphs`),
//! not automatically on model load, so an "eager" run is simply a pool that
//! never had capture called on it -- `ignis_program_decode`'s internal
//! `decode_graph_ready` bits all stay clear and every round takes the
//! untouched eager path. This lets one test compare both paths on the
//! identical model/pool geometry without a second code path of its own.
//!
//! One `#[test]`, several `Model`/`SeqPool` loads against one shared
//! `materialize()` (mirrors `sampling_gpu.rs`'s own note: materialized
//! device weights are never freed until the artifact drops, so more than
//! one `materialize()` call in this process would exhaust VRAM).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    SamplingParams, capture_decode_graphs, decode_program_batch_sampled, prefill_program_sampled,
    program_stats,
};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
const ROUNDS: usize = 3;
/// The Qwen 3.8 27B topology's decoder layer count -- what one traversal of
/// the model dispatches, and therefore what a decode round of *any* width
/// must report (GitHub #111). Before that ticket a width-B round reported
/// `64 * B`, because it ran B complete per-lane forwards.
const LAYERS: u64 = 64;

fn new_pool(slot_count: u32) -> Result<SeqPool, String> {
    SeqPool::create(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
    )
}

/// One lane's distinct prompt, so a graph that accidentally mixed up which
/// physical slot belongs to which round row would show up as a mismatched
/// token stream rather than passing by coincidence.
fn prompt_for(lane: usize) -> Vec<i32> {
    (0..6).map(|i| (5 + lane * 7 + i) as i32).collect()
}

/// Runs `width` sequences through prefill and `ROUNDS` shared decode rounds,
/// greedy throughout, and returns each lane's token stream. `use_graph`
/// controls only whether `capture_decode_graphs` was called on this pool --
/// the call sequence is otherwise identical either way.
fn run_width(model: &Model, width: usize) -> Vec<Vec<i32>> {
    let pool = new_pool(width as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let mut sequences: Vec<Seq<'_>> = (0..width)
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for (lane, seq) in sequences.iter_mut().enumerate() {
        prefill_program_sampled(
            model,
            &pool,
            seq,
            &prompt_for(lane),
            0,
            SamplingParams::greedy(),
            None,
        )
        .unwrap_or_else(|e| panic!("prefill lane {lane}: {e}"));
    }
    let sampling = vec![SamplingParams::greedy(); width];
    let mut tokens = vec![Vec::with_capacity(ROUNDS); width];
    for _ in 0..ROUNDS {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let round = decode_program_batch_sampled(model, &pool, &mut refs, &sampling)
            .unwrap_or_else(|e| panic!("decode: {e}"));
        // GitHub #111's leaf instrumentation: the round's dispatch count is
        // the layer count, not the layer count times the width -- one
        // B-wide traversal of the model, not B batch-1 forwards. Checked on
        // both paths, since `run_width` serves the eager and the graph run.
        let stats = program_stats(model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(
            stats.kernel_count, LAYERS,
            "width {width}: a decode round traversed the model more than once"
        );
        for (lane, tok) in round.into_iter().enumerate() {
            tokens[lane].push(tok);
        }
    }
    tokens
}

/// Runs `width` sequences through prefill and `ROUNDS` shared decode rounds
/// with the lanes presented to the round in `order`, and returns each lane's
/// token stream keyed by its *prompt* (not by its row), so two runs that
/// differ only in row order are directly comparable.
///
/// `order[row]` is the prompt index the round's row `row` carries. The
/// sequences are still allocated in prompt order, so a permuted `order` also
/// permutes which physical pool slot each row addresses -- which is the
/// point: the round reads every per-sequence input (token id, position, pool
/// slot) from device staging indexed by row, and a row/slot mix-up would
/// show up here as a changed token stream rather than as a silent
/// cross-contamination.
///
/// The pool is the caller's, and it must be the one the graphs were captured
/// against: a captured graph holds the pool's KV planes, block-table matrix
/// and GDN state planes at their capture-time addresses (ADR 0019), so
/// replaying it against a pool allocated later reads freed memory. The
/// sequences are released when this returns, so two calls can share one pool.
fn run_width_in_order(
    model: &Model,
    pool: &SeqPool,
    width: usize,
    order: &[usize],
) -> Vec<Vec<i32>> {
    assert_eq!(order.len(), width, "order must name every row");
    let mut sequences: Vec<Seq<'_>> = (0..width)
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for (prompt, seq) in sequences.iter_mut().enumerate() {
        prefill_program_sampled(
            model,
            pool,
            seq,
            &prompt_for(prompt),
            0,
            SamplingParams::greedy(),
            None,
        )
        .unwrap_or_else(|e| panic!("prefill prompt {prompt}: {e}"));
    }
    let sampling = vec![SamplingParams::greedy(); width];
    let mut tokens = vec![Vec::with_capacity(ROUNDS); width];
    for _ in 0..ROUNDS {
        // `order` maps rows to sequences; the borrow checker needs the
        // disjointness spelled out, so the refs are taken by index.
        let mut remaining: Vec<Option<&mut Seq<'_>>> =
            sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> = Vec::with_capacity(width);
        for &prompt in order {
            refs.push(
                remaining[prompt]
                    .take()
                    .expect("order must be a permutation: every prompt appears once"),
            );
        }
        let round = decode_program_batch_sampled(model, pool, &mut refs, &sampling)
            .unwrap_or_else(|e| panic!("decode: {e}"));
        for (row, tok) in round.into_iter().enumerate() {
            tokens[order[row]].push(tok);
        }
    }
    tokens
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn decode_cuda_graphs_replay_matches_eager_and_survive_interleaving() {
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
    // One materialize() call for the whole test (mirrors sampling_gpu.rs's
    // own note): materialized device weights are never freed until `artifact`
    // drops, so a second materialize() in this process would exhaust VRAM.
    // Everything below -- the width loop and the interleaving check -- reuses
    // this one `reader`/`artifact`/`handles` across many `Model` loads.
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };

    // --- replay-vs-eager token equality at every exact width 1..=8 --------
    for width in 1..=8usize {
        // --- eager: same geometry, capture never called ---------------
        let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("model load (eager, width {width}): {e}"));
        let eager = run_width(&model, width);
        drop(model);

        // --- graph: capture_decode_graphs called right after the pool
        // would exist in production (StepLeaf::load_model's own sequence) --
        let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("model load (graph, width {width}): {e}"));
        let pool_for_capture =
            new_pool(width as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let capture = capture_decode_graphs(&model, &pool_for_capture)
            .unwrap_or_else(|e| panic!("decode graph capture: {e}"));
        drop(pool_for_capture);
        assert!(
            capture.is_ready(width as u32),
            "width {width} did not capture a decode graph (ready_mask {:#010b})",
            capture.ready_mask
        );
        let graph = run_width(&model, width);
        drop(model);

        assert_eq!(
            eager, graph,
            "width {width}: graph replay diverged from the same traversal run eagerly"
        );
    }

    // --- a lane's row in the round must not change what it generates -----
    // GitHub #111 made the round one B-wide traversal, so every
    // per-sequence input -- token id, absolute position, physical pool slot
    // -- is read from device staging indexed by the row. Presenting the same
    // four sequences in reverse row order must leave each one's token stream
    // untouched: a row-to-slot mix-up, or one row reading another row's
    // activations, changes it. This is the isolation property the per-lane
    // loop used to get for free by construction.
    const PERMUTED_WIDTH: usize = 4;
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load (row order): {e}"));
    let row_order_pool =
        new_pool(PERMUTED_WIDTH as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let capture = capture_decode_graphs(&model, &row_order_pool)
        .unwrap_or_else(|e| panic!("decode graph capture: {e}"));
    assert!(
        capture.is_ready(PERMUTED_WIDTH as u32),
        "row-order run: width {PERMUTED_WIDTH} did not capture a decode graph"
    );
    let in_row_order = run_width_in_order(&model, &row_order_pool, PERMUTED_WIDTH, &[0, 1, 2, 3]);
    let reversed_rows = run_width_in_order(&model, &row_order_pool, PERMUTED_WIDTH, &[3, 2, 1, 0]);
    drop(row_order_pool);
    drop(model);
    assert_eq!(
        in_row_order, reversed_rows,
        "which row of the decode round a sequence occupies changed what it generated"
    );

    // --- interleaving: a prefill chunk between two replays must not
    // disturb the second replay's output (ADR 0019) ------------------------
    const WIDTH: usize = 2;
    const PROMPT: [i32; 6] = [5, 9, 20, 42, 7, 3];

    // --- baseline: two replays back to back, nothing in between ---------
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load (baseline): {e}"));
    let pool = new_pool(WIDTH as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    assert!(
        capture.is_ready(WIDTH as u32),
        "interleaving baseline: width {WIDTH} did not capture a decode graph -- \
         both runs would silently take the eager path and prove nothing"
    );
    let mut lanes: Vec<Seq<'_>> = (0..WIDTH)
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for seq in lanes.iter_mut() {
        prefill_program_sampled(&model, &pool, seq, &PROMPT, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let sampling = vec![SamplingParams::greedy(); WIDTH];
    let mut baseline = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut refs: Vec<&mut Seq<'_>> = lanes.iter_mut().collect();
        baseline.push(
            decode_program_batch_sampled(&model, &pool, &mut refs, &sampling)
                .unwrap_or_else(|e| panic!("decode: {e}")),
        );
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(
            stats.graph_launches, 1,
            "interleaving baseline: decode round did not replay a graph"
        );
    }
    drop(lanes);
    drop(pool);
    drop(model);

    // --- interleaved: a prefill chunk on a third sequence lands between
    // the same two replays -- it only ever touches `scratch`, never the
    // decode graph's dedicated `decode_graph_scratch` (ADR 0019), so the
    // second replay must read exactly what it would have read anyway.
    let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
        .unwrap_or_else(|e| panic!("model load (interleaved): {e}"));
    let pool = new_pool(WIDTH as u32 + 1).unwrap_or_else(|e| panic!("seq pool create: {e}"));
    let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    assert!(
        capture.is_ready(WIDTH as u32),
        "interleaved run: width {WIDTH} did not capture a decode graph -- \
         the interleaving property would go unexercised"
    );
    let mut lanes: Vec<Seq<'_>> = (0..WIDTH)
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for seq in lanes.iter_mut() {
        prefill_program_sampled(&model, &pool, seq, &PROMPT, 0, SamplingParams::greedy(), None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let mut interleaved = Vec::with_capacity(2);
    {
        let mut refs: Vec<&mut Seq<'_>> = lanes.iter_mut().collect();
        interleaved.push(
            decode_program_batch_sampled(&model, &pool, &mut refs, &sampling)
                .unwrap_or_else(|e| panic!("decode 1: {e}")),
        );
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(
            stats.graph_launches, 1,
            "interleaved run: first decode round did not replay a graph"
        );
    }
    let mut interloper = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc interloper: {e}"));
    prefill_program_sampled(
        &model,
        &pool,
        &mut interloper,
        &PROMPT,
        0,
        SamplingParams::greedy(),
        None,
    )
    .unwrap_or_else(|e| panic!("interloper prefill: {e}"));
    drop(interloper);
    {
        let mut refs: Vec<&mut Seq<'_>> = lanes.iter_mut().collect();
        interleaved.push(
            decode_program_batch_sampled(&model, &pool, &mut refs, &sampling)
                .unwrap_or_else(|e| panic!("decode 2: {e}")),
        );
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(
            stats.graph_launches, 1,
            "interleaved run: second decode round (after the prefill chunk) did not replay a graph"
        );
    }
    drop(lanes);
    drop(pool);
    drop(model);

    assert_eq!(
        baseline, interleaved,
        "a prefill chunk between two decode graph replays must not change the second replay's output"
    );
}
