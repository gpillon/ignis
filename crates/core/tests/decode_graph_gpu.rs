//! GPU integration coverage for P3-05's decode CUDA graphs (GitHub #102,
//! ADR 0019) and the batch-wide decode round (GitHub #111): a decode
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

/// The identity row order for `width` lanes: row `b` carries prompt `b`.
fn in_prompt_order(width: usize) -> Vec<usize> {
    (0..width).collect()
}

/// Every lane greedy -- the settings for a run whose only question is
/// whether two code paths agree on a token stream.
fn greedy_for(_prompt: usize) -> SamplingParams {
    SamplingParams::greedy()
}

/// One lane's own sampling settings: a distinct seed, temperature, top-k
/// and penalties per prompt (GitHub #115).
///
/// Requirement 21 of `.scratch/runtime/specs/03-serving-loop.md` is that
/// what a request generates depends on its own seed and never on which
/// lanes shared its round. A greedy round cannot exercise that at all --
/// argmax reads neither the RNG nor the penalty counts -- so a row-order
/// test that used `greedy()` on every lane proved isolation only for the
/// KV, GDN and position state. These settings put the sampler's per-row
/// config and its per-slot penalty counts under the same test.
/// The temperature is high and the truncations are off on purpose: the
/// guard below requires a sampled stream to differ from the greedy one, and
/// a peaked distribution under a low temperature can draw the argmax token
/// every time over a handful of rounds. Spreading the distribution makes
/// that agreement vanishingly unlikely without making any lane's stream
/// less deterministic -- a seed still fixes it exactly.
fn per_lane_sampling_for(prompt: usize) -> SamplingParams {
    SamplingParams {
        temperature: 1.5 + prompt as f32 * 0.2,
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 0.1 * prompt as f32,
        frequency_penalty: 0.25 * prompt as f32,
        seed: 0xA5A5_0000 + prompt as u64,
    }
}

/// Runs `order.len()` sequences through prefill and `ROUNDS` shared decode
/// rounds, and returns each lane's token stream keyed by its *prompt*, not
/// by the row it occupied -- so two runs that differ only in row order are
/// directly comparable.
///
/// `order[row]` is the prompt index the round's row `row` carries. The
/// sequences are allocated in prompt order whatever `order` says, so a
/// permuted order also permutes which physical pool slot each row
/// addresses. That is the point: the round reads every per-sequence input
/// (token id, position, pool slot, sampling config, penalty counts) from
/// device staging indexed by row, and a row-to-slot mix-up shows up here as
/// a changed token stream rather than as silent cross-contamination.
///
/// `sampling_for` is keyed by prompt too, so a lane's settings travel with
/// the sequence rather than with the row -- otherwise permuting the rows
/// would permute the settings and the comparison would prove nothing.
///
/// The pool is the caller's, and it must be the one the graphs were
/// captured against: a captured graph holds the pool's KV planes,
/// block-table matrix and GDN state planes at their capture-time addresses
/// (ADR 0019), so replaying it against a pool allocated later reads freed
/// memory. The sequences are released when this returns, so several calls
/// can share one pool.
fn run_lanes(
    model: &Model,
    pool: &SeqPool,
    order: &[usize],
    sampling_for: fn(usize) -> SamplingParams,
) -> Vec<Vec<i32>> {
    let width = order.len();
    let mut seen = vec![false; width];
    for &prompt in order {
        assert!(prompt < width && !seen[prompt], "order must be a permutation of 0..{width}");
        seen[prompt] = true;
    }
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
            sampling_for(prompt),
            None,
        )
        .unwrap_or_else(|e| panic!("prefill prompt {prompt}: {e}"));
    }
    // Row-major, so row `b` carries prompt `order[b]`'s settings.
    let sampling: Vec<SamplingParams> = order.iter().map(|&prompt| sampling_for(prompt)).collect();
    let mut tokens = vec![Vec::with_capacity(ROUNDS); width];
    for _ in 0..ROUNDS {
        // `order` maps rows to sequences; the borrow checker needs the
        // disjointness spelled out, so the refs are taken by index.
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> = Vec::with_capacity(width);
        for &prompt in order {
            refs.push(remaining[prompt].take().expect("checked to be a permutation above"));
        }
        let round = decode_program_batch_sampled(model, pool, &mut refs, &sampling)
            .unwrap_or_else(|e| panic!("decode: {e}"));
        // GitHub #111's leaf instrumentation: the round's dispatch count is
        // the layer count, not the layer count times the width -- one
        // B-wide traversal of the model, not B batch-1 forwards. Checked on
        // whichever path this run took, eager or replayed.
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(
            stats.kernel_count, LAYERS,
            "width {width}: a decode round traversed the model more than once"
        );
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
        let order = in_prompt_order(width);

        // --- eager: same geometry, capture never called ---------------
        let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("model load (eager, width {width}): {e}"));
        let eager_pool = new_pool(width as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let eager = run_lanes(&model, &eager_pool, &order, greedy_for);
        drop(eager_pool);
        drop(model);

        // --- graph: capture_decode_graphs called right after the pool
        // would exist in production (StepLeaf::load_model's own sequence),
        // and the run stays on that same pool -- a captured graph holds the
        // pool's planes at their capture-time addresses (ADR 0019).
        let model = load_qwen38_27b(&reader, &artifact, &handles, MAX_CONTEXT, MAX_CONTEXT)
            .unwrap_or_else(|e| panic!("model load (graph, width {width}): {e}"));
        let graph_pool = new_pool(width as u32).unwrap_or_else(|e| panic!("seq pool create: {e}"));
        let capture = capture_decode_graphs(&model, &graph_pool)
            .unwrap_or_else(|e| panic!("decode graph capture: {e}"));
        assert!(
            capture.is_ready(width as u32),
            "width {width} did not capture a decode graph (ready_mask {:#010b})",
            capture.ready_mask
        );
        let graph = run_lanes(&model, &graph_pool, &order, greedy_for);
        drop(graph_pool);
        drop(model);

        assert_eq!(
            eager, graph,
            "width {width}: graph replay diverged from the same traversal run eagerly"
        );
    }

    // --- a lane's row in the round must not change what it generates -----
    // GitHub #111 made the round one B-wide traversal, so every
    // per-sequence input -- token id, absolute position, physical pool slot,
    // sampling config, penalty counts -- is read from device staging indexed
    // by the row. Presenting the same four sequences in reverse row order
    // must leave each one's token stream untouched: a row-to-slot mix-up, or
    // one row reading another row's activations, changes it. This is the
    // isolation property the per-lane loop used to get for free by
    // construction.
    //
    // The lanes sample with their own seeds, temperatures and penalties
    // (GitHub #115), so this covers requirement 21 -- what a request
    // generates depends on its own seed and never on which lanes shared its
    // round -- which a greedy round cannot exercise at all.
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
    let in_row_order = run_lanes(&model, &row_order_pool, &[0, 1, 2, 3], per_lane_sampling_for);
    let reversed_rows = run_lanes(&model, &row_order_pool, &[3, 2, 1, 0], per_lane_sampling_for);
    // The per-lane settings have to actually reach the sampler. If they were
    // dropped somewhere, every lane would fall back to argmax and the
    // comparison below would hold for a reason that has nothing to do with
    // row indexing -- the exact way this test could rot into proving
    // nothing. A greedy run over the same prompts must therefore differ.
    let greedy_rows = run_lanes(&model, &row_order_pool, &[0, 1, 2, 3], greedy_for);
    drop(row_order_pool);
    drop(model);
    assert_ne!(
        in_row_order, greedy_rows,
        "the per-lane seeds, temperatures and penalties never reached the sampler, so the          row-order comparison proves nothing about them"
    );
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
