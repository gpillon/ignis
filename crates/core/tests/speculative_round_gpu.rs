//! GPU coverage for the verify round (P5-04, GitHub #153, spec 05): the
//! speculative substrate -- verify inputs of k+1 columns per lane, the
//! record-mode GDN traversal, the vendored accept kernel, the stop-aware
//! commit, the ReplaySSM fold and one verify graph per batch width -- proven
//! with a **fake drafter** before the real one exists (P5-05).
//!
//! Two fake drafters drive it. The *oracle* proposes the spec-off greedy
//! continuation (what the same engine emits one token per round), so every
//! draft must be accepted and the committed text must equal the spec-off
//! text token for token: that equality is this phase's correctness floor,
//! and it exercises verify, accept, fold and the multi-token commit at once.
//! The *random* drafter proposes arbitrary ids, so the accept kernel must
//! reject at the first divergence (at most one draft can match by chance),
//! the fold must roll every rejected transition back, and the text must
//! still equal the spec-off text -- the rollback path, proven the same way.
//!
//! The model is loaded with `SpeculativeBackend::VerifyOnly`: the verify
//! substrate at window 7, no drafter bound, the drafts supplied per call
//! through [`decode_program_verify`] -- the internal seam the ticket names.
//!
//! **Equivalence at near-ties.** The engine's argmax is not route-invariant
//! at a near-tie: the verify traversal runs k+1 columns per lane through
//! different kernel tilings than a one-column decode, and the engine's own
//! two prefill routes (chunked, per-token) already disagree about the winner
//! at such positions -- measured while writing this test on a counting
//! prompt (`24 -> 15` vs `24 -> 19`: the chunked route ranks 19 first by
//! 0.69 logits, the per-token route 15 first by 0.25) and on a pair tied
//! exactly at BF16 resolution. Spec-off is not width-invariant there either.
//! So "spec-on equals spec-off" is checked as: identical streams, or
//! identical up to a first divergence where the two candidates are a
//! near-tie *by the engine's own measure* -- its two prefill routes disagree
//! about which wins, or rate the gap no wider than the spread between them
//! ([`assert_equivalent`]). The tolerance is never a constant; a real defect
//! in verify, accept or fold produces a divergence both routes agree on by a
//! wide margin, and fails.
//!
//! On acceptance criterion 2 (temperature > 0): exact token equality between
//! spec-on and spec-off sampling holds here only where the draw cannot
//! change the pick (`top_k = 1`, a one-candidate support). The vendored
//! accept kernel keys its RNG by the *speculative* purposes (accept,
//! correction, bonus), a different counter stream from the decode purpose
//! today's sampler draws from, so at a wider support the two engines draw
//! different uniforms and pick different tokens from the same
//! distribution -- a property of the reference's own rule (ADR 0010), not a
//! defect here. What is checked at a wide support is what the RNG design
//! guarantees: a fixed seed reproduces the spec-on stream exactly, and a
//! lane's stream does not depend on which row of its round it occupies. That
//! is checked at a fixed width: across widths the verify traversal's own
//! numerics differ (8 columns at B=1 against 32 at B=4 take different kernel
//! tilings), and a draw near a probability boundary follows that drift the
//! same way a greedy pick follows it at a near-tie.
//!
//! One `#[test]`, several `Model`/`SeqPool` loads against one shared
//! `materialize()` (see `decode_graph_gpu.rs`: materialized weights are
//! never freed until the artifact drops, so a second `materialize()` in
//! this process would exhaust VRAM).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact, GPU, or kernel error is a **skip**; under the profile
//! the same condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::path::Path;

use ignis_artifact::{CudaDevice, MaterializedArtifact, ObjectHandle, Reader, bind_text_scope_27b, materialize};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{Model, load_qwen38_27b, load_qwen38_27b_with_speculation};
use ignis_core::seq::{Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    PrefillRoute, SamplingParams, VerifyLane, capture_decode_graphs, decode_program_batch_sampled,
    decode_program_verify, prefill_program_sampled, prefill_program_with_route, program_stats,
};
use ignis_core::{KvFormat, Speculation, SpeculativeBackend};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 128;
/// The load's draft window: DFlash2-7, the reference lane G5 measures.
const WINDOW: u32 = 7;
/// Tokens each lane emits in a comparison -- several full windows.
const TOTAL: usize = 24;
/// The Qwen 3.8 27B topology's decoder layer count: one traversal, whatever
/// the width and however many columns per lane (GitHub #111's counter).
const LAYERS: u64 = 64;

/// A pool for a plain load.
fn new_pool(slot_count: u32) -> SeqPool {
    pool_for(KvFormat::Bf16, slot_count, None)
}

/// A pool for a verify-only load: the program entry points pair a pool with
/// a model of the same speculative backend (P5-03, GitHub #152).
fn verify_pool(slot_count: u32) -> SeqPool {
    verify_pool_in(KvFormat::Bf16, slot_count)
}

fn verify_pool_in(kv_format: KvFormat, slot_count: u32) -> SeqPool {
    pool_for(kv_format, slot_count, Some(SpeculativeBackend::VerifyOnly))
}

fn pool_for(kv_format: KvFormat, slot_count: u32, backend: Option<SpeculativeBackend>) -> SeqPool {
    SeqPool::create_with_speculation(
        &ModelConfig::qwen38_27b(),
        &SeqPoolBudget {
            kv_format,
            kv_page_group_count: MAX_CONTEXT.div_ceil(64) * slot_count,
            max_context_tokens: MAX_CONTEXT,
            slot_count,
        },
        backend,
    )
    .unwrap_or_else(|e| panic!("seq pool create: {e}"))
}

/// One lane's distinct prompt (as in `decode_graph_gpu.rs`), so a lane
/// reading another lane's columns, records or slot shows up as a changed
/// stream rather than passing by coincidence.
fn prompt_for(lane: usize) -> Vec<i32> {
    (0..6).map(|i| (5 + lane * 7 + i) as i32).collect()
}

fn load_plain(reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle]) -> Model {
    load_plain_in(KvFormat::Bf16, reader, artifact, handles)
}

fn load_plain_in(kv_format: KvFormat, reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle]) -> Model {
    load_qwen38_27b(reader, artifact, handles, MAX_CONTEXT, MAX_CONTEXT, kv_format)
        .unwrap_or_else(|e| panic!("model load (spec-off, {}): {e}", kv_format.as_str()))
}

fn load_verify(reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle]) -> Model {
    load_verify_in(KvFormat::Bf16, reader, artifact, handles)
}

fn load_verify_in(kv_format: KvFormat, reader: &Reader, artifact: &MaterializedArtifact, handles: &[ObjectHandle]) -> Model {
    let spec = Speculation::new(SpeculativeBackend::VerifyOnly, WINDOW).unwrap();
    load_qwen38_27b_with_speculation(
        reader,
        artifact,
        handles,
        MAX_CONTEXT,
        MAX_CONTEXT,
        kv_format,
        Some(spec),
    )
    .unwrap_or_else(|e| panic!("model load (verify-only, {}, window {WINDOW}): {e}", kv_format.as_str()))
}

/// The spec-off stream: `total` tokens per lane, one per round, every lane
/// in one batch. The oracle the fake drafter reads and the text every
/// spec-on run is held to.
fn spec_off_streams(model: &Model, pool: &SeqPool, prompts: &[Vec<i32>], sampling: SamplingParams, total: usize) -> Vec<Vec<i32>> {
    let mut sequences: Vec<Seq<'_>> = prompts
        .iter()
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for (seq, prompt) in sequences.iter_mut().zip(prompts) {
        prefill_program_sampled(model, pool, seq, prompt, 0, sampling, None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let params = vec![sampling; prompts.len()];
    let mut streams = vec![Vec::with_capacity(total); prompts.len()];
    for _ in 0..total {
        let mut refs: Vec<&mut Seq<'_>> = sequences.iter_mut().collect();
        let round = decode_program_batch_sampled(model, pool, &mut refs, &params)
            .unwrap_or_else(|e| panic!("spec-off decode: {e}"));
        for (stream, token) in streams.iter_mut().zip(round) {
            stream.push(token);
        }
    }
    streams
}

/// A fake drafter's proposal for a lane that has emitted `emitted` tokens so
/// far.
#[derive(Clone, Copy)]
enum Drafter<'a> {
    /// The spec-off continuation: 100% acceptance expected.
    Oracle(&'a [Vec<i32>]),
    /// Arbitrary ids: at most one accepted per round expected.
    Random,
    /// No proposal: every lane at extent 0, today's round.
    Nothing,
}

fn propose(drafter: Drafter<'_>, lane: usize, emitted: usize, round: usize, budget: usize) -> Vec<i32> {
    let want = (WINDOW as usize).min(budget.saturating_sub(1));
    match drafter {
        Drafter::Oracle(streams) => {
            let stream = &streams[lane];
            let start = (emitted + 1).min(stream.len());
            let end = (start + want).min(stream.len());
            stream[start..end].to_vec()
        }
        Drafter::Random => {
            // A small counter-based generator, keyed by lane and round, so
            // a run is reproducible and no two lanes propose alike.
            let mut state = 0x9E37_79B9u64 ^ ((lane as u64) << 32) ^ (round as u64);
            (0..want)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) % 200_000) as i32
                })
                .collect()
        }
        Drafter::Nothing => Vec::new(),
    }
}

/// What one spec-on run recorded: each lane's emitted text, and per round
/// the (proposed, accepted) draft counts of every lane, in order.
struct VerifyRun {
    emitted: Vec<Vec<i32>>,
    /// Per round, per lane that rode it: (lane, tokens emitted before the
    /// round, drafts proposed, drafts accepted).
    rounds: Vec<Vec<(usize, usize, usize, usize)>>,
}

/// Runs `prompts` through the verify round with `drafter` until every lane
/// has emitted `total` tokens, budgeting each lane to exactly `total`
/// (`remaining_tokens`), with `stop_ids` on every lane. Rounds are shared:
/// every lane rides every round until it is done, so widths shrink as
/// lanes finish -- the exact-width graphs at every width get exercised.
fn run_verify(
    model: &Model,
    pool: &SeqPool,
    prompts: &[Vec<i32>],
    sampling: SamplingParams,
    drafter: Drafter<'_>,
    stop_ids: &[i32],
    total: usize,
    expect_graph: bool,
) -> VerifyRun {
    let mut sequences: Vec<Seq<'_>> = prompts
        .iter()
        .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
        .collect();
    for (seq, prompt) in sequences.iter_mut().zip(prompts) {
        prefill_program_sampled(model, pool, seq, prompt, 0, sampling, None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
    }
    let mut emitted: Vec<Vec<i32>> = vec![Vec::new(); prompts.len()];
    let mut done = vec![false; prompts.len()];
    let mut rounds = Vec::new();
    let mut round = 0usize;
    while done.iter().any(|d| !d) {
        let active: Vec<usize> = (0..prompts.len()).filter(|&i| !done[i]).collect();
        let drafts: Vec<Vec<i32>> = active
            .iter()
            .map(|&i| propose(drafter, i, emitted[i].len(), round, total - emitted[i].len()))
            .collect();
        let lanes: Vec<VerifyLane<'_>> = active
            .iter()
            .zip(&drafts)
            .map(|(&i, drafts)| VerifyLane {
                sampling,
                remaining_tokens: (total - emitted[i].len()) as u32,
                stop_ids,
                drafts,
            })
            .collect();
        let mut remaining: Vec<Option<&mut Seq<'_>>> = sequences.iter_mut().map(Some).collect();
        let mut refs: Vec<&mut Seq<'_>> = active
            .iter()
            .map(|&i| remaining[i].take().expect("each active lane once"))
            .collect();
        let runs = decode_program_verify(model, pool, &mut refs, &lanes, WINDOW)
            .unwrap_or_else(|e| panic!("verify round {round}: {e}"));
        let stats = program_stats(model, pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.kernel_count, LAYERS, "a verify round traversed the model more than once");
        assert_eq!(
            stats.graph_launches,
            u64::from(expect_graph),
            "round {round} at width {}: graph replay expectation",
            active.len()
        );
        let mut this_round = Vec::new();
        for ((&i, run), drafts) in active.iter().zip(runs).zip(&drafts) {
            assert!(!run.is_empty() && run.len() <= WINDOW as usize + 1, "lane {i}: run of {}", run.len());
            assert!(run.len() <= total - emitted[i].len(), "lane {i}: the budget was exceeded");
            this_round.push((i, emitted[i].len(), drafts.len(), run.len() - 1));
            let stopped = run.iter().any(|t| stop_ids.contains(t));
            emitted[i].extend(run);
            if emitted[i].len() >= total || stopped {
                done[i] = true;
            }
        }
        rounds.push(this_round);
        round += 1;
        assert!(round < 4 * total, "the run does not converge");
    }
    VerifyRun { emitted, rounds }
}

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// `a` (the spec-off pick) against `b` (the spec-on pick) after `prompt` +
/// `agreed`, by the engine's own two prefill routes: a near-tie when the
/// routes disagree about which wins, or rate the gap no wider than the
/// spread between them. Panics naming both gaps otherwise.
fn assert_near_tie(kv_format: KvFormat, model: &Model, prompt: &[i32], agreed: &[i32], a: i32, b: i32, what: &str) {
    let vocab = ModelConfig::qwen38_27b().vocab as usize;
    let mut tokens = prompt.to_vec();
    tokens.extend_from_slice(agreed);
    let gap = |route: PrefillRoute| {
        let pool = verify_pool_in(kv_format, 1);
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        let mut logits = vec![0f32; vocab];
        prefill_program_with_route(model, &pool, &mut seq, &tokens, 0, route, Some(&mut logits))
            .unwrap_or_else(|e| panic!("near-tie probe prefill: {e}"));
        logits[a as usize] - logits[b as usize]
    };
    let chunked = gap(PrefillRoute::Chunked);
    let per_token = gap(PrefillRoute::PerToken);
    let routes_disagree = (chunked > 0.0) != (per_token > 0.0) || chunked == 0.0 || per_token == 0.0;
    let within_spread = chunked.abs().min(per_token.abs()) <= (chunked - per_token).abs();
    assert!(
        routes_disagree || within_spread,
        "{what}: spec-on picked {b} where spec-off picked {a} after {} tokens, and it is not a near-tie: \
         logit[{a}] - logit[{b}] = {chunked} (chunked route), {per_token} (per-token route)",
        agreed.len()
    );
}

/// Spec-on `on` against spec-off `off`, lane by lane: identical, or
/// identical up to a near-tie divergence ([`assert_near_tie`]). Returns each
/// lane's agreeing prefix length.
fn assert_equivalent(kv_format: KvFormat, model: &Model, prompts: &[Vec<i32>], off: &[Vec<i32>], on: &[Vec<i32>], what: &str) -> Vec<usize> {
    assert_eq!(off.len(), on.len(), "{what}: lane count");
    prompts
        .iter()
        .zip(off.iter().zip(on))
        .enumerate()
        .map(|(lane, (prompt, (off, on)))| match first_divergence(off, on) {
            None => off.len(),
            Some(d) => {
                assert!(d < off.len() && d < on.len(), "{what} lane {lane}: one stream ended early at {d}");
                assert_near_tie(kv_format, model, prompt, &off[..d], off[d], on[d], &format!("{what} lane {lane}"));
                d
            }
        })
        .collect()
}

/// The oracle drafter's rounds wholly inside a lane's agreeing prefix must
/// accept every draft.
fn assert_oracle_accepts(run: &VerifyRun, agreed: &[usize], what: &str) {
    for (round, lanes) in run.rounds.iter().enumerate() {
        for &(lane, start, proposed, accepted) in lanes {
            if start + proposed < agreed[lane] {
                assert_eq!(accepted, proposed, "{what} round {round} lane {lane}: the oracle's drafts were not all accepted");
            }
        }
    }
}

/// A prompt of `len` tokens, distinct from every `prompt_for` lane.
fn page_prompt(len: usize) -> Vec<i32> {
    (0..len).map(|i| (300 + 11 * i) as i32).collect()
}

/// `tokens` greedy window-0 rounds on one sequence, alone.
fn continue_greedy(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, tokens: usize) -> Vec<i32> {
    let greedy = SamplingParams::greedy();
    (0..tokens)
        .map(|_| {
            decode_program_batch_sampled(model, pool, &mut [&mut *seq], &[greedy])
                .unwrap_or_else(|e| panic!("window-0 round: {e}"))[0]
        })
        .collect()
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_verify_round_commits_the_spec_off_text_under_any_drafter() {
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
    let greedy = SamplingParams::greedy();
    let prompts8: Vec<Vec<i32>> = (0..8).map(prompt_for).collect();

    // --- the spec-off oracle: 8 greedy streams, one token per round -------
    let model = load_plain(&reader, &artifact, &handles);
    let pool = new_pool(8);
    let greedy_streams = spec_off_streams(&model, &pool, &prompts8, greedy, TOTAL);
    drop(pool);
    // Spec-off at each width AC 1 compares, from that width's own round:
    // spec-off is no more width-invariant at a near-tie than spec-on is.
    let mut streams_at = std::collections::HashMap::new();
    for width in [1usize, 4] {
        let pool = new_pool(width as u32);
        streams_at.insert(width, spec_off_streams(&model, &pool, &prompts8[..width], greedy, TOTAL));
        drop(pool);
    }
    // AC 3's candidates: a prompt of 64 - (s + 1) tokens whose run is cut at
    // index s lands the sequence exactly on the first KV page boundary, the
    // one place a prefix can be published. s in 1..=6 keeps the stop inside
    // one window with a licensed token after it; these spec-off streams only
    // supply the drafts.
    let page_prompts: Vec<Vec<i32>> = (1..=6usize).map(|s| page_prompt(64 - (s + 1))).collect();
    let pool = new_pool(page_prompts.len() as u32);
    let page_streams = spec_off_streams(&model, &pool, &page_prompts, greedy, TOTAL);
    drop(pool);
    drop(model);

    // --- AC 1: greedy spec-on == spec-off at widths 1 and 4, both drafters -
    let model = load_verify(&reader, &artifact, &handles);
    for width in [1usize, 4] {
        let pool = verify_pool(width as u32);
        let capture = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert!(capture.is_ready(width as u32), "width {width}: no decode graph");
        assert!(
            stats.verify_graph_ready_mask & (1 << (width - 1)) != 0,
            "width {width}: no verify graph at window {WINDOW} (mask {:#010b})",
            stats.verify_graph_ready_mask
        );
        let prompts = &prompts8[..width];
        let off = &streams_at[&width];
        let oracle = run_verify(&model, &pool, prompts, greedy, Drafter::Oracle(off), &[], TOTAL, true);
        let agreed = assert_equivalent(KvFormat::Bf16, &model, prompts, off, &oracle.emitted, &format!("width {width} oracle"));
        assert_oracle_accepts(&oracle, &agreed, &format!("width {width}"));
        // The multi-token commit actually happened.
        assert!(
            oracle.rounds.iter().flatten().any(|&(_, _, _, accepted)| accepted > 0),
            "width {width}: no round committed more than one token"
        );

        let random = run_verify(&model, &pool, prompts, greedy, Drafter::Random, &[], TOTAL, true);
        assert_equivalent(KvFormat::Bf16, &model, prompts, off, &random.emitted, &format!("width {width} random drafter (rollback)"));
        for (round, lanes) in random.rounds.iter().enumerate() {
            for &(lane, _, _, accepted) in lanes {
                assert!(accepted <= 1, "width {width} round {round} lane {lane}: {accepted} random drafts accepted");
            }
        }

        // No proposal at all: today's round, one token each.
        let nothing = run_verify(&model, &pool, prompts, greedy, Drafter::Nothing, &[], TOTAL, true);
        assert_equivalent(KvFormat::Bf16, &model, prompts, off, &nothing.emitted, &format!("width {width} extent 0"));
        assert_eq!(nothing.rounds.len(), TOTAL, "width {width}: extent 0 must commit one token per round");
        drop(pool);
    }
    drop(model);

    // --- AC 2: temperature > 0 with a fixed seed -----------------------------
    // (a) a one-candidate support: the draw cannot change the pick, so
    // spec-on equals spec-off exactly, through the sampling branch of the
    // accept kernel (its RNG, its penalty overlay, its token-count update).
    // No penalties: with top_k = 1 the pick is then the argmax of the raw
    // logits, which is what the near-tie probe measures.
    let peaked = SamplingParams {
        temperature: 0.9,
        top_k: 1,
        top_p: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 77,
    };
    let model = load_plain(&reader, &artifact, &handles);
    let pool = new_pool(4);
    let peaked_streams = spec_off_streams(&model, &pool, &prompts8[..4], peaked, TOTAL);
    drop(pool);
    drop(model);
    let model = load_verify(&reader, &artifact, &handles);
    let pool = verify_pool(4);
    let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let oracle = run_verify(&model, &pool, &prompts8[..4], peaked, Drafter::Oracle(&peaked_streams), &[], TOTAL, true);
    let agreed = assert_equivalent(KvFormat::Bf16, &model, &prompts8[..4], &peaked_streams, &oracle.emitted, "top_k=1 sampling");
    assert_oracle_accepts(&oracle, &agreed, "top_k=1 sampling");
    let random = run_verify(&model, &pool, &prompts8[..4], peaked, Drafter::Random, &[], TOTAL, true);
    assert_equivalent(KvFormat::Bf16, &model, &prompts8[..4], &peaked_streams, &random.emitted, "top_k=1 sampling, random drafts");

    // (b) a wide support: a fixed seed reproduces the stream, and a lane's
    // stream is its own whatever shares its round.
    let wide = SamplingParams {
        temperature: 1.2,
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
        seed: 4242,
    };
    let together = run_verify(&model, &pool, &prompts8[..4], wide, Drafter::Oracle(&greedy_streams), &[], TOTAL, true);
    let again = run_verify(&model, &pool, &prompts8[..4], wide, Drafter::Oracle(&greedy_streams), &[], TOTAL, true);
    assert_eq!(together.emitted, again.emitted, "a fixed seed must reproduce the spec-on stream");
    assert_ne!(together.emitted, greedy_streams[..4], "the wide support never reached the sampler: the stream is the greedy one");
    // The same four lanes in reverse row order: same width, same columns,
    // same routes, so each lane's sampled stream must be bit-identical --
    // a row reading another row's config, position, counts, records or slot
    // changes it.
    let reversed_prompts: Vec<Vec<i32>> = prompts8[..4].iter().rev().cloned().collect();
    let reversed_oracle: Vec<Vec<i32>> = greedy_streams[..4].iter().rev().cloned().collect();
    let reversed = run_verify(&model, &pool, &reversed_prompts, wide, Drafter::Oracle(&reversed_oracle), &[], TOTAL, true);
    let unreversed: Vec<Vec<i32>> = reversed.emitted.into_iter().rev().collect();
    assert_eq!(unreversed, together.emitted, "a lane's sampled stream changed with the row it occupied");
    drop(pool);
    drop(model);

    // --- AC 3: the stop-aware commit -------------------------------------
    // Measured against the round itself, not against spec-off, so no
    // near-tie can hide or fake a result: a probe round with no stop id
    // yields its run R; the identical round with a stop id at R[s] runs the
    // identical device pass and must commit exactly R[..=s] -- through the
    // stop and no further -- with the frontier at the emitted length, R[s+1]
    // pending, and a prefix published at the cut continuing the same text on
    // a claimant.
    let model = load_verify(&reader, &artifact, &handles);
    let pool = verify_pool(3);
    let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
    let mut exercised = false;
    for (stop_at, (prompt, stream)) in (1..=6usize).zip(page_prompts.iter().zip(&page_streams)) {
        let drafts = stream[1..=WINDOW as usize].to_vec();
        let probe_run = {
            let mut probe = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
            prefill_program_sampled(&model, &pool, &mut probe, prompt, 0, greedy, None)
                .unwrap_or_else(|e| panic!("prefill: {e}"));
            let lanes = [VerifyLane { sampling: greedy, remaining_tokens: TOTAL as u32, stop_ids: &[], drafts: &drafts }];
            decode_program_verify(&model, &pool, &mut [&mut probe], &lanes, WINDOW)
                .unwrap_or_else(|e| panic!("probe verify round: {e}"))
                .remove(0)
        };
        // The stop must sit inside the run, first occur there, and have a
        // licensed token after it.
        if probe_run.len() <= stop_at + 1 || probe_run[..stop_at].contains(&probe_run[stop_at]) {
            continue;
        }
        let stop = probe_run[stop_at];
        let mut publisher = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_sampled(&model, &pool, &mut publisher, prompt, 0, greedy, None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
        let stops = [stop];
        let lanes = [VerifyLane { sampling: greedy, remaining_tokens: TOTAL as u32, stop_ids: &stops, drafts: &drafts }];
        let runs = decode_program_verify(&model, &pool, &mut [&mut publisher], &lanes, WINDOW)
            .unwrap_or_else(|e| panic!("verify round with a stop id: {e}"));
        exercised = true;
        assert_eq!(runs[0], probe_run[..=stop_at], "the run must commit through the stop id and no further");
        assert_eq!(
            publisher.stats().position,
            (prompt.len() + runs[0].len()) as u64,
            "position equals the emitted length"
        );
        assert_eq!(publisher.stats().position, 64, "the cut lands on the page boundary");
        // A prefix published at the cut: the claimant stands exactly where
        // the publisher stands, so the two continue with the same text.
        let prefix = publisher.publish_prefix(64).unwrap_or_else(|e| panic!("publish the prefix at the cut: {e}"));
        let mut claimant = pool.alloc_shared(MAX_CONTEXT, &prefix).unwrap_or_else(|e| panic!("claim: {e}"));
        let from_publisher = continue_greedy(&model, &pool, &mut publisher, 8);
        let from_claimant = continue_greedy(&model, &pool, &mut claimant, 8);
        assert_eq!(
            from_publisher[0],
            probe_run[stop_at + 1],
            "the token pending after the cut is not the run's next licensed token"
        );
        assert_eq!(from_claimant, from_publisher, "a claimant of the prefix published at the cut yields different text");
        drop(claimant);
        drop(prefix);
        drop(publisher);
        // And a sequence that prefilled exactly the emitted text continues
        // the same way -- up to a near-tie between its route and decode's.
        let mut text = prompt.clone();
        text.extend(&runs[0]);
        let mut fresh = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_sampled(&model, &pool, &mut fresh, &text, 0, greedy, None)
            .unwrap_or_else(|e| panic!("prefill: {e}"));
        let from_fresh = continue_greedy(&model, &pool, &mut fresh, 8);
        drop(fresh);
        assert_equivalent(KvFormat::Bf16, &model, &[text], &[from_publisher], &[from_fresh], "a fresh prefill of the emitted text");
        break;
    }
    assert!(exercised, "no stop candidate reached its stop: the stop-aware commit went unexercised");
    // Window 0 on a windowed load is today's round (the AC-3 check above
    // already used it); a foreign window is refused naming both.
    {
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_sampled(&model, &pool, &mut seq, &prompts8[0], 0, greedy, None).unwrap_or_else(|e| panic!("prefill: {e}"));
        let lanes = [VerifyLane::greedy(&[])];
        let err = decode_program_verify(&model, &pool, &mut [&mut seq], &lanes, 3).expect_err("a foreign window must be refused");
        assert!(err.contains('3') && err.contains(&WINDOW.to_string()), "{err}");
    }
    drop(pool);
    drop(model);
    {
        let model = load_plain(&reader, &artifact, &handles);
        let pool = new_pool(1);
        let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
        prefill_program_sampled(&model, &pool, &mut seq, &prompts8[0], 0, greedy, None).unwrap_or_else(|e| panic!("prefill: {e}"));
        let lanes = [VerifyLane::greedy(&[])];
        let err = decode_program_verify(&model, &pool, &mut [&mut seq], &lanes, WINDOW).expect_err("a window on a load without one must be refused");
        assert!(err.contains(&WINDOW.to_string()) && err.contains("loaded with"), "{err}");
        let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert_eq!(stats.verify_graph_ready_mask, 0, "no verify graph without a window");
    }

    // --- AC 4: graph replay == eager at every width 1..8, interleaving ---
    for width in 1..=8usize {
        let prompts = &prompts8[..width];
        let model = load_verify(&reader, &artifact, &handles);
        let eager_pool = verify_pool(width as u32);
        let eager = run_verify(&model, &eager_pool, prompts, greedy, Drafter::Random, &[], TOTAL, false);
        drop(eager_pool);
        drop(model);

        let model = load_verify(&reader, &artifact, &handles);
        let graph_pool = verify_pool(width as u32);
        let _ = capture_decode_graphs(&model, &graph_pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let stats = program_stats(&model, &graph_pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert!(stats.verify_graph_ready_mask & (1 << (width - 1)) != 0, "width {width}: no verify graph");
        let graph = run_verify(&model, &graph_pool, prompts, greedy, Drafter::Random, &[], TOTAL, true);
        drop(graph_pool);
        drop(model);
        assert_eq!(eager.emitted, graph.emitted, "width {width}: verify graph replay diverged from eager");
        assert_eq!(eager.rounds, graph.rounds, "width {width}: replay accepted differently from eager");
    }
    // An interleaved prefill chunk between two replays changes nothing: the
    // verify traversal never touches `scratch`, only its own reservations.
    {
        const WIDTH: usize = 2;
        let model = load_verify(&reader, &artifact, &handles);
        let pool = verify_pool(WIDTH as u32 + 1);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        let two_rounds = |interleave: bool| -> Vec<Vec<Vec<i32>>> {
            let mut lanes: Vec<Seq<'_>> = (0..WIDTH)
                .map(|_| pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}")))
                .collect();
            for (seq, prompt) in lanes.iter_mut().zip(&prompts8) {
                prefill_program_sampled(&model, &pool, seq, prompt, 0, greedy, None).unwrap_or_else(|e| panic!("prefill: {e}"));
            }
            let mut out = Vec::new();
            for round in 0..2 {
                if interleave && round == 1 {
                    let mut interloper = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("alloc: {e}"));
                    prefill_program_sampled(&model, &pool, &mut interloper, &prompts8[7], 0, greedy, None).unwrap_or_else(|e| panic!("prefill: {e}"));
                }
                let drafts: Vec<Vec<i32>> = (0..WIDTH).map(|i| propose(Drafter::Random, i, round * 3, round, TOTAL)).collect();
                let verify_lanes: Vec<VerifyLane<'_>> = drafts.iter().map(|d| VerifyLane::greedy(d)).collect();
                let mut refs: Vec<&mut Seq<'_>> = lanes.iter_mut().collect();
                out.push(decode_program_verify(&model, &pool, &mut refs, &verify_lanes, WINDOW).unwrap_or_else(|e| panic!("decode: {e}")));
                let stats = program_stats(&model, &pool).unwrap_or_else(|e| panic!("stats: {e}"));
                assert_eq!(stats.graph_launches, 1, "the round did not replay a verify graph");
            }
            out
        };
        let baseline = two_rounds(false);
        let interleaved = two_rounds(true);
        assert_eq!(baseline, interleaved, "a prefill chunk between two verify replays changed the second replay");
        drop(pool);
        drop(model);
    }

    // --- hq-e8-2b, the serving format ------------------------------------
    // The same substrate under the codec: the GQA verify columns take the hq
    // width-8 tile (one pass for draft 7 + bonus) instead of BF16's two
    // passes of six. BF16 above is the correctness oracle (ADR 0022). hq is
    // not held to its own spec-off stream: the one-column decode tile and the
    // verify tile read the lossy codec through different kernels, and that
    // moves logits beyond the two prefill routes' spread -- measured while
    // writing this leg, the hq decode picked 6061 where the verify round and
    // both hq prefill routes pick 5, by 0.44-0.56 logits.
    //
    // Nor is hq held to rollback equality against its own extent-0 text. Under
    // hq, column 0's pick depends on how many later columns are valid, which
    // cannot happen causally and does not happen under BF16 on the identical
    // verify code (column 0 is exactly independent of the extent there, at
    // every lane). Measured while writing this leg, at width 4 and 8: lane 3
    // (after `2`) picks 44370 with a masked tail and 5 with the full window;
    // lane 5 (after `198`) picks 2 masked and 46 full. Both hq prefill routes
    // rank 44370 and 2 first (46 third), so the masked picks are the ones the
    // references agree with, and the difference sits inside the ~0.5-logit
    // route spread already measured. That points at the vendored hq small-T
    // tile's masked tiling, not at this round; it is recorded for its own
    // investigation rather than patched here (ADR 0010).
    //
    // So hq is checked on what does hold: a multi-token commit under the
    // codec, and replay against eager at every width, strictly.
    let hq = KvFormat::HqE8_2b;
    {
        let model = load_verify_in(hq, &reader, &artifact, &handles);
        let pool = verify_pool_in(hq, 4);
        let _ = capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("hq capture: {e}"));
        let nothing = run_verify(&model, &pool, &prompts8[..4], greedy, Drafter::Nothing, &[], TOTAL, true);
        let own = run_verify(&model, &pool, &prompts8[..4], greedy, Drafter::Oracle(&nothing.emitted), &[], TOTAL, true);
        assert!(
            own.rounds.iter().flatten().any(|&(_, _, _, accepted)| accepted > 0),
            "hq width 4: no round committed more than one token"
        );
        drop(pool);
        drop(model);
    }
    for width in 1..=8usize {
        let prompts = &prompts8[..width];
        let model = load_verify_in(hq, &reader, &artifact, &handles);
        let eager_pool = verify_pool_in(hq, width as u32);
        let eager = run_verify(&model, &eager_pool, prompts, greedy, Drafter::Random, &[], TOTAL, false);
        drop(eager_pool);
        drop(model);

        let model = load_verify_in(hq, &reader, &artifact, &handles);
        let graph_pool = verify_pool_in(hq, width as u32);
        let _ = capture_decode_graphs(&model, &graph_pool).unwrap_or_else(|e| panic!("hq capture: {e}"));
        let stats = program_stats(&model, &graph_pool).unwrap_or_else(|e| panic!("stats: {e}"));
        assert!(stats.verify_graph_ready_mask & (1 << (width - 1)) != 0, "hq width {width}: no verify graph");
        let graph = run_verify(&model, &graph_pool, prompts, greedy, Drafter::Random, &[], TOTAL, true);
        drop(graph_pool);
        drop(model);
        assert_eq!(eager.emitted, graph.emitted, "hq width {width}: verify graph replay diverged from eager");
        assert_eq!(eager.rounds, graph.rounds, "hq width {width}: replay accepted differently from eager");
    }
}
