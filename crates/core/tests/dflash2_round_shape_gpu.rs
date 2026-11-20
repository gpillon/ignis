//! The shape a two-shape DFlash2 load's verify rounds take, on real rounds
//! (GitHub #689, #690, spec #684): the shape rule as the leaf applies it, the
//! shape policy `all` (a round is one shape for all its lanes, extended at 1
//! to 4 lanes only when every lane asks), a lane's ask as part of its
//! sequence through a snapshot and a restore, and the shape policies
//! `aggregate` and `all` on the same two lanes' forced accepts.
//!
//! The rule and the policy are pure functions with their own host-only
//! tables (`kernel/tests/test_round_shape.cpp`). What those tables cannot hold
//! is the round around them: that the leaf feeds the rule each lane's own
//! extent and accept, that a round runs the shape its lanes' asks give it,
//! that the lanes limit is the extended shape's, that what a lane asks is
//! the sequence's, not the handle's, and that the policy reads each lane's
//! own recent rounds. Those are here, on two loads of one materialized
//! artifact: the NVFP4 body, DFlash2 at 15 under the sampled draft, hq-e8-2b,
//! FP8 exception weights, 8 lanes, the rule at its default thresholds and
//! `IGNIS_ROUND_SHAPE` unset unless a round names it; the first load under
//! the policy `all` (parts 1 to 5), the second under `aggregate`, the default
//! (part 5).
//!
//! The rounds are live and greedy, on a text the drafter predicts far ahead
//! (counting), so the accept is the text's, not the test's: the test keeps a
//! mirror of every lane's ask, written from the spec's rule (enter after a
//! base round that drafted all 7 and had all 7 taken, stay while an extended
//! round takes 6, a base round its budget cut short changes nothing), feeds
//! it each round's extent and accept, and holds every round's shape
//! ([`LaneVerifyRun::extended`]) to what the mirror predicts. Where the text
//! must not decide, the round's token budget does: an extended round under a
//! budget of 4 tokens takes at most 3 drafts and sends its lane back. Each
//! part also asserts that the cases it is about happened, so a text that
//! never took a full base round fails instead of passing on nothing.
//!
//! 1. **One lane's course**: base with no history; entry and stay as the
//!    text gives them; an exit by budget; a base round its budget cut short,
//!    forced by `IGNIS_ROUND_SHAPE=base`, that keeps the lane's ask; and
//!    `extended` forcing a lane that does not ask.
//! 2. **One shape a round, and the lanes limit**: a lane that asks beside
//!    one that does not runs base; two that ask run extended; five lanes
//!    that all ask run base, and four of them extended.
//! 3. **A lane's ask is its sequence's**: a lane that asks, snapshotted and
//!    restored onto a fresh handle and onto its own, still asks; a blob that
//!    does not ask, restored onto a handle that did, does not.
//! 4. **A cut round asks by the target's accept** (#691): a lane whose every
//!    run the host cuts to its anchor alone, as a grammar's first draft
//!    refused would, takes the shape its own rounds give it, the rule reading
//!    the drafts the target accepted and not the none the cut kept: it enters
//!    after a base round whose 7 drafts were taken and stays while its
//!    extended rounds take 6.
//! 5. **The policy on forced accepts** (#690): two lanes, each round
//!    teacher-forced with drafts the recording agrees with for as many as the
//!    test says ([`forced_round`]), so the accepts are the test's, not the
//!    text's. A lane that takes every draft beside one that takes 2 (a high
//!    expected gain beside a conversation): under `aggregate` its first round
//!    asking is granted the extended shape and every round after stays
//!    extended, under `all` every round is base. A lane that takes all 7 of
//!    a base round and nothing past them (a low one, still asking): under
//!    `aggregate` its granted round is extended and every round after base,
//!    its window holding that round; under `all` every round is base.
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a **skip**; under the profile the same
//! condition is a **hard failure**. Run via `scripts/gpu-profile.ps1`. Its own
//! binary: it writes the process environment the leaf reads.

#![cfg(feature = "cuda")]

#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b_with, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_decode_lanes, model_scope, Model};
use ignis_core::seq::{snapshot_format_version, Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{
    capture_decode_graphs, decode_program_verify_cut, decode_program_verify_forced_with, decode_program_verify_runs,
    prefill_program_sampled, LaneVerifyRun, SamplingParams, TeacherForcing, VerifyLane,
};
use ignis_core::{
    DraftRule, DrafterSampling, ExceptionWeights, KvFormat, PrefillAttention, PrefillLinear, RopeScaling, ShapeEstimate, ShapePolicy,
    ShapeRule, Speculation, SpeculativeBackend,
};

const NVFP4: &str = "F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer";
/// The load's window (the extended shape's drafts a lane), the base shape's,
/// and the most lanes an extended round runs.
const WINDOW: u32 = 15;
const BASE_WINDOW: u32 = 7;
const EXTENDED_LANES: usize = 4;
const LANES: u32 = 8;
const CHUNK: u32 = 128;
/// Each sequence's context: its prompt and every round below at the window.
const CONTEXT: u32 = 1536;
/// The switch that forces a round's shape over the rule.
const ROUND_SHAPE: &str = "IGNIS_ROUND_SHAPE";
/// The token budget of an exit: the anchor and 3 drafts, fewer than the stay
/// threshold an extended round could take.
const EXIT_BUDGET: u32 = 4;
/// The most rounds a part waits for the text to give a lane its ask.
const PATIENCE: usize = 16;

fn set(name: &str, value: Option<&str>) {
    // SAFETY (test-only): this binary's one test reads and writes the
    // environment on its own thread, where the leaf reads it too.
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

/// The spec's shape rule, at the load's thresholds: what a lane asks of its
/// next round after one at `extended` (else the base shape) of `extent`
/// drafts of which the target took `accepted`, having asked `asked`.
fn asks_after(rule: ShapeRule, asked: bool, extended: bool, extent: u32, accepted: u32) -> bool {
    if extended {
        accepted >= rule.stay_accepted()
    } else if extent < BASE_WINDOW {
        asked
    } else {
        accepted >= rule.enter_accepted()
    }
}

/// The test's mirror of the round: the lanes riding it with what each asks,
/// and what the round must run under `forced` (`IGNIS_ROUND_SHAPE`'s value).
fn predicted(asks: &[bool], forced: Option<&str>) -> bool {
    asks.len() <= EXTENDED_LANES
        && match forced {
            Some("extended") => true,
            Some(_) => false,
            None => asks.iter().all(|&ask| ask),
        }
}

/// One live greedy verify round over `seqs`, `asks` their mirror, each lane
/// under a budget of `budget` tokens (0: none) and the shape forced by
/// `forced`: the round's shape, held to the mirror's prediction, each lane's
/// extent held to its shape's window under the budget, and the mirror moved
/// by the round. Failures are collected, not panicked, so every part runs.
#[allow(clippy::too_many_arguments)]
fn round(
    model: &Model,
    pool: &SeqPool,
    rule: ShapeRule,
    seqs: &mut [&mut Seq<'_>],
    asks: &mut [bool],
    budget: u32,
    forced: Option<&str>,
    failures: &mut Vec<String>,
    what: &str,
) -> bool {
    let want = predicted(asks, forced);
    set(ROUND_SHAPE, forced);
    let lanes = vec![VerifyLane { remaining_tokens: budget, ..VerifyLane::greedy(&[]) }; seqs.len()];
    let runs: Vec<LaneVerifyRun> = decode_program_verify_runs(model, pool, seqs, &lanes, WINDOW).unwrap_or_else(|e| panic!("{what}: {e}"));
    set(ROUND_SHAPE, None);
    let extended = runs[0].extended;
    if runs.iter().any(|run| run.extended != extended) {
        failures.push(format!("{what}: one round, lanes of two shapes"));
    }
    println!(
        "dflash2_round_shape_gpu {what}: {} lanes asking {asks:?}, {} (expected {}), accepts {:?}",
        asks.len(),
        if extended { "extended" } else { "base" },
        if want { "extended" } else { "base" },
        runs.iter().map(|run| run.tokens.len() - 1).collect::<Vec<_>>()
    );
    if extended != want {
        failures.push(format!("{what}: lanes asking {asks:?} ran {} where the rule gives {}", shape(extended), shape(want)));
    }
    let window = if extended { WINDOW } else { BASE_WINDOW };
    let expected_extent = if budget == 0 { window } else { window.min(budget - 1) };
    for (lane, run) in runs.iter().enumerate() {
        if run.extent != expected_extent {
            failures.push(format!("{what}: lane {lane} drafted {} at its round's shape, not {expected_extent}", run.extent));
        }
        let accepted = run.tokens.len() as u32 - 1;
        asks[lane] = asks_after(rule, asks[lane], extended, run.extent, accepted);
    }
    extended
}

fn shape(extended: bool) -> &'static str {
    if extended { "the extended shape" } else { "the base shape" }
}

/// #690: one teacher-forced verify round of `seqs` whose shape the load's
/// rule and policy choose, lane i's drafts agreeing with its recording for
/// its first `agree[i]`, so that the target accepts that many of a window
/// that long: its shape, each lane's accept held to what the forcing gives
/// it. The recordings walk `text` from `at`.
fn forced_round(
    model: &Model,
    pool: &SeqPool,
    seqs: &mut [&mut Seq<'_>],
    agree: &[u32],
    text: &[i32],
    at: usize,
    failures: &mut Vec<String>,
    what: &str,
) -> bool {
    let recordings: Vec<Vec<i32>> = (0..seqs.len())
        .map(|lane| (0..=WINDOW as usize).map(|j| text[(at + 3 * lane + j) % text.len()]).collect())
        .collect();
    let drafts: Vec<Vec<i32>> = recordings
        .iter()
        .zip(agree)
        .map(|(recording, &agree)| {
            (0..WINDOW as usize)
                .map(|j| if j < agree as usize { recording[j] } else if recording[j] == 13 { 14 } else { 13 })
                .collect()
        })
        .collect();
    let recording_refs: Vec<&[i32]> = recordings.iter().map(Vec::as_slice).collect();
    let draft_refs: Vec<&[i32]> = drafts.iter().map(Vec::as_slice).collect();
    let lanes = vec![VerifyLane::greedy(&[]); seqs.len()];
    let forcing = TeacherForcing { shape_window: None, drafts: Some(&draft_refs) };
    let runs = decode_program_verify_forced_with(model, pool, seqs, &lanes, WINDOW, &recording_refs, forcing)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let extended = runs[0].extended;
    if runs.iter().any(|run| run.extended != extended) {
        failures.push(format!("{what}: one round, lanes of two shapes"));
    }
    let window = if extended { WINDOW } else { BASE_WINDOW };
    for (lane, (run, &agree)) in runs.iter().zip(agree).enumerate() {
        let accepted = run.tokens.len() as u32 - 1;
        if (run.extent, accepted) != (window, agree.min(window)) {
            failures.push(format!(
                "{what}: lane {lane} drafted {} and took {accepted}, not the forced {window} and {}",
                run.extent,
                agree.min(window)
            ));
        }
    }
    extended
}

/// #690: part 5 on a load under `policy`: two fresh lanes through the
/// courses of its note, each round's shape held to the course's.
fn the_policy_on_forced_accepts(model: &Model, pool: &SeqPool, policy: ShapePolicy, prompt: &[i32], failures: &mut Vec<String>) {
    const ROUNDS: usize = 6;
    let aggregate = policy == ShapePolicy::Aggregate;
    // Lane 0's agreement a round, and the shapes its rounds take under the
    // policy: the high gain's granted round and every one after extended, the
    // low gain's granted round alone; under `all` none, lane 1 never asking.
    let courses: [(&str, u32, [bool; ROUNDS]); 2] = [
        ("a high expected gain", WINDOW, [false, aggregate, aggregate, aggregate, aggregate, aggregate]),
        ("a low expected gain", BASE_WINDOW, [false, aggregate, false, false, false, false]),
    ];
    for (case, agree, shapes) in courses {
        let mut lanes: Vec<Seq<'_>> = (0..2)
            .map(|lane| {
                let mut seq = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("part 5: alloc lane {lane}: {e}"));
                prefill_program_sampled(model, pool, &mut seq, prompt, 0, SamplingParams::greedy(), None)
                    .unwrap_or_else(|e| panic!("part 5: prefill lane {lane}: {e}"));
                seq
            })
            .collect();
        let [a, b] = &mut lanes[..] else { unreachable!("two lanes") };
        for (at, &want) in shapes.iter().enumerate() {
            let what = format!("part 5, `{}`, {case} beside a lane that takes 2: round {at}", policy.as_str());
            let extended = forced_round(model, pool, &mut [&mut *a, &mut *b], &[agree, 2], prompt, 5 * at, failures, &what);
            println!("dflash2_round_shape_gpu {what}: {}", if extended { "extended" } else { "base" });
            if extended != want {
                failures.push(format!("{what}: ran {} where the policy gives {}", shape(extended), shape(want)));
            }
        }
    }
}

/// The ask the blob's progress image carries (format 6).
fn blob_asks(blob: &[u8]) -> u32 {
    let (at, _) = snapshot_blob::section(blob, snapshot_blob::SECTION_PROGRESS);
    snapshot_blob::read_u32(blob, at + snapshot_blob::PROGRESS_ASKS_EXTENDED_SHAPE)
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_two_shape_loads_rounds_take_the_shape_their_lanes_ask_one_shape_a_round_and_a_lanes_ask_is_its_sequences() {
    let path = Path::new(NVFP4);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {NVFP4}")) {
        return;
    }
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let (_, (major, minor)) = CudaDevice::identity(0).unwrap_or_else(|e| panic!("device identity: {e}"));
    if major != 12 && gpu_profile::skip_or_fail(&format!("compute capability {major}.{minor}: FP8 exception weights need 12.0")) {
        return;
    }
    assert_eq!(snapshot_format_version(), 6, "this test reads the version-6 blob layout");
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open {NVFP4}: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let prompt: Vec<i32> = frontend
        .tokenizer()
        .encode("<|im_start|>user\nCount from one to five hundred, one number per line.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        .unwrap_or_else(|e| panic!("tokenize: {e}"))
        .into_iter()
        .map(|id| i32::try_from(id).expect("token id fits i32"))
        .collect();

    let rule = ShapeRule::default();
    // #690: parts 1 to 3 hold `all`, named on the first load; the second is
    // the default's, `aggregate`.
    let aggregate = Speculation::new(SpeculativeBackend::Dflash2, WINDOW)
        .unwrap_or_else(|e| panic!("speculation: {e}"))
        .with_draft_rule(DraftRule::Sampled, DrafterSampling::default())
        .with_shape_rule(rule)
        .unwrap_or_else(|e| panic!("shape rule: {e}"));
    assert_eq!(aggregate.shape_policy(), ShapePolicy::Aggregate, "the default policy");
    let spec = aggregate.with_shape_policy(ShapePolicy::All, ShapeEstimate::default());
    let (plan, handles) =
        bind_model_scope_27b_with(&reader, model_scope(Some(spec), None, ExceptionWeights::Fp8)).unwrap_or_else(|e| panic!("bind: {e}"));
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut failures: Vec<String> = Vec::new();
    let load = |spec: Speculation| {
        let model = load_qwen38_27b_with_decode_lanes(
            &reader,
            &artifact,
            &handles,
            CHUNK,
            CONTEXT,
            KvFormat::HqE8_2b,
            Some(spec),
            None,
            RopeScaling::NONE,
            PrefillAttention::default(),
            PrefillLinear::default(),
            LANES,
        )
        .unwrap_or_else(|e| panic!("model load: {e}"));
        let pool = SeqPool::create_with_speculation(
            &ModelConfig::qwen38_27b(),
            &SeqPoolBudget {
                kv_format: KvFormat::HqE8_2b,
                kv_page_group_count: CONTEXT.div_ceil(64) * LANES,
                max_context_tokens: CONTEXT,
                slot_count: LANES,
                retained_slot_count: 0,
                retained_host_slot_count: 0,
            },
            Some(SpeculativeBackend::Dflash2),
        )
        .unwrap_or_else(|e| panic!("pool: {e}"));
        set(ROUND_SHAPE, None);
        capture_decode_graphs(&model, &pool).unwrap_or_else(|e| panic!("capture: {e}"));
        (model, pool)
    };
    {
        let (model, pool) = load(spec);
        let fresh = |what: &str| {
            let mut seq = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("{what}: alloc: {e}"));
            prefill_program_sampled(&model, &pool, &mut seq, &prompt, 0, SamplingParams::greedy(), None)
                .unwrap_or_else(|e| panic!("{what}: prefill: {e}"));
            seq
        };
        let mut round = |seqs: &mut [&mut Seq<'_>], asks: &mut [bool], budget: u32, forced: Option<&str>, what: &str| {
            round(&model, &pool, rule, seqs, asks, budget, forced, &mut failures, what)
        };
        let mut missing: Vec<String> = Vec::new();

        // 1. One lane's course.
        {
            let mut lane = fresh("part 1");
            let mut ask = [false];
            // No history: the base shape.
            let first = round(&mut [&mut lane], &mut ask, 0, None, "part 1: the lane's first round");
            let (mut entered, mut stayed) = (!first && ask[0], false);
            for at in 0..PATIENCE {
                let was = ask[0];
                let extended = round(&mut [&mut lane], &mut ask, 0, None, &format!("part 1: live round {at}"));
                entered |= !extended && ask[0];
                stayed |= extended && was && ask[0];
            }
            if !entered {
                missing.push("part 1: no base round took all 7 drafts, so no lane entered the extended shape".into());
            }
            if !stayed {
                missing.push("part 1: no extended round kept its lane asking".into());
            }
            // An exit by budget: wait for the lane to ask, then an extended
            // round of 3 drafts sends it back to the base shape.
            for at in 0..PATIENCE {
                if ask[0] {
                    break;
                }
                round(&mut [&mut lane], &mut ask, 0, None, &format!("part 1: towards the exit, round {at}"));
            }
            if ask[0] {
                round(&mut [&mut lane], &mut ask, EXIT_BUDGET, None, "part 1: an extended round under a budget of 4 tokens");
                if ask[0] {
                    missing.push("part 1: an extended round of 3 drafts left its lane asking".into());
                }
                round(&mut [&mut lane], &mut ask, 0, None, "part 1: after the exit, the base shape");
            } else {
                missing.push("part 1: the lane never asked again, so no exit by budget ran".into());
            }
            // A base round its budget cut short changes nothing: forced to the
            // base shape while the lane asks, then at the rule's shape.
            for at in 0..PATIENCE {
                if ask[0] {
                    break;
                }
                round(&mut [&mut lane], &mut ask, 0, None, &format!("part 1: towards a short base round, round {at}"));
            }
            if ask[0] {
                round(&mut [&mut lane], &mut ask, 3, Some("base"), "part 1: a base round of 2 drafts, forced, while the lane asks");
                if !ask[0] {
                    missing.push("part 1: a base round its budget cut short took the lane's ask".into());
                }
                round(&mut [&mut lane], &mut ask, 0, None, "part 1: after the short base round, still extended");
            } else {
                missing.push("part 1: the lane never asked again, so no short base round ran".into());
            }
            // And while it does not ask: a short base round leaves it at the
            // base shape, and `extended` forces it there.
            round(&mut [&mut lane], &mut ask, EXIT_BUDGET, None, "part 1: a round of 3 drafts, after which the lane does not ask");
            round(&mut [&mut lane], &mut ask, 3, None, "part 1: a short base round of a lane that does not ask");
            if ask[0] {
                missing.push("part 1: a lane that did not ask asks after a base round its budget cut short".into());
            }
            round(&mut [&mut lane], &mut ask, EXIT_BUDGET, Some("extended"), "part 1: `extended` forces a lane that does not ask");
            round(&mut [&mut lane], &mut ask, 0, None, "part 1: after it, the base shape");
        }

        // 2. One shape a round, and the lanes limit.
        {
            let mut lanes: Vec<Seq<'_>> = (0..EXTENDED_LANES + 1).map(|_| fresh("part 2")).collect();
            let mut asks = vec![false; lanes.len()];
            let [a, b, c, d, e] = &mut lanes[..] else { unreachable!("five lanes") };
            for at in 0..PATIENCE {
                if asks[0] {
                    break;
                }
                round(&mut [&mut *a], &mut asks[..1], 0, None, &format!("part 2: lane a alone, round {at}"));
            }
            if asks[0] {
                // b has no history: the round of both is at the base shape.
                round(&mut [&mut *a, &mut *b], &mut asks[..2], 0, None, "part 2: a asks and b does not");
            } else {
                missing.push("part 2: lane a never asked".into());
            }
            let mut both = false;
            for at in 0..PATIENCE {
                both |= asks[0] && asks[1];
                round(&mut [&mut *a, &mut *b], &mut asks[..2], 0, None, &format!("part 2: a and b, round {at}"));
                if both {
                    break;
                }
            }
            if !both {
                missing.push("part 2: a and b never both asked".into());
            }
            // Five lanes are past the extended shape's four: the base shape,
            // whatever they ask; then four that ask run extended.
            let (mut five, mut four) = (false, false);
            for at in 0..PATIENCE {
                five |= asks.iter().all(|&ask| ask);
                round(&mut [&mut *a, &mut *b, &mut *c, &mut *d, &mut *e], &mut asks, 0, None, &format!("part 2: five lanes, round {at}"));
                if five && asks[1..].iter().all(|&ask| ask) {
                    round(&mut [&mut *b, &mut *c, &mut *d, &mut *e], &mut asks[1..], 0, None, "part 2: four of them, every one asking");
                    four = true;
                    break;
                }
            }
            if !five {
                missing.push("part 2: five lanes never all asked".into());
            }
            if !four {
                missing.push("part 2: four lanes that all ask never rode together".into());
            }
        }

        // 3. A lane's ask is its sequence's.
        {
            let mut lane = fresh("part 3");
            let mut ask = [false];
            let cold = fresh("part 3, a lane with no history");
            let cold_blob = cold.snapshot().unwrap_or_else(|e| panic!("part 3: snapshot the cold lane: {e}"));
            drop(cold);
            if blob_asks(&cold_blob) != 0 {
                missing.push("part 3: a lane with no history's blob asks the extended shape".into());
            }
            for at in 0..PATIENCE {
                if ask[0] {
                    break;
                }
                round(&mut [&mut lane], &mut ask, 0, None, &format!("part 3: the lane, round {at}"));
            }
            if ask[0] {
                let blob = lane.snapshot().unwrap_or_else(|e| panic!("part 3: snapshot: {e}"));
                if blob_asks(&blob) != 1 {
                    missing.push("part 3: the blob of a lane that asks does not say so".into());
                }
                // Onto a fresh handle, as a scheduler restores an evicted request.
                let mut restored = pool.alloc(CONTEXT).unwrap_or_else(|e| panic!("part 3: alloc: {e}"));
                restored.restore(&blob).unwrap_or_else(|e| panic!("part 3: restore onto a fresh handle: {e}"));
                round(&mut [&mut restored], &mut [true], 0, None, "part 3: restored onto a fresh handle, the lane still asks");
                drop(restored);
                // A blob that does not ask, onto the handle of one that does:
                // the image's ask, not the handle's.
                lane.restore(&cold_blob).unwrap_or_else(|e| panic!("part 3: restore the cold blob onto the lane: {e}"));
                round(&mut [&mut lane], &mut [false], 0, None, "part 3: a cold blob restored onto an asking handle asks nothing");
                // And its own blob back onto its own handle.
                lane.restore(&blob).unwrap_or_else(|e| panic!("part 3: restore onto its own handle: {e}"));
                round(&mut [&mut lane], &mut [true], 0, None, "part 3: restored onto its own handle, the lane asks again");
            } else {
                missing.push("part 3: the lane never asked".into());
            }
        }

        // 4. A cut round asks by the target's accept (#691). Every run is cut
        // to its anchor, the draft after it pending: a rule that read the cut
        // would see no draft taken and never ask. Each round is held to the
        // mirror, which the round before moved by what the target accepted.
        {
            let mut lane = fresh("part 4");
            let mut ask = false;
            let (mut entered, mut stayed) = (false, false);
            for at in 0..PATIENCE {
                let what = format!("part 4: cut round {at}");
                let mut accepted = 0u32;
                let lanes = [VerifyLane::greedy(&[])];
                let runs = decode_program_verify_cut(&model, &pool, &mut [&mut lane], &lanes, WINDOW, &[true], &mut |_, run| {
                    accepted = run.len() as u32 - 1;
                    1
                })
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                let run = &runs[0];
                println!(
                    "dflash2_round_shape_gpu {what}: asking {ask}, {}, accepted {accepted}, kept {}",
                    shape(run.extended),
                    run.tokens.len()
                );
                if run.extended != ask {
                    failures.push(format!("{what}: a lane {} ran {}", if ask { "that asks" } else { "that does not ask" }, shape(run.extended)));
                }
                if run.tokens.len() != 1 {
                    failures.push(format!("{what}: the cut kept {} tokens, not the anchor alone", run.tokens.len()));
                }
                let was = ask;
                ask = asks_after(rule, ask, run.extended, run.extent, accepted);
                // Counted where the next round holds the ask to its shape.
                if at + 1 < PATIENCE {
                    entered |= !run.extended && ask;
                    stayed |= run.extended && was && ask;
                }
            }
            if !entered {
                missing.push("part 4: no base round the host cut had all 7 drafts taken, so no cut sent its lane extended".into());
            }
            if !stayed {
                missing.push("part 4: no extended round the host cut took 6 drafts, so no cut kept its lane asking".into());
            }
        }
        failures.extend(missing);

        // 5. The policy on forced accepts: `all` here, `aggregate` below.
        the_policy_on_forced_accepts(&model, &pool, ShapePolicy::All, &prompt, &mut failures);
    }
    {
        let (model, pool) = load(aggregate);
        the_policy_on_forced_accepts(&model, &pool, ShapePolicy::Aggregate, &prompt, &mut failures);
    }
    artifact.release_arena(&mut device).unwrap_or_else(|e| panic!("release: {e}"));
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
