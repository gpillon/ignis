//! GPU test for the drafter floor (GitHub #506, spec #503 "Drafter taps"): a
//! prefill job taps the DFlash2 drafter's window from the later of its own
//! last 2048 positions and the floor the scheduler hands it — the earliest
//! retained-image point at or past the job's end, minus the window — so the
//! positions no later window reads are never tapped or projected.
//!
//! What must not move, checked against today's rule (every job's floor 0) on
//! one prompt cut into several jobs, in both shapes a load cuts it in — the
//! 27B's two cuts (a prefix at the opener's page floor, then the checkpoint at
//! the opener) and the single cut whose capture lends the pages below it
//! (#306, what #508 gives the 27B on Ampere and Ada):
//!
//! - **the window at every retained-image point** — the published prefix's
//!   image, the checkpoint's image and the sequence at its prompt's end —
//!   holds the same rows as under today's rule;
//! - **what is restored from them proposes the same drafts**: a claimant of
//!   the prefix and one of the checkpoint, read at once and after a short tail
//!   of their own, and the prompt's end, propose what today's do;
//! - **a span wholly below its floor taps nothing** and still moves the
//!   drafter's frontier to its end, so the checks on that frontier hold.
//!
//! The windows are read out of snapshot blobs (`support/snapshot_blob.rs`,
//! version 5 only: a format change fails the version assertion).
//!
//! Explicit GPU profile (ADR 0006, GitHub #38): outside `IGNIS_GPU_PROFILE=1`
//! a missing artifact or GPU is a skip; under the profile, a hard failure.

#![cfg(feature = "cuda")]

#[path = "support/snapshot_blob.rs"]
mod snapshot_blob;

use std::path::Path;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, DraftModule, FrontendSet, Reader};
use ignis_core::compute::ModelConfig;
use ignis_core::dflash2_readout;
use ignis_core::gpu_profile;
use ignis_core::model_load::{load_qwen38_27b_with_speculation, Model};
use ignis_core::seq::{snapshot_format_version, Seq, SeqPool, SeqPoolBudget};
use ignis_core::step::{prefill_program_draw, LaneDraw, SamplingParams};
use ignis_core::{KvFormat, Speculation, SpeculativeBackend};

use snapshot_blob::{read_u64, section, PROGRESS_DRAFTER_FRONTIER, SECTION_DFLASH_WINDOW, SECTION_PROGRESS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MAX_CONTEXT: u32 = 4096;
/// The load's prefill chunk: every job below spans several, and the ones the
/// floor cuts into start mid-chunk.
const PREFILL_CHUNK: u32 = 128;
/// The leaf's KV page.
const PAGE: u32 = 64;
const WINDOW: u32 = 2048;
const DRAFT_TOKENS: u32 = 7;
/// The prompt: a prefix published at `PUBLISH` (40 whole pages), its
/// generation opener at `OPENER` inside the next page, and its end at
/// `PROMPT`, far enough past the window that most of its jobs tap little or
/// nothing.
const PUBLISH: u32 = 2560;
const OPENER: u32 = 2600;
const PROMPT: u32 = 3000;
/// Each claimant's own tail: shorter than the window, so the window it
/// proposes from reaches back into the image it resumed from.
const TAIL: u32 = 100;

/// One prefill job: its span's start and length, and the floor the scheduler
/// hands it.
#[derive(Clone, Copy)]
struct Job {
    start: u32,
    len: u32,
    floor: u32,
}

const fn job(start: u32, len: u32, floor: u32) -> Job {
    Job { start, len, floor }
}

/// The 27B's two cuts: 512-token jobs to the publish point, the opener's short
/// chunk, then the rest. Floors as `ConcreteScheduler` deals them
/// (`crates/core/tests/drafter_floor.rs`): the prefix's 2560 - 2048 before it,
/// the checkpoint's 2600 - 2048 for the opener's chunk, the end's 3000 - 2048
/// last. The first job lies wholly below its floor.
const TWO_CUTS: [Job; 7] = [
    job(0, 512, 512),
    job(512, 512, 512),
    job(1024, 512, 512),
    job(1536, 512, 512),
    job(2048, 512, 512),
    job(2560, 40, 552),
    job(2600, 400, 952),
];

/// The single cut: no prefix at the opener's page floor, so every job before
/// the opener reads its floor from the checkpoint, 2600 - 2048 — a floor that
/// starts mid-chunk in the second job.
const SINGLE_CUT: [Job; 6] = [
    job(0, 512, 552),
    job(512, 512, 552),
    job(1024, 512, 552),
    job(1536, 512, 552),
    job(2048, 552, 552),
    job(2600, 400, 952),
];

/// What one run of a prompt leaves: the window at each retained-image point
/// and the drafts proposed from each.
struct Run {
    prefix_window: Option<Vec<u8>>,
    checkpoint_window: Vec<u8>,
    end_window: Vec<u8>,
    /// Drafts at the prompt's end; from the checkpoint's claimant at once and
    /// after its tail; from the prefix's claimant after its tail.
    end_drafts: Vec<i32>,
    checkpoint_drafts: Vec<i32>,
    checkpoint_tail_drafts: Vec<i32>,
    prefix_tail_drafts: Option<Vec<i32>>,
}

fn window_of(blob: &[u8]) -> Vec<u8> {
    let (at, bytes) = section(blob, SECTION_DFLASH_WINDOW);
    blob[at..at + bytes].to_vec()
}

/// The program and drafter frontiers a snapshot of a live sequence carries.
fn frontiers(blob: &[u8]) -> (u64, u64) {
    let (at, _) = section(blob, SECTION_PROGRESS);
    (read_u64(blob, at), read_u64(blob, at + PROGRESS_DRAFTER_FRONTIER))
}

fn greedy() -> LaneDraw<'static> {
    LaneDraw::free(SamplingParams::greedy())
}

fn drafts(model: &Model, pool: &SeqPool, seq: &Seq<'_>, label: &str) -> Vec<i32> {
    dflash2_readout::propose(model, pool, seq, DRAFT_TOKENS, DRAFT_TOKENS, false)
        .unwrap_or_else(|e| panic!("{label}: propose: {e}"))
        .drafts
}

/// The span `[start, start + tokens.len())` on `seq`, with `floor`.
fn prefill(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, tokens: &[i32], start: u32, floor: u32, label: &str) {
    prefill_program_draw(model, pool, seq, tokens, u64::from(start), u64::from(floor), None, false, false, greedy(), None)
        .unwrap_or_else(|e| panic!("{label}: prefill at {start}: {e}"));
}

/// The claimant's tail at `start`, with the floor its own end gives it under
/// the floor rule (`with_floors`) and 0 under today's.
fn prefill_tail(model: &Model, pool: &SeqPool, seq: &mut Seq<'_>, tail: &[i32], start: u32, with_floors: bool, label: &str) {
    let floor = if with_floors { (start + tail.len() as u32).saturating_sub(WINDOW) } else { 0 };
    prefill(model, pool, seq, tail, start, floor, label);
}

/// Prefill `prompt` as `jobs` (their floors, or 0 for today's rule), taking
/// the prefix at [`PUBLISH`] where a job ends there and the checkpoint at
/// [`OPENER`]; then restore each image into a claimant and read what it
/// proposes.
fn run(
    model: &Model,
    pool: &SeqPool,
    prompt: &[i32],
    tail: &[i32],
    jobs: &[Job],
    with_floors: bool,
    label: &str,
) -> Run {
    let mut seq = pool.alloc(MAX_CONTEXT).unwrap_or_else(|e| panic!("{label}: alloc: {e}"));
    let mut prefix = None;
    let mut checkpoint = None;
    for job in jobs {
        let floor = if with_floors { job.floor } else { 0 };
        let end = job.start + job.len;
        prefill(model, pool, &mut seq, &prompt[job.start as usize..end as usize], job.start, floor, label);
        if floor >= end {
            // A span wholly below its floor taps nothing, and the drafter's
            // frontier still moves with the program's.
            let blob = seq.snapshot().unwrap_or_else(|e| panic!("{label}: snapshot at {end}: {e:?}"));
            assert_eq!(frontiers(&blob), (u64::from(end), u64::from(end)), "{label}: the frontiers at {end}");
            if job.start == 0 {
                assert!(
                    window_of(&blob).iter().all(|&b| b == 0),
                    "{label}: a fresh sequence's span below its floor wrote its window"
                );
            }
        }
        if end == PUBLISH {
            prefix = Some(seq.publish_prefix(PUBLISH, 0).unwrap_or_else(|e| panic!("{label}: publish: {e:?}")));
        }
        if end == OPENER {
            checkpoint =
                Some(seq.capture_checkpoint(OPENER, 1).unwrap_or_else(|e| panic!("{label}: capture: {e:?}")));
        }
    }
    let blob = seq.snapshot().unwrap_or_else(|e| panic!("{label}: snapshot at the end: {e:?}"));
    assert_eq!(frontiers(&blob), (u64::from(PROMPT), u64::from(PROMPT)), "{label}: the frontiers at the end");
    let end_window = window_of(&blob);
    let end_drafts = drafts(model, pool, &seq, &format!("{label}: the prompt's end"));
    drop(seq);

    let checkpoint = checkpoint.unwrap_or_else(|| panic!("{label}: no job ended at the opener"));
    let image = |bytes: u64| vec![0u8; usize::try_from(bytes).expect("an image fits usize")];
    let mut blob = image(checkpoint.snapshot_bytes().unwrap_or_else(|e| panic!("{label}: checkpoint size: {e:?}")));
    checkpoint.snapshot_into(&mut blob).unwrap_or_else(|e| panic!("{label}: checkpoint snapshot: {e:?}"));
    let checkpoint_window = window_of(&blob);
    let mut claimant =
        pool.alloc_from_checkpoint(MAX_CONTEXT, &checkpoint).unwrap_or_else(|e| panic!("{label}: claim: {e}"));
    let checkpoint_drafts = drafts(model, pool, &claimant, &format!("{label}: the checkpoint's claimant"));
    prefill_tail(model, pool, &mut claimant, tail, OPENER, with_floors, label);
    let checkpoint_tail_drafts = drafts(model, pool, &claimant, &format!("{label}: after the checkpoint's tail"));
    drop(claimant);
    drop(checkpoint);

    let (prefix_window, prefix_tail_drafts) = match prefix {
        None => (None, None),
        Some(prefix) => {
            let mut blob = image(prefix.snapshot_bytes().unwrap_or_else(|e| panic!("{label}: prefix size: {e:?}")));
            prefix.snapshot_into(&mut blob).unwrap_or_else(|e| panic!("{label}: prefix snapshot: {e:?}"));
            let mut claimant =
                pool.alloc_shared(MAX_CONTEXT, &prefix).unwrap_or_else(|e| panic!("{label}: claim the prefix: {e}"));
            prefill_tail(model, pool, &mut claimant, tail, PUBLISH, with_floors, label);
            let tail_drafts = drafts(model, pool, &claimant, &format!("{label}: after the prefix's tail"));
            drop(claimant);
            drop(prefix);
            (Some(window_of(&blob)), Some(tail_drafts))
        }
    };
    Run {
        prefix_window,
        checkpoint_window,
        end_window,
        end_drafts,
        checkpoint_drafts,
        checkpoint_tail_drafts,
        prefix_tail_drafts,
    }
}

/// The window under the floor rule holds today's rows, compared as BF16 words.
fn same_rows(today: &[u8], floored: &[u8], label: &str) {
    assert_eq!(today.len(), floored.len(), "{label}: one lane each");
    assert!(today.iter().any(|&b| b != 0), "{label}: today's window is written");
    let words = |bytes: &[u8]| -> Vec<u16> { bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let (a, b) = (words(today), words(floored));
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let zeroed = a.iter().zip(&b).filter(|(x, y)| **x != 0 && **y == 0).count();
    println!("{label}: {differing} of {} window words differ from today's, {zeroed} of them zeroed", a.len());
    assert_eq!(differing, 0, "{label}: the window's rows differ from today's");
}

fn compare(today: &Run, floored: &Run, shape: &str) {
    same_rows(&today.checkpoint_window, &floored.checkpoint_window, &format!("{shape}: the checkpoint's image"));
    same_rows(&today.end_window, &floored.end_window, &format!("{shape}: the prompt's end"));
    match (&today.prefix_window, &floored.prefix_window) {
        (Some(a), Some(b)) => same_rows(a, b, &format!("{shape}: the prefix's image")),
        (None, None) => {}
        _ => panic!("{shape}: one rule published a prefix and the other did not"),
    }
    assert_eq!(floored.end_drafts, today.end_drafts, "{shape}: drafts at the prompt's end");
    assert_eq!(floored.checkpoint_drafts, today.checkpoint_drafts, "{shape}: drafts from the checkpoint");
    assert_eq!(
        floored.checkpoint_tail_drafts, today.checkpoint_tail_drafts,
        "{shape}: drafts after the checkpoint's tail"
    );
    assert_eq!(floored.prefix_tail_drafts, today.prefix_tail_drafts, "{shape}: drafts after the prefix's tail");
    println!(
        "{shape}: end drafts {:?}, checkpoint drafts {:?} / {:?}, prefix drafts {:?}",
        floored.end_drafts, floored.checkpoint_drafts, floored.checkpoint_tail_drafts, floored.prefix_tail_drafts
    );
}

/// `sentence` repeated until it is at least `tokens` long, cut to `tokens`.
fn repeated(sentence: &[i32], tokens: u32) -> Vec<i32> {
    sentence.iter().copied().cycle().take(tokens as usize).collect()
}

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn the_drafter_floor_leaves_every_retained_window_as_today() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    assert_eq!(snapshot_format_version(), 5, "this test reads the version-5 blob layout");
    assert_eq!(PUBLISH % PAGE, 0, "a prefix is whole pages");
    assert_eq!(OPENER / PAGE, PUBLISH / PAGE, "the opener lies in the page after the prefix");
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let frontend = FrontendSet::from_reader(&reader).unwrap_or_else(|e| panic!("frontend: {e}"));
    let encode = |text: &str| -> Vec<i32> {
        frontend
            .tokenizer()
            .encode(text)
            .unwrap_or_else(|e| panic!("tokenize: {e}"))
            .into_iter()
            .map(|id| i32::try_from(id).expect("token id fits i32"))
            .collect()
    };
    let prompt = repeated(
        &encode("A paged KV cache keeps each sequence's keys and values in fixed pages, so prefixes share them. "),
        PROMPT,
    );
    let tail = repeated(&encode("Now say the same thing about a ring buffer, in one sentence. "), TAIL);

    let (plan, handles) =
        bind_model_scope_27b(&reader, Some(DraftModule::Dflash2)).unwrap_or_else(|e| panic!("bind with dflash2: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let mut artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + dflash2: {e}")) {
                return;
            }
            unreachable!("skip_or_fail panics under the profile");
        }
    };
    let speculation = Speculation::new(SpeculativeBackend::Dflash2, DRAFT_TOKENS).unwrap();
    let model = load_qwen38_27b_with_speculation(
        &reader,
        &artifact,
        &handles,
        PREFILL_CHUNK,
        MAX_CONTEXT,
        KvFormat::Bf16,
        Some(speculation),
    )
    .unwrap_or_else(|e| panic!("load with dflash2: {e}"));
    // The prefilled sequence and one claimant at a time, at full context; the
    // prefix and the checkpoint each in a retained slot.
    let budget = SeqPoolBudget {
        kv_format: KvFormat::Bf16,
        kv_page_group_count: 160,
        max_context_tokens: MAX_CONTEXT,
        slot_count: 2,
        retained_slot_count: 2,
        retained_host_slot_count: 0,
    };
    let pool = SeqPool::create_with_speculation(&ModelConfig::qwen38_27b(), &budget, Some(SpeculativeBackend::Dflash2))
        .unwrap_or_else(|e| panic!("drafter pool: {e}"));

    for (shape, jobs) in [("two cuts", &TWO_CUTS[..]), ("a single cut", &SINGLE_CUT[..])] {
        assert_eq!(jobs.iter().map(|j| j.len).sum::<u32>(), PROMPT, "{shape}: the jobs cover the prompt");
        let today = run(&model, &pool, &prompt, &tail, jobs, false, &format!("{shape}, today's rule"));
        let floored = run(&model, &pool, &prompt, &tail, jobs, true, &format!("{shape}, the floor"));
        assert_eq!(today.prefix_window.is_some(), shape == "two cuts", "{shape}: where the prefix is published");
        compare(&today, &floored, shape);
    }

    drop(pool);
    drop(model);
    let _ = artifact.release_arena(&mut device);
}
