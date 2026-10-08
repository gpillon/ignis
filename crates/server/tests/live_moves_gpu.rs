//! A live move resumes bit-exact, on the card (spec vram-budget/03 AC 36, the
//! fixed branch; ADR 0045, GitHub #309): a greedy sequence reserved its whole
//! bound is moved off the device at two points of its generation and restored
//! each time, through KV-RAM and through KV-disk (windowed), and its tokens are
//! the ones it generates when nothing moves it.
//!
//! On the fixed branch only an admission moves a sequence, so the moves are
//! made the way serving makes them: A (`agent`, greedy, `ignore_eos`, an
//! explicit `max_tokens`) decodes, and an `interactive` request whose
//! reservation does not fit beside it arrives, once a third of the way into
//! A's tokens and once two thirds. Each time A leaves the device at a round
//! boundary, the arrival runs alone to its end, and A comes back and goes on.
//! A decodes alone throughout, in both runs, so every compared round is at
//! width 1 (finding 2026-09-14: batched decode width drift).
//!
//! - **KV-RAM leg.** The disk tier off and an arena that holds A's blob: each
//!   move is the synchronous snapshot and restore.
//! - **KV-disk leg.** No arena: each move is a windowed spill to a file and a
//!   windowed read back.
//!
//! The admission moving a decoding victim *through the server's own load* is
//! AC 23's first leg, `kv_disk_gpu.rs`. Machine-local artifacts
//! (`IGNIS_ARTIFACT_27B`, `IGNIS_FLASH_NEXT_DIR`); explicit GPU profile (ADR
//! 0006): a missing artifact or GPU is a skip outside it, a failure under it.
//! The files go under this checkout's `.scratch/kv-disk-gpu/` (or
//! `IGNIS_KV_DISK_TEST_DIR`), emptied after each leg.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_artifact::{FrontendSet, Reader};
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::scheduler::DiskSource;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{gpu_profile, ConcreteScheduler, Scheduler};
use ignis_server::runtime::{cuda_scheduler_with_thinking_close, flash_next_scheduler_with_ngram_cache, EngineShape};

const ARTIFACT_27B: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL_27B: &str = "qwen3.8-27b";
const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL_FLASH_NEXT: &str = "qwen3.8-flash-next";
/// Flash-Next's EOS (generation_config.json).
const EOS_FLASH_NEXT: u32 = 248_044;
const PAGE_TOKENS: u32 = 64;
const CONTEXT: u32 = 8192;
/// A's prompt and tokens: over half the context, so no arrival of the same
/// size fits beside it.
const A_PROMPT: u32 = 4400;
const A_TOKENS: u32 = 120;
const B_PROMPT: u32 = 4300;
const B_TOKENS: u32 = 16;
/// The arena of the KV-RAM leg: A's blob at this context is ~150 MB on
/// Flash-Next and ~270 MB on the 27B.
const ARENA_BYTES: u64 = 512 << 20;
/// No run takes longer than this; a move that never lands fails here instead
/// of spinning.
const MAX_RUN: std::time::Duration = std::time::Duration::from_secs(600);

#[derive(Debug, Clone, Copy)]
enum Family {
    Qwen27b,
    FlashNext,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    KvRam,
    KvDisk,
}

fn prompt(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 1000 + (i * 7919 + seed * 104_729) % 60_000).collect()
}

fn input(model: &str, tokens: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        model: model.into(),
        tokens,
        params: DecodeParams { max_tokens: Some(max_tokens), ignore_eos: true, ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        warm_up: false,
    }
}

/// Where a leg's files go; emptied before the leg and when it drops.
struct BlobDir(PathBuf);

impl BlobDir {
    fn new(leg: &str) -> Self {
        let root = std::env::var_os("IGNIS_KV_DISK_TEST_DIR").map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-gpu"),
            PathBuf::from,
        );
        let dir = root.join(leg);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the leg's blob directory");
        Self(dir)
    }
}

impl Drop for BlobDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The shape a leg loads: one context of pool (the floor the plan accepts),
/// no prompt reuse (a retained slot would only move pages around live work),
/// and the tier the leg moves through.
fn shape(family: Family, tier: Tier) -> EngineShape {
    let (host_pool_bytes, kv_disk_bytes) = match tier {
        Tier::KvRam => (ARENA_BYTES, 0),
        Tier::KvDisk => (0, 8 << 30),
    };
    let base = EngineShape {
        max_context: CONTEXT,
        host_pool_bytes,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        kv_disk_bytes: Some(kv_disk_bytes),
        kv_pool: Some(ignis_core::KvPoolSize::Tokens(u64::from(CONTEXT))),
        ..EngineShape::default()
    };
    match family {
        Family::Qwen27b => base,
        Family::FlashNext => EngineShape { decode_lanes: 2, prefill_chunk: 2048, ..base },
    }
}

fn load(family: Family, tier: Tier, blobs: &BlobDir) -> Option<(ConcreteScheduler, ignis_server::metrics::LoadReservations, String)> {
    let location = CacheLocation::Directory(blobs.0.clone());
    let loaded = match family {
        Family::Qwen27b => {
            let path = std::env::var_os("IGNIS_ARTIFACT_27B").map_or_else(|| PathBuf::from(ARTIFACT_27B), PathBuf::from);
            if !path.exists() && gpu_profile::skip_or_fail(&format!("no 27B artifact at {}", path.display())) {
                return None;
            }
            let reader = Reader::open(&path).expect("open the 27B artifact");
            let eos = FrontendSet::from_reader(&reader).expect("frontend").eos_token_id().expect("an eos token");
            drop(reader);
            cuda_scheduler_with_thinking_close(&path, MODEL_27B.into(), eos, shape(family, tier), None, &location)
                .map(|(s, r)| (s, r, MODEL_27B.to_owned()))
        }
        Family::FlashNext => {
            let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
            let path = dir.join(ARTIFACT_FILE_NAME);
            if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
                return None;
            }
            flash_next_scheduler_with_ngram_cache(
                &path,
                MODEL_FLASH_NEXT.into(),
                EOS_FLASH_NEXT,
                shape(family, tier),
                None,
                Default::default(),
                &location,
            )
            .map(|(s, r)| (s, r, MODEL_FLASH_NEXT.to_owned()))
        }
    };
    match loaded {
        Ok(loaded) => Some(loaded),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load {family:?} for the {tier:?} leg: {e}"));
            None
        }
    }
}

/// Every event of a run, each request's tokens in order, and when each
/// event came.
struct Run {
    events: Vec<(Instant, SchedEvent)>,
    tokens: HashMap<RequestId, Vec<u32>>,
    started: Instant,
}

impl Default for Run {
    fn default() -> Self {
        Self { events: Vec::new(), tokens: HashMap::new(), started: Instant::now() }
    }
}

impl Run {
    fn step(&mut self, sched: &mut ConcreteScheduler) {
        let events = sched.advance();
        let now = Instant::now();
        for event in events {
            if let SchedEvent::Token { request, token, .. } = &event {
                self.tokens.entry(*request).or_default().push(*token);
            }
            self.events.push((now, event));
        }
        if let Some(error) = sched.last_error() {
            panic!("the leaf failed a step: {error}");
        }
        if self.started.elapsed() >= MAX_RUN {
            let last: Vec<&SchedEvent> =
                self.events.iter().map(|(_, e)| e).filter(|e| !matches!(e, SchedEvent::Token { .. })).rev().take(12).collect();
            panic!(
                "the run never settled: disk busy {}, KV pages {}, the last events (newest first) {last:?}",
                sched.disk_busy(),
                sched.kv_used_pages()
            );
        }
    }

    fn to_idle(&mut self, sched: &mut ConcreteScheduler) {
        while !sched.is_idle() {
            self.step(sched);
        }
    }

    fn until(&mut self, sched: &mut ConcreteScheduler, done: impl Fn(&Self) -> bool) {
        while !done(self) {
            assert!(!sched.is_idle(), "the run went idle first");
            self.step(sched);
        }
    }

    fn tokens(&self, request: RequestId) -> &[u32] {
        self.tokens.get(&request).map_or(&[], Vec::as_slice)
    }

    fn count(&self, f: impl Fn(&SchedEvent) -> bool) -> usize {
        self.events.iter().filter(|(_, e)| f(e)).count()
    }

    /// When the first event `f` holds of came.
    fn when(&self, f: impl Fn(&SchedEvent) -> bool) -> Option<Instant> {
        self.events.iter().find(|(_, e)| f(e)).map(|(at, _)| *at)
    }

    fn has(&self, f: impl Fn(&SchedEvent) -> bool) -> bool {
        self.count(f) > 0
    }

    fn finished_at_length(&self, request: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Done { request: r, reason: FinishReason::Length, .. } if *r == request))
    }

    fn assert_no_work_lost(&self) {
        assert!(!self.has(|e| matches!(e, SchedEvent::SnapshotDropped { .. })), "a live snapshot was dropped");
        assert!(!self.has(|e| matches!(e, SchedEvent::Requeued { .. })), "a request lost its work");
        assert!(!self.has(|e| matches!(e, SchedEvent::DiskFailure { .. })), "a disk transfer failed");
    }
}

/// How many times `a` left the device, through `tier`.
fn moves_of(run: &Run, a: RequestId, tier: Tier) -> usize {
    match tier {
        Tier::KvRam => run.count(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == a)),
        Tier::KvDisk => run.count(|e| matches!(e, SchedEvent::DiskSpilled { request, from: DiskSource::Device } if *request == a)),
    }
}

fn restores_of(run: &Run, a: RequestId) -> usize {
    run.count(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a))
}

/// One leg: A alone, then A moved twice through `tier`.
fn a_live_move_resumes_bit_exact(family: Family, tier: Tier) {
    let blobs = BlobDir::new(&format!("live-moves-{family:?}-{tier:?}"));
    let Some((mut sched, reserved, model)) = load(family, tier, &blobs) else {
        return;
    };
    let pool = reserved.kv_pool_pages * PAGE_TOKENS;
    if pool >= A_PROMPT + A_TOKENS + B_PROMPT + B_TOKENS {
        gpu_profile::skip_or_fail(&format!(
            "the pool holds {pool} tokens, A and an arrival together: no move can be forced (the shape names one context)"
        ));
        return;
    }
    match tier {
        Tier::KvRam => assert!(reserved.kv_disk_bytes.is_none(), "the KV-RAM leg has no disk tier"),
        Tier::KvDisk => assert!(reserved.kv_disk_bytes.is_some(), "the disk leg has the tier"),
    }

    // A alone: the reference.
    let mut reference = Run::default();
    let a0 = sched.submit(input(&model, prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    reference.to_idle(&mut sched);
    assert!(reference.finished_at_length(a0));
    let want = reference.tokens(a0).to_vec();
    assert_eq!(want.len(), A_TOKENS as usize);

    // A again, moved at a third and at two thirds of its tokens.
    let mut run = Run::default();
    let a = sched.submit(input(&model, prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    let (mut at, mut out_ms, mut in_ms) = (Vec::new(), Vec::new(), Vec::new());
    for (k, point) in [A_TOKENS / 3, 2 * A_TOKENS / 3].into_iter().enumerate() {
        run.until(&mut sched, |run| run.tokens(a).len() >= point as usize);
        let asked = Instant::now();
        let b = sched
            .submit(input(&model, prompt(10 + k as u32, B_PROMPT), B_TOKENS), RequestClass::Interactive)
            .expect("submit the arrival");
        run.until(&mut sched, |run| moves_of(run, a, tier) > k);
        out_ms.push(asked.elapsed().as_millis());
        at.push(run.tokens(a).len());
        run.until(&mut sched, |run| restores_of(run, a) > k);
        let b_done = run
            .when(|e| matches!(e, SchedEvent::Done { request, .. } if *request == b))
            .expect("the arrival ended before A came back");
        in_ms.push(b_done.elapsed().as_millis());
        assert!(run.finished_at_length(b), "the arrival ran alone to its end before A came back");
        assert_eq!(run.tokens(b).len(), B_TOKENS as usize);
    }
    run.to_idle(&mut sched);

    assert_eq!(moves_of(&run, a, tier), 2, "A left the device twice through {tier:?}");
    assert_eq!(restores_of(&run, a), 2, "and came back twice");
    if tier == Tier::KvDisk {
        assert!(!run.has(|e| matches!(e, SchedEvent::Evicted { .. })), "KV-RAM is off: nothing went there");
    }
    assert!(run.finished_at_length(a));
    run.assert_no_work_lost();
    let got = run.tokens(a);
    let first_difference = want.iter().zip(got).position(|(w, g)| w != g);
    assert_eq!(
        got,
        want.as_slice(),
        "{family:?} {tier:?}: A's tokens across two moves differ from its lone run's (first at {first_difference:?})"
    );
    println!(
        "{family:?} {tier:?}: A's {} greedy tokens equal its lone run's across moves at tokens {at:?} \
         (out: the advance that reported A off the device returned {out_ms:?} ms after the arrival, its first \
         prefill chunk included; in: {in_ms:?} ms from the arrival's end to A back)",
        got.len()
    );
    assert_eq!(sched.kv_used_pages(), 0, "nothing is left charged");
}

#[test]
#[ignore = "GPU profile only: the real 27B artifact"]
fn qwen27b_a_live_move_resumes_bit_exact_through_kv_ram() {
    a_live_move_resumes_bit_exact(Family::Qwen27b, Tier::KvRam);
}

#[test]
#[ignore = "GPU profile only: the real 27B artifact"]
fn qwen27b_a_live_move_resumes_bit_exact_through_kv_disk() {
    a_live_move_resumes_bit_exact(Family::Qwen27b, Tier::KvDisk);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_a_live_move_resumes_bit_exact_through_kv_ram() {
    a_live_move_resumes_bit_exact(Family::FlashNext, Tier::KvRam);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact"]
fn flash_next_a_live_move_resumes_bit_exact_through_kv_disk() {
    a_live_move_resumes_bit_exact(Family::FlashNext, Tier::KvDisk);
}
