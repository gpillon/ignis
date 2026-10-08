//! KV-disk, Tier 2, on the card (spec vram-budget/03 ACs 23-24, ADR 0045):
//! a forced overflow moves a live sequence's real device state to a file and
//! back, and the request's greedy tokens are the ones it generates alone.
//!
//! - **Leg 1, bit-exact.** The pool holds one context and KV-RAM is off.
//!   Request A (`agent`, greedy, `ignore_eos`) runs alone first: its
//!   reference. Then the same A again, and B (`interactive`) arrives when A
//!   is mid-decode: neither fits beside the other, so A goes straight to the
//!   disk, B runs alone to its end, and A comes back from the file and
//!   finishes alone. Every decode round is at width 1 in both runs, so a
//!   width change cannot explain a difference (finding 2026-09-14).
//! - **Leg 2, the chain.** A KV-RAM arena that holds one blob. A and B
//!   (`agent`) decode together; C (`interactive`) needs both out. The first
//!   to leave lands in KV-RAM and is demoted to the disk to make room for the
//!   second. C runs alone, then A and B come back and finish; nothing is
//!   dropped or re-queued. C's tokens are its lone run's.
//!
//! The 27B runs both on this branch (it honours `kv_pool_bytes`). Flash-Next
//! needs its pool cut to one context, which #309 P1's pool policy brings:
//! until then its load holds both prompts and the overflow cannot be forced,
//! which the test says rather than passing.
//!
//! Machine-local artifacts (`IGNIS_ARTIFACT_27B`, `IGNIS_FLASH_NEXT_DIR`).
//! Explicit GPU profile (ADR 0006): a missing artifact or GPU is a skip
//! outside it, a failure under it. The files go under this checkout's
//! `.scratch/kv-disk-gpu/` (or `IGNIS_KV_DISK_TEST_DIR`), which the test
//! empties after itself.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

/// Which model a leg loads.
#[derive(Debug, Clone, Copy)]
enum Family {
    Qwen27b,
    FlashNext,
}

/// A distinct, deterministic prompt of `n` tokens.
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

/// Where a leg's files go; emptied before the leg and by [`BlobDir::drop`].
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

    fn location(&self) -> CacheLocation {
        CacheLocation::Directory(self.0.clone())
    }

    /// The per-process directories under the tier's own (AC 13).
    fn process_dirs(&self) -> Vec<PathBuf> {
        std::fs::read_dir(self.0.join(ignis_runtime::kv_disk::DIR_NAME))
            .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
            .unwrap_or_default()
    }
}

impl Drop for BlobDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A load of `family` with `shape` and the tier under `blobs`; `None` on a
/// skip.
fn load(family: Family, shape: EngineShape, blobs: &BlobDir) -> Option<(ConcreteScheduler, ignis_server::metrics::LoadReservations, String)> {
    let loaded = match family {
        Family::Qwen27b => {
            let path = std::env::var_os("IGNIS_ARTIFACT_27B").map_or_else(|| PathBuf::from(ARTIFACT_27B), PathBuf::from);
            if !path.exists() && gpu_profile::skip_or_fail(&format!("no 27B artifact at {}", path.display())) {
                return None;
            }
            let reader = Reader::open(&path).expect("open the 27B artifact");
            let eos = FrontendSet::from_reader(&reader).expect("frontend").eos_token_id().expect("an eos token");
            drop(reader);
            cuda_scheduler_with_thinking_close(&path, MODEL_27B.into(), eos, shape, None, &blobs.location())
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
                shape,
                None,
                Default::default(),
                &blobs.location(),
            )
            .map(|(s, r)| (s, r, MODEL_FLASH_NEXT.to_owned()))
        }
    };
    match loaded {
        Ok(loaded) => Some(loaded),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load {family:?}: {e}"));
            None
        }
    }
}

/// Every event of a run, and each request's tokens in order.
#[derive(Default)]
struct Run {
    events: Vec<SchedEvent>,
    tokens: HashMap<RequestId, Vec<u32>>,
}

impl Run {
    fn step(&mut self, sched: &mut ConcreteScheduler) {
        for event in sched.advance() {
            if let SchedEvent::Token { request, token, .. } = &event {
                self.tokens.entry(*request).or_default().push(*token);
            }
            self.events.push(event);
        }
    }

    fn to_idle(&mut self, sched: &mut ConcreteScheduler) {
        while !sched.is_idle() {
            self.step(sched);
        }
    }

    /// Advance until `request` has `n` tokens.
    fn until_tokens(&mut self, sched: &mut ConcreteScheduler, request: RequestId, n: usize) {
        while self.tokens.get(&request).map_or(0, Vec::len) < n {
            assert!(!sched.is_idle(), "{request} ended before {n} tokens");
            self.step(sched);
        }
    }

    fn tokens(&self, request: RequestId) -> &[u32] {
        self.tokens.get(&request).map_or(&[], Vec::as_slice)
    }

    fn has(&self, f: impl Fn(&SchedEvent) -> bool) -> bool {
        self.events.iter().any(f)
    }

    fn count(&self, f: impl Fn(&SchedEvent) -> bool) -> usize {
        self.events.iter().filter(|e| f(e)).count()
    }

    fn finished_at_length(&self, request: RequestId) -> bool {
        self.has(|e| matches!(e, SchedEvent::Done { request: r, reason: FinishReason::Length, .. } if *r == request))
    }

    /// Nothing live lost: no snapshot dropped, no request re-queued, no
    /// file refused (ADR 0045).
    fn assert_no_work_lost(&self) {
        assert!(!self.has(|e| matches!(e, SchedEvent::SnapshotDropped { .. })), "a live snapshot was dropped");
        assert!(!self.has(|e| matches!(e, SchedEvent::Requeued { .. })), "a request lost its work");
        assert!(!self.has(|e| matches!(e, SchedEvent::DiskFailure { .. })), "a disk transfer failed");
    }
}

/// The pool's tokens on this load.
fn pool_tokens(reserved: &ignis_server::metrics::LoadReservations) -> u32 {
    reserved.kv_pool_pages * PAGE_TOKENS
}

/// The shape both legs share: `max_context`, KV-RAM at `host_pool_bytes`,
/// no prompt reuse (these legs are about live work, and a retained slot
/// would only move pages around them), the tier on.
fn shape(family: Family, max_context: u32, host_pool_bytes: u64) -> EngineShape {
    let base = EngineShape {
        max_context,
        host_pool_bytes,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        kv_disk_bytes: Some(8 << 30),
        ..EngineShape::default()
    };
    match family {
        // The pool holds exactly one context: the floor the plan accepts.
        Family::Qwen27b => {
            let page_bytes = base.kv_format.page_bytes(ignis_core::KvGeometry::qwen38_27b());
            EngineShape { kv_pool_bytes: Some(u64::from(max_context.div_ceil(PAGE_TOKENS)) * page_bytes), ..base }
        }
        // Two lanes, the pool at its floor -- #309 P1's policy; on a tree
        // without it the load says so (see `forced_overflow_restores_bit_exact`).
        Family::FlashNext => EngineShape { decode_lanes: 2, prefill_chunk: 2048, ..base },
    }
}

/// What leg 1 measured of a live blob, which sizes leg 2's arena: the file
/// and the tokens its sequence held.
struct Spilled {
    file_bytes: u64,
    tokens: u32,
    kv_page_bytes: u64,
}

impl Spilled {
    /// A blob of `tokens` on the same load shape: the state image (the
    /// measured file less its header and its pages) plus `tokens`' pages.
    fn blob_bytes(&self, tokens: u32) -> u64 {
        let pages = |t: u32| u64::from(t.div_ceil(PAGE_TOKENS)) * self.kv_page_bytes;
        let image = (self.file_bytes - ignis_runtime::kv_disk::HEADER_BYTES as u64).saturating_sub(pages(self.tokens));
        image + pages(tokens)
    }
}

/// Leg 1 (AC 23, AC 24): the forced overflow, bit-exact.
fn forced_overflow_restores_bit_exact(family: Family) -> Option<Spilled> {
    const CONTEXT: u32 = 8192;
    const A_PROMPT: u32 = 4400;
    const A_TOKENS: u32 = 96;
    const B_PROMPT: u32 = 4300;
    const B_TOKENS: u32 = 32;
    let blobs = BlobDir::new(&format!("leg1-{family:?}"));
    let (mut sched, reserved, model) = load(family, shape(family, CONTEXT, 0), &blobs)?;
    assert!(reserved.kv_disk_bytes.is_some(), "the tier is on");
    let both = A_PROMPT + A_TOKENS + B_PROMPT + B_TOKENS;
    if pool_tokens(&reserved) >= both {
        gpu_profile::skip_or_fail(&format!(
            "the pool holds {} tokens, both requests' {both}: the overflow cannot be forced on this tree \
             (Flash-Next's pool at one context is #309 P1's)",
            pool_tokens(&reserved)
        ));
        return None;
    }
    assert_eq!(blobs.process_dirs().len(), 1, "the load holds its own directory");

    // A alone: the reference.
    let mut reference = Run::default();
    let a0 = sched.submit(input(&model, prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    reference.to_idle(&mut sched);
    assert!(reference.finished_at_length(a0));
    let want = reference.tokens(a0).to_vec();
    assert_eq!(want.len(), A_TOKENS as usize);

    // A again, and B when A is mid-decode.
    let mut run = Run::default();
    let a = sched.submit(input(&model, prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    run.until_tokens(&mut sched, a, (A_TOKENS / 3) as usize);
    let b = sched.submit(input(&model, prompt(2, B_PROMPT), B_TOKENS), RequestClass::Interactive).expect("submit B");
    let (mut file_bytes, mut generated_at_spill) = (0, None);
    while !sched.is_idle() {
        run.step(&mut sched);
        file_bytes = file_bytes.max(sched.disk_tier().map_or(0, |d| d.used_bytes()));
        if generated_at_spill.is_none()
            && run.has(|e| matches!(e, SchedEvent::DiskSpilled { request, from: DiskSource::Device } if *request == a))
        {
            generated_at_spill = Some(run.tokens(a).len() as u32);
        }
    }
    let generated_at_spill = generated_at_spill.expect("A went straight to the disk");
    assert!(!run.has(|e| matches!(e, SchedEvent::Evicted { .. })), "KV-RAM is off: nothing went there");
    assert!(run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a)), "A came back");
    assert!(run.finished_at_length(a) && run.finished_at_length(b));
    run.assert_no_work_lost();
    assert_eq!(run.tokens(b).len(), B_TOKENS as usize);
    let got = run.tokens(a);
    let first_difference = want.iter().zip(got).position(|(w, g)| w != g);
    assert_eq!(
        got,
        want.as_slice(),
        "{family:?}: A's tokens after the disk differ from its lone run's (first at {first_difference:?})"
    );
    println!(
        "{family:?} leg 1: A's {} greedy tokens equal its lone run's across a device -> disk -> device move \
         at token {generated_at_spill}; the file held {file_bytes} bytes",
        got.len()
    );
    assert_eq!(sched.disk_tier().map(|d| d.used_bytes()), Some(0), "A's file went once it landed");
    drop(sched);
    assert!(blobs.process_dirs().is_empty(), "a clean shutdown removes the process's directory");
    Some(Spilled { file_bytes, tokens: A_PROMPT + generated_at_spill, kv_page_bytes: reserved.kv_page_bytes })
}

/// Leg 2 (AC 23, AC 24): the chain, device -> KV-RAM -> disk.
fn the_chain_demotes_kv_ram_to_the_disk(family: Family, spilled: &Spilled) {
    const CONTEXT: u32 = 8192;
    const AB_PROMPT: u32 = 2450; // ~30% of the context
    const AB_TOKENS: u32 = 820; // ~10%
    const AB_BEFORE_C: u32 = 32;
    const C_PROMPT: u32 = 6550; // ~80%
    const C_TOKENS: u32 = 32;
    // An arena that holds A's or B's blob at C's arrival, and not both: if
    // it held none, nothing would land in KV-RAM; if two, nothing would be
    // demoted -- both are asserted below, so a wrong size fails loudly.
    let arena = spilled.blob_bytes(AB_PROMPT + AB_BEFORE_C + PAGE_TOKENS) * 3 / 2;
    let leg_shape = match family {
        Family::FlashNext => EngineShape { decode_lanes: 3, ..shape(family, CONTEXT, arena) },
        Family::Qwen27b => shape(family, CONTEXT, arena),
    };
    let blobs = BlobDir::new(&format!("leg2-{family:?}"));
    let Some((mut sched, _, model)) = load(family, leg_shape, &blobs) else {
        return;
    };

    // C alone first: its reference.
    let mut reference = Run::default();
    let c0 = sched.submit(input(&model, prompt(30, C_PROMPT), C_TOKENS), RequestClass::Interactive).expect("submit C");
    reference.to_idle(&mut sched);
    let want_c = reference.tokens(c0).to_vec();

    let mut run = Run::default();
    let a = sched.submit(input(&model, prompt(10, AB_PROMPT), AB_TOKENS), RequestClass::Agent).expect("submit A");
    let b = sched.submit(input(&model, prompt(20, AB_PROMPT), AB_TOKENS), RequestClass::Agent).expect("submit B");
    run.until_tokens(&mut sched, a, AB_BEFORE_C as usize);
    run.until_tokens(&mut sched, b, AB_BEFORE_C as usize);
    let c = sched.submit(input(&model, prompt(30, C_PROMPT), C_TOKENS), RequestClass::Interactive).expect("submit C");
    run.to_idle(&mut sched);

    let to_kv_ram = run.count(|e| matches!(e, SchedEvent::Evicted { request, .. } if *request == a || *request == b));
    let demoted = run.count(|e| matches!(e, SchedEvent::DiskSpilled { from: DiskSource::KvRam, .. }));
    println!("{family:?} leg 2: {to_kv_ram} left the device for KV-RAM, {demoted} demoted to the disk; arena {arena} bytes");
    assert!(to_kv_ram >= 1, "the first to leave landed in KV-RAM");
    assert!(demoted >= 1, "and was demoted to the disk to make room for the second");
    for (r, n) in [(a, AB_TOKENS), (b, AB_TOKENS), (c, C_TOKENS)] {
        assert!(run.finished_at_length(r), "{r} generated its full max_tokens");
        assert_eq!(run.tokens(r).len(), n as usize);
    }
    run.assert_no_work_lost();
    assert_eq!(run.tokens(c), want_c.as_slice(), "C decoded alone throughout: its lone run's tokens");
    drop(sched);
    assert!(blobs.process_dirs().is_empty(), "a clean shutdown removes the process's directory");
}

fn both_legs(family: Family) {
    if let Some(spilled) = forced_overflow_restores_bit_exact(family) {
        the_chain_demotes_kv_ram_to_the_disk(family, &spilled);
    }
}

#[test]
#[ignore = "GPU profile only: the real 27B artifact"]
fn qwen27b_a_forced_overflow_moves_through_the_disk_bit_exact_and_kv_ram_demotes_to_it() {
    both_legs(Family::Qwen27b);
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, and #309 P1's pool at the floor"]
fn flash_next_a_forced_overflow_moves_through_the_disk_bit_exact_and_kv_ram_demotes_to_it() {
    both_legs(Family::FlashNext);
}

/// AC 41: the 27B at its defaults has no tier -- no directory, nothing on
/// the load's reservations, so `/metrics` renders none of it.
#[test]
#[ignore = "GPU profile only: the real 27B artifact"]
fn qwen27b_at_its_defaults_opens_no_kv_disk() {
    let blobs = BlobDir::new("defaults-Qwen27b");
    let Some((sched, reserved, _)) = load(Family::Qwen27b, EngineShape::default(), &blobs) else {
        return;
    };
    assert_eq!(reserved.kv_disk_bytes, None);
    assert!(sched.disk_tier().is_none(), "the scheduler has no disk ledger");
    assert!(!blobs.0.join(ignis_runtime::kv_disk::DIR_NAME).exists(), "and no directory was made");
}
