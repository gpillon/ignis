//! KV-disk on a 27B load that drafts with its MTP head (GitHub #622, spec
//! #618), on the card: a forced overflow moves a live sequence's real device
//! state -- its pages with the head's layer, its hq residual window with the
//! head's ring words, its continuation hidden -- to a file and back, and the
//! request generates and drafts what it does alone.
//!
//! The pool holds one context and KV-RAM is off. Request A (`agent`, greedy,
//! `ignore_eos`) runs alone first: its reference, tokens and speculative
//! counters. Then the same A again, and B (`interactive`) arrives when A is
//! mid-decode: neither fits beside the other, so A goes straight to the disk,
//! B runs alone to its end, and A comes back from the file and finishes
//! alone. Every round is at width 1 in both runs.
//!
//! Driven through the scheduler and the runtime's compute seam, as
//! `ignis-server`'s `kv_disk_gpu.rs` drives a served load: the server still
//! refuses the MTP backend on the 27B (#623). The file names the load's
//! drafter, `mtp:7:full` (the CPU tests hold a file of another drafter to a
//! refusal by name). The files go under this checkout's
//! `.scratch/kv-disk-mtp-gpu/`, which the test empties after itself.
//!
//! Explicit GPU profile (ADR 0006): a missing artifact or GPU is a skip
//! outside it, a failure under it.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ignis_artifact::{bind_model_scope_27b, materialize, CudaDevice, FrontendSet, Reader};
use ignis_core::compute::{ModelConfig, ModelFamily};
use ignis_core::scheduler::DiskSource;
use ignis_core::types::{DecodeParams, FinishReason, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{
    gpu_profile, Compute, ConcreteScheduler, KvFormat, RopeScaling, Scheduler, SchedulerConfig, SpecCounters,
    Speculation, SpeculativeBackend,
};
use ignis_runtime::kv_disk::{self, DiskIdentity, DiskStore};
use ignis_runtime::{CudaLeaf, CudaLeafConfig, Model, RuntimeCompute, StagingWindow, TransferPace, KV_PAGE_TOKENS};

const ARTIFACT: &str = r"F:\ai\q38\ninfer-models\qwen3_8_27b_nvfp4full-v2.ninfer";
const MODEL: &str = "qwen3.8-27b";
const CONTEXT: u32 = 4096;
const CHUNK: u32 = 1024;
const A_PROMPT: u32 = 2200;
const A_TOKENS: u32 = 96;
const B_PROMPT: u32 = 2150;
const B_TOKENS: u32 = 32;

/// A distinct, deterministic prompt of `n` tokens.
fn prompt(seed: u32, n: u32) -> Vec<u32> {
    (0..n).map(|i| 1000 + (i * 7919 + seed * 104_729) % 60_000).collect()
}

fn input(tokens: Vec<u32>, max_tokens: u32) -> RequestInput {
    RequestInput {
        decision: None,
        model: MODEL.into(),
        tokens,
        params: DecodeParams { max_tokens: Some(max_tokens), ignore_eos: true, ..DecodeParams::default() },
        multimodal: None,
        opener_tokens: None,
        user_turn_tokens: None,
        system_block_tokens: None,
        reuse_boundaries: Vec::new(),
        constrained: None,
        forced_literal: None,
        grammar: None,
        warm_up: false,
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
        if let Some(error) = sched.last_error() {
            panic!("the leaf failed a step: {error}");
        }
    }

    fn to_idle(&mut self, sched: &mut ConcreteScheduler) {
        while !sched.is_idle() {
            self.step(sched);
        }
    }

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

    fn drafting(&self, request: RequestId) -> Option<SpecCounters> {
        self.events.iter().find_map(|e| match e {
            SchedEvent::Done { request: r, reason: FinishReason::Length, spec, .. } if *r == request => *spec,
            _ => None,
        })
    }
}

/// Where the leg's files go; emptied before the leg and on drop.
struct BlobDir(PathBuf);

impl BlobDir {
    fn new() -> Self {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-mtp-gpu");
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

#[test]
#[ignore = "GPU profile only: scripts/gpu-profile.ps1"]
fn a_live_sequence_moved_to_the_disk_and_back_generates_and_drafts_as_it_does_alone_on_an_mtp_load() {
    let path = Path::new(ARTIFACT);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("artifact absent: {ARTIFACT}")) {
        return;
    }
    let reader = Reader::open(path).unwrap_or_else(|e| panic!("open artifact: {e}"));
    let eos = FrontendSet::from_reader(&reader)
        .unwrap_or_else(|e| panic!("frontend: {e}"))
        .eos_token_id()
        .expect("an eos token");
    let speculation = Speculation::new(SpeculativeBackend::Mtp, 7).unwrap();
    let (plan, handles) = bind_model_scope_27b(&reader, ignis_core::model_load::draft_module(Some(speculation)))
        .unwrap_or_else(|e| panic!("bind with the MTP head: {e}"));
    let mut device = match CudaDevice::create(0) {
        Ok(device) => device,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("CUDA unavailable: {e}")) {
                return;
            }
            unreachable!();
        }
    };
    let artifact = match materialize(&reader, &plan, &mut device, None) {
        Ok(artifact) => artifact,
        Err(e) => {
            if gpu_profile::skip_or_fail(&format!("materialize text + mtp: {e}")) {
                return;
            }
            unreachable!();
        }
    };

    // The pool holds one context, and the pages an MTP pool's holders may
    // copy past the ledger (GitHub #622) -- which admission keeps back, so the
    // scheduler sees one context and cannot hold A and B together.
    let kv_format = KvFormat::default();
    let cfg = ModelConfig::qwen38_27b();
    let page_bytes = kv_format.page_bytes(ignis_core::seq::kv_geometry(&cfg, Some(SpeculativeBackend::Mtp)));
    let base = CudaLeafConfig::default();
    let reserve = ignis_core::seq::holder_page_reserve(&cfg, Some(SpeculativeBackend::Mtp), base.slot_count, 0);
    let context_pages = CONTEXT / KV_PAGE_TOKENS;
    let config = CudaLeafConfig {
        max_context_tokens: CONTEXT,
        kv_format,
        kv_pool_bytes: page_bytes * u64::from(context_pages + reserve),
        prefill_chunk_tokens: CHUNK,
        speculation: Some(speculation),
        retained_device_slots: 0,
        retained_host_slots: 0,
        ..base
    };
    let lanes = config.slot_count;
    let leaf = CudaLeaf::new(device, reader, artifact, handles, config);
    let model = Arc::new(Model::load(Arc::new(leaf)).unwrap_or_else(|e| panic!("model load: {e:?}")));
    let compute =
        RuntimeCompute::new(model, eos).with_transfer_pace(TransferPace::for_family(ModelFamily::Qwen38_27b));
    let identity = DiskIdentity::of_load(MODEL, &compute.blob_identity(), RopeScaling::default(), None);
    assert_eq!(identity.drafter, "mtp:7:full", "the disk files name the load's drafter");
    let blobs = BlobDir::new();
    let store = DiskStore::open_on_volume(&blobs.0, 8 << 30, identity, Default::default())
        .unwrap_or_else(|e| panic!("open the disk tier: {e}"))
        .unwrap_or_else(|| panic!("the volume under {} has no room for the tier", blobs.0.display()));
    let budget_bytes = store.budget_bytes();
    let arena = ignis_core::seq::HostArena::create(kv_disk::STAGING_BYTES).unwrap_or_else(|e| panic!("staging: {e}"));
    let staging: Vec<StagingWindow> = (0..kv_disk::STAGING_WINDOWS)
        .map(|_| Box::new(arena.alloc(kv_disk::WINDOW_BYTES).unwrap_or_else(|e| panic!("staging window: {e}"))) as StagingWindow)
        .collect();
    let compute = compute.with_kv_disk(store, staging);
    let mut sched = ConcreteScheduler::with_config(
        SchedulerConfig {
            model: MODEL.into(),
            max_in_flight: lanes as usize,
            max_prefill_batch: lanes as usize,
            resident_slot_capacity: lanes,
            kv_page_tokens: KV_PAGE_TOKENS,
            max_sequence_tokens: CONTEXT,
            kv_capacity_pages: context_pages,
            host_capacity_bytes: 0,
            serving_chunk_tokens: CHUNK,
            prompt_reuse: false,
            retained_slots: 0,
            kv_disk_capacity_bytes: budget_bytes,
            kv_disk_restore_floor_tokens: kv_disk::restore_floor_tokens(ModelFamily::Qwen38_27b),
            ..SchedulerConfig::default()
        },
        Arc::new(compute),
    );

    // A alone: the reference.
    let mut reference = Run::default();
    let a0 = sched.submit(input(prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    reference.to_idle(&mut sched);
    let want = reference.tokens(a0).to_vec();
    assert_eq!(want.len(), A_TOKENS as usize);
    let want_drafts = reference.drafting(a0).expect("A's reference ran verify rounds");
    assert!(want_drafts.accepted > 0, "the head landed no draft alone: {want_drafts:?}");

    // A again, and B when A is mid-decode.
    let mut run = Run::default();
    let a = sched.submit(input(prompt(1, A_PROMPT), A_TOKENS), RequestClass::Agent).expect("submit A");
    run.until_tokens(&mut sched, a, (A_TOKENS / 3) as usize);
    let b = sched.submit(input(prompt(2, B_PROMPT), B_TOKENS), RequestClass::Interactive).expect("submit B");
    let mut file_bytes = 0;
    while !sched.is_idle() {
        run.step(&mut sched);
        file_bytes = file_bytes.max(sched.disk_tier().map_or(0, |d| d.used_bytes()));
    }
    assert!(
        run.has(|e| matches!(e, SchedEvent::DiskSpilled { request, from: DiskSource::Device } if *request == a)),
        "A went straight to the disk"
    );
    assert!(!run.has(|e| matches!(e, SchedEvent::Evicted { .. })), "KV-RAM is off: nothing went there");
    assert!(run.has(|e| matches!(e, SchedEvent::Restored { request, .. } if *request == a)), "A came back");
    assert!(!run.has(|e| matches!(e, SchedEvent::Requeued { .. })), "no request lost its work");
    assert!(!run.has(|e| matches!(e, SchedEvent::DiskFailure { .. })), "no disk transfer failed");
    assert_eq!(run.tokens(b).len(), B_TOKENS as usize);
    let got = run.tokens(a);
    let first_difference = want.iter().zip(got).position(|(w, g)| w != g);
    assert_eq!(got, want.as_slice(), "A's tokens after the disk differ from its lone run's (first at {first_difference:?})");
    let got_drafts = run.drafting(a).expect("A ran verify rounds");
    assert_eq!(got_drafts, want_drafts, "A drafts across the disk as it does alone: the head's state came back");
    println!(
        "A's {} greedy tokens and {} rounds ({}/{} drafts accepted) equal its lone run's across a device -> disk -> \
         device move; the file held {file_bytes} bytes",
        got.len(),
        got_drafts.rounds,
        got_drafts.accepted,
        got_drafts.drafted
    );
    assert_eq!(sched.disk_tier().map(|d| d.used_bytes()), Some(0), "A's file went once it landed");
}
