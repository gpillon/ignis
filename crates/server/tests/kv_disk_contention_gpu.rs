//! KV-disk contention on the volume the n-gram table is read from, measured
//! (spec vram-budget/03 AC 25, ADR 0045): a >= 1 GB live blob spilled to the
//! disk twice on Flash-Next, once while a prompt prefills against an
//! uncovered n-gram table (every row read from the artifact) and once while
//! two lanes decode, each against the same work without the spill.
//!
//! - **Prefill.** F (16,384 tokens, two chunks) prefills alone; then again
//!   with G arriving to move A2 (another ~1 GB `agent`) to the disk. Each
//!   chunk's wall time (the advance that ran it) and its n-gram gather time
//!   are recorded, and whether the spill was in flight on both sides of it:
//!   G waits for the prefill F holds, so none is (2026-10-08). The volume's
//!   share of a chunk is measured by `ignis-runtime`'s
//!   `kv_disk_volume_contention`.
//! - **Decode.** B and C (`interactive`) decode at width 2 alone; then again
//!   while A is written to the disk to make room for E. Their inter-token
//!   latencies over the spill window give p50 and p99.
//!
//! Starting bounds, for the owner to confirm: chunk wall time and ITL p50
//! each within +10% of the run without the spill. The test records and
//! prints; it asserts only that each spill happened and that the run lost
//! no work. The figures go in a finding. `IGNIS_KV_P2_RAW` names a file the
//! raw samples are written to as JSON.
//!
//! The pool is cut to one context (`--kv-pool-bytes`'s token form, #309 P1)
//! so that E cannot fit beside A. Machine-local: the Flash-Next artifact
//! (`IGNIS_FLASH_NEXT_DIR`), its files on F:.

#![cfg(feature = "cuda")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ignis_artifact::packer::ARTIFACT_FILE_NAME;
use ignis_core::ngram_cache::CacheLocation;
use ignis_core::ngram_table::HotBudget;
use ignis_core::scheduler::DiskSource;
use ignis_core::types::{DecodeParams, RequestClass, RequestId, RequestInput, SchedEvent};
use ignis_core::{gpu_profile, ConcreteScheduler, Scheduler};
use ignis_server::runtime::{flash_next_scheduler_with_ngram_cache, EngineShape};

const FLASH_NEXT_DIR: &str = "F:/ai/models/Qwen3.8-Flash-Next-ignis";
const MODEL: &str = "qwen3.8-flash-next";
const EOS: u32 = 248_044;
const CONTEXT: u32 = 262_144;
const CHUNK: u32 = 8192;
/// ~1 GB of Flash-Next state: a 130 MB image and 4,224 bytes a token.
const A_PROMPT: u32 = 210_000;

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
        warm_up: false,
    }
}

/// One advance's wall time, its events, and the n-gram gather time it took.
struct Step {
    wall: Duration,
    gather: Duration,
    events: Vec<SchedEvent>,
    busy_before: bool,
    busy_after: bool,
    at: Instant,
}

struct Rig {
    sched: ConcreteScheduler,
    gather_nanos: Box<dyn Fn() -> u64>,
    steps: Vec<Step>,
}

impl Rig {
    fn step(&mut self) -> &Step {
        let busy_before = self.sched.disk_busy();
        let gather_before = (self.gather_nanos)();
        let started = Instant::now();
        let events = self.sched.advance();
        let wall = started.elapsed();
        let gather = Duration::from_nanos((self.gather_nanos)() - gather_before);
        let busy_after = self.sched.disk_busy();
        self.steps.push(Step { wall, gather, events, busy_before, busy_after, at: Instant::now() });
        self.steps.last().unwrap()
    }

    fn to_idle(&mut self) {
        while !self.sched.is_idle() {
            self.step();
        }
    }

    fn tokens_of(&self, request: RequestId) -> usize {
        self.steps
            .iter()
            .flat_map(|s| &s.events)
            .filter(|e| matches!(e, SchedEvent::Token { request: r, .. } if *r == request))
            .count()
    }

    fn until_tokens(&mut self, request: RequestId, n: usize) {
        while self.tokens_of(request) < n {
            assert!(!self.sched.is_idle(), "{request} ended early");
            self.step();
        }
    }

    fn mark(&self) -> usize {
        self.steps.len()
    }

    /// Each chunk of `request` after `from`: (wall, gather, under a spill).
    fn chunks(&self, from: usize, request: RequestId) -> Vec<(Duration, Duration, bool)> {
        self.steps[from..]
            .iter()
            .filter(|s| s.events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { request: r, .. } if *r == request)))
            .map(|s| (s.wall, s.gather, s.busy_before && s.busy_after))
            .collect()
    }

    /// The gaps between consecutive tokens of `requests` after `from`,
    /// while `keep` says of the step.
    fn itl(&self, from: usize, requests: &[RequestId], keep: impl Fn(&Step) -> bool) -> Vec<Duration> {
        let mut last: HashMap<RequestId, Instant> = HashMap::new();
        let mut gaps = Vec::new();
        for step in &self.steps[from..] {
            for event in &step.events {
                if let SchedEvent::Token { request, .. } = event {
                    if requests.contains(request) {
                        if let Some(previous) = last.insert(*request, step.at) {
                            if keep(step) {
                                gaps.push(step.at - previous);
                            }
                        }
                    }
                }
            }
        }
        gaps
    }

    fn spilled_at(&self, from: usize, request: RequestId) -> Option<usize> {
        self.steps[from..]
            .iter()
            .position(|s| s.events.iter().any(|e| matches!(e, SchedEvent::DiskSpilled { request: r, from: DiskSource::Device } if *r == request)))
            .map(|i| from + i)
    }
}

fn quantile(samples: &[Duration], q: f64) -> f64 {
    let mut ms: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if ms.is_empty() {
        return f64::NAN;
    }
    ms[((ms.len() - 1) as f64 * q).round() as usize]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
#[ignore = "GPU profile only: the real Flash-Next artifact, ~5 min"]
fn a_one_gigabyte_spill_beside_a_prefill_and_beside_two_decoding_lanes() {
    let dir = std::env::var_os("IGNIS_FLASH_NEXT_DIR").map_or_else(|| PathBuf::from(FLASH_NEXT_DIR), PathBuf::from);
    let path = dir.join(ARTIFACT_FILE_NAME);
    if !path.exists() && gpu_profile::skip_or_fail(&format!("no Flash-Next artifact at {}", path.display())) {
        return;
    }
    let blobs = std::env::var_os("IGNIS_KV_DISK_TEST_DIR")
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch/kv-disk-gpu"), PathBuf::from)
        .join("contention");
    let _ = std::fs::remove_dir_all(&blobs);
    std::fs::create_dir_all(&blobs).unwrap();
    let shape = EngineShape {
        max_context: CONTEXT,
        prefill_chunk: CHUNK,
        // Four in flight: A, B and C, and the arrival that moves A.
        decode_lanes: 4,
        host_pool_bytes: 0,
        prompt_reuse: false,
        retained_device_slots: 0,
        retained_host_slots: 0,
        retained_host_named: true,
        // Uncovered: every row a prefill stages is read from the artifact.
        ngram_hot_bytes: HotBudget::Bytes(0),
        kv_disk_bytes: Some(8 << 30),
        // The pool at one context (#309 P1's token form): A, B, C and E
        // cannot all fit, so E moves A.
        kv_pool: Some(ignis_core::KvPoolSize::Tokens(u64::from(CONTEXT))),
        ..EngineShape::default()
    };
    let (sched, reserved) = match flash_next_scheduler_with_ngram_cache(
        &path,
        MODEL.into(),
        EOS,
        shape,
        None,
        ignis_core::ngram_cache::PersistenceOptions { enabled: false, ..Default::default() },
        &CacheLocation::Directory(blobs.clone()),
    ) {
        Ok(loaded) => loaded,
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("load the Flash-Next scheduler: {e}"));
            return;
        }
    };
    let pool = reserved.kv_pool_pages * 64;
    // A, B, C and E at once: what the pool must not hold for A to move.
    if pool >= (A_PROMPT + 400) + 2 * (2000 + 400) + (50_000 + 16) {
        gpu_profile::skip_or_fail(&format!(
            "the pool holds {pool} tokens: the spill cannot be forced (the shape names a pool of one context)"
        ));
        return;
    }
    let counters = reserved.flash_next.clone().expect("a Flash-Next load's counters");
    let mut rig = Rig { sched, gather_nanos: Box::new(move || counters.ngram_prefill_gather_nanos()), steps: Vec::new() };

    // ── decode: B and C at width 2, alone, then beside A's spill ──────────
    let from = rig.mark();
    let b0 = rig.sched.submit(input(prompt(2, 2000), 200), RequestClass::Interactive).unwrap();
    let c0 = rig.sched.submit(input(prompt(3, 2000), 200), RequestClass::Interactive).unwrap();
    rig.to_idle();
    // Steady decode only: past both prompts' prefill.
    let baseline_itl = rig.itl(from, &[b0, c0], |s| {
        !s.events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { .. }))
    });

    let a = rig.sched.submit(input(prompt(1, A_PROMPT), 400), RequestClass::Agent).unwrap();
    rig.until_tokens(a, 4);
    let b = rig.sched.submit(input(prompt(2, 2000), 400), RequestClass::Interactive).unwrap();
    let c = rig.sched.submit(input(prompt(3, 2000), 400), RequestClass::Interactive).unwrap();
    rig.until_tokens(b, 8);
    rig.until_tokens(c, 8);
    let from = rig.mark();
    let e = rig.sched.submit(input(prompt(4, 50_000), 16), RequestClass::Interactive).unwrap();
    while rig.spilled_at(from, a).is_none() {
        rig.step();
    }
    // The spill's window: from E's arrival to the step its file committed
    // in, less a step that also ran a prefill chunk -- E's first starts in
    // the step that gave it the room, and its seconds are E's, not the
    // spill's.
    let spilled_at = rig.spilled_at(from, a).unwrap();
    let decode_only = |s: &Step| !s.events.iter().any(|e| matches!(e, SchedEvent::PrefillChunk { .. }));
    let spill_window: Duration = rig.steps[from..=spilled_at].iter().filter(|s| decode_only(s)).map(|s| s.wall).sum();
    let spill_itl = {
        let (first, last) = (rig.steps[from].at, rig.steps[spilled_at].at);
        rig.itl(from, &[b, c], |s| s.at >= first && s.at <= last && decode_only(s))
    };
    rig.to_idle();
    assert!(rig.tokens_of(e) > 0, "E ran once A was on the disk");

    // ── prefill: F alone, then with G arriving to move A2 ─────────────────
    // What the run records is whether any of F's chunks ran with the spill
    // in flight: G, the arrival that needs the room, waits for the prefill
    // F holds, so the spill starts only once F's prefill is done. The
    // volume's share of a chunk -- its n-gram gather beside the tier's
    // writes -- is `ignis-runtime`'s `kv_disk_volume_contention`.
    let from = rig.mark();
    let f0 = rig.sched.submit(input(prompt(5, 16_384), 1), RequestClass::Interactive).unwrap();
    rig.to_idle();
    let baseline_chunks = rig.chunks(from, f0);

    // A2's budget outlasts F's prefill, so it is still on the device when G
    // needs the room; it is cancelled once G is done.
    let a2 = rig.sched.submit(input(prompt(6, A_PROMPT), 8000), RequestClass::Agent).unwrap();
    rig.until_tokens(a2, 4);
    let from = rig.mark();
    let f = rig.sched.submit(input(prompt(5, 16_384), 1), RequestClass::Interactive).unwrap();
    let g = rig.sched.submit(input(prompt(7, 40_000), 1), RequestClass::Interactive).unwrap();
    while rig.tokens_of(g) == 0 {
        assert!(!rig.sched.is_idle(), "G never ran");
        rig.step();
    }
    rig.sched.cancel(a2);
    rig.to_idle();
    let spill_chunks = rig.chunks(from, f);
    let a2_spilled = rig.spilled_at(from, a2).is_some();
    assert!(rig.tokens_of(g) > 0, "G ran once A was on the disk");

    // ── what happened ─────────────────────────────────────────────────────
    let lost = rig.steps.iter().flat_map(|s| &s.events).any(|e| {
        matches!(e, SchedEvent::Requeued { .. } | SchedEvent::SnapshotDropped { .. } | SchedEvent::DiskFailure { .. })
    });
    let under: Vec<_> = spill_chunks.iter().filter(|c| c.2).collect();
    println!("decode, width 2: baseline ITL p50 {:.2} ms p99 {:.2} ms over {} gaps", quantile(&baseline_itl, 0.5), quantile(&baseline_itl, 0.99), baseline_itl.len());
    println!(
        "decode beside a {A_PROMPT}-token spill ({:.0} ms window): ITL p50 {:.2} ms p99 {:.2} ms over {} gaps",
        ms(spill_window),
        quantile(&spill_itl, 0.5),
        quantile(&spill_itl, 0.99),
        spill_itl.len()
    );
    for (i, (wall, gather, _)) in baseline_chunks.iter().enumerate() {
        println!("prefill chunk {i} alone: wall {:.1} ms, n-gram gather {:.1} ms", ms(*wall), ms(*gather));
    }
    for (i, (wall, gather, busy)) in spill_chunks.iter().enumerate() {
        println!("prefill chunk {i} beside the spill: wall {:.1} ms, n-gram gather {:.1} ms, spill in flight {busy}", ms(*wall), ms(*gather));
    }
    if let Some(raw) = std::env::var_os("IGNIS_KV_P2_RAW") {
        let list = |v: &[Duration]| v.iter().map(|d| format!("{:.3}", ms(*d))).collect::<Vec<_>>().join(",");
        let chunks = |v: &[(Duration, Duration, bool)]| {
            v.iter().map(|(w, g, b)| format!("[{:.3},{:.3},{b}]", ms(*w), ms(*g))).collect::<Vec<_>>().join(",")
        };
        let json = format!(
            "{{\"a_prompt_tokens\":{A_PROMPT},\"spill_window_ms\":{:.3},\"baseline_itl_ms\":[{}],\"spill_itl_ms\":[{}],\"baseline_chunks\":[{}],\"spill_chunks\":[{}]}}\n",
            ms(spill_window),
            list(&baseline_itl),
            list(&spill_itl),
            chunks(&baseline_chunks),
            chunks(&spill_chunks)
        );
        std::fs::write(&raw, json).expect("write the raw samples");
    }
    drop(rig);
    let _ = std::fs::remove_dir_all(&blobs);
    assert!(!lost, "a spill lost work");
    assert!(a2_spilled, "A2 went to the disk to make room for G");
    println!("F's chunks with A2's spill in flight: {}", under.len());
    assert!(!spill_itl.is_empty(), "B and C decoded during A's spill");
}
